'''
Unrestricted RE-BWs2

Combines Retaining-the-Excitation-Degree (RE) perturbation theory with
size-consistent Brillouin-Wigner second-order perturbation theory (BWs2)
for open-shell (UHF) references.

Hamiltonian partitioning:
    H^(0) = H^(0)_MP + alpha * H^(0)_BWs + beta * H^(0)_DeltaRE

Three amplitude blocks are solved simultaneously:
    t2aa[i,a,j,b] : alpha-alpha, fully antisymmetric
    t2ab[i,a,J,B] : alpha-beta,  non-antisymmetric
    t2bb[I,A,J,B] : beta-beta,   fully antisymmetric

Working equations: pyscf/rebws/the_method.md
'''

import numpy
from pyscf import ao2mo, lib
from pyscf.lib import logger


DIIS_SPACE       = 10
DIIS_START_CYCLE = 0


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------

def get_nocc(urebws):
    if urebws._nocc is not None:
        return urebws._nocc
    nocca = int(numpy.count_nonzero(urebws.mo_occ[0] > 0))
    noccb = int(numpy.count_nonzero(urebws.mo_occ[1] > 0))
    return nocca, noccb


def get_nmo(urebws):
    if urebws._nmo is not None:
        return urebws._nmo
    return urebws.mo_occ[0].size, urebws.mo_occ[1].size


def get_e_hf(urebws, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = urebws.mo_coeff
    if mo_coeff is urebws._scf.mo_coeff and urebws._scf.converged:
        return urebws._scf.e_tot
    dm  = urebws._scf.make_rdm1(mo_coeff, urebws.mo_occ)
    vhf = urebws._scf.get_veff(urebws.mol, dm)
    return urebws._scf.energy_tot(dm=dm, vhf=vhf)


# ---------------------------------------------------------------------------
# ERI storage and transformation
# ---------------------------------------------------------------------------

def _mem_usage(nocca, nvira, noccb, nvirb):
    '''Estimate memory (MB) for all 13 ERI blocks.'''
    aa  = (nocca * nvira)**2 + (nocca * nvira)**2 + nocca**4 + nvira**4
    bb  = (noccb * nvirb)**2 + (noccb * nvirb)**2 + noccb**4 + nvirb**4
    ab  = nocca * nvira * noccb * nvirb
    mix = (nocca**2 * nvirb**2 + noccb**2 * nvira**2
           + nocca**2 * noccb**2 + nvira**2 * nvirb**2)
    return (aa + bb + ab + mix) * 8.0 / 1e6


class _ChemistsERIs:
    '''Stores all 13 ERI blocks required by UREBWS2.

    Index conventions (chemist notation):
        α-α:  ovov[i,a,j,b]=(ia|jb)  oovv[i,j,a,b]=(ij|ab)
              oooo[i,j,k,l]=(ij|kl)  vvvv[a,b,c,d]=(ab|cd)
        β-β:  OVOV[I,A,J,B]=(IA|JB)  OOVV[I,J,A,B]=(IJ|AB)
              OOOO[I,J,K,L]=(IJ|KL)  VVVV[A,B,C,D]=(AB|CD)
        mixed:
              ovOV[i,a,J,B]=(ia|JB)   H intermediates + cross-spin R_aa/R_bb
              ooVV[i,j,A,B]=(ij|AB)   R_ab L6
              OOvv[I,J,a,b]=(IJ|ab)   R_ab L7
              ooOO[i,j,I,J]=(ij|IJ)   R_ab L9
              vvVV[a,b,A,B]=(ab|AB)   R_ab L14
    '''

    def __init__(self):
        self.mol       = None
        self.mo_coeff  = None
        self.mo_energy = None   # tuple (mo_ea, mo_eb)
        self.fock      = None   # tuple (focka, fockb)
        self.ovov = self.oovv = self.oooo = self.vvvv = None
        self.OVOV = self.OOVV = self.OOOO = self.VVVV = None
        self.ovOV = self.ooVV = self.OOvv = self.ooOO = self.vvVV = None

    def _common_init_(self, urebws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = urebws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. Run mf.kernel() first.')
        mo_a, mo_b = mo_coeff
        self.mo_coeff = (mo_a, mo_b)
        self.mol = urebws.mol

        if mo_coeff is urebws._scf.mo_coeff and urebws._scf.converged:
            self.mo_energy = (urebws._scf.mo_energy[0], urebws._scf.mo_energy[1])
            self.fock = (numpy.diag(self.mo_energy[0]),
                         numpy.diag(self.mo_energy[1]))
        else:
            dm     = urebws._scf.make_rdm1(mo_coeff, urebws.mo_occ)
            vhf    = urebws._scf.get_veff(urebws.mol, dm)
            fockao = urebws._scf.get_fock(vhf=vhf, dm=dm)
            focka  = mo_a.conj().T @ fockao[0] @ mo_a
            fockb  = mo_b.conj().T @ fockao[1] @ mo_b
            self.fock      = (focka, fockb)
            self.mo_energy = (focka.diagonal().real, fockb.diagonal().real)
        return self


def _make_eris(urebws, mo_coeff=None, verbose=None):
    '''Transform AO ERIs to all 13 MO blocks via 3 full AO->MO transforms.

    eri_aa  (α-α-α-α) → ovov, oovv, oooo, vvvv
    eri_ab  (α-α-β-β) → ovOV, ooVV, ooOO, vvVV;  OOvv via transpose
    eri_bb  (β-β-β-β) → OVOV, OOVV, OOOO, VVVV
    '''
    log   = logger.new_logger(urebws, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris  = _ChemistsERIs()
    eris._common_init_(urebws, mo_coeff)
    mo_a, mo_b = eris.mo_coeff

    nocca, noccb = urebws.get_nocc()
    nmoa,  nmob  = urebws.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb

    mem_total = _mem_usage(nocca, nvira, noccb, nvirb)
    mem_now   = lib.current_memory()[0]
    if mem_total > max(0, urebws.max_memory - mem_now):
        log.warn('ERI blocks require %.1f MB but only %.1f MB is available.',
                 mem_total, max(0, urebws.max_memory - mem_now))

    log.debug('Transforming ERI blocks: eri_aa, eri_ab, eri_bb')

    # α-α blocks
    eri_aa = ao2mo.restore(1, ao2mo.full(urebws.mol, mo_a, verbose=verbose), nmoa)
    eris.ovov = numpy.asarray(eri_aa[:nocca, nocca:, :nocca, nocca:])
    eris.oovv = numpy.asarray(eri_aa[:nocca, :nocca, nocca:, nocca:])
    eris.oooo = numpy.asarray(eri_aa[:nocca, :nocca, :nocca, :nocca])
    eris.vvvv = numpy.asarray(eri_aa[nocca:, nocca:, nocca:, nocca:])
    eri_aa = None

    # β-β blocks
    eri_bb = ao2mo.restore(1, ao2mo.full(urebws.mol, mo_b, verbose=verbose), nmob)
    eris.OVOV = numpy.asarray(eri_bb[:noccb, noccb:, :noccb, noccb:])
    eris.OOVV = numpy.asarray(eri_bb[:noccb, :noccb, noccb:, noccb:])
    eris.OOOO = numpy.asarray(eri_bb[:noccb, :noccb, :noccb, :noccb])
    eris.VVVV = numpy.asarray(eri_bb[noccb:, noccb:, noccb:, noccb:])
    eri_bb = None

    # mixed α-β blocks: eri_ab[p,q,P,Q] = (pq|PQ), p,q ∈ α-MOs, P,Q ∈ β-MOs
    eri_ab = (ao2mo.general(urebws.mol, (mo_a, mo_a, mo_b, mo_b), compact=False)
              .reshape(nmoa, nmoa, nmob, nmob))
    eris.ovOV = numpy.asarray(eri_ab[:nocca, nocca:, :noccb, noccb:])
    eris.ooVV = numpy.asarray(eri_ab[:nocca, :nocca, noccb:, noccb:])
    eris.ooOO = numpy.asarray(eri_ab[:nocca, :nocca, :noccb, :noccb])
    eris.vvVV = numpy.asarray(eri_ab[nocca:, nocca:, noccb:, noccb:])
    # OOvv[I,J,a,b] = (IJ|ab) = (ab|IJ) = eri_ab[a,b,I,J]  (8-fold symmetry)
    eris.OOvv = numpy.asarray(
        eri_ab[nocca:, nocca:, :noccb, :noccb].transpose(2, 3, 0, 1))
    eri_ab = None

    log.timer('UREBWS2 ERI transformation', *cput0)
    return eris


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def init_amps(urebws, eris):
    '''UMP2 amplitudes as the UREBWS2 starting point.

    Same-spin amplitudes are fully antisymmetric:
        t2aa = (ovov - ovov.T(0,3,2,1)) / D_aa
        t2bb = (OVOV - OVOV.T(0,3,2,1)) / D_bb

    Opposite-spin amplitude is non-antisymmetric:
        t2ab = ovOV / D_ab
    '''
    nocca, noccb = urebws.get_nocc()
    nmoa,  nmob  = urebws.get_nmo()

    eps_oa = eris.mo_energy[0][:nocca]
    eps_va = eris.mo_energy[0][nocca:]
    eps_ob = eris.mo_energy[1][:noccb]
    eps_vb = eris.mo_energy[1][noccb:]

    eia_a = eps_oa[:, None] - eps_va[None, :]   # (nocca, nvira), negative
    eia_b = eps_ob[:, None] - eps_vb[None, :]   # (noccb, nvirb), negative
    D_aa  = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
    D_ab  = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
    D_bb  = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)

    ovov = numpy.asarray(eris.ovov)
    ovOV = numpy.asarray(eris.ovOV)
    OVOV = numpy.asarray(eris.OVOV)

    t2aa = (ovov - ovov.transpose(0, 3, 2, 1)) / D_aa
    t2ab = ovOV / D_ab
    t2bb = (OVOV - OVOV.transpose(0, 3, 2, 1)) / D_bb
    t2   = (t2aa, t2ab, t2bb)
    return energy(urebws, t2, (ovov, ovOV, OVOV)), t2


def energy(urebws, t2, ovovs):
    '''UREBWS2 correlation energy.

    E_aa = (1/2) sum_{iajb} t2aa * ((ia|jb) - (ib|ja))
    E_ab = sum_{iaJB} t2ab * (ia|JB)
    E_bb = (1/2) sum_{IAJB} t2bb * ((IA|JB) - (IB|JA))
    '''
    t2aa, t2ab, t2bb = t2
    ovov, ovOV, OVOV = ovovs
    E_aa = 0.5 * numpy.einsum('iajb,iajb->', t2aa, ovov).real
    E_bb = 0.5 * numpy.einsum('iajb,iajb->', t2bb, OVOV).real
    E_ab = numpy.einsum('iaJB,iaJB->', t2ab, ovOV).real
    return E_aa + E_ab + E_bb


def _compute_H_aa(t2aa, t2ab, ovov, ovOV, eps_oa, alpha):
    '''Build the (nocca, nocca) alpha H intermediate.

    From the_method.md H^i_j:
        X_aa[i,j] = sum_{k,a,b} t2aa[i,a,k,b] * ovov[j,b,k,a]   (jb|ka)
        Y_aa[i,j] = sum_{k,a,b} t2aa[i,a,k,b] * ovov[j,a,k,b]   (ja|kb)
        Z_ab[i,j] = sum_{K,a,B} t2ab[i,a,K,B] * ovOV[j,a,K,B]   (ja|KB)

    H_aa = -(alpha/8)(X+X.T) + (alpha/8)(Y+Y.T) + (alpha/4)(Z+Z.T) + diag(eps_oa)
    '''
    nocca = t2aa.shape[0]
    if nocca == 0:
        return numpy.diag(eps_oa)
    X_aa  = numpy.einsum('iakb,jbka->ij', t2aa, ovov)
    Y_aa  = t2aa.reshape(nocca, -1) @ ovov.reshape(nocca, -1).T
    Z_ab  = t2ab.reshape(nocca, -1) @ ovOV.reshape(nocca, -1).T
    return (-(alpha / 8.0) * (X_aa + X_aa.T)
            + (alpha / 8.0) * (Y_aa + Y_aa.T)
            + (alpha / 4.0) * (Z_ab + Z_ab.T)
            + numpy.diag(eps_oa))


def _compute_H_bb(t2bb, t2ab, OVOV, ovOV, eps_ob, alpha):
    '''Build the (noccb, noccb) beta H intermediate.

    Exact alpha<->beta analogue of _compute_H_aa:
        X_bb[I,J] = sum_{K,A,B} t2bb[I,A,K,B] * OVOV[J,B,K,A]   (JB|KA)
        Y_bb[I,J] = sum_{K,A,B} t2bb[I,A,K,B] * OVOV[J,A,K,B]   (JA|KB)
        Z_ba[I,J] = sum_{k,a,A} t2ab[k,a,I,A] * ovOV[k,a,J,A]   (ka|JA)

    H_bb = -(alpha/8)(X+X.T) + (alpha/8)(Y+Y.T) + (alpha/4)(Z+Z.T) + diag(eps_ob)
    '''
    noccb     = t2bb.shape[0]
    if noccb == 0:
        return numpy.diag(eps_ob)
    X_bb      = numpy.einsum('IAKB,JBKA->IJ', t2bb, OVOV)
    Y_bb      = t2bb.reshape(noccb, -1) @ OVOV.reshape(noccb, -1).T
    # Z_ba[I,J] = sum_{k,a,A} t2ab[k,a,I,A] * ovOV[k,a,J,A]
    t2ab_perm = t2ab.transpose(2, 0, 1, 3)   # (noccb, nocca, nvira, nvirb)
    ovOV_perm = ovOV.transpose(2, 0, 1, 3)   # (noccb, nocca, nvira, nvirb)
    Z_ba      = t2ab_perm.reshape(noccb, -1) @ ovOV_perm.reshape(noccb, -1).T
    return (-(alpha / 8.0) * (X_bb + X_bb.T)
            + (alpha / 8.0) * (Y_bb + Y_bb.T)
            + (alpha / 4.0) * (Z_ba + Z_ba.T)
            + numpy.diag(eps_ob))


def compute_residual_aa(t2aa, t2ab, H_aa, eris, eps_va, beta):
    '''Alpha-alpha RE-BWs2 residual (the_method.md lines 60-81).

    The driving terms (lines 60 and 81) have the same sign correction as R_ab:
    the document writes -(ib|ja) + (ia|jb) but the correct residual sign is
    +(ib|ja) - (ia|jb) so that R=0 at the UMP2 starting point.

    With antisymmetric t2aa, the H_vv terms (lines 62,64) both reduce to
    -t2aa * eps_va (see derivation in urebws2.py docstring).
    '''
    ovov = numpy.asarray(eris.ovov)
    oovv = numpy.asarray(eris.oovv)
    oooo = numpy.asarray(eris.oooo)
    vvvv = numpy.asarray(eris.vvvv)
    ovOV = numpy.asarray(eris.ovOV)

    R = numpy.zeros_like(t2aa)

    # Driving: +(ib|ja) - (ia|jb)   (sign-corrected from the_method.md)
    R -= ovov - ovov.transpose(0, 3, 2, 1)

    # H-occ (lines 61, 63)
    R += numpy.einsum('iakb,jk->iajb', t2aa, H_aa)     # +t^ab_ik H^j_k
    R -= numpy.einsum('jakb,ik->iajb', t2aa, H_aa)     # -t^ab_jk H^i_k

    # H-vir canonical (lines 62, 64); antisymmetry of t2aa collapses both to -eps_va*t2aa
    R -= t2aa * (eps_va[None, :, None, None] + eps_va[None, None, None, :])

    if abs(beta) > 1e-14:
        R += beta * numpy.einsum('iakc,jkbc->iajb', t2aa, oovv)           # 65: (jk|bc)
        R += beta * numpy.einsum('jakc,ibkc->iajb', t2aa, ovov)           # 66: (ib|kc)
        R += beta * numpy.einsum('jaKC,ibKC->iajb', t2ab, ovOV)           # 67: (ib|KC)
        R += beta * numpy.einsum('ibkc,jakc->iajb', t2aa, ovov)           # 68: (ja|kc)
        R += beta * numpy.einsum('jbkc,ikac->iajb', t2aa, oovv)           # 69: (ik|ac)
        R += beta * numpy.einsum('ibKC,jaKC->iajb', t2ab, ovOV)           # 70: (ja|KC)
        R += 0.5 * beta * numpy.einsum('kalb,iljk->iajb', t2aa, oooo)     # 71: (il|jk)
        R += 0.5 * beta * numpy.einsum('icjd,adbc->iajb', t2aa, vvvv)     # 72: (ad|bc)
        R -= beta * numpy.einsum('iakc,jbkc->iajb', t2aa, ovov)           # 73: (jb|kc)
        R -= beta * numpy.einsum('jakc,ikbc->iajb', t2aa, oovv)           # 74: (ik|bc)
        R -= beta * numpy.einsum('iaKC,jbKC->iajb', t2ab, ovOV)           # 75: (jb|KC)
        R -= beta * numpy.einsum('ibkc,jkac->iajb', t2aa, oovv)           # 76: (jk|ac)
        R -= beta * numpy.einsum('jbkc,iakc->iajb', t2aa, ovov)           # 77: (ia|kc)
        R -= beta * numpy.einsum('jbKC,iaKC->iajb', t2ab, ovOV)           # 78: (ia|KC)
        R -= 0.5 * beta * numpy.einsum('kalb,ikjl->iajb', t2aa, oooo)     # 79: (ik|jl)
        R -= 0.5 * beta * numpy.einsum('icjd,acbd->iajb', t2aa, vvvv)     # 80: (ac|bd)

    return R


def compute_residual_bb(t2bb, t2ab, H_bb, eris, eps_vb, beta):
    '''Beta-beta RE-BWs2 residual.

    Exact alpha<->beta analogue of compute_residual_aa.
    Cross-spin t2ab terms access t2ab[k,c,J,A] (beta occ/vir in positions 3,4).
    '''
    OVOV = numpy.asarray(eris.OVOV)
    OOVV = numpy.asarray(eris.OOVV)
    OOOO = numpy.asarray(eris.OOOO)
    VVVV = numpy.asarray(eris.VVVV)
    ovOV = numpy.asarray(eris.ovOV)

    R = numpy.zeros_like(t2bb)

    # Driving: +(IB|JA) - (IA|JB)   (sign-corrected from the_method.md)
    R -= OVOV - OVOV.transpose(0, 3, 2, 1)

    # H-occ
    R += numpy.einsum('IAKB,JK->IAJB', t2bb, H_bb)     # +t^AB_IK H^J_K
    R -= numpy.einsum('JAKB,IK->IAJB', t2bb, H_bb)     # -t^AB_JK H^I_K

    # H-vir canonical
    R -= t2bb * (eps_vb[None, :, None, None] + eps_vb[None, None, None, :])

    if abs(beta) > 1e-14:
        R += beta * numpy.einsum('IAKC,JKBC->IAJB', t2bb, OOVV)           # 65β
        R += beta * numpy.einsum('JAKC,IBKC->IAJB', t2bb, OVOV)           # 66β
        R += beta * numpy.einsum('kcJA,kcIB->IAJB', t2ab, ovOV)           # 67β: (IB|kc)
        R += beta * numpy.einsum('IBKC,JAKC->IAJB', t2bb, OVOV)           # 68β
        R += beta * numpy.einsum('JBKC,IKAC->IAJB', t2bb, OOVV)           # 69β
        R += beta * numpy.einsum('kcIB,kcJA->IAJB', t2ab, ovOV)           # 70β: (JA|kc)
        R += 0.5 * beta * numpy.einsum('KALB,ILJK->IAJB', t2bb, OOOO)     # 71β
        R += 0.5 * beta * numpy.einsum('ICJD,ADBC->IAJB', t2bb, VVVV)     # 72β
        R -= beta * numpy.einsum('IAKC,JBKC->IAJB', t2bb, OVOV)           # 73β
        R -= beta * numpy.einsum('JAKC,IKBC->IAJB', t2bb, OOVV)           # 74β
        R -= beta * numpy.einsum('kcIA,kcJB->IAJB', t2ab, ovOV)           # 75β: (JB|kc)
        R -= beta * numpy.einsum('IBKC,JKAC->IAJB', t2bb, OOVV)           # 76β
        R -= beta * numpy.einsum('JBKC,IAKC->IAJB', t2bb, OVOV)           # 77β
        R -= beta * numpy.einsum('kcJB,kcIA->IAJB', t2ab, ovOV)           # 78β: (IA|kc)
        R -= 0.5 * beta * numpy.einsum('KALB,IKJL->IAJB', t2bb, OOOO)     # 79β
        R -= 0.5 * beta * numpy.einsum('ICJD,ACBD->IAJB', t2bb, VVVV)     # 80β

    return R


def compute_residual_ab(t2aa, t2ab, t2bb, H_aa, H_bb, eris, eps_va, eps_vb, beta):
    '''Opposite-spin (alpha-beta) RE-BWs2 residual (the_method.md lines 85-99).

    The driving term (line 99) carries the corrected sign: -(ia|JB) = -ovOV.
    '''
    ovov = numpy.asarray(eris.ovov)
    oovv = numpy.asarray(eris.oovv)
    ovOV = numpy.asarray(eris.ovOV)
    ooVV = numpy.asarray(eris.ooVV)
    OOvv = numpy.asarray(eris.OOvv)
    ooOO = numpy.asarray(eris.ooOO)
    OVOV = numpy.asarray(eris.OVOV)
    OOVV = numpy.asarray(eris.OOVV)
    vvVV = numpy.asarray(eris.vvVV)

    R = numpy.zeros_like(t2ab)

    # H contractions (lines 85-88)
    R += numpy.einsum('iaKB,JK->iaJB', t2ab, H_bb)    # L1: t^aB_iK H^J_K
    R += numpy.einsum('kaJB,ik->iaJB', t2ab, H_aa)    # L2: t^aB_kJ H^i_k
    R -= t2ab * eps_vb[None, None, None, :]             # L3: canonical H^B_C
    R -= t2ab * eps_va[None, :, None, None]             # L4: canonical H^a_c

    if abs(beta) > 1e-14:
        R += beta * numpy.einsum('iaKC,JKBC->iaJB', t2ab, OOVV)    # L5:  (JK|BC)
        R += beta * numpy.einsum('kaJC,ikBC->iaJB', t2ab, ooVV)    # L6:  (ik|BC)
        R += beta * numpy.einsum('icKB,JKac->iaJB', t2ab, OOvv)    # L7:  (JK|ac)
        R += beta * numpy.einsum('kcJB,ikac->iaJB', t2ab, oovv)    # L8:  (ik|ac)
        R -= beta * numpy.einsum('kaLB,ikJL->iaJB', t2ab, ooOO)    # L9:  (ik|JL)
        R -= beta * numpy.einsum('iakc,kcJB->iaJB', t2aa, ovOV)    # L10: (kc|JB)
        R -= beta * numpy.einsum('iaKC,JBKC->iaJB', t2ab, OVOV)    # L11: (JB|KC)
        R -= beta * numpy.einsum('JBKC,iaKC->iaJB', t2bb, ovOV)    # L12: (ia|KC)
        R -= beta * numpy.einsum('kcJB,iakc->iaJB', t2ab, ovov)    # L13: (ia|kc)
        R -= beta * numpy.einsum('icJD,acBD->iaJB', t2ab, vvVV)    # L14: (ac|BD)

    # L15: driving term, sign-corrected
    R -= ovOV

    return R


def kernel(urebws, eris=None, verbose=None):
    '''Iterative UREBWS2 solver with separate CC-DIIS per amplitude block.

    Algorithm:
        1. Initialise (t2aa, t2ab, t2bb) from UMP2 amplitudes.
        2. Build H_aa and H_bb intermediates.
        3. Compute residuals R_aa, R_ab, R_bb.
        4. Normalise: R_norm = R / D (positive denominator).
        5. Update: t2_new = t2 - R_norm; apply per-block DIIS.
        6. Repeat until max|R_norm| < conv_tol across all three blocks.
    '''
    if eris is None:
        eris = _make_eris(urebws)
    log = logger.new_logger(urebws, verbose)

    nocca, noccb = urebws.get_nocc()
    nmoa,  nmob  = urebws.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    eps_oa = eris.mo_energy[0][:nocca]
    eps_va = eris.mo_energy[0][nocca:]
    eps_ob = eris.mo_energy[1][:noccb]
    eps_vb = eris.mo_energy[1][noccb:]

    ovov = numpy.asarray(eris.ovov)
    ovOV = numpy.asarray(eris.ovOV)
    OVOV = numpy.asarray(eris.OVOV)

    # Positive denominators: (eps_vir - eps_occ) + (eps_vir - eps_occ)
    eia_a = eps_va[None, :] - eps_oa[:, None]    # positive
    eia_b = eps_vb[None, :] - eps_ob[:, None]    # positive
    D_aa  = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
    D_ab  = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
    D_bb  = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)
    D_aa[D_aa < 5.e-2] = 5.e-2
    D_ab[D_ab < 5.e-2] = 5.e-2
    D_bb[D_bb < 5.e-2] = 5.e-2

    # MP2 starting point
    e_corr, t2 = init_amps(urebws, eris)
    t2aa, t2ab, t2bb = t2
    log.info('Init E_corr(UREBWS2) = %.15g  [UMP2 starting point]', e_corr)

    # Separate DIIS per block
    adiis_aa = lib.diis.DIIS(urebws)
    adiis_aa.space = urebws.diis_space
    adiis_ab = lib.diis.DIIS(urebws)
    adiis_ab.space = urebws.diis_space
    adiis_bb = lib.diis.DIIS(urebws)
    adiis_bb.space = urebws.diis_space

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    conv = False

    for cycle in range(urebws.max_cycle):
        e_prev = e_corr

        H_aa = _compute_H_aa(t2aa, t2ab, ovov, ovOV, eps_oa, urebws.alpha)
        H_bb = _compute_H_bb(t2bb, t2ab, OVOV, ovOV, eps_ob, urebws.alpha)

        R_aa = compute_residual_aa(t2aa, t2ab, H_aa, eris, eps_va, urebws.beta)
        R_ab = compute_residual_ab(t2aa, t2ab, t2bb, H_aa, H_bb, eris,
                                   eps_va, eps_vb, urebws.beta)
        R_bb = compute_residual_bb(t2bb, t2ab, H_bb, eris, eps_vb, urebws.beta)

        R_aa_norm = R_aa / D_aa
        R_ab_norm = R_ab / D_ab
        R_bb_norm = R_bb / D_bb

        conv_check = max(float(numpy.max(numpy.abs(R_aa_norm), initial=0.0)),
                         float(numpy.max(numpy.abs(R_ab_norm), initial=0.0)),
                         float(numpy.max(numpy.abs(R_bb_norm), initial=0.0)))

        t2aa_new = t2aa + R_aa_norm
        t2ab_new = t2ab + R_ab_norm
        t2bb_new = t2bb + R_bb_norm

        if cycle >= urebws.diis_start_cycle:
            t2aa_new = adiis_aa.update(
                t2aa_new.ravel(), xerr=R_aa_norm.ravel()
            ).reshape(nocca, nvira, nocca, nvira)
            t2ab_new = adiis_ab.update(
                t2ab_new.ravel(), xerr=R_ab_norm.ravel()
            ).reshape(nocca, nvira, noccb, nvirb)
            t2bb_new = adiis_bb.update(
                t2bb_new.ravel(), xerr=R_bb_norm.ravel()
            ).reshape(noccb, nvirb, noccb, nvirb)

        t2aa, t2ab, t2bb = t2aa_new, t2ab_new, t2bb_new
        e_corr = energy(urebws, (t2aa, t2ab, t2bb), (ovov, ovOV, OVOV))

        log.info('cycle = %d  E_corr(UREBWS2) = %.15g  dE = %.9g  |R|_max = %.6g',
                 cycle + 1, e_corr, e_corr - e_prev, conv_check)
        cput1 = log.timer('UREBWS2 iter', *cput1)

        if conv_check < urebws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('UREBWS2 did not converge after %d cycles.', urebws.max_cycle)

    log.timer('UREBWS2', *cput0)
    return conv, e_corr, (t2aa, t2ab, t2bb)


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class UREBWS2(lib.StreamObject):
    '''Unrestricted RE-BWs2 for open-shell (UHF) references.

    Attributes
    ----------
    alpha : float
        BW-s2 H-intermediate scaling. Default 1.0. alpha=0 with beta=0 gives UMP2.
    beta : float
        RE contribution scaling. Default 1.0.
        beta=0 disables all oovv/oooo/vvvv contractions in the residuals.
    max_cycle : int
        Maximum amplitude iterations. Default 50.
    conv_tol : float
        Convergence threshold on max|R_norm| across all blocks. Default 1e-7.
    diis_space : int
        DIIS subspace size per block. Default 6.
    diis_start_cycle : int
        First iteration using DIIS. Default 3.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.rebws import UREBWS2
    >>> mol = gto.M(atom='O', basis='sto-3g', spin=2)
    >>> mf = scf.UHF(mol).run()
    >>> urebws = UREBWS2(mf).run()
    >>> print(urebws.e_corr)
    '''

    max_cycle        = 50
    conv_tol         = 1e-7
    alpha            = 1.0
    beta             = 1.0
    diis_space       = DIIS_SPACE
    diis_start_cycle = DIIS_START_CYCLE

    _keys = {
        'max_cycle', 'conv_tol', 'alpha', 'beta',
        'diis_space', 'diis_start_cycle',
        'mol', 'max_memory', 'mo_coeff', 'mo_occ',
        'e_hf', 'e_corr', 't2', 'converged',
    }

    def __init__(self, mf, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mol        = mf.mol
        self._scf       = mf
        self.verbose    = self.mol.verbose
        self.stdout     = self.mol.stdout
        self.max_memory = mf.max_memory

        self.mo_coeff   = mo_coeff
        self.mo_occ     = mo_occ
        self._nocc      = None
        self._nmo       = None

        self.converged  = False
        self.e_hf       = None
        self.e_corr     = None
        self.t2         = None   # tuple (t2aa, t2ab, t2bb)

    @property
    def mo_energy(self):
        return self._scf.mo_energy

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    @property
    def e_tot(self):
        return self._scf.e_tot + (self.e_corr or 0.0)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        nocca, noccb = self.get_nocc()
        nmoa,  nmob  = self.get_nmo()
        log.info('nocca = %d  noccb = %d  nmoa = %d  nmob = %d',
                 nocca, noccb, nmoa, nmob)
        log.info('alpha = %.6g  beta = %.6g', self.alpha, self.beta)
        log.info('max_cycle = %d  conv_tol = %g', self.max_cycle, self.conv_tol)
        log.info('diis_space = %d  diis_start_cycle = %d',
                 self.diis_space, self.diis_start_cycle)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def ao2mo(self, mo_coeff=None):
        '''Transform ERIs into the _ChemistsERIs object used internally.'''
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    def _full_ao2mo(self, mo_coeff=None):
        '''Return (eri_aa, eri_ab, eri_bb) full MO ERI tensors.

        Preserves the interface from the skeleton ao2mo method.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_a, mo_b = mo_coeff
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        eri_aa = ao2mo.restore(1, ao2mo.full(self.mol, mo_a), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(self.mol, mo_b), nmob)
        eri_ab = (ao2mo.general(self.mol, (mo_a, mo_a, mo_b, mo_b), compact=False)
                  .reshape(nmoa, nmoa, nmob, nmob))
        return eri_aa, eri_ab, eri_bb

    def kernel(self, mo_coeff=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        self.e_hf = get_e_hf(self, mo_coeff=mo_coeff)
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.t2 = kernel(self, eris)
        self._finalize()
        return self.e_corr, self.t2

    def _finalize(self):
        log = logger.new_logger(self)
        if self.converged:
            log.note('E(%s) = %.15g  E_corr = %.15g',
                     self.__class__.__name__, self.e_tot, self.e_corr)
        else:
            log.warn('%s did not converge', self.__class__.__name__)
            log.note('E(%s) = %.15g  E_corr = %.15g',
                     self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    get_nocc  = get_nocc
    get_nmo   = get_nmo
    get_e_hf  = get_e_hf
    init_amps = init_amps
    energy    = energy


from pyscf import scf
scf.uhf.UHF.UREBWS2 = lib.class_as_method(UREBWS2)
