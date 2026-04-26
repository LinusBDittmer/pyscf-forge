'''
Density-fitted Unrestricted RE-BWs2 (DF-URE-BWs2)

Uses the symmetric Coulomb-metric DF decomposition:
    (pq|rs) = sum_Q B^Q_pq B^Q_rs

Six B tensors stored per instance:
    B_ov[Q, i, a]   alpha occ-vir   (naux, nocca, nvira)
    B_oo[Q, i, j]   alpha occ-occ   (naux, nocca, nocca)
    B_vv[Q, a, b]   alpha vir-vir   (naux, nvira, nvira)
    B_OV[Q, I, A]   beta  occ-vir   (naux, noccb, nvirb)
    B_OO[Q, I, J]   beta  occ-occ   (naux, noccb, noccb)
    B_VV[Q, A, B]   beta  vir-vir   (naux, nvirb, nvirb)

Per-iteration intermediates:
    Theta_aa[Q,i,a] = einsum('iajb,Qjb->Qia', t2aa, B_ov)
    Xi_aa[Q,i,a]    = einsum('icka,Qkc->Qia', t2aa, B_ov)
    Phi_ab[Q,i,a]   = einsum('iaJB,QJB->Qia', t2ab, B_OV)
    Theta_bb[Q,I,A] = einsum('IAJB,QJB->QIA', t2bb, B_OV)
    Xi_bb[Q,I,A]    = einsum('ICKA,QKC->QIA', t2bb, B_OV)
    Phi_ba[Q,I,A]   = einsum('kaIA,Qka->QIA', t2ab, B_ov)

No pair symmetry exists for antisymmetric t2aa/t2bb, so the Q-loop
implements each oovv/oooo/vvvv term separately.

Driving term L15 carries a NEGATIVE sign (sign error in the_method.md corrected).
Working equations: pyscf/rebws/the_method.md
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.rebws.urebws2 import (get_nocc, get_nmo, get_e_hf,
                                   DIIS_SPACE, DIIS_START_CYCLE)


# ---------------------------------------------------------------------------
# ERI storage and transformation
# ---------------------------------------------------------------------------

class _DF_UChemistsERIs:
    '''Stores B-tensor slices for DF-URE-BWs2.'''

    def __init__(self):
        self.mol       = None
        self.mo_coeff  = None
        self.mo_energy = None   # tuple (mo_ea, mo_eb)
        self.fock      = None   # tuple (focka, fockb)
        self.B_ov = None   # (naux, nocca, nvira)
        self.B_oo = None   # (naux, nocca, nocca)
        self.B_vv = None   # (naux, nvira, nvira)
        self.B_OV = None   # (naux, noccb, nvirb)
        self.B_OO = None   # (naux, noccb, noccb)
        self.B_VV = None   # (naux, nvirb, nvirb)

    def _common_init_(self, dfurebws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = dfurebws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. Run mf.kernel() first.')
        mo_a, mo_b = mo_coeff
        self.mo_coeff = (mo_a, mo_b)
        self.mol = dfurebws.mol

        if mo_coeff is dfurebws._scf.mo_coeff and dfurebws._scf.converged:
            self.mo_energy = (dfurebws._scf.mo_energy[0], dfurebws._scf.mo_energy[1])
            self.fock = (numpy.diag(self.mo_energy[0]),
                         numpy.diag(self.mo_energy[1]))
        else:
            dm     = dfurebws._scf.make_rdm1(mo_coeff, dfurebws.mo_occ)
            vhf    = dfurebws._scf.get_veff(dfurebws.mol, dm)
            fockao = dfurebws._scf.get_fock(vhf=vhf, dm=dm)
            focka  = mo_a.conj().T @ fockao[0] @ mo_a
            fockb  = mo_b.conj().T @ fockao[1] @ mo_b
            self.fock      = (focka, fockb)
            self.mo_energy = (focka.diagonal().real, fockb.diagonal().real)
        return self


def _make_df_eris(dfurebws, mo_coeff=None, verbose=None):
    '''AO→MO B-tensor transform via two passes through the DF auxiliary integrals.

    Pass 1: alpha MOs  → B_ov, B_oo, B_vv
    Pass 2: beta  MOs  → B_OV, B_OO, B_VV
    '''
    log   = logger.new_logger(dfurebws, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris  = _DF_UChemistsERIs()
    eris._common_init_(dfurebws, mo_coeff)
    mo_a, mo_b = eris.mo_coeff

    nocca, noccb = dfurebws.nocc
    nmoa,  nmob  = dfurebws.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    if dfurebws.with_df is not None:
        with_df = dfurebws.with_df
    else:
        from pyscf import df as _df_module
        from pyscf.df import addons
        auxbasis = dfurebws.auxbasis
        if auxbasis is None:
            auxbasis = addons.make_auxbasis(dfurebws.mol, mp2fit=True)
        with_df = _df_module.DF(dfurebws.mol, auxbasis=auxbasis)
        with_df.build()

    naux = with_df.get_naoaux()

    log.debug('DF-URE-BWs2: alpha B_ov, B_oo, B_vv (pass 1)')
    B_ov_buf = numpy.zeros((naux, nocca * nvira))
    B_oo_buf = numpy.zeros((naux, nocca * nocca))
    B_vv_buf = numpy.zeros((naux, nvira * nvira))
    p0 = 0
    for eri1 in with_df.loop():
        p1 = p0 + eri1.shape[0]
        B_ov_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_a, (0, nocca, nocca, nmoa),
                                        aosym='s2', mosym='s1')
        B_oo_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_a, (0, nocca, 0, nocca),
                                        aosym='s2', mosym='s1')
        B_vv_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_a, (nocca, nmoa, nocca, nmoa),
                                        aosym='s2', mosym='s1')
        p0 = p1
    eris.B_ov = B_ov_buf.reshape(naux, nocca, nvira)
    eris.B_oo = B_oo_buf.reshape(naux, nocca, nocca)
    eris.B_vv = B_vv_buf.reshape(naux, nvira, nvira)

    log.debug('DF-URE-BWs2: beta B_OV, B_OO, B_VV (pass 2)')
    B_OV_buf = numpy.zeros((naux, noccb * nvirb))
    B_OO_buf = numpy.zeros((naux, noccb * noccb))
    B_VV_buf = numpy.zeros((naux, nvirb * nvirb))
    p0 = 0
    for eri1 in with_df.loop():
        p1 = p0 + eri1.shape[0]
        B_OV_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_b, (0, noccb, noccb, nmob),
                                        aosym='s2', mosym='s1')
        B_OO_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_b, (0, noccb, 0, noccb),
                                        aosym='s2', mosym='s1')
        B_VV_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_b, (noccb, nmob, noccb, nmob),
                                        aosym='s2', mosym='s1')
        p0 = p1
    eris.B_OV = B_OV_buf.reshape(naux, noccb, nvirb)
    eris.B_OO = B_OO_buf.reshape(naux, noccb, noccb)
    eris.B_VV = B_VV_buf.reshape(naux, nvirb, nvirb)

    log.timer('DF-URE-BWs2 integral transform', *cput0)
    return eris


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def init_amps(dfurebws, eris):
    '''DF-UMP2 amplitudes as the DF-URE-BWs2 starting point.'''
    nocca, noccb = dfurebws.nocc
    eps_oa = eris.mo_energy[0][:nocca]
    eps_va = eris.mo_energy[0][nocca:]
    eps_ob = eris.mo_energy[1][:noccb]
    eps_vb = eris.mo_energy[1][noccb:]

    eia_a = eps_oa[:, None] - eps_va[None, :]
    eia_b = eps_ob[:, None] - eps_vb[None, :]
    D_aa  = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
    D_ab  = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
    D_bb  = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)

    B_ov = eris.B_ov
    B_OV = eris.B_OV
    ovov = numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)
    ovOV = numpy.einsum('Qia,QJB->iaJB', B_ov, B_OV)
    OVOV = numpy.einsum('QIA,QJB->IAJB', B_OV, B_OV)

    t2aa = (ovov - ovov.transpose(0, 3, 2, 1)) / D_aa
    t2ab = ovOV / D_ab
    t2bb = (OVOV - OVOV.transpose(0, 3, 2, 1)) / D_bb
    t2 = (t2aa, t2ab, t2bb)
    return energy(dfurebws, t2, eris), t2


def energy(dfurebws, t2, eris):
    '''DF-URE-BWs2 correlation energy.

    E_aa = (1/2) * einsum('iajb,Qia,Qjb->', t2aa, B_ov, B_ov)  -- via Theta_aa
    E_bb = (1/2) * einsum('IAJB,QIA,QJB->', t2bb, B_OV, B_OV)  -- via Theta_bb
    E_ab = einsum('iaJB,Qia,QJB->', t2ab, B_ov, B_OV)           -- via Phi_ab
    '''
    t2aa, t2ab, t2bb = t2
    B_ov = eris.B_ov
    B_OV = eris.B_OV
    Theta_aa = numpy.einsum('iajb,Qjb->Qia', t2aa, B_ov)
    Theta_bb = numpy.einsum('IAJB,QJB->QIA', t2bb, B_OV)
    Phi_ab   = numpy.einsum('iaJB,QJB->Qia', t2ab, B_OV)
    E_aa = 0.5 * numpy.einsum('Qia,Qia->', Theta_aa, B_ov).real
    E_bb = 0.5 * numpy.einsum('QIA,QIA->', Theta_bb, B_OV).real
    E_ab = numpy.einsum('Qia,Qia->', Phi_ab, B_ov).real
    return E_aa + E_ab + E_bb


def _compute_H_aa_df(Xi_aa, Theta_aa, Phi_ab, B_ov, eps_oa, alpha):
    '''Alpha H intermediate from DF B-tensor contractions.

    H_aa = -(alpha/8)*(X+X.T) + (alpha/8)*(Y+Y.T) + (alpha/4)*(Z+Z.T) + diag(eps_oa)

    X_aa[i,j] = einsum('Qia,Qja->ij', Xi_aa, B_ov)    -- iakb contraction
    Y_aa[i,j] = einsum('Qia,Qja->ij', Theta_aa, B_ov) -- iajb contraction
    Z_ab[i,j] = einsum('Qia,Qja->ij', Phi_ab, B_ov)   -- cross-spin t2ab|ovOV
    '''
    if len(eps_oa) == 0:
        return numpy.diag(eps_oa)
    X_aa = numpy.einsum('Qia,Qja->ij', Xi_aa, B_ov)
    Y_aa = numpy.einsum('Qia,Qja->ij', Theta_aa, B_ov)
    Z_ab = numpy.einsum('Qia,Qja->ij', Phi_ab, B_ov)
    return (-(alpha / 8.0) * (X_aa + X_aa.T)
            + (alpha / 8.0) * (Y_aa + Y_aa.T)
            + (alpha / 4.0) * (Z_ab + Z_ab.T)
            + numpy.diag(eps_oa))


def _compute_H_bb_df(Xi_bb, Theta_bb, Phi_ba, B_OV, eps_ob, alpha):
    '''Beta H intermediate from DF B-tensor contractions.

    Exact alpha<->beta analogue of _compute_H_aa_df.
    '''
    if len(eps_ob) == 0:
        return numpy.diag(eps_ob)
    X_bb = numpy.einsum('QIA,QJA->IJ', Xi_bb, B_OV)
    Y_bb = numpy.einsum('QIA,QJA->IJ', Theta_bb, B_OV)
    Z_ba = numpy.einsum('QIA,QJA->IJ', Phi_ba, B_OV)
    return (-(alpha / 8.0) * (X_bb + X_bb.T)
            + (alpha / 8.0) * (Y_bb + Y_bb.T)
            + (alpha / 4.0) * (Z_ba + Z_ba.T)
            + numpy.diag(eps_ob))


def compute_residual_aa_df(t2aa, H_aa, Theta_aa, Phi_ab, eris, eps_va, beta):
    '''Alpha-alpha DF residual.

    Because t2aa is antisymmetric (not pair-symmetric), the Q-loop terms
    L65/L69 and L74/L76 are implemented separately rather than combined.
    Vectorised ovov terms use Theta_aa and Phi_ab intermediates.
    Driving term carries NEGATIVE sign (sign correction from the_method.md).
    '''
    B_ov = eris.B_ov
    B_oo = eris.B_oo
    B_vv = eris.B_vv
    naux = B_ov.shape[0]
    R = numpy.zeros_like(t2aa)

    # H contractions (no ERIs)
    R += numpy.einsum('iakb,jk->iajb', t2aa, H_aa)
    R -= numpy.einsum('jakb,ik->iajb', t2aa, H_aa)
    R -= t2aa * (eps_va[None, :, None, None] + eps_va[None, None, None, :])

    # Driving: -(ovov - ovov.T(0,3,2,1))   sign-corrected
    R -= numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov) - numpy.einsum('Qib,Qja->iajb', B_ov, B_ov)

    if abs(beta) > 1e-14:
        # Vectorised ovov-only terms
        R -= beta * numpy.einsum('Qia,Qjb->iajb', Theta_aa, B_ov)   # L73
        R -= beta * numpy.einsum('Qia,Qjb->iajb', B_ov, Theta_aa)   # L77
        R -= beta * numpy.einsum('Qia,Qjb->iajb', Phi_ab, B_ov)     # L75
        R -= beta * numpy.einsum('Qia,Qjb->iajb', B_ov, Phi_ab)     # L78
        R += beta * numpy.einsum('Qib,Qja->iajb', B_ov, Theta_aa)   # L66
        R += beta * numpy.einsum('Qja,Qib->iajb', B_ov, Theta_aa)   # L68
        R += beta * numpy.einsum('Qib,Qja->iajb', B_ov, Phi_ab)     # L67
        R += beta * numpy.einsum('Qja,Qib->iajb', B_ov, Phi_ab)     # L70

        for Q in range(naux):
            Boo_Q = B_oo[Q]
            Bvv_Q = B_vv[Q]

            # L65: (jk|bc)  R += beta * t2aa[i,a,k,c]*Boo[j,k]*Bvv[b,c]
            Z65 = numpy.einsum('iakc,jk->iajc', t2aa, Boo_Q)
            R += beta * numpy.einsum('iajc,bc->iajb', Z65, Bvv_Q)

            # L69: (ik|ac)  R += beta * t2aa[j,b,k,c]*Boo[i,k]*Bvv[a,c]
            Z69 = numpy.einsum('jbkc,ik->ibjc', t2aa, Boo_Q)
            R += beta * numpy.einsum('ibjc,ac->iajb', Z69, Bvv_Q)

            # L74: (ik|bc)  R -= beta * t2aa[j,a,k,c]*Boo[i,k]*Bvv[b,c]
            Z74 = numpy.einsum('jakc,ik->iajc', t2aa, Boo_Q)
            R -= beta * numpy.einsum('iajc,bc->iajb', Z74, Bvv_Q)

            # L76: (jk|ac)  R -= beta * t2aa[i,b,k,c]*Boo[j,k]*Bvv[a,c]
            Z76 = numpy.einsum('ibkc,jk->ibjc', t2aa, Boo_Q)
            R -= beta * numpy.einsum('ibjc,ac->iajb', Z76, Bvv_Q)

            # L71: (il|jk)  R += 0.5*beta * t2aa[k,a,l,b]*Boo[i,l]*Boo[j,k]
            Z71 = numpy.einsum('kalb,jk->lajb', t2aa, Boo_Q)
            R += 0.5 * beta * numpy.einsum('lajb,il->iajb', Z71, Boo_Q)

            # L79: (ik|jl)  R -= 0.5*beta * t2aa[k,a,l,b]*Boo[i,k]*Boo[j,l]
            Z79 = numpy.einsum('kalb,jl->kajb', t2aa, Boo_Q)
            R -= 0.5 * beta * numpy.einsum('kajb,ik->iajb', Z79, Boo_Q)

            # L72: (ad|bc)  R += 0.5*beta * t2aa[i,c,j,d]*Bvv[a,d]*Bvv[b,c]
            Z72 = numpy.einsum('icjd,ad->icja', t2aa, Bvv_Q)
            R += 0.5 * beta * numpy.einsum('icja,bc->iajb', Z72, Bvv_Q)

            # L80: (ac|bd)  R -= 0.5*beta * t2aa[i,c,j,d]*Bvv[a,c]*Bvv[b,d]
            Z80 = numpy.einsum('icjd,ac->iajd', t2aa, Bvv_Q)
            R -= 0.5 * beta * numpy.einsum('iajd,bd->iajb', Z80, Bvv_Q)

    return R


def compute_residual_bb_df(t2bb, H_bb, Theta_bb, Phi_ba, eris, eps_vb, beta):
    '''Beta-beta DF residual. Exact alpha<->beta analogue of compute_residual_aa_df.'''
    B_OV = eris.B_OV
    B_OO = eris.B_OO
    B_VV = eris.B_VV
    naux = B_OV.shape[0]
    R = numpy.zeros_like(t2bb)

    R += numpy.einsum('IAKB,JK->IAJB', t2bb, H_bb)
    R -= numpy.einsum('JAKB,IK->IAJB', t2bb, H_bb)
    R -= t2bb * (eps_vb[None, :, None, None] + eps_vb[None, None, None, :])

    R -= numpy.einsum('QIA,QJB->IAJB', B_OV, B_OV) - numpy.einsum('QIB,QJA->IAJB', B_OV, B_OV)

    if abs(beta) > 1e-14:
        R -= beta * numpy.einsum('QIA,QJB->IAJB', Theta_bb, B_OV)   # L73β
        R -= beta * numpy.einsum('QIA,QJB->IAJB', B_OV, Theta_bb)   # L77β
        R -= beta * numpy.einsum('QIA,QJB->IAJB', Phi_ba, B_OV)     # L75β
        R -= beta * numpy.einsum('QIA,QJB->IAJB', B_OV, Phi_ba)     # L78β
        R += beta * numpy.einsum('QIB,QJA->IAJB', B_OV, Theta_bb)   # L66β
        R += beta * numpy.einsum('QJA,QIB->IAJB', B_OV, Theta_bb)   # L68β
        R += beta * numpy.einsum('QIB,QJA->IAJB', B_OV, Phi_ba)     # L67β
        R += beta * numpy.einsum('QJA,QIB->IAJB', B_OV, Phi_ba)     # L70β

        for Q in range(naux):
            BOO_Q = B_OO[Q]
            BVV_Q = B_VV[Q]

            Z65 = numpy.einsum('IAKC,JK->IAJC', t2bb, BOO_Q)
            R += beta * numpy.einsum('IAJC,BC->IAJB', Z65, BVV_Q)

            Z69 = numpy.einsum('JBKC,IK->IBJC', t2bb, BOO_Q)
            R += beta * numpy.einsum('IBJC,AC->IAJB', Z69, BVV_Q)

            Z74 = numpy.einsum('JAKC,IK->IAJC', t2bb, BOO_Q)
            R -= beta * numpy.einsum('IAJC,BC->IAJB', Z74, BVV_Q)

            Z76 = numpy.einsum('IBKC,JK->IBJC', t2bb, BOO_Q)
            R -= beta * numpy.einsum('IBJC,AC->IAJB', Z76, BVV_Q)

            Z71 = numpy.einsum('KALB,JK->LAJB', t2bb, BOO_Q)
            R += 0.5 * beta * numpy.einsum('LAJB,IL->IAJB', Z71, BOO_Q)

            Z79 = numpy.einsum('KALB,JL->KAJB', t2bb, BOO_Q)
            R -= 0.5 * beta * numpy.einsum('KAJB,IK->IAJB', Z79, BOO_Q)

            Z72 = numpy.einsum('ICJD,AD->ICJA', t2bb, BVV_Q)
            R += 0.5 * beta * numpy.einsum('ICJA,BC->IAJB', Z72, BVV_Q)

            Z80 = numpy.einsum('ICJD,AC->IAJD', t2bb, BVV_Q)
            R -= 0.5 * beta * numpy.einsum('IAJD,BD->IAJB', Z80, BVV_Q)

    return R


def compute_residual_ab_df(t2ab, H_aa, H_bb, Theta_aa, Theta_bb, Phi_ab, Phi_ba,
                           eris, eps_va, eps_vb, beta):
    '''Opposite-spin DF residual.

    Vectorised cross-spin terms use Theta_aa, Theta_bb, Phi_ab, Phi_ba.
    Q-loop handles the mixed-spin oovv/ooOO/vvVV blocks (L5-L9, L14).
    L5+L6 and L7+L8 can be combined since t2ab has no pair-symmetry constraint.
    Driving term L15 = -ovOV carries NEGATIVE sign.
    '''
    B_ov = eris.B_ov
    B_OV = eris.B_OV
    B_oo = eris.B_oo
    B_vv = eris.B_vv
    B_OO = eris.B_OO
    B_VV = eris.B_VV
    naux = B_ov.shape[0]
    R = numpy.zeros_like(t2ab)

    # H contractions
    R += numpy.einsum('iaKB,JK->iaJB', t2ab, H_bb)
    R += numpy.einsum('kaJB,ik->iaJB', t2ab, H_aa)
    R -= t2ab * eps_vb[None, None, None, :]
    R -= t2ab * eps_va[None, :, None, None]

    # Driving term L15 (NEGATIVE sign)
    R -= numpy.einsum('Qia,QJB->iaJB', B_ov, B_OV)

    if abs(beta) > 1e-14:
        # Vectorised cross-spin terms
        R -= beta * numpy.einsum('Qia,QJB->iaJB', Theta_aa, B_OV)   # L10
        R -= beta * numpy.einsum('Qia,QJB->iaJB', Phi_ab, B_OV)     # L11
        R -= beta * numpy.einsum('Qia,QJB->iaJB', B_ov, Theta_bb)   # L12
        R -= beta * numpy.einsum('Qia,QJB->iaJB', B_ov, Phi_ba)     # L13

        for Q in range(naux):
            Boo_Q = B_oo[Q]
            Bvv_Q = B_vv[Q]
            BOO_Q = B_OO[Q]
            BVV_Q = B_VV[Q]

            # L5+L6: OOVV and ooVV contributions combined
            Z56 = (numpy.einsum('iaKC,JK->iaJC', t2ab, BOO_Q)
                 + numpy.einsum('kaJC,ik->iaJC', t2ab, Boo_Q))
            R += beta * numpy.einsum('iaJC,BC->iaJB', Z56, BVV_Q)

            # L7+L8: OOvv and oovv contributions combined
            Z78 = (numpy.einsum('icKB,JK->icJB', t2ab, BOO_Q)
                 + numpy.einsum('kcJB,ik->icJB', t2ab, Boo_Q))
            R += beta * numpy.einsum('icJB,ac->iaJB', Z78, Bvv_Q)

            # L9: ooOO
            Z9 = numpy.einsum('kaLB,JL->kaJB', t2ab, BOO_Q)
            R -= beta * numpy.einsum('kaJB,ik->iaJB', Z9, Boo_Q)

            # L14: vvVV
            Z14 = numpy.einsum('icJD,ac->iaJD', t2ab, Bvv_Q)
            R -= beta * numpy.einsum('iaJD,BD->iaJB', Z14, BVV_Q)

    return R


def kernel(dfurebws, eris=None, verbose=None):
    '''Iterative DF-URE-BWs2 solver with separate CC-DIIS per amplitude block.'''
    if eris is None:
        eris = _make_df_eris(dfurebws)
    log = logger.new_logger(dfurebws, verbose)

    nocca, noccb = dfurebws.nocc
    nmoa,  nmob  = dfurebws.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    eps_oa = eris.mo_energy[0][:nocca]
    eps_va = eris.mo_energy[0][nocca:]
    eps_ob = eris.mo_energy[1][:noccb]
    eps_vb = eris.mo_energy[1][noccb:]
    B_ov, B_OV = eris.B_ov, eris.B_OV

    eia_a = eps_va[None, :] - eps_oa[:, None]
    eia_b = eps_vb[None, :] - eps_ob[:, None]
    D_aa  = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
    D_ab  = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
    D_bb  = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)
    D_aa[D_aa < 5.e-2] = 5.e-2
    D_ab[D_ab < 5.e-2] = 5.e-2
    D_bb[D_bb < 5.e-2] = 5.e-2

    e_corr, t2 = init_amps(dfurebws, eris)
    t2aa, t2ab, t2bb = t2
    log.info('Init E_corr(DF-URE-BWs2) = %.15g  [DF-UMP2 starting point]', e_corr)

    adiis_aa = lib.diis.DIIS(dfurebws)
    adiis_aa.space = dfurebws.diis_space
    adiis_ab = lib.diis.DIIS(dfurebws)
    adiis_ab.space = dfurebws.diis_space
    adiis_bb = lib.diis.DIIS(dfurebws)
    adiis_bb.space = dfurebws.diis_space

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    conv = False

    for cycle in range(dfurebws.max_cycle):
        e_prev = e_corr

        # Per-cycle intermediates
        Theta_aa = numpy.einsum('iajb,Qjb->Qia', t2aa, B_ov)
        Xi_aa    = numpy.einsum('icka,Qkc->Qia', t2aa, B_ov)
        Phi_ab   = numpy.einsum('iaJB,QJB->Qia', t2ab, B_OV)
        Theta_bb = numpy.einsum('IAJB,QJB->QIA', t2bb, B_OV)
        Xi_bb    = numpy.einsum('ICKA,QKC->QIA', t2bb, B_OV)
        Phi_ba   = numpy.einsum('kaIA,Qka->QIA', t2ab, B_ov)

        H_aa = _compute_H_aa_df(Xi_aa, Theta_aa, Phi_ab, B_ov, eps_oa, dfurebws.alpha)
        H_bb = _compute_H_bb_df(Xi_bb, Theta_bb, Phi_ba, B_OV, eps_ob, dfurebws.alpha)

        R_aa = compute_residual_aa_df(t2aa, H_aa, Theta_aa, Phi_ab,
                                      eris, eps_va, dfurebws.beta)
        R_ab = compute_residual_ab_df(t2ab, H_aa, H_bb,
                                      Theta_aa, Theta_bb, Phi_ab, Phi_ba,
                                      eris, eps_va, eps_vb, dfurebws.beta)
        R_bb = compute_residual_bb_df(t2bb, H_bb, Theta_bb, Phi_ba,
                                      eris, eps_vb, dfurebws.beta)

        R_aa_norm = R_aa / D_aa
        R_ab_norm = R_ab / D_ab
        R_bb_norm = R_bb / D_bb

        conv_check = max(float(numpy.max(numpy.abs(R_aa_norm), initial=0.0)),
                         float(numpy.max(numpy.abs(R_ab_norm), initial=0.0)),
                         float(numpy.max(numpy.abs(R_bb_norm), initial=0.0)))

        t2aa_new = t2aa + R_aa_norm
        t2ab_new = t2ab + R_ab_norm
        t2bb_new = t2bb + R_bb_norm

        if cycle >= dfurebws.diis_start_cycle:
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
        e_corr = energy(dfurebws, (t2aa, t2ab, t2bb), eris)

        log.info('cycle = %d  E_corr(DF-URE-BWs2) = %.15g  dE = %.9g  |R|_max = %.6g',
                 cycle + 1, e_corr, e_corr - e_prev, conv_check)
        cput1 = log.timer('DF-URE-BWs2 iter', *cput1)

        if conv_check < dfurebws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('DF-URE-BWs2 did not converge after %d cycles.', dfurebws.max_cycle)

    log.timer('DF-URE-BWs2', *cput0)
    return conv, e_corr, (t2aa, t2ab, t2bb)


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class DFUREBWS2(lib.StreamObject):
    '''Density-fitted RE-BWs2 for open-shell (UHF) references.

    Attributes
    ----------
    alpha : float
        BW-s2 H-intermediate scaling. Default 1.0. alpha=0+beta=0 → DF-UMP2.
    beta : float
        RE contribution scaling. Default 1.0.
    auxbasis : str or None
        Auxiliary basis. None → auto-selected via make_auxbasis (mp2fit).
    max_cycle : int
        Maximum amplitude iterations. Default 50.
    conv_tol : float
        Convergence threshold on max|R_norm|. Default 1e-7.
    diis_space : int
        DIIS subspace size per block. Default 10.
    diis_start_cycle : int
        First cycle with DIIS extrapolation. Default 0.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.rebws import DFUREBWS2
    >>> mol = gto.M(atom='O', basis='sto-3g', spin=2)
    >>> mf = scf.UHF(mol).run()
    >>> dfurebws = DFUREBWS2(mf).run()
    >>> print(dfurebws.e_corr)
    '''

    max_cycle        = 50
    conv_tol         = 1e-7
    alpha            = 1.0
    beta             = 1.0
    auxbasis         = None
    diis_space       = DIIS_SPACE
    diis_start_cycle = DIIS_START_CYCLE

    _keys = {
        'max_cycle', 'conv_tol', 'alpha', 'beta', 'auxbasis',
        'diis_space', 'diis_start_cycle',
        'mol', 'max_memory', 'mo_coeff', 'mo_occ',
        'e_hf', 'e_corr', 't2', 'converged', 'with_df',
    }

    def __init__(self, mf, mo_coeff=None, mo_occ=None, auxbasis=None):
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

        if auxbasis is not None:
            self.auxbasis = auxbasis
        self.with_df = getattr(mf, 'with_df', None)

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
        log.info('auxbasis = %s', self.auxbasis)
        log.info('max_cycle = %d  conv_tol = %g', self.max_cycle, self.conv_tol)
        log.info('diis_space = %d  diis_start_cycle = %d',
                 self.diis_space, self.diis_start_cycle)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff, verbose=self.verbose)

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
scf.uhf.UHF.DFUREBWS2 = lib.class_as_method(DFUREBWS2)
