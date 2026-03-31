'''
Unrestricted BW-s2

Reference:
    Kevin Carter-Fenk and Martin Head-Gordon,
    J. Chem. Phys. 158, 234108 (2023); doi: 10.1063/5.0150033
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.bws2.rbws2 import BWSBase, _rotate_ovov, get_e_hf


# ---------------------------------------------------------------------------
# UBW-s2 computational functions
# ---------------------------------------------------------------------------

def kernel(bws, mo_energy=None, mo_coeff=None, eris=None, verbose=None):
    '''Iterative unrestricted BW-s2 solver.

    Returns
    -------
    conv : bool
    e_corr : float
    t2 : tuple of ndarray
        (t2aa, t2ab, t2bb) in the converged dressed-orbital basis.
        Shapes: (nocca,nvira,nocca,nvira), (nocca,nvira,noccb,nvirb),
                (noccb,nvirb,noccb,nvirb). Non-antisymmetrized.
    '''
    if eris is None:
        eris = bws.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocca, noccb = bws.get_nocc()
    nmoa, nmob   = bws.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_ea, mo_eb = mo_energy
    eps_oa = mo_ea[:nocca]
    eps_va = mo_ea[nocca:]
    eps_ob = mo_eb[:noccb]
    eps_vb = mo_eb[noccb:]

    log = logger.new_logger(bws, verbose)

    # Load canonical integrals once; kept throughout all iterations.
    ovov_can = numpy.asarray(eris.ovov).reshape(nocca, nvira, nocca, nvira)
    ovOV_can = numpy.asarray(eris.ovOV).reshape(nocca, nvira, noccb, nvirb)
    OVOV_can = numpy.asarray(eris.OVOV).reshape(noccb, nvirb, noccb, nvirb)

    # --- iteration 0: canonical UMP2 starting point ---
    e_corr, t2 = init_amps(bws, eris)
    t2aa, t2ab, t2bb = t2
    log.info('Init E_corr(UBW-s2) = %.15g  [UMP2]', e_corr)

    # Working copies of dressed integrals
    ovov = ovov_can.copy()
    ovOV = ovOV_can.copy()
    OVOV = OVOV_can.copy()

    # Canonical Fock occupied blocks (diagonal for canonical UHF)
    F_oo_a_can = numpy.diag(eps_oa)
    F_oo_b_can = numpy.diag(eps_ob)
    F_oo_a     = F_oo_a_can.copy()
    F_oo_b     = F_oo_b_can.copy()

    U_cum_a = numpy.eye(nocca)
    U_cum_b = numpy.eye(noccb)

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())

    conv = False
    for cycle in range(bws.max_cycle):
        e_prev = e_corr

        # 1. Build W for each spin block.
        #    W_bb reuses _compute_W by permuting indices so beta indices come first.
        W_aa = _compute_W(t2aa, t2ab,
                          ovov, ovOV)
        W_bb = _compute_W(t2bb, t2ab.transpose(2, 3, 0, 1),
                          OVOV, ovOV.transpose(2, 3, 0, 1))

        # 2. Diagonalize M = F_oo + (alpha/2)*W for dressed occupied eigenvalues
        M_a = F_oo_a + (bws.alpha / 2) * W_aa
        M_b = F_oo_b + (bws.alpha / 2) * W_bb
        eps_oa_dressed, U_a_new = numpy.linalg.eigh(M_a)
        eps_ob_dressed, U_b_new = numpy.linalg.eigh(M_b)

        # 3. Accumulate rotations; re-dress all three integral blocks from canonical
        U_cum_a = U_cum_a @ U_a_new
        U_cum_b = U_cum_b @ U_b_new
        ovov = _rotate_ovov(U_cum_a, ovov_can)
        OVOV = _rotate_ovov(U_cum_b, OVOV_can)
        ovOV = _rotate_ovOV(U_cum_a, U_cum_b, ovOV_can)
        F_oo_a = U_cum_a.T @ F_oo_a_can @ U_cum_a
        F_oo_b = U_cum_b.T @ F_oo_b_can @ U_cum_b

        # 4. Dressed amplitudes
        eia_a = eps_oa_dressed[:, None] - eps_va[None, :]   # (nocca, nvira)
        eia_b = eps_ob_dressed[:, None] - eps_vb[None, :]   # (noccb, nvirb)
        Daa = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
        Dab = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
        Dbb = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)
        t2aa = ovov / Daa
        t2ab = ovOV / Dab
        t2bb = OVOV / Dbb

        # 5. Correlation energy (converge on total)
        e_corr = energy(bws, (t2aa, t2ab, t2bb), (ovov, ovOV, OVOV))

        log.info('cycle = %d  E_corr(UBW-s2) = %.15g  dE = %.9g',
                 cycle + 1, e_corr, e_corr - e_prev)
        cput1 = log.timer('UBW-s2 iter', *cput1)

        if abs(e_corr - e_prev) < bws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('UBW-s2 did not converge after %d cycles.', bws.max_cycle)

    log.timer('UBW-s2', *cput0)
    return conv, e_corr, (t2aa, t2ab, t2bb)


def init_amps(bws, eris):
    '''Canonical UMP2 amplitudes as the UBW-s2 starting point.

    Returns
    -------
    e_corr : float
        UMP2 correlation energy.
    t2 : tuple (t2aa, t2ab, t2bb)
        Non-antisymmetrized, shapes (nocca,nvira,nocca,nvira),
        (nocca,nvira,noccb,nvirb), (noccb,nvirb,noccb,nvirb).
    '''
    nocca, noccb = bws.get_nocc()
    nmoa, nmob   = bws.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_ea, mo_eb = eris.mo_energy

    eia_a = mo_ea[:nocca, None] - mo_ea[None, nocca:]   # (nocca, nvira)
    eia_b = mo_eb[:noccb, None] - mo_eb[None, noccb:]   # (noccb, nvirb)

    ovov = numpy.asarray(eris.ovov).reshape(nocca, nvira, nocca, nvira)
    ovOV = numpy.asarray(eris.ovOV).reshape(nocca, nvira, noccb, nvirb)
    OVOV = numpy.asarray(eris.OVOV).reshape(noccb, nvirb, noccb, nvirb)

    Daa = lib.direct_sum('ia,jb->iajb', eia_a, eia_a)
    Dab = lib.direct_sum('ia,JB->iaJB', eia_a, eia_b)
    Dbb = lib.direct_sum('IA,JB->IAJB', eia_b, eia_b)

    t2aa = ovov / Daa
    t2ab = ovOV / Dab
    t2bb = OVOV / Dbb
    t2   = (t2aa, t2ab, t2bb)
    return energy(bws, t2, (ovov, ovOV, OVOV)), t2


def energy(bws, t2, ovovs):
    '''UBW-s2 correlation energy from dressed amplitudes and integrals.

    E_aa = (1/2) sum_{iajb} t2aa[i,a,j,b] * [(ia|jb) - (ib|ja)]
    E_ab =       sum_{iaJB} t2ab[i,a,J,B] * (ia|JB)
    E_bb = (1/2) sum_{IAJB} t2bb[I,A,J,B] * [(IA|JB) - (IB|JA)]
    '''
    t2aa, t2ab, t2bb = t2
    ovov, ovOV, OVOV = ovovs
    E_aa = 0.5 * numpy.einsum('iajb,iajb->', t2aa,
                               ovov - ovov.transpose(0, 3, 2, 1)).real
    E_ab = numpy.einsum('iaJB,iaJB->', t2ab, ovOV).real
    E_bb = 0.5 * numpy.einsum('IAJB,IAJB->', t2bb,
                               OVOV - OVOV.transpose(0, 3, 2, 1)).real
    return E_aa + E_ab + E_bb


def _compute_W(t2_ss, t2_os, ovov_ss, ovov_os):
    '''Compute one spin block of W.

    Can compute both W_aa and W_bb via index permutation at the call site:
        W_aa = _compute_W(t2aa, t2ab, ovov, ovOV)
        W_bb = _compute_W(t2bb, t2ab.transpose(2,3,0,1),
                          OVOV, ovOV.transpose(2,3,0,1))

    Parameters
    ----------
    t2_ss  : (nocc, nvir, nocc, nvir)    same-spin amps, non-antisymmetrized
    t2_os  : (nocc, nvir, nocc', nvir')  opposite-spin amps
    ovov_ss: (nocc, nvir, nocc, nvir)    same-spin ERIs
    ovov_os: (nocc, nvir, nocc', nvir')  opposite-spin ERIs;
             first two indices belong to THIS spin block.

    The W formula (equations.md):
        W_ij = (1/2) * (X_ss[i,j] + X_ss[j,i] + X_os[i,j] + X_os[j,i])
    where
        X_ss[i,j] = sum_{k,a,b} ovov_ss[i,a,k,b] * T_ss[j,a,k,b]
        X_os[i,j] = sum_{K,a,B} ovov_os[i,a,K,B] * t2_os[j,a,K,B]
        T_ss[j,a,k,b] = t2_ss[j,a,k,b] - t2_ss[j,b,k,a]  (factor 1, not 2)

    Key property: tr(W_aa) + tr(W_bb) = E_corr (size-consistency condition).
    '''
    T_ss = t2_ss - t2_ss.transpose(0, 3, 2, 1)
    X_ss = numpy.einsum('iakb,jakb->ij', ovov_ss, T_ss)
    X_os = numpy.einsum('iaKB,jaKB->ij', ovov_os, t2_os)
    return 0.5 * (X_ss + X_ss.T + X_os + X_os.T)


def _rotate_ovOV(U_cum_a, U_cum_b, ovOV_can):
    '''Rotate canonical ovOV to the dressed occupied-orbital basis.

    (ia|JB)_dressed = sum_{I,J0} U_cum_a[I,i] * U_cum_b[J0,J] * (Ia|J0 B)_can

    Alpha occ index rotates with U_cum_a; beta occ index rotates with U_cum_b.
    Virtual indices (a, B) remain canonical throughout.
    '''
    tmp = numpy.einsum('Ii,IaJB->iaJB', U_cum_a, ovOV_can)
    return numpy.einsum('Jj,iaJB->iajB', U_cum_b, tmp)


# ---------------------------------------------------------------------------
# Bookkeeping helpers (no frozen-orbital support)
# ---------------------------------------------------------------------------

def get_nocc(bws):
    if bws._nocc is not None:
        return bws._nocc
    nocca = numpy.count_nonzero(bws.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(bws.mo_occ[1] > 0)
    return nocca, noccb


def get_nmo(bws):
    if bws._nmo is not None:
        return bws._nmo
    return bws.mo_occ[0].size, bws.mo_occ[1].size


# ---------------------------------------------------------------------------
# ERI storage and transformation
# ---------------------------------------------------------------------------

class _ChemistsERIs:
    '''Stores the three (ov|ov) integral blocks for UBW-s2.

    mo_energy : tuple (mo_ea, mo_eb)  — full MO energy arrays
    fock      : tuple (focka, fockb)  — MO-basis Fock matrices
    ovov      : alpha-alpha (ia|jb)
    ovOV      : alpha-beta  (ia|JB)
    OVOV      : beta-beta   (IA|JB)
    '''

    def __init__(self, mol=None):
        self.mol       = mol
        self.mo_coeff  = None
        self.mo_energy = None
        self.fock      = None
        self.ovov      = None
        self.ovOV      = None
        self.OVOV      = None

    def _common_init_(self, bws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = bws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. '
                               'Run mf.kernel() first.')
        self.mol = bws.mol
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]
        self.mo_coeff = (mo_a, mo_b)

        if mo_coeff is bws._scf.mo_coeff and bws._scf.converged:
            self.mo_energy = (bws._scf.mo_energy[0], bws._scf.mo_energy[1])
            self.fock = (numpy.diag(self.mo_energy[0]),
                         numpy.diag(self.mo_energy[1]))
        else:
            dm     = bws._scf.make_rdm1(mo_coeff, bws.mo_occ)
            vhf    = bws._scf.get_veff(bws.mol, dm)
            fockao = bws._scf.get_fock(vhf=vhf, dm=dm)
            focka  = mo_a.conj().T.dot(fockao[0]).dot(mo_a)
            fockb  = mo_b.conj().T.dot(fockao[1]).dot(mo_b)
            self.fock      = (focka, fockb)
            self.mo_energy = (focka.diagonal().real, fockb.diagonal().real)
        return self


def _make_eris(bws, mo_coeff=None, ao2mofn=None, verbose=None):
    log   = logger.new_logger(bws, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris  = _ChemistsERIs()
    eris._common_init_(bws, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocca, noccb = bws.get_nocc()
    nmoa,  nmob  = bws.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nao          = mo_coeff[0].shape[0]

    moa, mob = mo_coeff
    orboa = numpy.asarray(moa[:, :nocca], order='F')
    orbva = numpy.asarray(moa[:, nocca:], order='F')
    orbob = numpy.asarray(mob[:, :noccb], order='F')
    orbvb = numpy.asarray(mob[:, noccb:], order='F')

    nao_pair = nao * (nao + 1) // 2
    mem_incore = nao_pair**2 * 8 / 1e6
    mem_now    = lib.current_memory()[0]

    if (bws.mol.incore_anyway or
            (bws._scf._eri is not None and
             mem_incore + mem_now < bws.max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((orboa, orbva, orboa, orbva)).reshape(
                nocca * nvira, nocca * nvira)
            eris.ovOV = ao2mofn((orboa, orbva, orbob, orbvb)).reshape(
                nocca * nvira, noccb * nvirb)
            eris.OVOV = ao2mofn((orbob, orbvb, orbob, orbvb)).reshape(
                noccb * nvirb, noccb * nvirb)
        else:
            eris.ovov = ao2mo.general(bws._scf._eri, (orboa, orbva, orboa, orbva))
            eris.ovOV = ao2mo.general(bws._scf._eri, (orboa, orbva, orbob, orbvb))
            eris.OVOV = ao2mo.general(bws._scf._eri, (orbob, orbvb, orbob, orbvb))

    elif getattr(bws._scf, 'with_df', None):
        log.warn('DF-UHF detected. (ia|jb) computed from DF 3-index tensors.\n'
                 'Consider using a DF-UBW-s2 implementation for better performance.')
        eris.ovov = bws._scf.with_df.ao2mo((orboa, orbva, orboa, orbva))
        eris.ovOV = bws._scf.with_df.ao2mo((orboa, orbva, orbob, orbvb))
        eris.OVOV = bws._scf.with_df.ao2mo((orbob, orbvb, orbob, orbvb))

    else:
        from pyscf.mp import ump2 as _ump2
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        _ump2._ao2mo_ovov(bws, (orboa, orbva, orbob, orbvb),
                          eris.feri, max(2000, bws.max_memory), log)
        eris.ovov = (eris.feri['ovov'] if nocca * nvira > 0
                     else numpy.zeros((nocca * nvira, nocca * nvira)))
        eris.ovOV = (eris.feri['ovOV'] if nocca * nvira * noccb * nvirb > 0
                     else numpy.zeros((nocca * nvira, noccb * nvirb)))
        eris.OVOV = (eris.feri['OVOV'] if noccb * nvirb > 0
                     else numpy.zeros((noccb * nvirb, noccb * nvirb)))

    log.timer('Integral transformation', *time0)
    return eris


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class UBWS2(BWSBase):
    '''Unrestricted BW-s2.

    Second-order size-consistent Brillouin-Wigner perturbation theory
    for open-shell (UHF) references. Alpha and beta occupied subspaces
    are dressed independently via separate W matrices.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.bws2 import UBWS2
    >>> mol = gto.M(atom='O', basis='sto-3g', spin=2)
    >>> mf = scf.UHF(mol).run()
    >>> bws = UBWS2(mf).run()
    >>> print(bws.e_corr)
    '''

    get_nocc = get_nocc
    get_nmo  = get_nmo
    get_e_hf = get_e_hf

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        nocca, noccb = self.get_nocc()
        nmoa,  nmob  = self.get_nmo()
        log.info('nocca = %s  noccb = %s  nmoa = %s  nmob = %s',
                 nocca, noccb, nmoa, nmob)
        log.info('max_cycle = %d  conv_tol = %g  alpha = %g',
                 self.max_cycle, self.conv_tol, self.alpha)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.t2 = kernel(
            self, mo_energy, mo_coeff, eris)
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

    init_amps = init_amps
    energy    = energy


from pyscf import scf
scf.uhf.UHF.BWS2 = lib.class_as_method(UBWS2)
