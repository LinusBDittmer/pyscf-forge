'''
Density-fitted Restricted RE-BWs2 (DF-RE-BWs2)

Uses the symmetric Coulomb-metric DF decomposition:
    (pq|rs) = sum_Q B^Q_pq B^Q_rs

B tensors stored per instance:
    B_ov[Q, i, a]   occupied-virtual   (naux, nocc, nvir)
    B_oo[Q, i, j]   occupied-occupied  (naux, nocc, nocc)
    B_vv[Q, a, b]   virtual-virtual    (naux, nvir, nvir)  -- always in-core

Per-iteration intermediates exploit t2 pair symmetry t^{ab}_{ij} = t^{ba}_{ji}
to reduce 4 B_ov contractions to 2:
    Theta[Q, i, a] = einsum('iajb,Qjb->Qia', t2, B_ov)   Upsilon = Theta
    Xi[Q, i, a]    = einsum('icka,Qkc->Qia', t2, B_ov)   Psi     = Xi

Driving term L15 carries a NEGATIVE sign (sign error in the_method.md corrected).
Working equations: pyscf/rebws/the_method.md
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.rebws.rebws2 import (get_nocc, get_nmo, get_e_hf,
                                 DIIS_SPACE, DIIS_START_CYCLE)


# ---------------------------------------------------------------------------
# ERI storage and transformation
# ---------------------------------------------------------------------------

class _DF_ChemistsERIs:
    '''Stores B-tensor slices for DF-RE-BWs2.'''

    def __init__(self):
        self.mol       = None
        self.mo_coeff  = None
        self.mo_energy = None
        self.fock      = None
        self.B_ov      = None   # (naux, nocc, nvir)
        self.B_oo      = None   # (naux, nocc, nocc)
        self.B_vv      = None   # (naux, nvir, nvir)

    def _common_init_(self, dfrebws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = dfrebws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. Run mf.kernel() first.')
        self.mo_coeff = mo_coeff
        self.mol      = dfrebws.mol

        if mo_coeff is dfrebws._scf.mo_coeff and dfrebws._scf.converged:
            self.mo_energy = dfrebws._scf.mo_energy
            self.fock      = numpy.diag(self.mo_energy)
        else:
            dm     = dfrebws._scf.make_rdm1(mo_coeff, dfrebws.mo_occ)
            vhf    = dfrebws._scf.get_veff(dfrebws.mol, dm)
            fockao = dfrebws._scf.get_fock(vhf=vhf, dm=dm)
            self.fock      = mo_coeff.conj().T @ fockao @ mo_coeff
            self.mo_energy = self.fock.diagonal().real
        return self


def _make_df_eris(dfrebws, mo_coeff=None, verbose=None):
    '''AO→MO B-tensor transform for DF-RE-BWs2 via a single AO integral pass.'''
    log   = logger.new_logger(dfrebws, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris  = _DF_ChemistsERIs()
    eris._common_init_(dfrebws, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = dfrebws.nocc
    nmo  = dfrebws.nmo
    nvir = nmo - nocc

    # Build or reuse DF object
    if dfrebws.with_df is not None:
        with_df = dfrebws.with_df
    else:
        from pyscf import df as _df_module
        from pyscf.df import addons
        auxbasis = dfrebws.auxbasis
        if auxbasis is None:
            auxbasis = addons.make_auxbasis(dfrebws.mol, mp2fit=True)
        with_df = _df_module.DF(dfrebws.mol, auxbasis=auxbasis)
        with_df.build()

    naux = with_df.get_naoaux()

    mem_B_ov  = naux * nocc * nvir * 8 / 1e6
    mem_B_oo  = naux * nocc * nocc * 8 / 1e6
    mem_B_vv  = naux * nvir * nvir * 8 / 1e6
    mem_total = mem_B_ov + mem_B_oo + mem_B_vv
    mem_now   = lib.current_memory()[0]
    max_memory = max(0, dfrebws.max_memory - mem_now)
    if mem_total > max_memory:
        log.warn('B tensors require %.1f MB but only %.1f MB available.',
                 mem_total, max_memory)
    if mem_B_vv > 0.8 * max_memory:
        log.warn('B_vv alone requires %.1f MB. Consider a smaller basis.', mem_B_vv)

    log.debug('DF-RE-BWs2: transforming B_ov, B_oo, B_vv (single AO pass)')

    B_ov_buf = numpy.zeros((naux, nocc * nvir))
    B_oo_buf = numpy.zeros((naux, nocc * nocc))
    B_vv_buf = numpy.zeros((naux, nvir * nvir))

    p0 = 0
    for eri1 in with_df.loop():
        p1 = p0 + eri1.shape[0]
        B_ov_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_coeff, (0, nocc, nocc, nmo),
                                        aosym='s2', mosym='s1')
        B_oo_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_coeff, (0, nocc, 0, nocc),
                                        aosym='s2', mosym='s1')
        B_vv_buf[p0:p1] = _ao2mo.nr_e2(eri1, mo_coeff, (nocc, nmo, nocc, nmo),
                                        aosym='s2', mosym='s1')
        p0 = p1

    eris.B_ov = B_ov_buf.reshape(naux, nocc, nvir)
    eris.B_oo = B_oo_buf.reshape(naux, nocc, nocc)
    eris.B_vv = B_vv_buf.reshape(naux, nvir, nvir)

    log.timer('DF-RE-BWs2 integral transform', *cput0)
    return eris


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def init_amps(dfrebws, eris):
    '''MP2 amplitudes as the DF-RE-BWs2 starting point.'''
    nocc    = dfrebws.nocc
    eps_occ = eris.mo_energy[:nocc]
    eps_vir = eris.mo_energy[nocc:]
    eia     = eps_occ[:, None] - eps_vir[None, :]
    Dij     = lib.direct_sum('ia,jb->iajb', eia, eia)

    B_ov = eris.B_ov
    ovov = numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)
    t2   = ovov / Dij
    return energy(dfrebws, t2, eris), t2


def energy(dfrebws, t2, eris):
    '''DF-RE-BWs2 correlation energy using pair-symmetry reduced intermediates.

    E = (2*einsum('Qia,Qia->', Theta, B_ov) - einsum('Qia,Qia->', Xi, B_ov)).real
    where Upsilon = Theta and Psi = Xi by pair symmetry.
    '''
    B_ov  = eris.B_ov
    Theta = numpy.einsum('iajb,Qjb->Qia', t2, B_ov)
    Xi    = numpy.einsum('icka,Qkc->Qia', t2, B_ov)
    return (2.0 * numpy.einsum('Qia,Qia->', Theta, B_ov)
            - numpy.einsum('Qia,Qia->', Xi, B_ov)).real


def _compute_H_ij_df(Theta, Xi, B_ov, eps_occ, alpha):
    '''DF H intermediate from pair-symmetry reduced intermediates.

    H = -(alpha/4)*(X + X.T) + (alpha/2)*(Y + Y.T) + diag(eps_occ)
    X[i,j] = einsum('Qia,Qja->ij', Xi, B_ov)     (from iakb contraction)
    Y[i,j] = einsum('Qia,Qja->ij', Theta, B_ov)  (from iajb contraction)
    '''
    X = numpy.einsum('Qia,Qja->ij', Xi, B_ov)
    Y = numpy.einsum('Qia,Qja->ij', Theta, B_ov)
    return (-(alpha / 4.0) * (X + X.T)
            + (alpha / 2.0) * (Y + Y.T)
            + numpy.diag(eps_occ))


def compute_residual_df(t2, H, Theta, Xi, B_ov, B_oo, B_vv, eps_vir, alpha, beta):
    '''DF restricted opposite-spin RE-BWs2 residual.

    B_ov-only terms are vectorised using pair-symmetry reduced intermediates:
        L12+L13 merged: -2*beta * outer_Q(B_ov, Theta)  [Upsilon=Theta]
        M1+M2 present independently: both use Xi        [Psi=Xi]

    Q-loop exploits pair symmetry for the oovv terms:
        Z56 = Z5 + Z5.T(2,3,0,1)   and   Z78 = Z7 + Z7.T(2,3,0,1)

    Driving term L15 carries NEGATIVE sign (sign correction from the_method.md).
    '''
    naux = B_ov.shape[0]
    R    = numpy.zeros_like(t2)

    # L1-L4: H contractions (no ERIs involved)
    R += numpy.einsum('iakb,jk->iajb', t2, H)      # L1
    R += numpy.einsum('kajb,ik->iajb', t2, H)      # L2
    R -= t2 * (eps_vir[None, :, None, None] + eps_vir[None, None, None, :])   # L3+L4

    if abs(beta) > 1e-14:
        # B_ov-only terms (vectorised)
        R -= 2.0 * beta * numpy.einsum('Qia,Qjb->iajb', Theta, B_ov)   # L10+L11
        R -= 2.0 * beta * numpy.einsum('Qia,Qjb->iajb', B_ov, Theta)   # L12+L13 (Upsilon=Theta)
        R += beta * numpy.einsum('Qia,Qjb->iajb', Xi, B_ov)            # M1
        R += beta * numpy.einsum('Qia,Qjb->iajb', B_ov, Xi)            # M2 (Psi=Xi)

        # Q-loop for B_oo/B_vv terms
        for Q in range(naux):
            Boo_Q = B_oo[Q]   # (nocc, nocc)
            Bvv_Q = B_vv[Q]   # (nvir, nvir)

            # L5+L6: Z6 = Z5.T(2,3,0,1) by pair symmetry
            Z5  = numpy.einsum('iakc,jk->iajc', t2, Boo_Q)
            Z56 = Z5 + Z5.transpose(2, 3, 0, 1)
            R  += beta * numpy.einsum('iajc,bc->iajb', Z56, Bvv_Q)

            # L7+L8: Z8 = Z7.T(2,3,0,1) by pair symmetry
            Z7  = numpy.einsum('ickb,jk->icjb', t2, Boo_Q)
            Z78 = Z7 + Z7.transpose(2, 3, 0, 1)
            R  += beta * numpy.einsum('icjb,ac->iajb', Z78, Bvv_Q)

            # L9: oooo contribution
            Z9 = numpy.einsum('kalb,jl->kajb', t2, Boo_Q)
            R -= beta * numpy.einsum('kajb,ik->iajb', Z9, Boo_Q)

            # L14: vvvv contribution
            Z14 = numpy.einsum('icjd,ac->iajd', t2, Bvv_Q)
            R  -= beta * numpy.einsum('iajd,bd->iajb', Z14, Bvv_Q)

    # L15: driving term (NEGATIVE sign — corrects sign error in the_method.md)
    R -= numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)

    return R


def kernel(dfrebws, eris=None, verbose=None):
    '''Iterative DF-RE-BWs2 solver with CC-DIIS acceleration.'''
    if eris is None:
        eris = _make_df_eris(dfrebws)
    log = logger.new_logger(dfrebws, verbose)

    nocc    = dfrebws.nocc
    nvir    = dfrebws.nmo - nocc
    eps_occ = eris.mo_energy[:nocc]
    eps_vir = eris.mo_energy[nocc:]
    B_ov    = eris.B_ov
    B_oo    = eris.B_oo
    B_vv    = eris.B_vv

    eia = eps_vir[None, :] - eps_occ[:, None]       # positive
    Dab = lib.direct_sum('ia,jb->iajb', eia, eia)   # positive denominator
    Dab[Dab < 5.e-2] = 5.e-2

    e_corr, t2 = init_amps(dfrebws, eris)
    log.info('Init E_corr(DF-RE-BWs2) = %.15g  [MP2 starting point]', e_corr)

    adiis = lib.diis.DIIS(dfrebws)
    adiis.space = dfrebws.diis_space

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    conv = False

    for cycle in range(dfrebws.max_cycle):
        e_prev = e_corr

        Theta = numpy.einsum('iajb,Qjb->Qia', t2, B_ov)
        Xi    = numpy.einsum('icka,Qkc->Qia', t2, B_ov)

        H = _compute_H_ij_df(Theta, Xi, B_ov, eps_occ, dfrebws.alpha)
        R = compute_residual_df(t2, H, Theta, Xi, B_ov, B_oo, B_vv,
                                eps_vir, dfrebws.alpha, dfrebws.beta)
        R_norm     = R / Dab
        conv_check = float(numpy.max(numpy.abs(R_norm)))

        t2_new = t2 + R_norm
        if cycle >= dfrebws.diis_start_cycle:
            t2_new = adiis.update(t2_new.ravel(),
                                  xerr=R_norm.ravel()).reshape(nocc, nvir, nocc, nvir)
        t2 = t2_new

        e_corr = energy(dfrebws, t2, eris)

        log.info('cycle = %d  E_corr(DF-RE-BWs2) = %.15g  dE = %.9g  |R|_max = %.6g',
                 cycle + 1, e_corr, e_corr - e_prev, conv_check)
        cput1 = log.timer('DF-RE-BWs2 iter', *cput1)

        if conv_check < dfrebws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('DF-RE-BWs2 did not converge after %d cycles.', dfrebws.max_cycle)

    log.timer('DF-RE-BWs2', *cput0)
    return conv, e_corr, t2


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class DFREBWS2(lib.StreamObject):
    '''Density-fitted RE-BWs2 for closed-shell (RHF) references.

    Attributes
    ----------
    alpha : float
        BW-s2 H-intermediate scaling. Default 1.0. alpha=0+beta=0 → MP2.
    beta : float
        RE contribution scaling. Default 1.0.
    auxbasis : str or None
        Auxiliary basis. None → auto-selected via make_auxbasis (mp2fit).
    max_cycle : int
        Maximum amplitude iterations. Default 50.
    conv_tol : float
        Convergence threshold on max|R_norm|. Default 1e-7.
    diis_space : int
        DIIS subspace size. Default 10.
    diis_start_cycle : int
        First cycle with DIIS extrapolation. Default 0.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.rebws import DFREBWS2
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> dfrebws = DFREBWS2(mf).run()
    >>> print(dfrebws.e_corr)
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
        self.t2         = None

    @property
    def mo_energy(self):
        return self._scf.mo_energy

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

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
        log.info('nocc = %d  nmo = %d', self.nocc, self.nmo)
        log.info('alpha = %.6g  beta = %.6g', self.alpha, self.beta)
        log.info('auxbasis = %s', self.auxbasis)
        log.info('max_cycle = %d  conv_tol = %g', self.max_cycle, self.conv_tol)
        log.info('diis_space = %d  diis_start_cycle = %d',
                 self.diis_space, self.diis_start_cycle)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def ao2mo(self, mo_coeff=None):
        '''Transform to B-tensor DF-ERIs.'''
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
scf.hf.RHF.DFREBWS2 = lib.class_as_method(DFREBWS2)
