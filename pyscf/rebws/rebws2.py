'''
Restricted RE-BWs2

Combines Retaining-the-Excitation-Degree (RE) perturbation theory with
size-consistent Brillouin-Wigner second-order perturbation theory (BWs2).

Hamiltonian partitioning:
    H^(0) = H^(0)_MP + alpha * H^(0)_BWs + beta * H^(0)_DeltaRE

The amplitude equation is solved via the restricted opposite-spin residual
R^{ab}_{ij} (derived by setting all beta-spin indices to alpha in the
unrestricted R^{aB}_{iJ}). Non-antisymmetrised amplitudes t2[i,a,j,b] are
used throughout, mirroring the BW-s2 convention.

Working equations and derivation: pyscf/rebws/the_method.md
'''

import numpy
from pyscf import ao2mo, lib
from pyscf.lib import logger

DIIS_SPACE       = 10
DIIS_START_CYCLE = 0


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------

def get_nocc(rebws):
    if rebws._nocc is not None:
        return rebws._nocc
    nocc = int(numpy.count_nonzero(rebws.mo_occ > 0))
    assert nocc > 0
    return nocc


def get_nmo(rebws):
    if rebws._nmo is not None:
        return rebws._nmo
    return len(rebws.mo_occ)


def get_e_hf(rebws, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = rebws.mo_coeff
    if mo_coeff is rebws._scf.mo_coeff and rebws._scf.converged:
        return rebws._scf.e_tot
    dm  = rebws._scf.make_rdm1(mo_coeff, rebws.mo_occ)
    vhf = rebws._scf.get_veff(rebws.mol, dm)
    return rebws._scf.energy_tot(dm=dm, vhf=vhf)


# ---------------------------------------------------------------------------
# ERI storage and transformation
# ---------------------------------------------------------------------------

def _mem_usage(nocc, nvir):
    '''Estimate memory (MB) for all four ERI blocks.

    Returns (total_mb, basic_mb, vvvv_mb) where basic excludes vvvv.
    '''
    ovov  = (nocc * nvir)**2 * 8 / 1e6
    oovv  = (nocc * nvir)**2 * 8 / 1e6   # same size as ovov
    oooo  = nocc**4 * 8 / 1e6
    vvvv  = nvir**4 * 8 / 1e6
    basic = ovov + oovv + oooo
    return basic + vvvv, basic, vvvv


class _ChemistsERIs:
    '''Stores the four ERI blocks required by RE-BWs2.

    Index conventions (chemist notation, all arrays contiguous C order):
        ovov[i,a,j,b] = (ia|jb)
        oovv[i,j,a,b] = (ij|ab)
        oooo[i,j,k,l] = (ij|kl)
        vvvv[a,b,c,d] = (ab|cd)
    '''
    def __init__(self):
        self.mol       = None
        self.mo_coeff  = None
        self.mo_energy = None
        self.fock      = None
        self.ovov      = None
        self.oovv      = None
        self.oooo      = None
        self.vvvv      = None

    def _common_init_(self, rebws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = rebws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. Run mf.kernel() first.')
        self.mo_coeff = mo_coeff
        self.mol      = rebws.mol

        if mo_coeff is rebws._scf.mo_coeff and rebws._scf.converged:
            self.mo_energy = rebws._scf.mo_energy
            self.fock      = numpy.diag(self.mo_energy)
        else:
            dm     = rebws._scf.make_rdm1(mo_coeff, rebws.mo_occ)
            vhf    = rebws._scf.get_veff(rebws.mol, dm)
            fockao = rebws._scf.get_fock(vhf=vhf, dm=dm)
            self.fock      = mo_coeff.conj().T @ fockao @ mo_coeff
            self.mo_energy = self.fock.diagonal().real
        return self


def _make_eris(rebws, mo_coeff=None, verbose=None):
    '''Transform AO ERIs into the four MO blocks required by RE-BWs2.

    Builds the full (nmo, nmo, nmo, nmo) MO tensor once and slices the
    required blocks from it. For large systems a block-by-block transform
    using ao2mo.general should replace this approach.
    '''
    log   = logger.new_logger(rebws, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris  = _ChemistsERIs()
    eris._common_init_(rebws, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = rebws.nocc
    nmo  = rebws.nmo
    nvir = nmo - nocc

    mem_incore, mem_basic, mem_vvvv = _mem_usage(nocc, nvir)
    mem_now    = lib.current_memory()[0]
    max_memory = max(0, rebws.max_memory - mem_now)

    if mem_incore > max_memory:
        log.warn('ERI blocks require %.1f MB but only %.1f MB is available.',
                 mem_incore, max_memory)
    if mem_vvvv > 0.8 * max_memory:
        log.warn('vvvv block alone requires %.1f MB. Consider a smaller basis.',
                 mem_vvvv)

    log.debug('Transforming ERI blocks: ovov, oovv, oooo, vvvv')
    eri_full = ao2mo.restore(1, ao2mo.full(rebws.mol, mo_coeff), nmo)
    eris.ovov = numpy.asarray(eri_full[:nocc, nocc:, :nocc, nocc:])
    eris.oovv = numpy.asarray(eri_full[:nocc, :nocc, nocc:, nocc:])
    eris.oooo = numpy.asarray(eri_full[:nocc, :nocc, :nocc, :nocc])
    eris.vvvv = numpy.asarray(eri_full[nocc:, nocc:, nocc:, nocc:])
    eri_full  = None

    log.timer('RE-BWs2 ERI transformation', *cput0)
    return eris


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def init_amps(rebws, eris):
    '''MP2 amplitudes as the RE-BWs2 starting point.

    t2[i,a,j,b] = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)

    Returns
    -------
    e_corr : float
        MP2 correlation energy (= RE-BWs2 energy at cycle 0).
    t2 : ndarray, shape (nocc, nvir, nocc, nvir)
    '''
    nocc    = rebws.nocc
    eps_occ = eris.mo_energy[:nocc]
    eps_vir = eris.mo_energy[nocc:]
    eia     = eps_occ[:, None] - eps_vir[None, :]       # (nocc, nvir), negative
    Dij     = lib.direct_sum('ia,jb->iajb', eia, eia)   # (nocc, nvir, nocc, nvir), negative
    ovov    = numpy.asarray(eris.ovov)
    t2      = ovov / Dij
    return energy(rebws, t2, ovov), t2


def energy(rebws, t2, ovov):
    '''RE-BWs2 correlation energy.

    E_corr = sum_{i,a,j,b} t2[i,a,j,b] * (2*(ia|jb) - (ib|ja))
    '''
    return numpy.einsum('iajb,iajb->', t2,
                        2.0 * ovov - ovov.transpose(0, 3, 2, 1)).real


def _compute_H_ij(t2, ovov, eps_occ, alpha):
    '''Build the (nocc, nocc) H intermediate (restricted, canonical).

    Derived by restricting the unrestricted H^alpha_ij to closed-shell.
    In the unrestricted formulation, H_aa has three contributions:
        - same-spin X_aa/Y_aa terms (coeff alpha/8) using antisymmetric t2aa
        - opposite-spin Z_ab term (coeff alpha/4) using t2ab = t2_R

    Substituting t2aa = t2_R - t2_R.T_0321 and t2ab = t2_R at closed shell,
    X_aa = X - Y, Y_aa = Y - X, Z_ab = Y (where X, Y use t2_R), and the
    combined result is:

    Let:
        X[i,j] = sum_{k,a,b} t2[i,a,k,b] * ovov[j,b,k,a]   (jb|ka) in chemist notation
        Y[i,j] = sum_{k,a,b} t2[i,a,k,b] * ovov[j,a,k,b]   (ja|kb)

    Then:
        H = -(alpha/4)*(X + X.T) + (alpha/2)*(Y + Y.T) + diag(eps_occ)

    Reduces to diag(eps_occ) at alpha=0. H is symmetric.
    '''
    nocc = t2.shape[0]
    # X[i,j] = sum_{k,a,b} t2[i,a,k,b] * ovov[j,b,k,a]
    X = numpy.einsum('iakb,jbka->ij', t2, ovov)
    # Y[i,j] = sum_{k,a,b} t2[i,a,k,b] * ovov[j,a,k,b]
    # Written as a matrix multiply over the flattened (a,k,b) dimension.
    Y = t2.reshape(nocc, -1) @ ovov.reshape(nocc, -1).T
    H = (-(alpha / 4.0) * (X + X.T)
         + (alpha / 2.0) * (Y + Y.T)
         + numpy.diag(eps_occ))
    return H


def compute_residual(t2, H, eris, eps_vir, alpha, beta):
    '''Restricted opposite-spin RE-BWs2 residual.

    Derived from the unrestricted R^{aB}_{iJ} by setting all beta-spin
    indices to alpha. In canonical orbitals the H_vv = diag(eps_vir)
    contractions (L3, L4) simplify to element-wise multiplications.

    The driving term (L15) carries a NEGATIVE sign — the document
    the_method.md has a sign error on this term.

    ERI block access (chemist notation):
        ovov[i,a,j,b] = (ia|jb)
        oovv[i,j,a,b] = (ij|ab),  e.g. (jk|bc) = oovv[j,k,b,c]
        oooo[i,j,k,l] = (ij|kl),  e.g. (ik|jl) = oooo[i,k,j,l]
        vvvv[a,b,c,d] = (ab|cd),  e.g. (ac|bd) = vvvv[a,c,b,d]

    Parameters
    ----------
    t2 : ndarray, shape (nocc, nvir, nocc, nvir)
    H  : ndarray, shape (nocc, nocc) — H intermediate from _compute_H_ij
    eris : _ChemistsERIs
    eps_vir : ndarray, shape (nvir,) — canonical virtual orbital energies
    alpha, beta : float — RE-BWs2 scaling parameters
    '''
    ovov = numpy.asarray(eris.ovov)
    oovv = numpy.asarray(eris.oovv)
    oooo = numpy.asarray(eris.oooo)
    vvvv = numpy.asarray(eris.vvvv)

    R = numpy.zeros_like(t2)

    # L1: sum_k t2[i,a,k,b] * H[j,k]   <- t^ab_ik H^j_k
    R += numpy.einsum('iakb,jk->iajb', t2, H)
    # L2: sum_k t2[k,a,j,b] * H[i,k]   <- t^ab_kj H^i_k
    R += numpy.einsum('kajb,ik->iajb', t2, H)

    # L3+L4: canonical H_vv = diag(eps_vir)
    #   L3: -t2[i,a,j,b] * eps_vir[b]   (from -t^ac_ij H^b_c, canonical)
    #   L4: -t2[i,a,j,b] * eps_vir[a]   (from -t^cb_ij H^a_c, canonical)
    R -= t2 * (eps_vir[None, :, None, None] + eps_vir[None, None, None, :])

    if abs(beta) > 1e-14:
        # L5: +beta * sum_{k,c} t2[i,a,k,c] * oovv[j,k,b,c]   (jk|bc)
        R += beta * numpy.einsum('iakc,jkbc->iajb', t2, oovv)
        # L6: +beta * sum_{k,c} t2[k,a,j,c] * oovv[i,k,b,c]   (ik|bc)
        R += beta * numpy.einsum('kajc,ikbc->iajb', t2, oovv)
        # L7: +beta * sum_{k,c} t2[i,c,k,b] * oovv[j,k,a,c]   (jk|ac)
        R += beta * numpy.einsum('ickb,jkac->iajb', t2, oovv)
        # L8: +beta * sum_{k,c} t2[k,c,j,b] * oovv[i,k,a,c]   (ik|ac)
        R += beta * numpy.einsum('kcjb,ikac->iajb', t2, oovv)
        # L9: -beta * sum_{k,l} t2[k,a,l,b] * oooo[i,k,j,l]   (ik|jl)
        R -= beta * numpy.einsum('kalb,ikjl->iajb', t2, oooo)
        # L10+L11: -2*beta * sum_{k,c} t2[i,a,k,c] * ovov[j,b,k,c]
        #   Two identical terms collapse: (kc|jb) = (jb|kc) by 8-fold symmetry
        R -= 2.0 * beta * numpy.einsum('iakc,jbkc->iajb', t2, ovov)
        # L12: -beta * sum_{k,c} t2[j,b,k,c] * ovov[i,a,k,c]   (ia|kc)
        R -= beta * numpy.einsum('jbkc,iakc->iajb', t2, ovov)
        # L13: -beta * sum_{k,c} t2[k,c,j,b] * ovov[i,a,k,c]   (ia|kc), different t2 access
        R -= beta * numpy.einsum('kcjb,iakc->iajb', t2, ovov)
        # L14: -beta * sum_{c,d} t2[i,c,j,d] * vvvv[a,c,b,d]   (ac|bd)
        R -= beta * numpy.einsum('icjd,acbd->iajb', t2, vvvv)
        # M1/M2: antisymmetry corrections from t2aa/t2bb in the opposite-spin residual.
        # In the unrestricted R_ab, L10 uses antisymmetric t2aa and L12 uses antisymmetric
        # t2bb.  At closed shell, the extra terms from antisymmetrisation are:
        #   M1 from L10:  +beta * sum_{k,c} t2[i,c,k,a] * ovov[j,b,k,c]
        #   M2 from L12:  +beta * sum_{k,c} t2[j,c,k,b] * ovov[i,a,k,c]
        R += beta * numpy.einsum('icka,jbkc->iajb', t2, ovov)
        R += beta * numpy.einsum('jckb,iakc->iajb', t2, ovov)

    # L15: driving term — NEGATIVE sign (corrected from the_method.md)
    R -= ovov

    return R


def kernel(rebws, eris=None, verbose=None):
    '''Iterative RE-BWs2 solver.

    Algorithm:
        1. Initialise t2 from MP2 amplitudes.
        2. Build H intermediate from current t2 and ovov.
        3. Compute and normalise residual R_norm = R / Delta_eps.
        4. Update: t2_new = t2 - R_norm.
        5. Apply CC-DIIS from cycle diis_start_cycle onward
           (error vector = R_norm, trial vector = t2_new).
        6. Repeat until max|R_norm| < conv_tol.

    Returns
    -------
    conv : bool
    e_corr : float
    t2 : ndarray, shape (nocc, nvir, nocc, nvir)
    '''
    if eris is None:
        eris = _make_eris(rebws)
    log = logger.new_logger(rebws, verbose)

    nocc    = rebws.nocc
    nvir    = rebws.nmo - nocc
    eps_occ = eris.mo_energy[:nocc]
    eps_vir = eris.mo_energy[nocc:]
    ovov    = numpy.asarray(eris.ovov)

    # Delta_eps[i,a,j,b] = (eps_a - eps_i) + (eps_b - eps_j)  > 0
    eia = eps_vir[None, :] - eps_occ[:, None]       # (nocc, nvir), positive
    Dab = lib.direct_sum('ia,jb->iajb', eia, eia)   # (nocc, nvir, nocc, nvir), positive
    Dab[Dab < 5.e-2] = 5.e-2

    # Iteration 0: MP2 starting point
    e_corr, t2 = init_amps(rebws, eris)
    log.info('Init E_corr(RE-BWs2) = %.15g  [MP2 starting point]', e_corr)

    adiis = lib.diis.DIIS(rebws)
    adiis.space = rebws.diis_space

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    conv = False

    for cycle in range(rebws.max_cycle):
        e_prev = e_corr

        # Build H intermediate
        H = _compute_H_ij(t2, ovov, eps_occ, rebws.alpha)

        # Compute and normalise the residual
        R      = compute_residual(t2, H, eris, eps_vir, rebws.alpha, rebws.beta)
        R_norm = R / Dab

        # Convergence check: max absolute element of the normalised residual
        conv_check = float(numpy.max(numpy.abs(R_norm)))

        # Amplitude update
        t2_new = t2 + R_norm

        # CC-style DIIS: error vector = R_norm, trial vector = t2_new
        if cycle >= rebws.diis_start_cycle:
            t2_new = adiis.update(t2_new.ravel(),
                                  xerr=R_norm.ravel()).reshape(nocc, nvir, nocc, nvir)

        t2     = t2_new
        e_corr = energy(rebws, t2, ovov)

        log.info('cycle = %d  E_corr(RE-BWs2) = %.15g  dE = %.9g  |R|_max = %.6g',
                 cycle + 1, e_corr, e_corr - e_prev, conv_check)
        cput1 = log.timer('RE-BWs2 iter', *cput1)

        if conv_check < rebws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('RE-BWs2 did not converge after %d cycles.', rebws.max_cycle)

    log.timer('RE-BWs2', *cput0)
    return conv, e_corr, t2


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class REBWS2(lib.StreamObject):
    '''Restricted RE-BWs2 for closed-shell (RHF) references.

    Attributes
    ----------
    alpha : float
        Scaling of the BW-s2 H-intermediate contribution. Default 1.0.
        alpha=0 with any beta recovers MP2.
    beta : float
        Scaling of the RE contribution. Default 1.0.
        beta=0 turns off all oovv/oooo/vvvv contractions in the residual.
    max_cycle : int
        Maximum number of amplitude iterations. Default 50.
    conv_tol : float
        Convergence threshold on max|R_norm|. Default 1e-7.
    diis_space : int
        DIIS subspace size. Default 6.
    diis_start_cycle : int
        First iteration on which DIIS extrapolation is applied. Default 3.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.rebws import REBWS2
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> rebws = REBWS2(mf).run()
    >>> print(rebws.e_corr)
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

        # Results
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
        '''Return full (nmo, nmo, nmo, nmo) MO ERI tensor.

        Utility method preserved from the original skeleton for external use.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo    = mo_coeff.shape[1]
        eri_mo = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose)
        return ao2mo.restore(1, eri_mo, nmo)

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
scf.hf.RHF.REBWS2 = lib.class_as_method(REBWS2)
