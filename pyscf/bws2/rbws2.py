'''
Restricted BW-s2

Reference:
    Kevin Carter-Fenk and Martin Head-Gordon,
    J. Chem. Phys. 158, 234108 (2023); doi: 10.1063/5.0150033
'''

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo


# ---------------------------------------------------------------------------
# BW-s2 specific computational functions
# ---------------------------------------------------------------------------

def kernel(bws, mo_energy=None, mo_coeff=None, eris=None, verbose=None):
    '''Iterative BW-s2 solver.

    Returns
    -------
    conv : bool
    e_corr : float
    t2 : ndarray, shape (nocc, nvir, nocc, nvir)
        Final amplitudes in the converged dressed-orbital basis.
    '''
    if eris is None:
        eris = bws.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = bws.nocc
    nvir = bws.nmo - nocc
    eps_occ = mo_energy[:nocc]
    eps_vir = mo_energy[nocc:]

    log = logger.new_logger(bws, verbose)

    # Load canonical ovov into memory once; kept throughout all iterations.
    # Shape: (nocc, nvir, nocc, nvir)
    ovov_can = numpy.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)

    # --- iteration 0: canonical MP2 starting point ---
    e_corr, t2 = init_amps(bws, eris)
    log.info('Init E_corr(BW-s2) = %.15g  [MP2]', e_corr)

    F_oo            = numpy.diag(eps_occ)
    U_cum           = numpy.eye(nocc)
    eps_occ_dressed = eps_occ.copy()
    ovov            = ovov_can.copy()
    F_oo_can        = F_oo.copy()

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())

    conv = False
    for cycle in range(bws.max_cycle):
        e_prev = e_corr

        # 1. Build W from current dressed amplitudes and integrals
        W = _compute_W(t2, ovov)

        # 2. Diagonalize F_oo + (alpha/2)*W for new dressed occupied eigenvalues
        M = F_oo + (bws.alpha / 2) * W
        eps_occ_dressed, U_new = numpy.linalg.eigh(M)

        # 3. Compose cumulative rotation and re-dress ovov from canonical
        U_cum = U_cum @ U_new
        ovov  = _rotate_ovov(U_cum, ovov_can)
        F_oo = U_cum.T @ F_oo_can @ U_cum

        # 4. Dressed amplitudes: t_ij^ab = (ia|jb)_dressed / Delta_ij^ab
        #    Delta[i,a,j,b] = eps_occ_dressed[i] + eps_occ_dressed[j]
        #                     - eps_vir[a] - eps_vir[b]  (negative)
        eia = eps_occ_dressed[:, None] - eps_vir[None, :]   # (nocc, nvir)
        Dij = lib.direct_sum('ia,jb->iajb', eia, eia)
        t2  = ovov / Dij

        # 5. BW-s2 correlation energy
        e_corr = energy(bws, t2, ovov)

        log.info('cycle = %d  E_corr(BW-s2) = %.15g  dE = %.9g',
                 cycle + 1, e_corr, e_corr - e_prev)
        cput1 = log.timer('BW-s2 iter', *cput1)

        if abs(e_corr - e_prev) < bws.conv_tol:
            conv = True
            break

    if not conv:
        log.warn('BW-s2 did not converge after %d cycles.', bws.max_cycle)

    log.timer('BW-s2', *cput0)
    return conv, e_corr, t2


def init_amps(bws, eris):
    '''Compute canonical MP2 amplitudes as the BW-s2 starting point.

    Returns
    -------
    e_corr : float
        MP2 correlation energy (= BW-s2 energy at iteration 0).
    t2 : ndarray, shape (nocc, nvir, nocc, nvir)
    '''
    nocc     = bws.nocc
    nvir     = bws.nmo - nocc
    mo_energy = eris.mo_energy
    eia      = mo_energy[:nocc, None] - mo_energy[None, nocc:]   # (nocc, nvir)
    Dij      = lib.direct_sum('ia,jb->iajb', eia, eia)
    ovov     = numpy.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
    t2       = ovov / Dij
    return energy(bws, t2, ovov), t2


def energy(bws, t2, ovov):
    '''BW-s2 correlation energy from current dressed amplitudes and integrals.

    E_c = sum_{ijab} t_ij^ab [2*(ia|jb) - (ib|ja)]
    '''
    return numpy.einsum('iajb,iajb->', t2,
                        2 * ovov - ovov.transpose(0, 3, 2, 1)).real


def _compute_W(t2, ovov):
    '''Build the W matrix (occ x occ) from current amplitudes and ovov.

    W_ij = (1/2) sum_{k,a,b} [T_jk^ab (ia|kb) + T_ik^ab (ja|kb)]

    where T_ij^ab = 2*t_ij^ab - t_ij^ba  (scaled amplitudes).
    Key property: tr(W) = E_corr (ensures size-consistency cancellation).
    '''
    T = 2 * t2 - t2.transpose(0, 3, 2, 1)       # T[i,a,j,b] = 2t - t^{ba}
    X = numpy.einsum('jakb,iakb->ij', T, ovov)   # asymmetric intermediate
    return 0.5 * (X + X.T)


def _rotate_ovov(U_cum, ovov_can):
    '''Rotate canonical ovov to the current dressed occupied-orbital basis.

    (ia|jb)_dressed = sum_{I,J} U_cum[I,i] * U_cum[J,j] * (IA|JB)_can

    Only the occupied indices rotate; virtual energies stay canonical.
    Cost: O(N_occ^2 * N_vir^2) — no AO->MO transform required.
    '''
    tmp = numpy.einsum('Ii,IaJb->iaJb', U_cum, ovov_can)
    return numpy.einsum('Jj,iaJb->iajb', U_cum, tmp)


# ---------------------------------------------------------------------------
# Bookkeeping helpers (no frozen-orbital support in this version)
# ---------------------------------------------------------------------------

def get_nocc(bws):
    if bws._nocc is not None:
        return bws._nocc
    nocc = numpy.count_nonzero(bws.mo_occ > 0)
    assert nocc > 0
    return nocc


def get_nmo(bws):
    if bws._nmo is not None:
        return bws._nmo
    return len(bws.mo_occ)


def get_e_hf(bws, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = bws.mo_coeff
    if mo_coeff is bws._scf.mo_coeff and bws._scf.converged:
        return bws._scf.e_tot
    dm  = bws._scf.make_rdm1(mo_coeff, bws.mo_occ)
    vhf = bws._scf.get_veff(bws._scf.mol, dm)
    return bws._scf.energy_tot(dm=dm, vhf=vhf)


# ---------------------------------------------------------------------------
# ERI transformation (identical strategy to mp2._make_eris / _ao2mo_ovov)
# ---------------------------------------------------------------------------

def _mem_usage(nocc, nvir):
    nmo     = nocc + nvir
    basic   = ((nocc * nvir)**2 + nocc * nvir**2 * 2) * 8 / 1e6
    incore  = nocc * nvir * nmo**2 / 2 * 8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic


class _ChemistsERIs:
    def __init__(self, mol=None):
        self.mol      = mol
        self.mo_coeff = None
        self.mo_energy = None
        self.fock     = None
        self.ovov     = None

    def _common_init_(self, bws, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = bws.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff is not initialised. '
                               'Run mf.kernel() first.')
        self.mo_coeff = mo_coeff
        self.mol      = bws.mol

        if mo_coeff is bws._scf.mo_coeff and bws._scf.converged:
            # Canonical MOs from a converged SCF: Fock is diagonal.
            self.mo_energy = bws._scf.mo_energy
            self.fock      = numpy.diag(self.mo_energy)
        else:
            dm     = bws._scf.make_rdm1(mo_coeff, bws.mo_occ)
            vhf    = bws._scf.get_veff(bws.mol, dm)
            fockao = bws._scf.get_fock(vhf=vhf, dm=dm)
            self.fock      = mo_coeff.conj().T.dot(fockao).dot(mo_coeff)
            self.mo_energy = self.fock.diagonal().real
        return self


def _make_eris(bws, mo_coeff=None, ao2mofn=None, verbose=None):
    log   = logger.new_logger(bws, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris  = _ChemistsERIs()
    eris._common_init_(bws, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = bws.nocc
    nmo  = bws.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now    = lib.current_memory()[0]
    max_memory = max(0, bws.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:, :nocc], order='F')
    cv = numpy.asarray(mo_coeff[:, nocc:], order='F')

    if (bws.mol.incore_anyway or
            (bws._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co, cv, co, cv)).reshape(nocc * nvir,
                                                           nocc * nvir)
        else:
            eris.ovov = ao2mo.general(bws._scf._eri, (co, cv, co, cv))

    elif getattr(bws._scf, 'with_df', None):
        log.warn('DF-HF detected. (ia|jb) computed from DF 3-index tensors.\n'
                 'Consider using a DF-BW-s2 implementation for better '
                 'performance.')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = bws._scf.with_df.ao2mo((co, cv, co, cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        eris.ovov = _ao2mo_ovov(bws, co, cv, eris.feri,
                                max(2000, max_memory), log)

    log.timer('Integral transformation', *time0)
    return eris


def _ao2mo_ovov(bws, orbo, orbv, feri, max_memory=2000, verbose=None):
    '''Outcore AO->MO transformation for the (ov|ov) block.

    Identical algorithm to mp2._ao2mo_ovov.
    '''
    from pyscf.scf.hf import RHF
    assert isinstance(bws._scf, RHF)
    time0 = (logger.process_clock(), logger.perf_counter())
    log   = logger.new_logger(bws, verbose)

    mol    = bws.mol
    int2e  = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                              'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir      = orbv.shape[1]
    nbas      = mol.nbas
    assert nvir <= nao

    ao_loc    = mol.ao_loc_nr()
    dmax      = max(4, min(nao / 3,
                           numpy.sqrt(max_memory * .95e6 / 8 / (nao + nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax      = max(x[2] for x in sh_ranges)
    eribuf    = numpy.empty((nao, dmax, dmax, nao))
    ftmp      = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax,
              nocc**2 * (nao * (nao + dmax) / 2 + nvir**2) * 8 / 1e6)

    buf_i  = numpy.empty((nocc * dmax**2 * nao,))
    buf_li = numpy.empty((nocc**2 * dmax**2,))
    buf1   = numpy.empty_like(buf_li)

    fint           = gto.moleintor.getints4c
    jk_blk_slices  = []
    count          = 0
    time1          = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip + 1]:
                i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
                j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])
                jk_blk_slices.append((i0, i1, j0, j1))

                eri    = fint(int2e, mol._atm, mol._bas, mol._env,
                              shls_slice=(0, nbas, ish0, ish1,
                                          jsh0, jsh1, 0, nbas),
                              aosym='s1', ao_loc=ao_loc,
                              cintopt=ao2mopt._cintopt, out=eribuf)
                tmp_i  = numpy.ndarray((nocc, (i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc, nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao, (i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T,
                         tmp_i.reshape(nocc*(i1-i0)*(j1-j0), nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc, nocc, (i1-i0), (j1-j0))
                save(str(count), tmp_li.transpose(1, 0, 2, 3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1(
                    'partial ao2mo [%d:%d,%d:%d]' % (ish0, ish1, jsh0, jsh1),
                    *time1)

    time1 = time0 = log.timer('bws2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    h5dat   = feri.create_dataset('ovov', (nocc * nvir, nocc * nvir), 'f8',
                                   chunks=(nvir, nvir))
    occblk  = int(min(nocc,
                      max(4, 250 / nocc,
                          max_memory * .9e6 / 8 / (nao**2 * nocc) / 5)))

    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0 + occblk, nocc)
            for k, (p0, p1, q0, q1) in enumerate(jk_blk_slices):
                eri[:i1-i0, :, p0:p1, q0:q1] = ftmp[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)][:, i0:i1])
                    eri[:i1-i0, :, q0:q1, p0:p1] = dat.transpose(1, 0, 3, 2)

    def save_blk(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i * nvir:(i + 1) * nvir] = dat[i - i0].reshape(nvir,
                                                                    nocc * nvir)

    orbv           = numpy.asarray(orbv, order='F')
    buf_prefetch   = numpy.empty((occblk, nocc, nao, nao))
    buf            = numpy.empty_like(buf_prefetch)
    bufw           = numpy.empty((occblk * nocc, nvir**2))
    bufw1          = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save_blk) as bsave:
            load(0, buf_prefetch)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefetch = buf_prefetch, buf
                prefetch(i1, buf_prefetch)
                eri = buf[:i1-i0].reshape((i1-i0) * nocc, nao, nao)
                dat = _ao2mo.nr_e2(eri, orbv, (0, nvir, 0, nvir),
                                   's1', 's1', out=bufw)
                bsave(i0, i1,
                      dat.reshape(i1-i0, nocc, nvir, nvir).transpose(0, 2, 1, 3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0, i1),
                                         *time1)

    log.timer('bws2 ao2mo_ovov pass2', *time0)
    return h5dat


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class BWSBase(lib.StreamObject):
    '''Base class for BW-s2 methods.

    Attributes
    ----------
    max_cycle : int
        Maximum number of BW-s2 iterations. Default 50.
    conv_tol : float
        Energy convergence threshold (hartree). Default 1e-7.
    '''

    max_cycle = 50
    conv_tol  = 1e-7
    alpha     = 1.0   # W scaling: M = F_oo + (alpha/2)*W; alpha=1 -> BW-s2, alpha=0 -> MP2

    _keys = {
        'max_cycle', 'conv_tol', 'alpha', 'mol', 'max_memory',
        'mo_coeff', 'mo_occ', 'e_hf', 'e_corr', 't2',
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

        self.mo_coeff = mo_coeff
        self.mo_occ   = mo_occ
        self._nocc    = None
        self._nmo     = None

        # results
        self.converged = False
        self.e_hf      = None
        self.e_corr    = None
        self.t2        = None

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
        return self.e_hf + self.e_corr

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        log.info('max_cycle = %d  conv_tol = %g  alpha = %g',
                 self.max_cycle, self.conv_tol, self.alpha)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    # abstract slots — subclasses must provide concrete implementations
    ao2mo     = NotImplemented
    init_amps = NotImplemented
    energy    = NotImplemented


class RBWS2(BWSBase):
    '''Restricted BW-s2.

    Second-order size-consistent Brillouin-Wigner perturbation theory
    for closed-shell (RHF) references. Self-consistent in the correlation
    energy: the occupied orbital energies are dressed each iteration by the
    W matrix (related to correlated ionization potentials) until the
    correlation energy converges.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.bws2 import RBWS2
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> bws = RBWS2(mf).run()
    >>> print(bws.e_corr)
    '''

    get_nocc = get_nocc
    get_nmo  = get_nmo
    get_e_hf = get_e_hf

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

    # Assign module-level functions as methods so they can be called on the
    # instance (e.g. bws.init_amps(eris)) and overridden in subclasses.
    init_amps = init_amps
    energy    = energy


from pyscf import scf
scf.hf.RHF.BWS2 = lib.class_as_method(RBWS2)
