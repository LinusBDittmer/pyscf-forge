#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Linus B. Dittmer <linus.dittmer@stud.uni-heidelberg.de>
#

'''
SF-XTDA: spin-complete spin-flip extended TDA (ROKS SF-XCIS) on a ROKS/ROHF M_S = +1 triplet.

Model (spin-free, all states spin-pure):
    H = sum_pq h'_pq E_pq + (c_H/2) sum (pq|rs) e_pqrs,   h' = Fbar - c_H (J - K_C - K_O/2)[Phi0],
evaluated in the SF-XCIS space (all M_S = 0 determinants with at most one hole in C and at most
one particle in V) in a CSF basis.  Every matrix element is taken from the generated formulas in
_blocks.py.  All two-electron integrals are density fitted and no N^4 quantity is built: the
blocks that touch an OO/CO/OV CSF are stored explicitly (at most N^3), the CV-CV block is applied
as a sigma vector through one-index operators and DF J/K contractions.
'''

import numpy
from pyscf import lib, scf, dft, df, symm
from pyscf.ao2mo import _ao2mo
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from pyscf.sfxtda._blocks import BLOCKS, ORDER, CSF_INFO

# axis of each C/V label in the broadcast shape (bra i, bra a, ket j, ket b)
_AX = {'i': 0, 'a': 1, 'j': 2, 'b': 3}
_SPACE = {'i': 'C', 'j': 'C', 'a': 'V', 'b': 'V', 'O1': 'O1', 'O2': 'O2'}


def _reference(mf):
    '''Orbital partition, spin-averaged KS Fock matrix (MO basis) and exact-exchange fraction.'''
    if not isinstance(mf, scf.rohf.ROHF):
        raise TypeError('SF-XTDA requires a ROHF or ROKS reference')
    mol = mf.mol
    mo_occ = mf.mo_occ
    core = numpy.where(mo_occ > 1.5)[0]
    somo = numpy.where((mo_occ > .5) & (mo_occ < 1.5))[0]
    virt = numpy.where(mo_occ < .5)[0]
    if mol.spin != 2 or len(somo) != 2:
        raise ValueError('SF-XTDA requires the M_S = +1 triplet reference (two singly occupied orbitals)')
    if not mf.converged:
        raise RuntimeError('reference SCF not converged')
    if isinstance(mf, dft.rks.KohnShamDFT):
        omega, alpha, c_h = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        if abs(omega) > 1e-12:
            raise NotImplementedError('range-separated hybrids')
    else:
        c_h = 1.
    mo_coeff = mf.mo_coeff
    f = mf.get_fock()
    fbar = .5 * mo_coeff.T @ (f.focka + f.fockb) @ mo_coeff
    return mo_coeff, core, somo, virt, fbar, c_h


def _labels(s):
    out, k = [], 0
    while k < len(s):
        if s[k] == 'O':
            out.append(s[k:k+2])
            k += 2
        else:
            out.append(s[k])
            k += 1
    return out


class _DFInts:
    '''Fock elements and c_H-scaled DF integrals as arrays broadcastable to (nC, nV, nC, nV).

    The three-index factors are scaled by sqrt(c_H), so [pq|rs] = sum_L B^L_pq B^L_rs.
    Integrals with four free C/V indices (N^4) are refused unless allow_4index is set (tests only).
    For c_H = 0 (pure functionals) every two-electron integral vanishes and no DF tensor is built
    (naux = 0).
    '''
    def __init__(self, with_df, mo_coeff, core, somo, virt, fbar, c_h):
        nmo = mo_coeff.shape[1]
        naux = with_df.get_naoaux() if c_h != 0 else 0
        self.core, self.somo, self.virt, self.fbar = core, somo, virt, fbar
        nC, nV = len(core), len(virt)
        self.nC, self.nV, self.naux = nC, nV, naux
        self.allow_4index = False
        self._orb = {'C': core, 'V': virt, 'O1': somo[:1], 'O2': somo[1:]}
        self._cache = {}
        self.bcc = numpy.empty((naux, nC, nC))
        self.bcv = numpy.empty((naux, nC, nV))
        self.bvv = numpy.empty((naux, nV, nV))
        self.bo = numpy.empty((naux, 2, nmo))
        mo = numpy.asarray(mo_coeff, order='F')
        p1 = 0
        for eri1 in (with_df.loop() if naux else ()):
            p0, p1 = p1, p1 + eri1.shape[0]
            buf = _ao2mo.nr_e2(eri1, mo, (0, nmo, 0, nmo), aosym='s2', mosym='s1').reshape(-1, nmo, nmo)
            self.bcc[p0:p1] = buf[:, core[:, None], core]
            self.bcv[p0:p1] = buf[:, core[:, None], virt]
            self.bvv[p0:p1] = buf[:, virt[:, None], virt]
            self.bo[p0:p1] = buf[:, somo]
        scale = numpy.sqrt(c_h)
        for b in (self.bcc, self.bcv, self.bvv, self.bo):
            b *= scale

    def _b(self, p, q):
        '''Three-index factor (naux, n_p, n_q) for the label pair (p, q).'''
        sp, sq = _SPACE[p], _SPACE[q]
        if sp[0] == 'O':
            return self.bo[:, int(sp[1]) - 1, self._orb[sq]][:, None, :]
        if sq[0] == 'O':
            return self.bo[:, int(sq[1]) - 1, self._orb[sp]][:, :, None]
        return {('C', 'C'): self.bcc, ('C', 'V'): self.bcv,
                ('V', 'C'): self.bcv.transpose(0, 2, 1), ('V', 'V'): self.bvv}[sp, sq]

    def value(self, name):
        if name == 'delta_ij':
            return numpy.eye(self.nC)[:, None, :, None]
        if name == 'delta_ab':
            return numpy.eye(self.nV)[None, :, None, :]
        if name in self._cache:
            return self._cache[name]
        if name.startswith('F_'):
            labs = _labels(name[2:])
            arr = self.fbar[numpy.ix_(self._orb[_SPACE[labs[0]]], self._orb[_SPACE[labs[1]]])]
        else:
            left, right = name[1:-1].split('|')
            labs = _labels(left) + _labels(right)
            if not self.allow_4index and all(lab in _AX for lab in labs):
                raise RuntimeError(f'N^4 integral {name} requested')
            arr = numpy.tensordot(self._b(*labs[:2]), self._b(*labs[2:]), axes=(0, 0))
        idx = [k for k, lab in enumerate(labs) if lab in _AX]
        lab_idx = [labs[k] for k in idx]
        arr = arr.reshape([arr.shape[k] for k in idx])
        order = sorted(range(len(lab_idx)), key=lambda k: _AX[lab_idx[k]])
        arr = numpy.transpose(arr, order)
        shape = [1, 1, 1, 1]
        for pos, k in enumerate(order):
            shape[_AX[lab_idx[k]]] = arr.shape[pos]
        arr = arr.reshape(shape)
        if len(idx) <= 2:
            self._cache[name] = arr
        return arr


def csf_list(spin, nC, nV):
    '''CSF types of total spin S in the fixed order (OO, CO, OV, CV) with their (rows, cols) shape.'''
    out = []
    for cl in ORDER:
        names, spins = CSF_INFO[cl]
        for nm, s in zip(names, spins):
            if s == spin:
                out.append((cl, nm, {'OO': (1, 1), 'CO': (nC, 1), 'OV': (1, nV), 'CV': (nC, nV)}[cl]))
    return out


def _split(spin, nC, nV):
    blocks = csf_list(spin, nC, nV)
    return [b for b in blocks if b[0] != 'CV'], [b for b in blocks if b[0] == 'CV']


def build_small(ints, spin):
    '''Explicit blocks <small|H-E0|small> (n_small, n_small) and <small|H-E0|CV> (n_small, nblk*nC*nV).

    Every formula touching an OO/CO/OV CSF has at most three free C/V indices, so each evaluated
    block is at most max(nC, nV) x nC*nV.
    '''
    nC, nV = ints.nC, ints.nV
    nCV = nC * nV
    small, large = _split(spin, nC, nV)
    soff = numpy.cumsum([0] + [r * c for _, _, (r, c) in small])
    hss = numpy.zeros((soff[-1], soff[-1]))
    hsl = numpy.zeros((soff[-1], len(large) * nCV))
    for x, (A, na, (ni, nai)) in enumerate(small):
        rows = slice(soff[x], soff[x+1])
        for y, (B, nb, (nj, nbj)) in enumerate(small + large):
            if (A, na, B, nb) not in BLOCKS:
                continue
            args, f = BLOCKS[(A, na, B, nb)]
            blk = numpy.broadcast_to(f(*[ints.value(a) for a in args]), (ni, nai, nj, nbj))
            blk = blk.reshape(ni * nai, nj * nbj)
            if B == 'CV':
                q = y - len(small)
                hsl[rows, q*nCV:(q+1)*nCV] = blk
            else:
                cols = slice(soff[y], soff[y+1])
                hss[rows, cols] = blk
                hss[cols, rows] = blk.T
    return hss, hsl


def cvcv_operator(ints, spin):
    '''Decompose the CV-CV formulas into one-index operators and J/K coefficients.

    Every CV-CV term is c*s*delta_ij*delta_ab, c*X_ij*delta_ab, c*Y_ab*delta_ij, c*[ia|jb] or c*[ij|ab].
    The coefficient c of an argument is found by probing the formula (deltas = 1, that argument = 1,
    all others = 0).  Returns the bra-ket matrices lc (nblk*nC, nblk*nC) and rv (nblk*nV, nblk*nV)
    and the J/K coefficient matrices cj, ck (nblk, nblk).
    '''
    nC, nV = ints.nC, ints.nV
    large = [nm for cl, nm, _ in csf_list(spin, nC, nV) if cl == 'CV']
    nblk = len(large)
    lc = numpy.zeros((nblk, nC, nblk, nC))
    rv = numpy.zeros((nblk, nV, nblk, nV))
    cj = numpy.zeros((nblk, nblk))
    ck = numpy.zeros((nblk, nblk))
    for p, na in enumerate(large):
        for q, nb in enumerate(large):
            if ('CV', na, 'CV', nb) not in BLOCKS:
                continue
            args, f = BLOCKS[('CV', na, 'CV', nb)]
            for name in args:
                if name.startswith('delta'):
                    continue
                c = f(*[1. if a.startswith('delta') else float(a == name) for a in args])
                if name == '(ia|jb)':
                    cj[p, q] += c
                    continue
                if name == '(ij|ab)':
                    ck[p, q] += c
                    continue
                v = ints.value(name)
                free = {lab for lab in _labels(name[2:] if name.startswith('F_') else name[1:-1].replace('|', ''))
                        if lab in _AX}
                if not free:
                    mat, blk = lc, c * v.item() * numpy.eye(nC)
                elif free == {'i', 'j'}:
                    mat, blk = lc, c * v.reshape(nC, nC)
                elif free == {'a', 'b'}:
                    mat, blk = rv, c * v.reshape(nV, nV)
                else:
                    raise RuntimeError(f'unexpected CV-CV argument {name}')
                mat[p, :, q, :] += blk
                if p != q:
                    mat[q, :, p, :] += blk.T
            if p != q:
                cj[q, p] = cj[p, q]
                ck[q, p] = ck[p, q]
    return lc.reshape(nblk * nC, -1), rv.reshape(nblk * nV, -1), cj, ck


def hdiag_large(ints, lc, rv, cj, ck):
    '''Exact diagonal of the CV-CV block, shape (nblk, nC, nV).'''
    nC, nV = ints.nC, ints.nV
    nblk = cj.shape[0]
    r = numpy.arange(nblk)
    dlc = lc.reshape(nblk, nC, nblk, nC)[r, :, r, :].diagonal(0, 1, 2)
    drv = rv.reshape(nblk, nV, nblk, nV)[r, :, r, :].diagonal(0, 1, 2)
    jd = numpy.einsum('Lia,Lia->ia', ints.bcv, ints.bcv)
    kd = ints.bcc.diagonal(0, 1, 2).T @ ints.bvv.diagonal(0, 1, 2)
    return (dlc[:, :, None] + drv[:, None, :]
            + cj.diagonal()[:, None, None] * jd + ck.diagonal()[:, None, None] * kd)


def j_contract(ints, t):
    '''J[T]_ia = sum_jb [ia|jb] T_jb for t of shape (m, nC, nV).'''
    naux = ints.naux
    rho = numpy.tensordot(t, ints.bcv, axes=([1, 2], [1, 2]))
    return lib.dot(rho, ints.bcv.reshape(naux, -1)).reshape(t.shape)


def k_contract(ints, t):
    '''K[T]_ia = sum_jb [ij|ab] T_jb = sum_L (B^L_CC T B^L_VV)_ia for one t of shape (nC, nV).'''
    naux, nC, nV = ints.bcv.shape
    x = lib.dot(ints.bcc.reshape(naux * nC, nC), t).reshape(naux, nC, nV)
    x = x.transpose(1, 0, 2).reshape(nC, naux * nV)
    return lib.dot(x, ints.bvv.reshape(naux * nV, nV))


def damp_(hss, hsl, d_small, d_large, p1, p2):
    '''In-place damping p1*exp(-p2*(H_KK-H_LL)^4) of the off-diagonal of hss and of all of hsl.'''
    f = p1 * numpy.exp(-p2 * (d_small[:, None] - d_small[None, :])**4)
    numpy.fill_diagonal(f, 1.)
    hss *= f
    f = numpy.subtract(d_small[:, None], d_large[None, :])
    f **= 4
    f *= -p2
    numpy.exp(f, out=f)
    f *= p1
    hsl *= f


def csf_irreps(blocks, orbsym, core, somo, virt):
    '''D2h-subgroup irrep id of every CSF (XOR of the orbital irreps).'''
    orbsym = numpy.asarray(orbsym)
    s1, s2 = orbsym[somo[0]], orbsym[somo[1]]
    oc, ov = orbsym[core], orbsym[virt]
    out = []
    for cl, nm, _ in blocks:
        if cl == 'OO':
            out.append(numpy.array([0 if nm in ('G', 'D') else s1 ^ s2]))
        elif cl == 'CO':
            out.append(oc ^ (s2 if 'O1^2' in nm else s1))
        elif cl == 'OV':
            out.append((s1 if 'O1^1' in nm else s2) ^ ov)
        else:
            base = (oc[:, None] ^ ov[None, :]).ravel()
            out.append(base if '^2' in nm else base ^ s1 ^ s2)
    return numpy.concatenate(out)


class SFXTDA(lib.StreamObject):
    '''Spin-complete spin-flip extended TDA (ROKS SF-XCIS) on a ROKS or ROHF M_S = +1 triplet.

    Attributes:
        nstates : int
            Number of roots per spin (lowest over all irreps, or within wfnsym if set).
        spins : tuple
            Total spins to compute, subset of (0, 1, 2).
        damping : None or (p1, p2)
            DFT/MRCI-style damping p1*exp(-p2*dE^4) of every coupling that involves an OO, CO
            or OV configuration (CV-CV couplings are never damped).  (1.0, 8.0) is recommended
            for BH&HLYP.
        wfnsym : str or int
            Restrict the roots to one irrep (requires mol.symmetry).
        auxbasis : str
            Auxiliary basis, used only when the reference SCF is not density fitted.

    Saved results (dicts keyed by spin):
        omega : excitation energies (Hartree) relative to the reference energy E_ROKS
        e : excitation energies relative to the lowest singlet (equal to omega if S=0 is not
            requested or wfnsym is set)
        e_tot : total energies E_ROKS + omega
        irrep : irrep label of each root
        converged : convergence flag of each root
        xs : eigenvectors in the CSF basis, shape (nroots, n)
    '''
    conv_tol = 1e-5
    max_cycle = 100
    lindep = 1e-12
    deg_eia_thresh = 1e-3

    def __init__(self, mf, nstates=5, spins=(0, 1), damping=None):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.nstates = nstates
        self.spins = tuple(spins)
        self.damping = damping
        self.wfnsym = None
        self.auxbasis = None
        self._ints = None
        self._orbs = None
        self.omega = self.e = self.e_tot = self.irrep = self.converged = self.xs = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s for %s ********', self.__class__, self._scf.__class__)
        log.info('nstates = %d  spins = %s', self.nstates, self.spins)
        log.info('damping = %s (applied to couplings involving OO/CO/OV only)', self.damping)
        if self.wfnsym is not None:
            log.info('wfnsym = %s', self.wfnsym)
        log.info('conv_tol = %g  max_cycle = %d  max_memory %d MB', self.conv_tol, self.max_cycle, self.max_memory)
        return self

    def build(self):
        log = logger.new_logger(self)
        mf = self._scf
        mo_coeff, core, somo, virt, fbar, c_h = _reference(mf)
        with_df = getattr(mf, 'with_df', None)
        if with_df is None:
            log.warn('Reference SCF is not density fitted: F^KS uses exact J/K, '
                     'the response integrals use df.DF(auxbasis=%s)', self.auxbasis)
            with_df = df.DF(self.mol, auxbasis=self.auxbasis)
            with_df.max_memory = self.max_memory
        self._ints = ints = _DFInts(with_df, mo_coeff, core, somo, virt, fbar, c_h)
        self._orbs = (core, somo, virt)
        self.c_h = c_h
        mem = sum(b.nbytes for b in (ints.bcc, ints.bcv, ints.bvv, ints.bo)) / 1e6
        log.info('SF-XTDA: nC = %d  nV = %d  naux = %d  c_H = %g  DF tensors %.1f MB',
                 ints.nC, ints.nV, ints.naux, c_h, mem)
        if mem > self.max_memory:
            log.warn('DF tensors (%.1f MB) exceed max_memory (%d MB)', mem, self.max_memory)
        return self

    def gen_vind(self, spin):
        '''Sigma-vector function vind(x) for x of shape (m, n) and the exact diagonal hdiag (n,).'''
        if self._ints is None:
            self.build()
        log = logger.new_logger(self)
        ints = self._ints
        nC, nV = ints.nC, ints.nV
        hss, hsl = build_small(ints, spin)
        lc, rv, cj, ck = cvcv_operator(ints, spin)
        hdl = hdiag_large(ints, lc, rv, cj, ck)
        if self.damping is not None:
            damp_(hss, hsl, hss.diagonal().copy(), hdl.ravel(), *self.damping)
        hdiag = numpy.concatenate([hss.diagonal(), hdl.ravel()])
        ns, nblk, n = hss.shape[0], cj.shape[0], hdiag.size
        # c_H = 0: all [pq|rs] vanish, skip the J/K contractions
        jrows = numpy.flatnonzero(abs(cj).sum(1)) if ints.naux else numpy.zeros(0, dtype=int)
        krows = numpy.flatnonzero(abs(ck).sum(1)) if ints.naux else numpy.zeros(0, dtype=int)
        log.info('S = %d: %d CSFs (%d small), explicit couplings %.1f MB', spin, n, ns, hsl.nbytes / 1e6)

        def vind(x):
            x = numpy.asarray(x).reshape(-1, n)
            m = x.shape[0]
            xs, xl = x[:, :ns], x[:, ns:]
            xl3 = xl.reshape(m, nblk, nC, nV)
            out = numpy.empty_like(x)
            out[:, :ns] = xs @ hss + xl @ hsl.T
            sl = (xs @ hsl).reshape(m, nblk, nC, nV)
            sl += (lc @ xl.reshape(m, nblk * nC, nV)).reshape(m, nblk, nC, nV)
            sl += (xl3.transpose(0, 2, 1, 3).reshape(m, nC, nblk * nV) @ rv
                   ).reshape(m, nC, nblk, nV).transpose(0, 2, 1, 3)
            if jrows.size:
                tj = numpy.einsum('PQ,mQia->mPia', cj[jrows], xl3)
                sl[:, jrows] += j_contract(ints, tj.reshape(-1, nC, nV)).reshape(m, -1, nC, nV)
            if krows.size:
                tk = numpy.einsum('PQ,mQia->mPia', ck[krows], xl3).reshape(-1, nC, nV)
                # ponytail: one K build per amplitude matrix keeps the intermediate at naux*nC*nV;
                # batch several per GEMM if profiling shows small-matrix overhead
                sl[:, krows] += numpy.stack([k_contract(ints, t) for t in tk]).reshape(m, -1, nC, nV)
            out[:, ns:] = sl.reshape(m, -1)
            return out

        forbid = self._forbid(self.get_csf_irreps(spin))
        if forbid is None:
            return vind, hdiag

        def vind_sym(x):
            x = numpy.array(x).reshape(-1, n)
            x[:, forbid] = 0
            out = vind(x)
            out[:, forbid] = 0
            return out
        return vind_sym, hdiag

    def get_csf_irreps(self, spin):
        '''Irrep id (D2h subgroup) of every CSF of spin S, or None without point-group symmetry.'''
        if not self.mol.symmetry:
            return None
        if self._orbs is None:
            self.build()
        core, somo, virt = self._orbs
        orbsym = numpy.asarray(scf.hf_symm.get_orbsym(self.mol, self._scf.mo_coeff)) % 10
        return csf_irreps(csf_list(spin, len(core), len(virt)), orbsym, core, somo, virt)

    def _forbid(self, x_sym):
        '''Mask of CSFs outside the target irrep wfnsym (None if no restriction).'''
        if self.wfnsym is None or x_sym is None:
            return None
        wfnsym = self.wfnsym
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(self.mol.groupname, wfnsym)
        return x_sym != wfnsym % 10

    def get_init_guess(self, hdiag, nstates, x_sym=None):
        '''Unit vectors on the lowest diagonal elements (allowed irrep only), as in pyscf TDA.'''
        h = hdiag.copy()
        forbid = self._forbid(x_sym)
        if forbid is not None:
            h[forbid] = 1e99
        nstates = min(nstates, h.size - (0 if forbid is None else numpy.count_nonzero(forbid)))
        thresh = numpy.partition(h, nstates - 1)[nstates - 1] + self.deg_eia_thresh
        idx = numpy.where(h <= thresh)[0]
        x0 = numpy.zeros((idx.size, h.size))
        x0[numpy.arange(idx.size), idx] = 1
        return x0, (None if x_sym is None else x_sym[idx])

    def kernel(self, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if not set(self.spins) <= {0, 1, 2}:
            raise ValueError(f'spins must be a subset of (0, 1, 2), got {self.spins}')
        if nstates is not None:
            self.nstates = nstates
        if self._ints is None:
            self.build()
        self.dump_flags()
        log = logger.new_logger(self)
        mol = self.mol
        self.omega, self.converged, self.xs, self.irrep, self.e_tot = {}, {}, {}, {}, {}
        for spin in self.spins:
            vind, hdiag = self.gen_vind(spin)
            x_sym = self.get_csf_irreps(spin)
            forbid = self._forbid(x_sym)
            nallowed = hdiag.size - (0 if forbid is None else numpy.count_nonzero(forbid))
            nroots = min(self.nstates, nallowed)
            if nroots == 0:
                log.note('SF-XTDA S=%d: no CSF in the requested irrep/space, no roots', spin)
                self.omega[spin] = self.e_tot[spin] = numpy.zeros(0)
                self.converged[spin] = numpy.zeros(0, dtype=bool)
                self.xs[spin] = numpy.zeros((0, hdiag.size))
                self.irrep[spin] = []
                continue
            x0, x0sym = self.get_init_guess(hdiag, nroots, x_sym)

            def precond(x, e, *args):
                if isinstance(e, numpy.ndarray):
                    e = e[0]
                d = hdiag - e
                d[abs(d) < 1e-8] = 1e-8
                return x / d

            def pick(w, v, nroots, envs):
                # all roots are physical: SF states may lie below the reference (omega < 0)
                return w, v, numpy.arange(w.size)

            conv, w, xs = lr_eigh(vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                                  nroots=nroots, x0sym=x0sym, pick=pick, max_cycle=self.max_cycle,
                                  max_memory=self.max_memory, verbose=log)
            xs = numpy.asarray(xs)
            self.omega[spin] = numpy.asarray(w)
            self.converged[spin] = numpy.asarray(conv)
            self.xs[spin] = xs
            self.e_tot[spin] = self._scf.e_tot + self.omega[spin]
            if x_sym is None:
                self.irrep[spin] = ['A'] * len(w)
            else:
                group = {'Dooh': 'D2h', 'Coov': 'C2v'}.get(mol.groupname, mol.groupname)
                ids = x_sym[abs(xs).argmax(axis=1)]
                self.irrep[spin] = [symm.irrep_id2name(group, i) for i in ids]
        # with wfnsym the lowest computed singlet is not the ground state: e falls back to omega
        if self.wfnsym is None and 0 in self.omega and self.omega[0].size:
            e0 = self.omega[0].min()
        else:
            e0 = 0.
        self.e = {spin: w - e0 for spin, w in self.omega.items()}
        log.timer('SF-XTDA', *cpu0)
        self._finalize()
        return self.e

    def _finalize(self):
        log = logger.new_logger(self)
        for spin in self.spins:
            if not all(self.converged[spin]):
                log.warn('SF-XTDA S=%d roots %s not converged', spin,
                         numpy.flatnonzero(~self.converged[spin]).tolist())
            for k, (w, e, g) in enumerate(zip(self.omega[spin], self.e[spin], self.irrep[spin])):
                log.note('S=%d  root %2d  %-4s  omega = %10.4f eV  e = %10.4f eV',
                         spin, k, g, w * HARTREE2EV, e * HARTREE2EV)
        return self
