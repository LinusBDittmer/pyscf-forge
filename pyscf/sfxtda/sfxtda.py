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
from pyscf import lib, scf, dft, df
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

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
    '''
    def __init__(self, with_df, mo_coeff, core, somo, virt, fbar, c_h):
        nmo = mo_coeff.shape[1]
        naux = with_df.get_naoaux()
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
        for eri1 in with_df.loop():
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
        e : excitation energies relative to the lowest singlet (equal to omega if S=0 not requested)
        e_tot : total energies E_ROKS + omega
        irrep : irrep label of each root
        converged : convergence flag of each root
        xs : eigenvectors in the CSF basis, shape (nroots, n)
    '''
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
