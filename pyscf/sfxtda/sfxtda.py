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
from pyscf import lib, scf, dft
from pyscf.lib import logger


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


class SFXTDA(lib.StreamObject):
    pass
