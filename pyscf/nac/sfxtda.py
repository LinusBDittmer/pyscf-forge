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
Analytic non-adiabatic couplings d = <bra|d ket/dR> between two SF-XTDA roots of one spin.

|K> = sum_mu x^K_mu |mu(R)> with CSFs on td.mo_coeff.  With dE = omega_ket - omega_bra and h = dE d:

    h = d/dR [x^bra.Htilde.x^ket]_x + dE sum_pq G_pq <phi_p|d phi_q/dR>,   G_pq = <bra|E_pq|ket>.

The first term is the gradient Lagrangian of pyscf.grad.sfxtda with x x^T replaced by the symmetric weight
(x^bra x^ket^T + x^ket x^bra^T)/2 (P, Ghat and the damping weight dW are bilinear; E_ROKS drops out since
x^bra.x^ket = 0).  With C -> C(1 + U), U = kappa - 1/2 C^T S^(A) C,

    <phi_p|d phi_q/dR> = kappa_pq + 1/2 [C^T (S^h - S^h^T) C]_pq,   S^h_uv = <chi_u|d chi_v/dR>,

so only G^A = (G - G^T)/2 enters.  Its kappa part is added to the orbital Lagrangian, X' = X + dE G^A, and passes
through the canonical constraint (damped model) and the Z-vector of the gradient; the AO part dE <C G^A C^T, S^h>
is dropped with use_etfs (electron translation factors, Fatehi & Subotnik, JPCL 3, 2039 (2012)).  Undamped roots
do not depend on rotations inside C, O and V, so the intra-space kappa is 0 in both terms.  A NAC transforms as
the product of the irreps of the two roots and is never symmetrised.  Grid response of the XC terms is omitted
(as in pyscf.grad.sfxtda).
'''

import numpy
from pyscf import dft
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import sfxtda as sfxtda_grad
from pyscf.sfxtda import sfxtda, _tdm


def nac_elec(nac, spin, ket, bra, atmlst=None, use_etfs=False):
    '''h = (omega_ket - omega_bra) <bra|d ket/dR>, shape (len(atmlst), 3); symmetric in ket and bra.'''
    td = nac.base
    mol, mo = nac.mol, td.mo_coeff
    if atmlst is None:
        atmlst = range(mol.natm)
    core, somo, virt = td._orbs
    xs = td.xs[spin]
    de = td.omega[spin][ket] - td.omega[spin][bra]
    auxmol = sfxtda_grad._auxmol(td)
    dfterms = sfxtda_grad._df_terms(mol, auxmol, mo, sfxtda._kernel_terms(*td.xc_coeffs))
    p, ghats = sfxtda_grad._state_densities(td, spin, bra, dfterms, ket)
    # G_pq = <bra|E_pq|ket>; transition_dm is <bra|a+_q a_p|ket>
    g = _tdm.transition_dm(xs[bra], xs[ket], sfxtda.csf_list(spin, len(core), len(virt)), core, somo, virt).T
    ga = .5 * (g - g.T)
    # ponytail: one Z-vector per pair; batch the right-hand sides of several pairs if many couplings are needed
    h = sfxtda_grad._relaxed(nac, p, ghats, auxmol, dfterms, de * ga, atmlst)
    if not use_etfs:
        s1 = rhf_grad.get_ovlp(mol)
        m = mo @ ga @ mo.T
        aoslices = mol.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            # s1[:, u, v] = <d chi_u/dR_A|chi_v> for u on A, so <chi_u|d chi_v/dR_A> = s1[:, v, u]
            h[k] += de * numpy.einsum('xvu,uv->x', s1[:, p0:p1], m[:, p0:p1])
    return h


class NonAdiabaticCouplings(sfxtda_grad.Gradients):
    '''SF-XTDA non-adiabatic couplings <bra|d ket/dR> between two roots of one spin (pyscf.nac conventions).

    Attributes:
        state : (ket, bra)
            0-based roots within the spin block (as in the SFXTDA log and transition_dm).
        spin : int
            Total spin S of both roots (default: first entry of td.spins).
        mult_ediff : bool
            Return (E_ket - E_bra) <bra|d ket/dR> (the h vector: symmetric, finite at degeneracies).
        use_etfs : bool
            Drop the AO overlap-derivative part of the CSF term (electron translation factors): translation
            invariant, no longer the derivative of the wavefunction overlap.
        cphf_conv_tol, cphf_max_cycle :
            Z-vector convergence settings.
    '''
    _keys = {'mult_ediff', 'use_etfs'}

    def __init__(self, td):
        sfxtda_grad.Gradients.__init__(self, td)
        self.state = (1, 0)
        self.mult_ediff = False
        self.use_etfs = False

    def dump_flags(self, verbose=None):
        rhf_grad.GradientsBase.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('S = %d  (ket, bra) = %s  mult_ediff = %s  use_etfs = %s  cphf_conv_tol = %g  cphf_max_cycle = %d',
                 self.spin, self.state, self.mult_ediff, self.use_etfs, self.cphf_conv_tol, self.cphf_max_cycle)
        return self

    def kernel(self, state=None, spin=None, atmlst=None, mult_ediff=None, use_etfs=None):
        td = self.base
        mf = td._scf
        # follow the SF-XTDA object: a gradient scanner sharing it moves td.mol without touching this object
        self.mol = td.mol
        if state is not None:
            self.state = state
        if spin is not None:
            self.spin = spin
        if mult_ediff is not None:
            self.mult_ediff = mult_ediff
        if use_etfs is not None:
            self.use_etfs = use_etfs
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if len(self.state) != 2:
            raise ValueError(f'state must be a pair (ket, bra), got {self.state}')
        (ket, bra), spin = self.state, self.spin
        if isinstance(mf, dft.rks.KohnShamDFT) and mf.do_nlc():
            raise NotImplementedError('SF-XTDA NACs with NLC')
        if td.xs is None:
            td.kernel()
        log = logger.new_logger(self)
        for k in (ket, bra):
            sfxtda_grad._check_root(td, spin, k, log)
        if ket == bra:
            self.de = numpy.zeros((self.mol.natm if atmlst is None else len(atmlst), 3))
            return self.de
        if self.verbose >= logger.INFO:
            self.dump_flags()
        sign = numpy.ones(len(td.xs[spin]))          # Task 4: self._phases(spin)
        h = nac_elec(self, spin, ket, bra, atmlst, self.use_etfs) * (sign[ket] * sign[bra])
        de = td.omega[spin][ket] - td.omega[spin][bra]
        if self.mult_ediff:
            self.de = h
        else:
            if abs(de) < 1e-4:
                log.warn('SF-XTDA S=%d roots %d and %d are %.2g Ha apart: the NAC diverges (use mult_ediff)',
                         spin, ket, bra, abs(de))
            self.de = h / de
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            ket, bra = self.state
            logger.note(self, '--------- SF-XTDA %s S=%d <%d|d%d/dR> ----------',
                        'h = dE * NAC' if self.mult_ediff else 'NAC', self.spin, bra, ket)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')


NAC = NonAdiabaticCouplings
