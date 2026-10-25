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
from pyscf import lib, gto, dft
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

    Root signs are aligned with the previous kernel() call through MO and CSF overlaps (for dynamics; reset_phase()
    forgets them), so the first call returns the NAC of the roots as computed.  For dynamics, bind the NAC object to
    a gradient scanner's SF-XTDA object (g.base.nac_method()) to share one SF-XTDA run per geometry.
    '''
    _keys = {'mult_ediff', 'use_etfs'}

    def __init__(self, td):
        sfxtda_grad.Gradients.__init__(self, td)
        self.state = (1, 0)
        self.mult_ediff = False
        self.use_etfs = False
        self._phase_ref = {}

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
        sign = self._phases(spin)
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

    def _phases(self, spin):
        '''Signs (nroots,) that align the roots of spin with the previous kernel() call (all +1 on the first).

        MO signs are aligned with the previous orbitals by sign diag(C0^T S(R0, R) C); a CSF then changes sign with
        the product of the signs of its singly occupied orbitals, the XOR of csf_irreps with the flip bits as irreps.
        '''
        td = self.base
        core, somo, virt = td._orbs
        mol, mo, xs = td.mol, td.mo_coeff, td.xs[spin]
        sign = numpy.ones(len(xs))
        prev = self._phase_ref.get(spin)
        if prev is not None:
            mol0, mo0, x0 = prev
            s = numpy.where((mo0.T @ gto.intor_cross('int1e_ovlp', mol0, mol) @ mo).diagonal() < 0, -1., 1.)
            flip = sfxtda.csf_irreps(sfxtda.csf_list(spin, len(core), len(virt)), (s < 0).astype(int),
                                     core, somo, virt)
            mo = mo * s
            xs = xs * (1 - 2. * flip)
            # ponytail: roots and orbitals matched by index; follow reordering by an overlap assignment if needed
            n = min(len(x0), len(xs))
            ov = numpy.einsum('ij,ij->i', x0[:n], xs[:n])
            sign[:n] = numpy.where(ov < 0, -1., 1.)
            bad = numpy.flatnonzero(abs(ov) < .5)
            if bad.size:
                logger.warn(self, 'SF-XTDA NAC S=%d roots %s: overlap < 0.5 with the previous call (root swap or '
                            'orbital reordering), their phases are not tracked', spin, bad.tolist())
        self._phase_ref[spin] = (mol, mo, xs * sign[:, None])
        return sign

    def reset_phase(self):
        '''Forget the root phases of earlier calls: the next kernel() takes the roots' signs as they are.'''
        self._phase_ref = {}
        return self

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            ket, bra = self.state
            logger.note(self, '--------- SF-XTDA %s S=%d <%d|d%d/dR> ----------',
                        'h = dE * NAC' if self.mult_ediff else 'NAC', self.spin, bra, ket)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def as_scanner(self, state=None, spin=None):
        '''Scanner: scanner(mol_or_geom) returns (e_tot of (ket, bra), NAC) at the new geometry, root phases tracked
        between calls.'''
        if isinstance(self, lib.GradScanner):
            return self
        logger.info(self, 'Create scanner for %s', self.__class__)
        name = self.__class__.__name__ + SFXTDA_NACScanner.__name_mixin__
        return lib.set_class(SFXTDA_NACScanner(self, state, spin), (SFXTDA_NACScanner, self.__class__), name)


class SFXTDA_NACScanner(lib.GradScanner):
    def __init__(self, nac, state=None, spin=None):
        lib.GradScanner.__init__(self, nac)
        if state is not None:
            self.state = state
        if spin is not None:
            self.spin = spin

    def __call__(self, mol_or_geom, state=None, spin=None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        self.base(mol)
        nac = self.kernel(state=state, spin=spin, **kwargs)
        return self.base.e_tot[self.spin][list(self.state)], nac

    @property
    def converged(self):
        td = self.base
        return bool(td._scf.converged and all(td.converged[self.spin][list(self.state)]))


NAC = NonAdiabaticCouplings
