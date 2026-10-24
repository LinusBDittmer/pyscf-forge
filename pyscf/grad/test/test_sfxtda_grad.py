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

import copy
import unittest
import numpy as np
from scipy.linalg import expm
from pyscf import gto, scf, dft, lib
from pyscf.sfxtda import sfxtda, _rdm2
from pyscf.grad import sfxtda as sfxtda_grad

CH2 = "C 0 0 0; H 0.1 1.0 0.5; H -0.2 -0.95 0.35"
MAKE = {
    'hf': lambda m: scf.ROHF(m).density_fit(),
    'hf_nodf': lambda m: scf.ROHF(m),
    'ks': lambda m: dft.ROKS(m, xc='BHANDHLYP').density_fit(),
    'pbe': lambda m: dft.ROKS(m, xc='PBE').density_fit(),
    'cam': lambda m: dft.ROKS(m, xc='CAM-B3LYP').density_fit(),
}

def setUpModule():
    global mol, MF
    mol = gto.M(atom=CH2, basis='6-31g', spin=2, verbose=0)
    MF = {k: f(mol).run(conv_tol=1e-12) for k, f in MAKE.items()}

def tearDownModule():
    global mol, MF
    del mol, MF

def _td(mf, nstates=3):
    td = sfxtda.SFXTDA(mf, nstates=nstates, spins=(0, 1, 2))
    td.conv_tol = 1e-7                 # well above the Davidson residual floor (~1e-9, S6)
    td.kernel()
    return td

def _frozen(grids):
    '''Grid of the reference geometry without AO screening: FD derivatives then exclude grid response.'''
    g = copy.copy(grids)
    g.non0tab = g.screen_index = None
    return g

def _omega_at(td, mo, x, spin):
    '''x.(H-E0).x with orbitals mo (Fbar from the densities of mo), x fixed.'''
    mf = td._scf
    core, somo, virt = td._orbs
    na, nb = sfxtda_grad._occupations(td)
    f = mf.get_fock(dm=np.array([(mo * na) @ mo.T, (mo * nb) @ mo.T]))
    fbar = .5 * mo.T @ (f.focka + f.fockb) @ mo
    ints0 = td._ints
    td._ints = sfxtda._DFInts(td._with_df, mo, core, somo, virt, fbar, sfxtda._kernel_terms(*td.xc_coeffs))
    try:
        return x @ td.gen_vind(spin)[0](x[None])[0]
    finally:
        td._ints = ints0

def _lagr(td, spin, state):
    mf = td._scf
    mo = mf.mo_coeff
    occs = sfxtda_grad._occupations(td)
    dfterms = sfxtda_grad._df_terms(td.mol, td._with_df.auxmol, mo, sfxtda._kernel_terms(*td.xc_coeffs))
    ghats = sfxtda_grad._two_body(td, td.xs[spin][state], spin, dfterms)
    fock = sfxtda_grad._fock_mo(mf)
    vresp = sfxtda_grad._uks_response(mf, *occs)
    p = sfxtda_grad._pdiff(td, spin, state)
    x = sfxtda_grad._orbital_lagrangian(fock, vresp, mo, occs, (p / 2, p / 2), dfterms, ghats)
    return x, fock, vresp, occs, dfterms, ghats, p

def _unpack(k, mask):
    kap = np.zeros(mask.shape)
    kap[mask] = k
    return kap - kap.T

def _rohf_g(td, mo, mask):
    '''ROKS orbital gradient (X^E - X^E^T)[mask] at orbitals mo.'''
    na, nb = sfxtda_grad._occupations(td)
    f = td._scf.get_fock(dm=np.array([(mo * na) @ mo.T, (mo * nb) @ mo.T]))
    xe = 2 * ((mo.T @ f.focka @ mo) * na + (mo.T @ f.fockb @ mo) * nb)
    return (xe - xe.T)[mask]


class KnownValues(unittest.TestCase):
    def test_df_terms_reproduce_omega(self):
        '''tr(P Fbar) + 1/2 sum_t <Ghat_t, T_t> == omega (own DF fit == with_df fit, S3).'''
        for key in ('hf', 'cam', 'hf_nodf'):
            td = _td(MF[key])
            for spin, state in ((0, 0), (1, 0), (2, 0)):
                x, fock, _, _, dfterms, ghats, p = _lagr(td, spin, state)
                e = np.einsum('pq,pq', p, .5 * (fock[0] + fock[1]))
                e += .5 * sum(np.sum(g * t) for g, (_, _, t, _) in zip(ghats, dfterms))
                self.assertAlmostEqual(e, td.omega[spin][state], 9)

    def test_orbital_lagrangian_vs_fd(self):
        rng = np.random.default_rng(11)
        for key in ('hf', 'ks', 'cam'):
            td = _td(MF[key])
            mo = td._scf.mo_coeff
            core, somo, virt = td._orbs
            for spin, state in ((0, 0), (1, 0)):
                xl = _lagr(td, spin, state)[0]
                k = rng.standard_normal(xl.shape) * 1e-5      # central FD error ~ h^3 (1.5e-10 at 1e-4)
                k -= k.T
                x = td.xs[spin][state]
                fd = (_omega_at(td, mo @ expm(k), x, spin) - _omega_at(td, mo @ expm(-k), x, spin)) / 2
                self.assertAlmostEqual(fd, np.sum(xl * k), delta=1e-11)
                asym = xl - xl.T
                for sp in (core, somo, virt):                 # redundant rotations: invariance
                    self.assertLess(abs(asym[np.ix_(sp, sp)]).max(), 1e-8)

    def test_rohf_hessian_vs_fd(self):
        rng = np.random.default_rng(5)
        for key in ('hf', 'ks'):
            td = _td(MF[key])
            mo = td._scf.mo_coeff
            occs = sfxtda_grad._occupations(td)
            mask, hop, hdiag = sfxtda_grad._rohf_hessian(sfxtda_grad._fock_mo(td._scf),
                                                         sfxtda_grad._uks_response(td._scf, *occs), mo, occs)
            n = np.count_nonzero(mask)
            jac = np.array([hop(e) for e in np.eye(n)]).T
            self.assertLess(abs(jac - jac.T).max(), 1e-7)
            k = rng.standard_normal(n) * 1e-4
            fd = (_rohf_g(td, mo @ expm(_unpack(k, mask)), mask) - _rohf_g(td, mo @ expm(-_unpack(k, mask)), mask)) / 2
            self.assertLess(abs(fd - jac @ k).max(), 1e-9)
            self.assertLess(abs(np.diag(jac) - hdiag).max() / abs(hdiag).max(), 1.)  # sane preconditioner

    def test_z_vector_stationary(self):
        '''After the Z-vector solve, X^L - X^L^T vanishes on every orbital pair.'''
        for key in ('hf', 'ks', 'cam'):
            td = _td(MF[key])
            mo = td._scf.mo_coeff
            for spin, state in ((0, 0), (1, 0)):
                xo, fock, vresp, occs = _lagr(td, spin, state)[:4]
                mask, hop, hdiag = sfxtda_grad._rohf_hessian(fock, vresp, mo, occs)
                z = sfxtda_grad.solve_z(hop, hdiag, -(xo - xo.T)[mask], 1e-10, 100, lib.logger.new_logger(td))
                zz = sfxtda_grad._z_densities(z, mask, occs)
                xl = xo + sfxtda_grad._orbital_lagrangian(fock, vresp, mo, occs, zz, [], [])
                self.assertLess(abs(xl - xl.T).max(), 1e-7)


if __name__ == "__main__":
    print("Full Tests for SF-XTDA gradients")
    unittest.main()
