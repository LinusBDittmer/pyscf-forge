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
    # omega is not variational in the orbitals: the default conv_tol_grad (1e-6) leaves ~1e-6 FD errors
    MF = {k: f(mol).run(conv_tol=1e-12, conv_tol_grad=1e-10) for k, f in MAKE.items()}

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

def _fd_dir(fun, mol, v, h=1e-4):
    c0 = mol.atom_coords()
    return (fun(mol.set_geom_(c0 + h * v, unit='Bohr', inplace=False))
            - fun(mol.set_geom_(c0 - h * v, unit='Bohr', inplace=False))) / (2 * h)

STATES = ((0, 0), (0, 1), (1, 0), (2, 0))

def _scf(key, mol2, grids=None, dm0=None):
    mf2 = MAKE[key](mol2)
    if grids is not None:
        mf2.grids = grids
    return mf2.run(dm0, conv_tol=1e-12, conv_tol_grad=1e-10)

def _e_states(key, mol2, grids, dm0, states=STATES):
    td = _td(_scf(key, mol2, grids, dm0))
    return np.array([td.e_tot[s][k] for s, k in states])

def _check_fd(test, key, states=STATES, tol=1e-6, full=False):
    mf = MF[key]
    td = _td(mf)
    for s, k in states:                           # roots well separated -> FD follows the same root
        w = td.omega[s]
        test.assertGreater(min(abs(w[k] - w[j]) for j in range(len(w)) if j != k), 1e-3)
    grids = _frozen(mf.grids) if hasattr(mf, 'grids') else None
    dm0 = mf.make_rdm1()
    fun = lambda m2: _e_states(key, m2, grids, dm0, states)
    g = td.nuc_grad_method()
    de = np.array([g.kernel(state=k, spin=s) for s, k in states])
    if full:
        for a in range(mol.natm):
            for c in range(3):
                v = np.zeros((mol.natm, 3))
                v[a, c] = 1
                test.assertLess(abs(_fd_dir(fun, mol, v) - de[:, a, c]).max(), tol)
    else:
        v = np.random.default_rng(19).standard_normal((mol.natm, 3))
        test.assertLess(abs(_fd_dir(fun, mol, v) - np.einsum('nax,ax->n', de, v)).max(), tol)


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

    def test_fock_skeleton_vs_fd(self):
        rng = np.random.default_rng(13)
        for key in ('hf', 'hf_nodf', 'ks', 'cam'):
            mf = MF[key]
            nao = mol.nao
            q = rng.standard_normal((2, nao, nao))
            q += q.transpose(0, 2, 1)
            d = np.asarray(mf.make_rdm1())
            grids = _frozen(mf.grids) if hasattr(mf, 'grids') else None
            xc = sfxtda._reference(mf)[5]

            def fun(mol2):
                mf2 = MAKE[key](mol2)
                if grids is not None:
                    mf2.grids = grids
                v = mf2.get_veff(mol2, d)
                h1 = mf2.get_hcore(mol2)
                return sum(np.sum(q[t] * (h1 + v[t])) for t in range(2))
            v = rng.standard_normal((mol.natm, 3))
            de = sfxtda_grad._fock_skeleton(mf, xc, q, d)
            # h = 2.5e-5: FD error ~ h^2 (2e-7 at 1e-4); cam keeps ~7e-8 from the pseudo-inverse LR metric (S3)
            self.assertAlmostEqual(_fd_dir(fun, mol, v, 2.5e-5), np.sum(de * v), delta=1e-7)

    def test_df_skeleton_vs_fd(self):
        rng = np.random.default_rng(17)
        for key in ('hf', 'cam'):
            td = _td(MF[key])
            mo = td._scf.mo_coeff
            terms = sfxtda._kernel_terms(*td.xc_coeffs)
            auxmol = td._with_df.auxmol
            x = td.xs[0][0]
            core, somo, virt = td._orbs
            blocks = sfxtda.csf_list(0, len(core), len(virt))

            def fun(mol2):
                aux2 = auxmol.set_geom_(mol2.atom_coords(), unit='Bohr', inplace=False)
                return sum(c * np.sum(_rdm2.contract(x, blocks, dd, core, somo, virt) * t)
                           for c, _, t, dd in sfxtda_grad._df_terms(mol2, aux2, mo, terms))
            dfterms = sfxtda_grad._df_terms(mol, auxmol, mo, terms)
            de = sfxtda_grad._df_skeleton(mol, auxmol, mo, dfterms, sfxtda_grad._two_body(td, x, 0, dfterms))
            v = rng.standard_normal((mol.natm, 3))
            # the numerically singular LR metric (pseudo-inverse) puts ~4e-8 of noise on the cam FD
            self.assertAlmostEqual(_fd_dir(fun, mol, v), np.sum(de * v), delta=1e-8 if key == 'hf' else 1e-7)

    def test_grad_vs_fd_full(self):
        _check_fd(self, 'hf', states=((0, 0), (1, 0)), full=True)

    def test_grad_vs_fd_direction(self):
        for key in ('hf_nodf', 'ks', 'pbe'):
            _check_fd(self, key)

    def test_grad_vs_fd_rsh_slow(self):
        _check_fd(self, 'cam', tol=1e-5)

    def test_grad_symmetry(self):
        atom = "C 0 0 0; H 0 0.95 0.55; H 0 -0.95 0.55"
        res = []
        for sym in (True, False):
            m = gto.M(atom=atom, basis='6-31g', spin=2, symmetry=sym, verbose=0)
            td = _td(scf.ROHF(m).density_fit().run(conv_tol=1e-12, conv_tol_grad=1e-10))
            res.append(td.nuc_grad_method().kernel(state=0, spin=1))
        self.assertAlmostEqual(abs(res[0] - res[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(res[0][:, 0]).max(), 0, 8)

    def test_grad_hooks_and_errors(self):
        td = _td(MF['hf'])
        self.assertIsInstance(td.nuc_grad_method(), sfxtda_grad.Gradients)
        self.assertIsInstance(td.Gradients(), sfxtda_grad.Gradients)
        with self.assertRaises(ValueError):
            td.nuc_grad_method().kernel(state=99, spin=0)
        td1 = sfxtda.SFXTDA(MF['hf'], spins=(1,))
        with self.assertRaises(ValueError):
            td1.nuc_grad_method().kernel(spin=0)
        tdd = sfxtda.SFXTDA(MF['hf'], damping=(1., 8.))
        with self.assertRaises(NotImplementedError):
            tdd.nuc_grad_method().kernel()
        mfn = dft.ROKS(mol, xc='B3LYP')
        mfn.nlc = 'vv10'                         # do_nlc() only recognises the lower-case value
        with self.assertRaises(NotImplementedError):
            sfxtda.SFXTDA(mfn).nuc_grad_method().kernel()

    def test_grad_refuses_collapsed_root(self):
        td = _td(MF['hf'])
        td.xs[1] = td.xs[1].copy()
        td.xs[1][1] = 1e-8                       # what a collapsed Davidson root looks like (S6)
        with self.assertRaises(RuntimeError):
            td.nuc_grad_method().kernel(state=1, spin=1)

    def test_grad_near_degenerate_warning(self):
        import io
        td = _td(MF['hf'])
        g = td.nuc_grad_method()
        g.stdout = io.StringIO()
        g.verbose = lib.logger.WARN
        g.kernel(state=1, spin=1)
        self.assertIn('near-degenerate', g.stdout.getvalue())

    def test_grad_scanner(self):
        coords2 = mol.atom_coords() + np.array([[0, 0, .03], [0, .02, 0], [.01, 0, -.02]])
        coords3 = mol.atom_coords() - .02
        # own SCF: the SCF scanner shares (and rebuilds) with_df with the object it was made from
        scanner = _td(_scf('hf', mol)).nuc_grad_method().as_scanner(state=1, spin=0)
        e2, de2 = scanner(mol.set_geom_(coords2, unit='Bohr', inplace=False))
        self.assertTrue(scanner.converged)
        scanner(coords3 * lib.param.BOHR)                # array path: coordinates in mol.unit (Angstrom)
        e2b, de2b = scanner(mol.set_geom_(coords2, unit='Bohr', inplace=False))
        mol2 = mol.set_geom_(coords2, unit='Bohr', inplace=False)
        td2 = _td(MAKE['hf'](mol2).run(conv_tol=1e-12, conv_tol_grad=1e-10))
        ref = td2.nuc_grad_method().kernel(state=1, spin=0)
        self.assertAlmostEqual(e2, td2.e_tot[0][1], 8)
        self.assertAlmostEqual(abs(de2 - ref).max(), 0, 6)
        self.assertAlmostEqual(abs(de2b - ref).max(), 0, 6)

    @unittest.skipUnless(__import__('importlib').util.find_spec('geometric'), 'geomeTRIC not installed')
    def test_optimizer_hook(self):
        from pyscf.geomopt import geometric_solver
        opt = _td(MF['hf']).nuc_grad_method().optimizer()
        self.assertIsInstance(opt, geometric_solver.GeometryOptimizer)


if __name__ == "__main__":
    print("Full Tests for SF-XTDA gradients")
    unittest.main()
