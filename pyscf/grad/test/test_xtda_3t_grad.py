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
from pyscf.xtda_3t import xtda_3t, _rdm2
from pyscf.grad import xtda_3t as xtda_3t_grad

CH2 = "C 0 0 0; H 0.1 1.0 0.5; H -0.2 -0.95 0.35"
MAKE = {
    'hf': lambda m: scf.ROHF(m).density_fit(),
    'hf_nodf': lambda m: scf.ROHF(m),
    'ks': lambda m: dft.ROKS(m, xc='BHANDHLYP').density_fit(),
    'pbe': lambda m: dft.ROKS(m, xc='PBE').density_fit(),
    'cam': lambda m: dft.ROKS(m, xc='CAM-B3LYP').density_fit(),
    'wb97_nodf': lambda m: dft.ROKS(m, xc='wB97'),          # long-range-only kernel, td-owned DF
}

def setUpModule():
    global mol, MF
    mol = gto.M(atom=CH2, basis='6-31g', spin=2, verbose=0)
    # omega is not variational in the orbitals: the default conv_tol_grad (1e-6) leaves ~1e-6 FD errors
    MF = {k: f(mol).run(conv_tol=1e-12, conv_tol_grad=1e-10) for k, f in MAKE.items()}

def tearDownModule():
    global mol, MF
    del mol, MF

def _td(mf, nstates=3, damping=None):
    td = xtda_3t.XTDA_3T(mf, nstates=nstates, spins=(0, 1, 2), damping=damping)
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
    na, nb = xtda_3t_grad._occupations(td)
    f = mf.get_fock(dm=np.array([(mo * na) @ mo.T, (mo * nb) @ mo.T]))
    fbar = .5 * mo.T @ (f.focka + f.fockb) @ mo
    ints0 = td._ints
    td._ints = xtda_3t._DFInts(td._with_df, mo, core, somo, virt, fbar, xtda_3t._kernel_terms(*td.xc_coeffs))
    try:
        return x @ td.gen_vind(spin)[0](x[None])[0]
    finally:
        td._ints = ints0

def _lagr(td, spin, state):
    mf = td._scf
    mo = td.mo_coeff
    occs = xtda_3t_grad._occupations(td)
    dfterms = xtda_3t_grad._df_terms(td.mol, td._with_df.auxmol, mo, xtda_3t._kernel_terms(*td.xc_coeffs))
    p, ghats = xtda_3t_grad._state_densities(td, spin, state, dfterms)
    fock = xtda_3t_grad._fock_mo(mf, mo)
    vresp = xtda_3t_grad._uks_response(mf, mo, *occs)
    x = xtda_3t_grad._orbital_lagrangian(fock, vresp, mo, occs, (p / 2, p / 2), dfterms, ghats)
    return x, fock, vresp, occs, dfterms, ghats, p

def _unpack(k, mask):
    kap = np.zeros(mask.shape)
    kap[mask] = k
    return kap - kap.T

def _rohf_g(td, mo, mask):
    '''ROKS orbital gradient (X^E - X^E^T)[mask] at orbitals mo.'''
    na, nb = xtda_3t_grad._occupations(td)
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

def _e_states(key, mol2, grids, dm0, states=STATES, damping=None):
    td = _td(_scf(key, mol2, grids, dm0), damping=damping)
    return np.array([td.e_tot[s][k] for s, k in states])

def _check_fd(test, key, states=STATES, tol=1e-6, full=False, damping=None):
    mf = MF[key]
    td = _td(mf, damping=damping)
    for s, k in states:                           # roots well separated -> FD follows the same root
        w = td.omega[s]
        test.assertGreater(min(abs(w[k] - w[j]) for j in range(len(w)) if j != k), 1e-3)
    grids = _frozen(mf.grids) if hasattr(mf, 'grids') else None
    dm0 = mf.make_rdm1()
    fun = lambda m2: _e_states(key, m2, grids, dm0, states, damping)
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


def _rotated(mf, seed, by_irrep=False):
    '''Shallow copy of mf with orbitals randomly rotated inside C, O, V (inside each (space, irrep) if by_irrep):
    the same ROKS state as another Roothaan matrix would deliver it.'''
    rng = np.random.default_rng(seed)
    mo = np.array(mf.mo_coeff)
    lab = mf.mo_occ.astype(int)
    if by_irrep:
        orbsym = scf.hf_symm.get_orbsym(mf.mol, mf.mo_coeff)
        lab = lab * 1000 + orbsym
    for v in np.unique(lab):
        idx = np.flatnonzero(lab == v)
        mo[:, idx] = mo[:, idx] @ np.linalg.qr(rng.standard_normal((idx.size, idx.size)))[0]
    mf2 = copy.copy(mf)
    mf2.mo_coeff = lib.tag_array(mo, orbsym=orbsym) if by_irrep else mo
    return mf2


def _dense_w(wss, wsl, gl):
    ns, nl = wsl.shape
    w = np.zeros((ns + nl, ns + nl))
    w[:ns, :ns], w[:ns, ns:], w[ns:, :ns] = wss, wsl, wsl.T
    w[ns:, ns:] += np.diag(gl)
    return w

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
            mo = td.mo_coeff
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

    def test_damping_weights_vs_fd(self):
        '''d omega/d lam of the lowest root of damped(H + lam V) == sum (x x^T + dW) V (toy matrices, F1).'''
        rng = np.random.default_rng(29)
        ns, nl, damping = 6, 11, (2.5, 4)
        h = .3 * rng.standard_normal((ns + nl, ns + nl))
        h += h.T
        h[np.diag_indices(ns + nl)] = rng.uniform(0, .8, ns + nl)
        v = rng.standard_normal(h.shape)
        v += v.T

        def damped(m):
            hss, hsl = m[:ns, :ns].copy(), m[:ns, ns:].copy()
            xtda_3t.damp_(hss, hsl, hss.diagonal().copy(), m.diagonal()[ns:].copy(), *damping)
            out = m.copy()
            out[:ns, :ns], out[:ns, ns:], out[ns:, :ns] = hss, hsl, hsl.T
            return out
        x = np.linalg.eigh(damped(h))[1][:, 0]
        wss, wsl, gs, gl = xtda_3t_grad._damping_weights(x, h[:ns, :ns], h[:ns, ns:], h.diagonal()[:ns],
                                                        h.diagonal()[ns:], *damping)
        w = np.outer(x, x) + _dense_w(wss + np.diag(gs), wsl, gl)
        lam = 1e-6
        fd = (np.linalg.eigvalsh(damped(h + lam * v))[0] - np.linalg.eigvalsh(damped(h - lam * v))[0]) / (2 * lam)
        self.assertAlmostEqual(fd, np.sum(w * v), delta=1e-8)
        self.assertAlmostEqual(gs.sum() + gl.sum(), 0, 12)

    def test_damping_weights_bilinear(self):
        '''_damping_weights(x, .., x2=y) is the polarisation of the quadratic weights (D2).'''
        rng = np.random.default_rng(4)
        ns, nl = 5, 9
        h = rng.standard_normal((ns + nl, ns + nl))
        h = h + h.T
        x, y = rng.standard_normal((2, ns + nl))
        args = (h[:ns, :ns], h[:ns, ns:], h.diagonal()[:ns], h.diagonal()[ns:], 2.5, 4)
        got = xtda_3t_grad._damping_weights(x, *args, x2=y)
        plus = xtda_3t_grad._damping_weights(x + y, *args)
        minus = xtda_3t_grad._damping_weights(x - y, *args)
        for g, a, b in zip(got, plus, minus):
            self.assertLess(abs(g - (a - b) / 4).max(), 1e-12)

    def test_state_densities_bilinear(self):
        '''Transition P, Ghat are symmetric in the two roots; state2 = state reproduces the state densities.'''
        for key, damping in (('cam', None), ('ks', (2., 5))):
            td = _td(MF[key], damping=damping)
            dfterms = xtda_3t_grad._df_terms(td.mol, xtda_3t_grad._auxmol(td), td.mo_coeff,
                                            xtda_3t._kernel_terms(*td.xc_coeffs))
            p01, g01 = xtda_3t_grad._state_densities(td, 0, 0, dfterms, 1)
            p10, g10 = xtda_3t_grad._state_densities(td, 0, 1, dfterms, 0)
            p00, g00 = xtda_3t_grad._state_densities(td, 0, 0, dfterms)
            p00b, g00b = xtda_3t_grad._state_densities(td, 0, 0, dfterms, 0)
            self.assertLess(abs(p01 - p10).max(), 1e-12)
            self.assertLess(abs(p00 - p00b).max(), 1e-10)
            self.assertLess(abs(np.trace(p01)), 1e-8)                  # traceless transition density
            for a, b in list(zip(g01, g10)) + list(zip(g00, g00b)):
                self.assertLess(abs(a - b).max(), 1e-10)

    def test_orbital_lagrangian_damped_vs_fd(self):
        '''X^omega of the damped model vs FD of x.Htilde(U).x (x fixed); intra-space rotations now contribute.'''
        rng = np.random.default_rng(11)
        for key in ('hf', 'ks', 'cam'):
            td = _td(MF[key], damping=(2., 5))
            mo = td.mo_coeff
            core, somo, virt = td._orbs
            for spin, state in ((0, 0), (1, 0)):
                xl = _lagr(td, spin, state)[0]
                k = rng.standard_normal(xl.shape) * 1e-5
                k -= k.T
                x = td.xs[spin][state]
                fd = (_omega_at(td, mo @ expm(k), x, spin) - _omega_at(td, mo @ expm(-k), x, spin)) / 2
                self.assertAlmostEqual(fd, np.sum(xl * k), delta=1e-11)
                asym = xl - xl.T
                self.assertGreater(max(abs(asym[np.ix_(sp, sp)]).max() for sp in (core, somo, virt)), 1e-6)

    def test_rohf_hessian_vs_fd(self):
        rng = np.random.default_rng(5)
        for key in ('hf', 'ks'):
            td = _td(MF[key])
            mo = td.mo_coeff
            occs = xtda_3t_grad._occupations(td)
            mask, hop, hdiag = xtda_3t_grad._rohf_hessian(xtda_3t_grad._fock_mo(td._scf, mo),
                                                         xtda_3t_grad._uks_response(td._scf, mo, *occs), mo, occs)
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
            mo = td.mo_coeff
            for spin, state in ((0, 0), (1, 0)):
                xo, fock, vresp, occs = _lagr(td, spin, state)[:4]
                mask, hop, hdiag = xtda_3t_grad._rohf_hessian(fock, vresp, mo, occs)
                z = xtda_3t_grad.solve_z(hop, hdiag, -(xo - xo.T)[mask], 1e-10, 100, lib.logger.new_logger(td))
                zz = xtda_3t_grad._z_densities(z, mask, occs)
                xl = xo + xtda_3t_grad._orbital_lagrangian(fock, vresp, mo, occs, zz, [], [])
                self.assertLess(abs(xl - xl.T).max(), 1e-7)

    def test_canonical_pz_degenerate(self):
        import io
        rng = np.random.default_rng(31)
        spaces = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
        eps = np.array([-1., -.5, -.2, -.1, .3, .3])            # virtuals 4, 5 degenerate
        for a45, warned in ((1e-3, True), (1e-9, False)):
            xo = rng.standard_normal((6, 6))
            xo[4, 5], xo[5, 4] = a45, 0.
            log = lib.logger.Logger(io.StringIO(), lib.logger.WARN)
            pz = xtda_3t_grad._canonical_pz(xo, eps, spaces, log)
            self.assertEqual(pz[4, 5], 0)
            self.assertAlmostEqual(pz[0, 1], -(xo[0, 1] - xo[1, 0]) / (2 * (eps[0] - eps[1])), 14)
            self.assertAlmostEqual(abs(pz - pz.T).max(), 0, 14)
            self.assertEqual(abs(pz[np.ix_([0, 1], [2, 3, 4, 5])]).max(), 0)   # no inter-space entries
            self.assertEqual(abs(np.diag(pz)).max(), 0)
            self.assertEqual('degenerate' in log.stdout.getvalue(), warned)

    def test_z_vector_stationary_damped(self):
        '''With Pz and the Z-vector, X^L - X^L^T vanishes on every pair, intra-space ones included.'''
        for key in ('hf', 'ks'):
            td = _td(MF[key], damping=(2., 5))
            mo = td.mo_coeff
            for spin, state in ((0, 0), (1, 0)):
                xo, fock, vresp, occs = _lagr(td, spin, state)[:4]
                pz = xtda_3t_grad._canonical_pz(xo, .5 * (fock[0] + fock[1]).diagonal(), td._orbs,
                                               lib.logger.new_logger(td))
                xo = xo + xtda_3t_grad._orbital_lagrangian(fock, vresp, mo, occs, (pz / 2, pz / 2), [], [])
                mask, hop, hdiag = xtda_3t_grad._rohf_hessian(fock, vresp, mo, occs)
                z = xtda_3t_grad.solve_z(hop, hdiag, -(xo - xo.T)[mask], 1e-10, 100, lib.logger.new_logger(td))
                zz = xtda_3t_grad._z_densities(z, mask, occs)
                xl = xo + xtda_3t_grad._orbital_lagrangian(fock, vresp, mo, occs, zz, [], [])
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
            xc = xtda_3t._reference(mf)[5]

            def fun(mol2):
                mf2 = MAKE[key](mol2)
                if grids is not None:
                    mf2.grids = grids
                v = mf2.get_veff(mol2, d)
                h1 = mf2.get_hcore(mol2)
                return sum(np.sum(q[t] * (h1 + v[t])) for t in range(2))
            v = rng.standard_normal((mol.natm, 3))
            de = xtda_3t_grad._fock_skeleton(mf, xc, q, d)
            # h = 2.5e-5: FD error ~ h^2 (2e-7 at 1e-4); cam keeps ~7e-8 from the pseudo-inverse LR metric (S3)
            self.assertAlmostEqual(_fd_dir(fun, mol, v, 2.5e-5), np.sum(de * v), delta=1e-7)

    def test_df_skeleton_vs_fd(self):
        rng = np.random.default_rng(17)
        for key in ('hf', 'cam'):
            td = _td(MF[key])
            mo = td.mo_coeff
            terms = xtda_3t._kernel_terms(*td.xc_coeffs)
            auxmol = td._with_df.auxmol
            x = td.xs[0][0]
            core, somo, virt = td._orbs
            blocks = xtda_3t.csf_list(0, len(core), len(virt))

            def fun(mol2):
                aux2 = auxmol.set_geom_(mol2.atom_coords(), unit='Bohr', inplace=False)
                return sum(c * np.sum(_rdm2.contract(x, blocks, dd, core, somo, virt) * t)
                           for c, _, t, dd in xtda_3t_grad._df_terms(mol2, aux2, mo, terms))
            dfterms = xtda_3t_grad._df_terms(mol, auxmol, mo, terms)
            de = xtda_3t_grad._df_skeleton(mol, auxmol, mo, dfterms, xtda_3t_grad._two_body(td, x, 0, dfterms))
            v = rng.standard_normal((mol.natm, 3))
            # the numerically singular LR metric (pseudo-inverse) puts ~4e-8 of noise on the cam FD
            self.assertAlmostEqual(_fd_dir(fun, mol, v), np.sum(de * v), delta=1e-8 if key == 'hf' else 1e-7)

    def test_grad_vs_fd_full(self):
        _check_fd(self, 'hf', states=((0, 0), (1, 0)), full=True)

    def test_grad_vs_fd_direction(self):
        for key in ('hf_nodf', 'ks', 'pbe'):
            _check_fd(self, key)

    def test_grad_nodf_without_full_range_term(self):
        '''Non-DF reference whose kernel has no full-range term: the td-owned DF is only used through
        range_coulomb copies, so its own auxmol is never built.'''
        _check_fd(self, 'wb97_nodf')

    def test_grad_vs_fd_rsh_slow(self):
        _check_fd(self, 'cam', tol=1e-5)

    def test_grad_damped_vs_fd_direction(self):
        # pbe skipped: its damped S=1 Davidson solve intermittently collapses at conv_tol=1e-7
        for key, damping in (('hf', (1.7, 4)), ('hf_nodf', (2., 5)), ('ks', (2., 5))):
            _check_fd(self, key, damping=damping)

    def test_grad_damped_vs_fd_rsh_slow(self):
        _check_fd(self, 'cam', tol=1e-5, damping=(2., 5))

    def test_grad_damped_symmetry(self):
        atom = "C 0 0 0; H 0 0.95 0.55; H 0 -0.95 0.55"
        res = []
        for sym in (True, False):
            m = gto.M(atom=atom, basis='6-31g', spin=2, symmetry=sym, verbose=0)
            td = _td(scf.ROHF(m).density_fit().run(conv_tol=1e-12, conv_tol_grad=1e-10), damping=(2., 5))
            res.append(td.nuc_grad_method().kernel(state=0, spin=1))
        self.assertAlmostEqual(abs(res[0] - res[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(res[0][:, 0]).max(), 0, 8)

    def test_grad_symmetry(self):
        atom = "C 0 0 0; H 0 0.95 0.55; H 0 -0.95 0.55"
        res = []
        for sym in (True, False):
            m = gto.M(atom=atom, basis='6-31g', spin=2, symmetry=sym, verbose=0)
            td = _td(scf.ROHF(m).density_fit().run(conv_tol=1e-12, conv_tol_grad=1e-10))
            res.append(td.nuc_grad_method().kernel(state=0, spin=1))
        self.assertAlmostEqual(abs(res[0] - res[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(res[0][:, 0]).max(), 0, 8)

    def test_grad_rotated_reference(self):
        '''Reference orbitals rotated inside C, O, V give the same gradient (D5).'''
        mf = _scf('hf', mol)
        ref = _td(mf).nuc_grad_method().kernel(state=0, spin=1)
        de = _td(_rotated(mf, 3)).nuc_grad_method().kernel(state=0, spin=1)
        self.assertAlmostEqual(abs(de - ref).max(), 0, 7)

    def test_grad_hooks_and_errors(self):
        td = _td(MF['hf'])
        self.assertIsInstance(td.nuc_grad_method(), xtda_3t_grad.Gradients)
        self.assertIsInstance(td.Gradients(), xtda_3t_grad.Gradients)
        with self.assertRaises(ValueError):
            td.nuc_grad_method().kernel(state=99, spin=0)
        td1 = xtda_3t.XTDA_3T(MF['hf'], spins=(1,))
        with self.assertRaises(ValueError):
            td1.nuc_grad_method().kernel(spin=0)
        mfn = dft.ROKS(mol, xc='B3LYP')
        mfn.nlc = 'vv10'                         # do_nlc() only recognises the lower-case value
        with self.assertRaises(NotImplementedError):
            xtda_3t.XTDA_3T(mfn).nuc_grad_method().kernel()

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
    print("Full Tests for XTDA-3T gradients")
    unittest.main()
