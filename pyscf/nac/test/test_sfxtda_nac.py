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
from pyscf import gto, scf, dft
from pyscf.sfxtda import sfxtda, _tdm
from pyscf.nac import sfxtda as sfxtda_nac

CH2 = "C 0 0 0; H 0.1 1.0 0.5; H -0.2 -0.95 0.35"
MAKE = {
    'hf': lambda m: scf.ROHF(m).density_fit(),
    'hf_nodf': lambda m: scf.ROHF(m),
    'ks': lambda m: dft.ROKS(m, xc='BHANDHLYP').density_fit(),
    'pbe': lambda m: dft.ROKS(m, xc='PBE').density_fit(),
    'cam': lambda m: dft.ROKS(m, xc='CAM-B3LYP').density_fit(),
}
PAIRS = ((1, 0), (2, 0), (3, 1))           # (ket, bra)

def setUpModule():
    global mol, MF
    mol = gto.M(atom=CH2, basis='6-31g', spin=2, verbose=0)
    MF = {k: f(mol).run(conv_tol=1e-12, conv_tol_grad=1e-10) for k, f in MAKE.items()}

def tearDownModule():
    global mol, MF
    del mol, MF

def _scf(key, mol2, grids=None, dm0=None):
    mf2 = MAKE[key](mol2)
    if grids is not None:
        mf2.grids = grids
    return mf2.run(dm0, conv_tol=1e-12, conv_tol_grad=1e-10)

def _frozen(grids):
    '''Grid of the reference geometry without AO screening: FD derivatives then exclude grid response.'''
    g = copy.copy(grids)
    g.non0tab = g.screen_index = None
    return g

def _td(mf, spin, damping=None, nstates=5):
    '''Roots of one spin with exact eigenvectors (dense vind(1), n ~ 140), signs as from Davidson (D9, F3).'''
    td = sfxtda.SFXTDA(mf, nstates=nstates, spins=(spin,), damping=damping)
    td.kernel()
    vind, _ = td.gen_vind(spin)
    w, v = np.linalg.eigh(vind(np.eye(td.xs[spin].shape[1])))
    v = v[:, :nstates].T
    td.xs[spin] = v * np.sign(np.einsum('ij,ij->i', v, td.xs[spin]))[:, None]
    td.omega[spin] = w[:nstates]
    return td

def _overlap(td0, td1, ket, bra, spin):
    '''<bra(R0)|ket(R1)> to first order in eps = C0^T S01 C1 - 1 (error even in the step), ket(R1) sign-aligned.'''
    core, somo, virt = td0._orbs
    blocks = sfxtda.csf_list(spin, len(core), len(virt))
    eps = td0.mo_coeff.T @ gto.intor_cross('int1e_ovlp', td0.mol, td1.mol) @ td1.mo_coeff
    eps -= np.eye(len(eps))
    xb = td0.xs[spin][bra]
    xk = td1.xs[spin][ket] * np.sign(td1.xs[spin][ket] @ td0.xs[spin][ket])
    return xb @ xk + np.sum(eps * _tdm.transition_dm(xb, xk, blocks, core, somo, virt).T)

def _check_fd(test, key, spin, damping=None, tol=1e-6, pairs=PAIRS, h=1e-4):
    mf = MF[key]
    td = _td(mf, spin, damping)
    w = td.omega[spin]
    for ket, bra in pairs:
        test.assertGreater(abs(w[ket] - w[bra]), 1e-3)
    grids = _frozen(mf.grids) if hasattr(mf, 'grids') else None
    dm0 = mf.make_rdm1()
    v = np.random.default_rng(19).standard_normal((mol.natm, 3))
    c0 = mol.atom_coords()
    tds = [_td(_scf(key, mol.set_geom_(c0 + s * h * v, unit='Bohr', inplace=False), grids, dm0), spin, damping)
           for s in (1, -1)]
    nac = td.nac_method()
    for ket, bra in pairs:
        d = nac.kernel(state=(ket, bra), spin=spin)
        fd = (_overlap(td, tds[0], ket, bra, spin) - _overlap(td, tds[1], ket, bra, spin)) / (2 * h)
        test.assertAlmostEqual(np.sum(d * v), fd, delta=tol)

def _rotated(mf, seed):
    '''Shallow copy of mf with orbitals randomly rotated inside C, O, V (another Roothaan matrix).'''
    rng = np.random.default_rng(seed)
    mo = np.array(mf.mo_coeff)
    for v in np.unique(mf.mo_occ):
        idx = np.flatnonzero(mf.mo_occ == v)
        mo[:, idx] = mo[:, idx] @ np.linalg.qr(rng.standard_normal((idx.size, idx.size)))[0]
    mf2 = copy.copy(mf)
    mf2.mo_coeff = mo
    return mf2

def _same_up_to_sign(a, b):
    return min(abs(a - b).max(), abs(a + b).max())

class KnownValues(unittest.TestCase):
    def test_nac_vs_fd(self):
        for key in ('hf', 'hf_nodf'):
            _check_fd(self, key, 0)

    def test_nac_vs_fd_s2(self):
        _check_fd(self, 'hf', 2)

    def test_nac_damped_vs_fd(self):
        for key, damping in (('ks', (1., 8.)), ('hf', (0.9, 5.0)), ('pbe', (1., 8.))):
            _check_fd(self, key, 0, damping, tol=5e-6)

    def test_nac_vs_fd_rsh_slow(self):
        _check_fd(self, 'cam', 1)
        _check_fd(self, 'ks', 1, (1., 8.))

    def test_nac_antisymmetry_and_etf(self):
        td = _td(MF['hf'], 0)
        nac = td.nac_method()
        d10 = nac.kernel(state=(1, 0), spin=0)
        self.assertLess(abs(d10 + nac.kernel(state=(0, 1))).max(), 1e-9)
        h10 = nac.kernel(state=(1, 0), mult_ediff=True)
        self.assertLess(abs(h10 - nac.kernel(state=(0, 1))).max(), 1e-10)     # mult_ediff persists
        self.assertLess(abs(h10 - (td.omega[0][1] - td.omega[0][0]) * d10).max(), 1e-10)
        self.assertGreater(abs(d10.sum(0)).max(), 1e-3)                        # AO term breaks translation inv.
        d_etf = nac.kernel(state=(1, 0), mult_ediff=False, use_etfs=True)
        self.assertLess(abs(d_etf.sum(0)).max(), 1e-10)                        # HF only (F2)
        self.assertEqual(abs(nac.kernel(state=(2, 2))).max(), 0)
        nac.reset_phase().kernel(state=(2, 2))
        self.assertIn(0, nac._phase_ref)                                       # phases follow every call

    def test_csf_sign_parity(self):
        '''MO sign flips s change each CSF by the parity of csf_irreps with the flip bits as irreps (F4).'''
        for spin in (0, 1, 2):
            td = sfxtda.SFXTDA(MF['ks'], spins=(spin,), damping=(1., 8.)).build()
            core, somo, virt = td._orbs
            blocks = sfxtda.csf_list(spin, len(core), len(virt))
            n = sum(r * c for _, _, (r, c) in blocks)
            h = td.gen_vind(spin)[0](np.eye(n))
            s = np.random.default_rng(spin).choice([-1., 1.], td.mo_coeff.shape[1])
            sig = 1 - 2. * sfxtda.csf_irreps(blocks, (s < 0).astype(int), core, somo, virt)
            td._ints = sfxtda._DFInts(td._with_df, td.mo_coeff * s, core, somo, virt, td._ints.fbar * np.outer(s, s),
                                      sfxtda._kernel_terms(*td.xc_coeffs))
            self.assertLess(abs(td.gen_vind(spin)[0](np.eye(n)) - sig[:, None] * h * sig).max(), 1e-12)

    def test_phase_tracking(self):
        '''Flipped MO signs (CSF vectors transformed with the parity) and a flipped root give the same NAC (D7).'''
        td = _td(MF['hf'], 0)
        nac = td.nac_method()
        ref = nac.kernel(state=(1, 0), spin=0)
        core, somo, virt = td._orbs
        s = np.where(np.arange(td.mo_coeff.shape[1]) % 3 == 0, -1., 1.)
        sig = 1 - 2. * sfxtda.csf_irreps(sfxtda.csf_list(0, len(core), len(virt)), (s < 0).astype(int),
                                         core, somo, virt)
        td.mo_coeff = td.mo_coeff * s              # undamped NAC path reads only mo_coeff and xs
        td.xs[0] = td.xs[0] * sig
        td.xs[0][1] *= -1
        self.assertLess(abs(nac.kernel(state=(1, 0), spin=0) - ref).max(), 1e-8)
        self.assertLess(abs(nac.reset_phase().kernel(state=(1, 0), spin=0) + ref).max(), 1e-8)

    def test_phase_warning_only_for_the_pair(self):
        '''A weak overlap with the previous call warns only for ket and bra, and says the phase is uncertain.'''
        import io
        td = _td(MF['hf'], 0)
        nac = td.nac_method()
        nac.kernel(state=(1, 0), spin=0)
        self.assertIsNot(nac._phase_ref[0][0], td.mol)          # moving td.mol in place keeps the reference
        for root, warned in ((3, False), (1, True)):
            mol0, mo0, x0 = nac._phase_ref[0]
            x0 = x0.copy()
            x0[root] = 0                                         # no overlap with the stored reference
            nac._phase_ref[0] = (mol0, mo0, x0)
            buf = io.StringIO()
            nac.stdout, nac.verbose = buf, 4
            nac.kernel(state=(1, 0), spin=0)
            self.assertEqual('weak overlap' in buf.getvalue(), warned)

    def test_phase_tracking_across_geometries(self):
        '''Phases tracked to a new geometry survive MO sign flips and a flipped root there (D7).'''
        td = _td(MF['hf'], 0)
        nac = td.nac_method()
        nac.kernel(state=(1, 0), spin=0)
        saved = nac._phase_ref[0]
        mol2 = mol.set_geom_(mol.atom_coords() + np.array([[0, 0, .03], [0, .02, 0], [.01, 0, -.02]]),
                             unit='Bohr', inplace=False)
        td2 = _td(_scf('hf', mol2), 0)
        nac2 = td2.nac_method()
        nac2._phase_ref = {0: saved}
        ref = nac2.kernel(state=(1, 0), spin=0)
        core, somo, virt = td2._orbs
        s = np.where(np.arange(td2.mo_coeff.shape[1]) % 3 == 1, -1., 1.)
        sig = 1 - 2. * sfxtda.csf_irreps(sfxtda.csf_list(0, len(core), len(virt)), (s < 0).astype(int),
                                         core, somo, virt)
        td2.mo_coeff = td2.mo_coeff * s              # undamped NAC path reads only mo_coeff and xs
        td2.xs[0] = td2.xs[0] * sig
        td2.xs[0][1] *= -1
        nac3 = td2.nac_method()
        nac3._phase_ref = {0: saved}
        self.assertLess(abs(nac3.kernel(state=(1, 0), spin=0) - ref).max(), 1e-8)

    def test_nac_hooks_and_errors(self):
        td = _td(MF['hf'], 0)
        nac = td.nac_method()
        self.assertIsInstance(nac, sfxtda_nac.NonAdiabaticCouplings)
        for bad in (1, (0, 1, 2)):
            with self.assertRaises(ValueError):
                nac.kernel(state=bad, spin=0)
        self.assertEqual(nac.state, (1, 0))                     # a rejected state is not kept
        with self.assertRaises(ValueError):
            nac.kernel(state=(1, 99), spin=0)
        with self.assertRaises(ValueError):
            nac.kernel(state=(0, 1, 2), spin=0)
        with self.assertRaises(ValueError):
            nac.kernel(state=(1, 0), spin=2)                     # spin not computed
        td.xs[0] = td.xs[0].copy()
        td.xs[0][1] *= .5
        with self.assertRaises(RuntimeError):
            nac.kernel(state=(1, 0), spin=0)
        mfn = dft.ROKS(mol, xc='B3LYP')
        mfn.nlc = 'vv10'                         # do_nlc() only recognises the lower-case value
        with self.assertRaises(NotImplementedError):
            sfxtda_nac.NonAdiabaticCouplings(sfxtda.SFXTDA(mfn)).kernel()

    def test_nac_near_degenerate_warning(self):
        import io
        td = sfxtda.SFXTDA(MF['hf'], nstates=4, spins=(1,))
        td.conv_tol = 1e-9
        td.kernel()
        self.assertLess(abs(td.omega[1][2] - td.omega[1][1]), 1e-4)  # 4.9e-5 Ha apart (test_tdm_vs_detci)
        for mult, warned in ((False, True), (True, False)):
            buf = io.StringIO()
            nac = td.nac_method()
            nac.stdout, nac.verbose = buf, 4
            nac.kernel(state=(2, 1), spin=1, mult_ediff=mult)
            self.assertEqual('use mult_ediff' in buf.getvalue(), warned)

    def test_nac_symmetry_irreps(self):
        '''With mol.symmetry, NACs between roots of different irreps survive and equal the C1 result (D6).'''
        atom = "C 0 0 0; H 0 0.95 0.55; H 0 -0.95 0.55"
        res = []
        for sym in (True, False):
            m = gto.M(atom=atom, basis='6-31g', spin=2, symmetry=sym, verbose=0)
            td = sfxtda.SFXTDA(scf.ROHF(m).density_fit().run(conv_tol=1e-12, conv_tol_grad=1e-10),
                               nstates=4, spins=(0,))
            td.conv_tol = 1e-8
            td.kernel()
            if sym:
                ket = next(k for k in range(1, 4) if td.irrep[0][k] != td.irrep[0][0])
            res.append(td.nac_method().kernel(state=(ket, 0), spin=0))
        self.assertGreater(abs(res[0]).max(), 1e-3)
        self.assertAlmostEqual(_same_up_to_sign(*res), 0, 6)

    def test_nac_rotated_reference(self):
        '''Undamped: a non-canonical td.mo_coeff (rotated inside C, O, V after build) gives the same NAC.  Damped: the
        model is defined in the Fbar-canonical basis, so rotated SCF orbitals give the same NAC.  A constant rotation
        leaves kappa_intra = 0 consistent in both terms; the R-dependent intra-space choice of Derivation step 6 is
        tested by test_nac_vs_fd (FD with canonical orbitals at R +- h).'''
        mf = MF['ks']
        res = []
        for rot in (False, True):
            td = sfxtda.SFXTDA(mf, nstates=4, spins=(0,))
            td.conv_tol = 1e-10
            td.build()
            if rot:
                core, somo, virt = td._orbs
                rng = np.random.default_rng(5)
                u = np.eye(td.mo_coeff.shape[1])
                for sp in (core, somo, virt):
                    u[np.ix_(sp, sp)] = np.linalg.qr(rng.standard_normal((sp.size, sp.size)))[0]
                td.mo_coeff = td.mo_coeff @ u
                td._ints = sfxtda._DFInts(td._with_df, td.mo_coeff, core, somo, virt, u.T @ td._ints.fbar @ u,
                                          sfxtda._kernel_terms(*td.xc_coeffs))
            td.kernel()
            res.append(td.nac_method().kernel(state=(1, 0), spin=0))
        self.assertGreater(abs(res[0]).max(), 1e-2)
        self.assertAlmostEqual(_same_up_to_sign(*res), 0, 7)
        a = _td(mf, 0, (1., 8.)).nac_method().kernel(state=(1, 0), spin=0)
        b = _td(_rotated(mf, 3), 0, (1., 8.)).nac_method().kernel(state=(1, 0), spin=0)
        self.assertAlmostEqual(_same_up_to_sign(a, b), 0, 7)

    def test_nac_scanner(self):
        coords2 = mol.atom_coords() + np.array([[0, 0, .03], [0, .02, 0], [.01, 0, -.02]])
        mol2 = mol.set_geom_(coords2, unit='Bohr', inplace=False)

        def fresh(m):
            # own SCF: the SCF scanner shares (and rebuilds) with_df with the object it was made from
            td = sfxtda.SFXTDA(_scf('hf', m), nstates=4, spins=(0,))
            td.conv_tol = 1e-7
            return td.run()
        nac0 = fresh(mol).nac_method()
        scanner = nac0.as_scanner(state=(1, 0), spin=0)
        _, d0 = scanner(mol)
        e2, d2 = scanner(mol2)
        self.assertTrue(scanner.converged)
        td2 = fresh(mol2)
        ref = td2.nac_method().kernel(state=(1, 0), spin=0)
        self.assertAlmostEqual(abs(e2 - td2.e_tot[0][[1, 0]]).max(), 0, 8)
        self.assertAlmostEqual(_same_up_to_sign(d2, ref), 0, 5)
        self.assertGreater(np.sum(d2 * d0), 0)                   # tracked phases: continuous along the path
        self.assertEqual(nac0._phase_ref, {})                    # the scanner tracks its own phases
        # NAMD loop without a second SF-XTDA run: NAC bound to the gradient scanner's SF-XTDA object
        g = fresh(mol).nuc_grad_method().as_scanner(state=1, spin=0)
        nac = g.base.nac_method()
        g(mol2)
        self.assertAlmostEqual(_same_up_to_sign(nac.kernel(state=(1, 0), spin=0), ref), 0, 5)
        g(mol)
        stale = g.base.nac_method()                              # bound at mol, then moved without kernel()
        g(mol2)
        w = g.base.omega[0]
        h = sfxtda_nac.nac_elec(stale, 0, 1, 0)
        self.assertAlmostEqual(_same_up_to_sign(h, (w[1] - w[0]) * ref), 0, 5)


if __name__ == "__main__":
    print("Full Tests for SF-XTDA non-adiabatic couplings")
    unittest.main()
