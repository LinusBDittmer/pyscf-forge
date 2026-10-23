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

import unittest
import numpy as np
from pyscf import gto, scf, dft, lib, symm
from pyscf.data.nist import HARTREE2EV
from pyscf.sfxtda import sfxtda
from pyscf.sfxtda._blocks import BLOCKS

CH2 = "C 0 0 0; H 0.1 1.0 0.5; H -0.2 -0.95 0.35"

def setUpModule():
    global mol, mf_hf, mf_ks, mf_pbe
    mol = gto.M(atom=CH2, basis='6-31g', spin=2, verbose=0)
    mf_hf = scf.ROHF(mol).density_fit().run(conv_tol=1e-11)
    mf_ks = dft.ROKS(mol, xc='BHANDHLYP').density_fit().run(conv_tol=1e-11)
    mf_pbe = dft.ROKS(mol, xc='PBE').density_fit().run(conv_tol=1e-11)

def tearDownModule():
    global mol, mf_hf, mf_ks, mf_pbe
    del mol, mf_hf, mf_ks, mf_pbe

def _config_ids(blocks, nC, nV):
    '''Spatial-configuration id of every CSF (as in the dense prototype).'''
    keys = []
    for cl, nm, (a, b) in blocks:
        if cl == "OO":
            keys += [nm if nm in ("G", "D") else "OO"]
        elif cl == "CO":
            keys += [("CO", nm[-4:], i) for i in range(nC)]
        elif cl == "OV":
            keys += [("OV", nm[-4:], c) for c in range(nV)]
        else:
            tag = nm[-4:] if "^2" in nm else "open"
            keys += [("CV", tag, i, c) for i in range(nC) for c in range(nV)]
    uniq = {k: n for n, k in enumerate(dict.fromkeys(keys))}
    return np.array([uniq[k] for k in keys])

def _dense(td, spin, damping=None):
    '''Dense <K|H-E0|L> from _blocks.py with the module's DF integrals; noCVCV damping.'''
    ints = td._ints
    nC, nV = ints.nC, ints.nV
    blocks = sfxtda.csf_list(spin, nC, nV)
    off = np.cumsum([0] + [r * c for _, _, (r, c) in blocks])
    M = np.zeros((off[-1], off[-1]))
    ints.allow_4index = True
    for x, (A, na, (ni, nai)) in enumerate(blocks):
        for y, (B, nb, (nj, nbj)) in enumerate(blocks):
            if (A, na, B, nb) in BLOCKS:
                args, f = BLOCKS[(A, na, B, nb)]
                blk = np.broadcast_to(f(*[ints.value(a) for a in args]), (ni, nai, nj, nbj))
                M[off[x]:off[x+1], off[y]:off[y+1]] = blk.reshape(ni*nai, nj*nbj)
                M[off[y]:off[y+1], off[x]:off[x+1]] = blk.reshape(ni*nai, nj*nbj).T
    ints.allow_4index = False
    if damping is not None:
        cid = _config_ids(blocks, nC, nV)
        cv = np.concatenate([np.full(r * c, cl == 'CV') for cl, _, (r, c) in blocks])
        d = M.diagonal().copy()
        fac = damping[0] * np.exp(-damping[1] * (d[:, None] - d[None, :])**4)
        M *= np.where((cid[:, None] != cid[None, :]) & ~(cv[:, None] & cv[None, :]), fac, 1.)
    return M

class KnownValues(unittest.TestCase):
    def test_reference(self):
        for mf, ch in ((mf_hf, 1.0), (mf_ks, 0.5), (mf_pbe, 0.0)):
            C, core, somo, virt, fbar, c_h = sfxtda._reference(mf)
            self.assertEqual((len(core), len(somo), len(virt)), (3, 2, 8))
            self.assertAlmostEqual(c_h, ch, 12)
            v = mf.get_veff(mol, mf.make_rdm1())
            ref = .5 * C.T @ (2 * mf.get_hcore() + v[0] + v[1]) @ C
            self.assertAlmostEqual(abs(fbar - ref).max(), 0, 10)

    def test_bad_reference(self):
        with self.assertRaises(NotImplementedError):
            sfxtda._reference(dft.ROKS(mol, xc='CAMB3LYP').density_fit().run())
        with self.assertRaises(TypeError):
            sfxtda._reference(dft.UKS(mol, xc='BHANDHLYP').run())
        with self.assertRaises(ValueError):
            sfxtda._reference(scf.ROHF(gto.M(atom=CH2, basis='6-31g', verbose=0)).run())

    def test_df_tensors(self):
        ints = sfxtda.SFXTDA(mf_ks).build()._ints
        C = mf_ks.mo_coeff
        cc, cv = C[:, ints.core], C[:, ints.virt]
        co1, co2 = C[:, ints.somo[:1]], C[:, ints.somo[1:]]
        def eri(*cs):
            shape = [c.shape[1] for c in cs]
            return .5 * mf_ks.with_df.ao2mo(cs, compact=False).reshape(shape)
        self.assertAlmostEqual(abs(lib.einsum('Lij,Lab->ijab', ints.bcc, ints.bvv) - eri(cc, cc, cv, cv)).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.einsum('Lia,Ljb->iajb', ints.bcv, ints.bcv) - eri(cc, cv, cc, cv)).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.einsum('Lxp,Lyq->xpyq', ints.bo, ints.bo)
                                   - eri(C[:, ints.somo], C, C[:, ints.somo], C)).max(), 0, 10)
        v = ints.value('(iO1|jO2)')
        self.assertEqual(v.shape, (3, 1, 3, 1))
        self.assertAlmostEqual(abs(v[:, 0, :, 0] - eri(cc, co1, cc, co2)[:, 0, :, 0]).max(), 0, 10)
        self.assertEqual(ints.value('(ij|O1b)').shape, (3, 1, 3, 8))
        fjb = ints.value('F_jb')
        self.assertAlmostEqual(abs(fjb[0, 0] - ints.fbar[np.ix_(ints.core, ints.virt)]).max(), 0, 14)
        with self.assertRaises(RuntimeError):
            ints.value('(ia|jb)')

    def test_explicit_blocks(self):
        for mf in (mf_hf, mf_ks):
            td = sfxtda.SFXTDA(mf).build()
            ints = td._ints
            for spin in (0, 1, 2):
                M = _dense(td, spin)
                hss, hsl = sfxtda.build_small(ints, spin)
                ns = hss.shape[0]
                self.assertEqual(ns, {0: 3 + 2*3 + 2*8, 1: 1 + 2*3 + 2*8, 2: 0}[spin])
                self.assertAlmostEqual(abs(hss - M[:ns, :ns]).max(initial=0), 0, 12)
                self.assertAlmostEqual(abs(hsl - M[:ns, ns:]).max(initial=0), 0, 12)
                lc, rv, cj, ck = sfxtda.cvcv_operator(ints, spin)
                self.assertAlmostEqual(abs(lc - lc.T).max(), 0, 12)
                self.assertAlmostEqual(abs(rv - rv.T).max(), 0, 12)
                hd = sfxtda.hdiag_large(ints, lc, rv, cj, ck)
                self.assertAlmostEqual(abs(hd.ravel() - M.diagonal()[ns:]).max(), 0, 12)

    def test_3oo_diagonal_zero(self):
        hss, _ = sfxtda.build_small(sfxtda.SFXTDA(mf_ks).build()._ints, 1)
        self.assertAlmostEqual(hss[0, 0], 0, 12)   # 3OO is the first S=1 CSF

    def test_jk_contract(self):
        ints = sfxtda.SFXTDA(mf_ks).build()._ints
        t = np.random.default_rng(1).random((2, 3, 8))
        ints.allow_4index = True
        iajb = np.broadcast_to(ints.value('(ia|jb)'), (3, 8, 3, 8))
        ijab = np.broadcast_to(ints.value('(ij|ab)'), (3, 8, 3, 8))
        ints.allow_4index = False
        self.assertAlmostEqual(abs(sfxtda.j_contract(ints, t) - np.einsum('iajb,mjb->mia', iajb, t)).max(), 0, 12)
        self.assertAlmostEqual(abs(sfxtda.k_contract(ints, t[0]) - np.einsum('iajb,jb->ia', ijab, t[0])).max(), 0, 12)

    def test_sigma_vs_dense(self):
        for mf in (mf_hf, mf_ks, mf_pbe):
            for damping in (None, (1.0, 8.0)):
                td = sfxtda.SFXTDA(mf, damping=damping).build()
                for spin in (0, 1, 2):
                    M = _dense(td, spin, damping)
                    vind, hdiag = td.gen_vind(spin)
                    self.assertAlmostEqual(abs(vind(np.eye(M.shape[0])) - M).max(), 0, 10)
                    self.assertAlmostEqual(abs(hdiag - M.diagonal()).max(), 0, 12)

    def test_eigenvalues(self):
        for mf in (mf_hf, mf_ks):
            for damping in (None, (1.0, 8.0)):
                td = sfxtda.SFXTDA(mf, nstates=4, spins=(0, 1, 2), damping=damping)
                td.conv_tol = 1e-7
                td.kernel()
                for spin in (0, 1, 2):
                    ref = np.linalg.eigvalsh(_dense(td, spin, damping))[:4]
                    self.assertAlmostEqual(abs(td.omega[spin] - ref).max(), 0, 9)
                    self.assertTrue(all(td.converged[spin]))
                self.assertAlmostEqual(abs(td.e_tot[1] - mf.e_tot - td.omega[1]).max(), 0, 12)
                self.assertAlmostEqual(td.e[0][0], 0, 12)

    def test_nstates_exceeds_dimension(self):
        td = sfxtda.SFXTDA(mf_ks, nstates=100, spins=(2,))
        td.conv_tol = 1e-7
        td.kernel()
        ref = np.linalg.eigvalsh(_dense(td, 2))
        self.assertEqual(len(td.omega[2]), 24)
        self.assertAlmostEqual(abs(td.omega[2] - ref).max(), 0, 9)
        self.assertAlmostEqual(abs(td.e[2] - td.omega[2]).max(), 0, 14)

    def test_symmetry(self):
        h2o = gto.M(atom="O 0 0 -0.0699; H 0 0.7575 0.5184; H 0 -0.7575 0.5184",
                    basis='6-31g', spin=2, symmetry=True, verbose=0)
        mf = dft.ROKS(h2o, xc='BHANDHLYP').density_fit().run(conv_tol=1e-11)
        td = sfxtda.SFXTDA(mf, nstates=6, spins=(0,))
        td.conv_tol = 1e-7
        td.kernel()
        M = _dense(td, 0)
        self.assertAlmostEqual(abs(td.omega[0] - np.linalg.eigvalsh(M)[:6]).max(), 0, 9)
        td2 = sfxtda.SFXTDA(mf, nstates=3, spins=(0,))
        td2.conv_tol = 1e-7
        td2.wfnsym = 'B1'
        td2.kernel()
        k = td2.get_csf_irreps(0) == symm.irrep_name2id('C2v', 'B1')
        self.assertAlmostEqual(abs(td2.omega[0] - np.linalg.eigvalsh(M[np.ix_(k, k)])[:3]).max(), 0, 9)
        self.assertEqual(set(td2.irrep[0]), {'B1'})

    def test_invalid_spins(self):
        with self.assertRaisesRegex(ValueError, 'spins'):
            sfxtda.SFXTDA(mf_ks, spins=(0, 3)).kernel()

    def test_wfnsym_edge_cases(self):
        h2o = gto.M(atom="O 0 0 -0.0699; H 0 0.7575 0.5184; H 0 -0.7575 0.5184",
                    basis='sto-3g', spin=2, symmetry=True, verbose=0)
        mf = dft.ROKS(h2o, xc='BHANDHLYP').density_fit().run(conv_tol=1e-11)
        td = sfxtda.SFXTDA(mf, nstates=2, spins=(0, 2))
        td.wfnsym = 'A1'
        td.kernel()
        # no A1 quintet CSF exists in STO-3G: empty result instead of a crash
        self.assertEqual(len(td.omega[2]), 0)
        self.assertEqual(len(td.omega[0]), 2)
        # lowest singlet of one irrep is not the ground state: e falls back to omega
        for s in (0, 2):
            self.assertAlmostEqual(abs(td.e[s] - td.omega[s]).max(initial=0), 0, 14)

    def test_guide_reference_values(self):
        # implementation guide Sec. 7: exact ERIs, undamped; DF differences expected <~1e-3 eV
        mf = dft.ROKS(mol, xc='BHANDHLYP')
        mf.grids.level = 5
        mf.conv_tol = 1e-11
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -39.115068731, 7)
        td = sfxtda.SFXTDA(mf, nstates=4, spins=(0, 1, 2))
        td.conv_tol = 1e-7
        td.kernel()
        ref = {0: [0.332450, 0.962664, 2.408297, 7.724930],
               1: [-0.061888, 7.146789, 7.332429, 7.819089],
               2: [12.280731, 14.581284, 17.984860, 20.212472]}
        for s in (0, 1, 2):
            self.assertLess(abs(td.omega[s] * HARTREE2EV - ref[s]).max(), 5e-3)

if __name__ == "__main__":
    print("Full Tests for SF-XTDA")
    unittest.main()
