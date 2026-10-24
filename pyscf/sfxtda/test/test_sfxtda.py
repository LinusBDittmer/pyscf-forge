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

import contextlib
import copy
import unittest
import numpy as np
from pyscf import gto, scf, dft, lib, symm
from pyscf.data.nist import HARTREE2EV
from pyscf.sfxtda import sfxtda
from pyscf.sfxtda._blocks import BLOCKS
from pyscf.sfxtda import _tdm
from pyscf.sfxtda import _rdm2

CH2 = "C 0 0 0; H 0.1 1.0 0.5; H -0.2 -0.95 0.35"

def setUpModule():
    global mol, mf_hf, mf_ks, mf_pbe, mf_cam, mf_lc, mf_hse, mf_neg
    mol = gto.M(atom=CH2, basis='6-31g', spin=2, verbose=0)
    mf_hf = scf.ROHF(mol).density_fit().run(conv_tol=1e-11)
    mf_ks = dft.ROKS(mol, xc='BHANDHLYP').density_fit().run(conv_tol=1e-11)
    mf_pbe = dft.ROKS(mol, xc='PBE').density_fit().run(conv_tol=1e-11)
    mf_cam = dft.ROKS(mol, xc='CAM-B3LYP').density_fit().run(conv_tol=1e-11)
    mf_lc = dft.ROKS(mol, xc='LRC-wPBE').density_fit().run(conv_tol=1e-11)        # hyb = 0
    mf_hse = dft.ROKS(mol, xc='HSE06').density_fit().run(conv_tol=1e-11)          # alpha = 0, erfc
    # alpha < hyb: negative long-range coefficient
    mf_neg = dft.ROKS(mol, xc='0.3*SR_HF(0.3)+0.1*LR_HF(0.3)+0.7*B88,LYP').density_fit().run(conv_tol=1e-11)

def tearDownModule():
    global mol, mf_hf, mf_ks, mf_pbe, mf_cam, mf_lc, mf_hse, mf_neg
    del mol, mf_hf, mf_ks, mf_pbe, mf_cam, mf_lc, mf_hse, mf_neg

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

def _full_b(td):
    '''Signed DF factors (naux, nmo, nmo) of td's kernel terms, stacked as in _DFInts.'''
    mo = td.mo_coeff
    bs, signs = [], []
    for c, o in sfxtda._kernel_terms(*td.xc_coeffs):
        with (contextlib.nullcontext(td._with_df) if o == 0 else td._with_df.range_coulomb(o)) as d:
            for eri1 in d.loop():
                b = lib.unpack_tril(eri1)
                bs.append(np.sqrt(abs(c)) * lib.einsum('Luv,up,vq->Lpq', b, mo, mo))
                signs.append(np.full(len(b), np.sign(c)))
    return np.concatenate(bs), np.concatenate(signs)

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

def _random_weights(rng, ns, nl):
    wss = rng.standard_normal((ns, ns))
    return wss + wss.T, rng.standard_normal((ns, nl)), rng.standard_normal(nl)

def _dense_w(wss, wsl, gl):
    ns, nl = wsl.shape
    w = np.zeros((ns + nl, ns + nl))
    w[:ns, :ns], w[:ns, ns:], w[ns:, :ns] = wss, wsl, wsl.T
    w[ns:, ns:] += np.diag(gl)
    return w

def _sizes(spin, nC, nV):
    small, large = sfxtda._split(spin, nC, nV)
    return sum(r * c for _, _, (r, c) in small), sum(r * c for _, _, (r, c) in large)

class KnownValues(unittest.TestCase):
    def test_reference(self):
        for mf, coeffs in ((mf_hf, (0, 1, 1)), (mf_ks, (0, .5, .5)), (mf_pbe, (0, 0, 0)),
                           (mf_cam, (.33, .65, .19)), (mf_hse, (.11, 0, .25))):
            C, core, somo, virt, fbar, xc = sfxtda._reference(mf)
            self.assertEqual((len(core), len(somo), len(virt)), (3, 2, 8))
            self.assertAlmostEqual(abs(np.subtract(xc, coeffs)).max(), 0, 12)
            v = mf.get_veff(mol, mf.make_rdm1())
            ref = .5 * C.T @ (2 * mf.get_hcore() + v[0] + v[1]) @ C
            self.assertAlmostEqual(abs(fbar - ref).max(), 0, 10)

    def test_bad_reference(self):
        with self.assertRaises(TypeError):
            sfxtda._reference(dft.UKS(mol, xc='BHANDHLYP').run())
        with self.assertRaises(ValueError):
            sfxtda._reference(scf.ROHF(gto.M(atom=CH2, basis='6-31g', verbose=0)).run())

    def test_df_tensors(self):
        td = sfxtda.SFXTDA(mf_ks).build()
        ints = td._ints
        C = td.mo_coeff
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

    def test_kernel_terms(self):
        kt = sfxtda._kernel_terms
        self.assertEqual(kt(0, .5, .5), [(.5, 0)])
        self.assertEqual(kt(0, 0, 0), [])
        self.assertEqual(kt(.11, 0, .25), [(.25, -.11)])
        self.assertEqual(kt(.3, 1., 0), [(1., .3)])
        self.assertEqual(kt(.3, .25, .25), [(.25, 0)])      # alpha = hyb: no erf term
        t = kt(.33, .65, .19)
        self.assertAlmostEqual(abs(np.subtract(t, [(.19, 0), (.46, .33)])).max(), 0, 12)
        t = kt(.3, .1, .3)
        self.assertAlmostEqual(abs(np.subtract(t, [(.3, 0), (-.2, .3)])).max(), 0, 12)

    def test_rsh_df_integrals(self):
        for mf in (mf_cam, mf_lc, mf_hse, mf_neg):
            td = sfxtda.SFXTDA(mf).build()
            self.assertEqual(mol.omega, 0)      # range_coulomb restored
            ints = td._ints
            C = td.mo_coeff
            cc, cv, co = C[:, ints.core], C[:, ints.virt], C[:, ints.somo]
            def w(*cs):
                '''[pq|rs] from the SCF's own fits via ao2mo.'''
                shape = [c.shape[1] for c in cs]
                out = 0
                for c, o in sfxtda._kernel_terms(*td.xc_coeffs):
                    if o == 0:
                        out = out + c * mf.with_df.ao2mo(cs, compact=False).reshape(shape)
                    else:
                        with mf.with_df.range_coulomb(o) as d:
                            out = out + c * d.ao2mo(cs, compact=False).reshape(shape)
                return out
            self.assertTrue(np.isin(ints.sign, (-1., 1.)).all())
            self.assertEqual(ints.sign.min() < 0, mf is mf_neg)
            ints.allow_4index = True
            # value() broadcasts to (i, a, j, b)
            self.assertAlmostEqual(abs(ints.value('(ia|jb)') - w(cc, cv, cc, cv)).max(), 0, 10)
            self.assertAlmostEqual(abs(ints.value('(ij|ab)') - w(cc, cc, cv, cv).transpose(0, 2, 1, 3)).max(), 0, 10)
            ints.allow_4index = False
            v = ints.value('(iO1|jO2)')[:, 0, :, 0]
            self.assertAlmostEqual(abs(v - w(cc, co[:, :1], cc, co[:, 1:])[:, 0, :, 0]).max(), 0, 10)
            self.assertEqual(mol.omega, 0)

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
        for mf in (mf_ks, mf_neg):
            ints = sfxtda.SFXTDA(mf).build()._ints
            nC, nV = ints.nC, ints.nV
            t = np.random.default_rng(1).random((2, nC, nV))
            ints.allow_4index = True
            iajb = np.broadcast_to(ints.value('(ia|jb)'), (nC, nV, nC, nV))
            ijab = np.broadcast_to(ints.value('(ij|ab)'), (nC, nV, nC, nV))
            ints.allow_4index = False
            self.assertAlmostEqual(abs(sfxtda.j_contract(ints, t) - np.einsum('iajb,mjb->mia', iajb, t)).max(), 0, 12)
            self.assertAlmostEqual(abs(sfxtda.k_contract(ints, t[0]) - np.einsum('iajb,jb->ia', ijab, t[0])).max(),
                                   0, 12)

    def test_rsh_sigma_vs_dense(self):
        for mf in (mf_cam, mf_lc, mf_hse, mf_neg):
            for damping in (None, (1.0, 8.0)):
                td = sfxtda.SFXTDA(mf, damping=damping).build()
                for spin in (0, 1, 2):
                    M = _dense(td, spin, damping)
                    vind, hdiag = td.gen_vind(spin)
                    self.assertAlmostEqual(abs(vind(np.eye(M.shape[0])) - M).max(), 0, 10)
                    self.assertAlmostEqual(abs(hdiag - M.diagonal()).max(), 0, 12)

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

    def test_rsh_reference_values(self):
        # independent determinant CI with exact w-ERIs, undamped; DF deviation measured <= 1.4e-4 eV
        cases = {
            'CAM-B3LYP': (-39.109992488,
                          {0: [-0.357136, 0.383902, 1.469059, 6.811046],
                           1: [-0.046383, 6.578226, 6.784253, 7.320196],
                           2: [12.811415, 15.041529, 18.240740, 20.376561]}),
            'HSE06': (-39.090381831,
                      {0: [-0.240935, 0.456509, 1.567146, 7.013247],
                       1: [-0.040819, 6.752383, 6.769513, 7.498453],
                       2: [12.884263, 15.150584, 18.115097, 20.348769]}),
        }
        for xc, (e_tot, ref) in cases.items():
            mf = dft.ROKS(mol, xc=xc)
            mf.grids.level = 5
            mf.conv_tol = 1e-11
            mf.run()
            self.assertAlmostEqual(mf.e_tot, e_tot, 7)
            td = sfxtda.SFXTDA(mf, nstates=4, spins=(0, 1, 2))
            td.conv_tol = 1e-7
            td.kernel()
            for s in (0, 1, 2):
                self.assertLess(abs(td.omega[s] * HARTREE2EV - ref[s]).max(), 1e-3)

    def test_tdm_table_vs_blocks(self):
        # one-body identity: with all integrals zero, <u|H-E0|v> = sum_pq F_pq dm(u,v)[q,p] - (u.v) tr(F gamma0)
        for spin in (0, 1, 2):
            td = sfxtda.SFXTDA(mf_hf).build()
            ints = td._ints
            core, somo, virt = td._orbs
            nmo = mf_hf.mo_coeff.shape[1]
            rng = np.random.default_rng(7 + spin)
            f = rng.standard_normal((nmo, nmo))
            ints.fbar = f + f.T
            ints.sign[:] = 0                        # every [pq|rs] -> 0
            m1 = _dense(td, spin)
            blocks = sfxtda.csf_list(spin, ints.nC, ints.nV)
            g0 = np.zeros(nmo)
            g0[core], g0[somo] = 2, 1
            for _ in range(3):
                u, v = rng.standard_normal((2, m1.shape[0]))
                dm = _tdm.transition_dm(u, v, blocks, core, somo, virt)
                ref = u @ m1 @ v + (u @ v) * (ints.fbar.diagonal() @ g0)
                self.assertAlmostEqual(np.einsum('pq,qp->', ints.fbar, dm), ref, 10)

    def test_tdm_keys(self):
        # table covers exactly the BLOCKS entries that contain an F_pq symbol
        with_f = {k for k, (args, _) in BLOCKS.items() if any(a.startswith('F_') for a in args)}
        self.assertEqual(set(_tdm.COUPLINGS), with_f)

    def test_tdm_vs_detci(self):
        # frozen determinant CI (DF integrals of mf_hf, explicit S^2 projection, <I|E_pq|J> from determinants)
        td = sfxtda.SFXTDA(mf_hf, nstates=4, spins=(0, 1, 2))
        td.conv_tol = 1e-9
        td.kernel()
        core, somo, virt = td._orbs
        ref = {  # tr(dm_0J dm_0J), |dm_0J|_F^2 for J = 0..3
            0: ([15.09034697, -1.32698052, 0.86629602, 0.0015433], [15.09034697, 1.99027892, 0.88195319, 1.68652078]),
            1: ([13.96273182, 0.005277, 0.06708164, 0.00165499], [13.96273182, 0.9724713, 0.96974053, 0.97816456]),
            2: ([11.89982431, 0.20624961, -0.23374966, 0.13427764], [11.89982431, 0.82596076, 0.97897885, 0.29174605])}
        for s in (0, 1, 2):
            blocks = sfxtda.csf_list(s, len(core), len(virt))
            dms = [_tdm.transition_dm(td.xs[s][0], td.xs[s][j], blocks, core, somo, virt) for j in range(4)]
            tr = np.array([np.trace(d @ d) for d in dms])
            fro = np.array([np.sum(d * d) for d in dms])
            for got, want in ((tr, ref[s][0]), (fro, ref[s][1])):
                if s == 1:   # roots 1, 2 are 4.9e-5 Ha apart: compare the pair sum
                    got, want = [got[0], got[1] + got[2], got[3]], [want[0], want[1] + want[2], want[3]]
                self.assertLess(abs(np.subtract(got, want)).max(), 1e-6)
            # state density: trace N, transitions traceless
            self.assertAlmostEqual(np.trace(dms[0]), mol.nelectron, 10)
            self.assertLess(np.abs([np.trace(d) for d in dms[1:]]).max(), 1e-10)
            occ = np.linalg.eigvalsh(dms[0])
            self.assertTrue(-1e-10 < occ.min() and occ.max() < 2 + 1e-10)

    def test_tdm_direction(self):
        # E_rs vs E_sr of every table entry: <X^T, dm(u, v)> for a non-symmetric X and random CSF vectors
        # (regression freeze, validated against determinant CI in the full CSF space); MO signs made canonical
        mf = mf_hf.copy()
        c = mf_hf.mo_coeff
        mf.mo_coeff = c * np.sign(c[abs(c).argmax(0), np.arange(c.shape[1])])
        td = sfxtda.SFXTDA(mf).build()
        core, somo, virt = td._orbs
        rng = np.random.default_rng(3)
        x = rng.standard_normal((c.shape[1],) * 2)
        for s, ref in zip((0, 1, 2), (-78.11584436, -25.77943742, -0.51561271)):
            blocks = sfxtda.csf_list(s, len(core), len(virt))
            u, v = rng.standard_normal((2, sum(r * q for _, _, (r, q) in blocks)))
            dm = _tdm.transition_dm(u, v, blocks, core, somo, virt)
            self.assertAlmostEqual(np.einsum('rs,sr->', x, dm), ref, 7)

    def test_oscillator_strength_vs_detci(self):
        td = sfxtda.SFXTDA(mf_hf, nstates=4, spins=(0, 1, 2))
        td.conv_tol = 1e-9
        td.kernel()
        ref = {  # frozen determinant CI: f_0J and |mu_0J|^2 (charge-centre origin), J = 0..3
            0: ([0., 0.00038228, 0.00545734, 0.02123599], [0.03146087, 0.08754974, 0.11005096]),
            1: ([0., 0.00165178, 0.00143328, 0.00022458], [0.00799955, 0.00694023, 0.00103085]),
            2: ([0., 0.03560197, 0.28409064, 0.00078773], [0.6173691, 1.6425702, 0.00345483])}
        omega = {0: [0.0535131769, 0.0717394227, 0.1470143867, 0.3429608363],
                 1: [-0.0076931411, 0.3020335937, 0.3020826064, 0.3190914264],
                 2: [0.4082409828, 0.4947418341, 0.6676734031, 0.7502526393]}
        for s in (0, 1, 2):
            self.assertLess(abs(td.omega[s] - omega[s]).max(), 1e-8)
            f = td.oscillator_strength(s)
            mu2 = (td.transition_dipole(s)[1:]**2).sum(1)
            if s == 1:   # near-degenerate roots 1, 2: pair sums
                f, mu2 = [f[0], f[1] + f[2], f[3]], [mu2[0] + mu2[1], mu2[2]]
                fr, mr = ref[s][0], ref[s][1]
                fr, mr = [fr[0], fr[1] + fr[2], fr[3]], [mr[0] + mr[1], mr[2]]
            else:
                fr, mr = ref[s]
            self.assertLess(abs(np.subtract(f, fr)).max(), 1e-7)
            self.assertLess(abs(np.subtract(mu2, mr)).max(), 1e-6)
        # f from root i: zero on the diagonal, antisymmetric energy factor
        f1 = td.oscillator_strength(0, i=1)
        self.assertEqual(f1[1], 0)
        self.assertAlmostEqual(f1[0], -td.oscillator_strength(0)[1], 10)

    def test_transition_dm_api(self):
        td = sfxtda.SFXTDA(mf_ks, nstates=3, spins=(0,))
        with self.assertRaises(RuntimeError):
            td.transition_dm(0, 0, 1)
        td.kernel()
        with self.assertRaises(ValueError):
            td.transition_dipole(spin=1)
        c = td.mo_coeff
        dm = td.transition_dm(0, 0, 1)
        self.assertLess(abs(td.transition_dm(0, 0, 1, ao_repr=True) - c @ dm @ c.T).max(), 1e-12)
        self.assertAlmostEqual(np.einsum('pq,qp->', mf_ks.get_ovlp(), td.transition_dm(0, 1, 1, ao_repr=True)),
                               mol.nelectron, 8)

    def test_root_guard(self):
        '''Non-normalised Davidson vectors (subspace collapse, S6) are flagged unconverged.'''
        xs = np.eye(3, 5)
        xs[1] = 1e-8
        conv = sfxtda._check_roots(xs, np.ones(3, dtype=bool), lib.logger.new_logger(mf_hf))
        self.assertEqual(conv.tolist(), [True, False, True])

    def test_energy_scanner(self):
        mol2 = gto.M(atom="C 0 0 0.05; H 0.1 1.02 0.5; H -0.2 -0.95 0.33", basis='6-31g', spin=2, verbose=0)
        for make in (lambda m: scf.ROHF(m).density_fit(), lambda m: scf.ROHF(m)):
            td = sfxtda.SFXTDA(make(mol).run(conv_tol=1e-11), nstates=3)
            scanner = td.as_scanner()
            e = scanner(mol2)
            ref = sfxtda.SFXTDA(make(mol2).run(conv_tol=1e-11), nstates=3)
            ref.kernel()
            for s in (0, 1):
                self.assertAlmostEqual(abs(e[s] - ref.e_tot[s]).max(), 0, 7)
            self.assertIs(scanner._with_df.mol, mol2)       # no stale DF tensors / auxmol

    def test_reset_clears(self):
        td = sfxtda.SFXTDA(scf.ROHF(mol).density_fit().run(conv_tol=1e-11), nstates=2)
        td.kernel()
        self.assertIsNotNone(td._with_df)
        td.reset()
        for attr in ('_ints', '_orbs', '_with_df', 'xs', 'omega', 'e_tot'):
            self.assertIsNone(getattr(td, attr))

    def test_rdm2_random_integrals(self):
        '''sum_L s_L <Y[B]^L, B^L> == x.(H-E0).x for random B, random signs, Fbar = 0 and random x.'''
        td = sfxtda.SFXTDA(mf_hf, spins=(0, 1, 2)).build()
        ints = td._ints
        core, somo, virt = td._orbs
        nmo = mf_hf.mo_coeff.shape[1]
        rng = np.random.default_rng(7)
        b = rng.standard_normal((9, nmo, nmo))
        b += b.transpose(0, 2, 1)
        sign = rng.choice([-1., 1.], 9)
        ints.naux, ints.sign = 9, sign
        ints.bcc, ints.bcv = b[:, core[:, None], core], b[:, core[:, None], virt]
        ints.bvv, ints.bo = b[:, virt[:, None], virt], b[:, somo]
        ints.fbar = np.zeros_like(ints.fbar)
        ints._cache = {}
        for spin in (0, 1, 2):
            vind, hdiag = td.gen_vind(spin)
            x = rng.standard_normal(hdiag.size)
            y = _rdm2.contract(x, sfxtda.csf_list(spin, len(core), len(virt)), b, core, somo, virt)
            self.assertAlmostEqual(np.einsum('L,Lpq,Lpq->', sign, y, b), x @ vind(x[None])[0], 9)

    def test_rdm2_eigvec_identity(self):
        '''omega_k == tr(P Fbar) + sum_L s_L <Y[B]^L, B^L> for converged roots (global and RSH kernels).'''
        for mf in (mf_hf, mf_cam, mf_neg):
            td = sfxtda.SFXTDA(mf, nstates=3, spins=(0, 1, 2))
            td.kernel()
            core, somo, virt = td._orbs
            b, sign = _full_b(td)
            g0 = np.zeros(mf.mo_coeff.shape[1])
            g0[core], g0[somo] = 2, 1
            for spin in (0, 1, 2):
                for k in range(len(td.omega[spin])):
                    x = td.xs[spin][k]
                    p = td.transition_dm(spin, k, k) - np.diag(g0)
                    y = _rdm2.contract(x, sfxtda.csf_list(spin, len(core), len(virt)), b, core, somo, virt)
                    fbar = sfxtda._reference(mf)[4]                       # full-MO Fbar
                    e = np.einsum('pq,pq', p, fbar) + np.einsum('L,Lpq,Lpq->', sign, y, b)
                    self.assertAlmostEqual(e, td.omega[spin][k], 8)

    def test_rdm2_pair_symmetry(self):
        '''<Y[F], G> == <F, Y[G]> and Y symmetric in pq.'''
        td = sfxtda.SFXTDA(mf_hf, nstates=2)
        td.kernel()
        core, somo, virt = td._orbs
        rng = np.random.default_rng(3)
        nmo = mf_hf.mo_coeff.shape[1]
        f, g = rng.standard_normal((2, 4, nmo, nmo))
        f += f.transpose(0, 2, 1)
        g += g.transpose(0, 2, 1)
        blocks = sfxtda.csf_list(1, len(core), len(virt))
        x = td.xs[1][0]
        yf = _rdm2.contract(x, blocks, f, core, somo, virt)
        yg = _rdm2.contract(x, blocks, g, core, somo, virt)
        self.assertAlmostEqual(abs(yf - yf.transpose(0, 2, 1)).max(), 0, 12)
        self.assertAlmostEqual(np.sum(yf * g), np.sum(f * yg), 10)

    def test_canonical_orbitals_invariance(self):
        '''Damped energies, AO state densities and oscillator strengths do not depend on rotations of the reference
        orbitals inside C, O, V; td.mo_coeff is Fbar-canonical and equals PySCF's own orbitals (D5, F2).'''
        for mf in (mf_hf, mf_ks):
            tds = []
            for m in (mf, _rotated(mf, 5)):
                td = sfxtda.SFXTDA(m, nstates=3, spins=(0, 1, 2), damping=(1., 8.))
                td.conv_tol = 1e-8
                td.kernel()
                tds.append(td)
            a, b = tds
            # no permutation or sign flip; the residual rotation follows the SCF convergence (conv_tol 1e-11 here)
            self.assertLess(abs(a.mo_coeff - mf.mo_coeff).max(), 1e-6)
            f = mf.get_fock()
            fbar = .5 * b.mo_coeff.T @ (f.focka + f.fockb) @ b.mo_coeff
            for sp in a._orbs:
                blk = fbar[np.ix_(sp, sp)]
                self.assertLess(abs(blk - np.diag(blk.diagonal())).max(), 1e-10)
            for spin in (0, 1, 2):
                self.assertAlmostEqual(abs(a.omega[spin] - b.omega[spin]).max(), 0, 9)
                self.assertLess(abs(a.transition_dm(spin, 0, 0, ao_repr=True)
                                    - b.transition_dm(spin, 0, 0, ao_repr=True)).max(), 1e-6)
                self.assertLess(abs(a.oscillator_strength(spin) - b.oscillator_strength(spin)).max(), 1e-6)

    def test_canonical_orbitals_symmetry(self):
        '''With mol.symmetry the canonicalisation keeps irreps: same irrep labels and damped energies as unrotated.'''
        m = gto.M(atom="C 0 0 0; H 0 0.95 0.55; H 0 -0.95 0.55", basis='6-31g', spin=2, symmetry=True, verbose=0)
        mf = scf.ROHF(m).density_fit().run(conv_tol=1e-11)
        tds = []
        for mm in (mf, _rotated(mf, 9, by_irrep=True)):
            td = sfxtda.SFXTDA(mm, nstates=3, spins=(0, 1), damping=(1., 8.))
            td.conv_tol = 1e-8
            td.kernel()
            tds.append(td)
        for spin in (0, 1):
            self.assertEqual(tds[0].irrep[spin], tds[1].irrep[spin])
            self.assertAlmostEqual(abs(tds[0].omega[spin] - tds[1].omega[spin]).max(), 0, 9)

    def test_weighted_dm_vs_dense(self):
        '''sum W.H == tr(P_W Fbar) for random symmetric CSF weights, random Fbar, all integrals zero.'''
        td = sfxtda.SFXTDA(mf_hf, spins=(0, 1, 2)).build()
        ints = td._ints
        core, somo, virt = td._orbs
        nmo = mf_hf.mo_coeff.shape[1]
        rng = np.random.default_rng(23)
        fbar = rng.standard_normal((nmo, nmo))
        fbar += fbar.T
        ints.naux, ints.sign = 0, np.zeros(0)
        ints.bcc, ints.bcv, ints.bvv, ints.bo = (np.zeros((0,) + b.shape[1:])
                                                 for b in (ints.bcc, ints.bcv, ints.bvv, ints.bo))
        ints.fbar, ints._cache = fbar, {}
        for spin in (0, 1, 2):
            blocks = sfxtda.csf_list(spin, len(core), len(virt))
            wss, wsl, gl = _random_weights(rng, *_sizes(spin, len(core), len(virt)))
            p = _tdm.weighted_dm(wss, wsl, gl, blocks, core, somo, virt)
            self.assertAlmostEqual(abs(p - p.T).max(), 0, 12)
            self.assertAlmostEqual(np.sum(_dense_w(wss, wsl, gl) * _dense(td, spin)), np.sum(p * fbar), 9)

    def test_contract_weighted_vs_dense(self):
        '''sum W.H == sum_L s_L <Y_W[B]^L, B^L> for random weights, random B and signs, Fbar = 0.'''
        td = sfxtda.SFXTDA(mf_hf, spins=(0, 1, 2)).build()
        ints = td._ints
        core, somo, virt = td._orbs
        nmo = mf_hf.mo_coeff.shape[1]
        rng = np.random.default_rng(29)
        b = rng.standard_normal((9, nmo, nmo))
        b += b.transpose(0, 2, 1)
        sign = rng.choice([-1., 1.], 9)
        ints.naux, ints.sign = 9, sign
        ints.bcc, ints.bcv = b[:, core[:, None], core], b[:, core[:, None], virt]
        ints.bvv, ints.bo = b[:, virt[:, None], virt], b[:, somo]
        ints.fbar = np.zeros_like(ints.fbar)
        ints._cache = {}
        for spin in (0, 1, 2):
            blocks = sfxtda.csf_list(spin, len(core), len(virt))
            wss, wsl, gl = _random_weights(rng, *_sizes(spin, len(core), len(virt)))
            y = _rdm2.contract_weighted(wss, wsl, gl, blocks, b, core, somo, virt)
            self.assertAlmostEqual(abs(y - y.transpose(0, 2, 1)).max(), 0, 12)
            self.assertAlmostEqual(np.einsum('L,Lpq,Lpq->', sign, y, b),
                                   np.sum(_dense_w(wss, wsl, gl) * _dense(td, spin)), 9)

if __name__ == "__main__":
    print("Full Tests for SF-XTDA")
    unittest.main()
