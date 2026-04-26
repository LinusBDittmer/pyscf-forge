import unittest
import numpy
from pyscf import gto, scf
from pyscf.rebws import DFUREBWS2
from pyscf.rebws.dfurebws2 import (
    _DF_UChemistsERIs, _make_df_eris,
    _compute_H_aa_df, _compute_H_bb_df,
    compute_residual_aa_df, compute_residual_ab_df, compute_residual_bb_df,
    init_amps, energy,
)
from pyscf import lib


def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.atom = '''
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
    '''
    mol.basis = 'sto-3g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


# ===========================================================================
# Group 0 — Skeleton / interface tests
# ===========================================================================

class TestDFUREBWS2Skeleton(unittest.TestCase):

    def test_instantiation(self):
        dfurebws = DFUREBWS2(mf)
        self.assertIs(dfurebws.mol, mol)
        self.assertIs(dfurebws._scf, mf)

    def test_attributes_from_mf(self):
        dfurebws = DFUREBWS2(mf)
        numpy.testing.assert_array_equal(dfurebws.mo_coeff[0], mf.mo_coeff[0])
        numpy.testing.assert_array_equal(dfurebws.mo_coeff[1], mf.mo_coeff[1])
        numpy.testing.assert_array_equal(dfurebws.mo_energy[0], mf.mo_energy[0])
        numpy.testing.assert_array_equal(dfurebws.mo_energy[1], mf.mo_energy[1])
        numpy.testing.assert_array_equal(dfurebws.mo_occ[0], mf.mo_occ[0])
        numpy.testing.assert_array_equal(dfurebws.mo_occ[1], mf.mo_occ[1])

    def test_e_tot_without_corr(self):
        dfurebws = DFUREBWS2(mf)
        self.assertAlmostEqual(dfurebws.e_tot, mf.e_tot, 12)

    def test_e_tot_with_corr(self):
        dfurebws = DFUREBWS2(mf)
        dfurebws.e_corr = -0.1
        self.assertAlmostEqual(dfurebws.e_tot, mf.e_tot - 0.1, 12)

    def test_with_df_passthrough(self):
        '''If mf has with_df, DFUREBWS2 reuses it without rebuilding.'''
        from pyscf import df as _df
        mf_df = scf.UHF(mol)
        mf_df.with_df = _df.DF(mol)
        mf_df.with_df.build()
        mf_df.run()
        dfurebws = DFUREBWS2(mf_df)
        self.assertIs(dfurebws.with_df, mf_df.with_df)

    def test_auxbasis_passthrough(self):
        dfurebws = DFUREBWS2(mf, auxbasis='weigend')
        self.assertEqual(dfurebws.auxbasis, 'weigend')

    def test_default_parameters(self):
        dfurebws = DFUREBWS2(mf)
        self.assertAlmostEqual(dfurebws.alpha, 1.0)
        self.assertAlmostEqual(dfurebws.beta, 1.0)
        self.assertEqual(dfurebws.max_cycle, 50)
        self.assertAlmostEqual(dfurebws.conv_tol, 1e-7)


# ===========================================================================
# Group 1 — B-tensor tests
# ===========================================================================

class TestBTensors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfurebws = DFUREBWS2(mf)
        cls.eris = _make_df_eris(cls.dfurebws)
        cls.nocca, cls.noccb = cls.dfurebws.get_nocc()
        nmoa, nmob = cls.dfurebws.get_nmo()
        cls.nvira = nmoa - cls.nocca
        cls.nvirb = nmob - cls.noccb
        cls.naux = cls.eris.B_ov.shape[0]

    def test_ao2mo_returns_eris_object(self):
        eris = DFUREBWS2(mf).ao2mo()
        self.assertIsInstance(eris, _DF_UChemistsERIs)

    def test_alpha_B_shapes(self):
        nocca, nvira, naux = self.nocca, self.nvira, self.naux
        self.assertEqual(self.eris.B_ov.shape, (naux, nocca, nvira))
        self.assertEqual(self.eris.B_oo.shape, (naux, nocca, nocca))
        self.assertEqual(self.eris.B_vv.shape, (naux, nvira, nvira))

    def test_beta_B_shapes(self):
        noccb, nvirb, naux = self.noccb, self.nvirb, self.naux
        self.assertEqual(self.eris.B_OV.shape, (naux, noccb, nvirb))
        self.assertEqual(self.eris.B_OO.shape, (naux, noccb, noccb))
        self.assertEqual(self.eris.B_VV.shape, (naux, nvirb, nvirb))

    def test_ovov_symmetry(self):
        '''Reconstructed alpha ovov: (ia|jb) == (jb|ia).'''
        B_ov = self.eris.B_ov
        ovov = numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)
        numpy.testing.assert_allclose(ovov, ovov.transpose(2, 3, 0, 1), atol=1e-12)

    def test_OVOV_symmetry(self):
        '''Reconstructed beta OVOV: (IA|JB) == (JB|IA).'''
        B_OV = self.eris.B_OV
        OVOV = numpy.einsum('QIA,QJB->IAJB', B_OV, B_OV)
        numpy.testing.assert_allclose(OVOV, OVOV.transpose(2, 3, 0, 1), atol=1e-12)

    def test_ovOV_symmetry(self):
        '''Reconstructed cross-spin ovOV: (ia|JB) in positions (ia,JB) sums consistently.'''
        B_ov = self.eris.B_ov
        B_OV = self.eris.B_OV
        ovOV = numpy.einsum('Qia,QJB->iaJB', B_ov, B_OV)
        # (ia|JB) == (JB|ia), i.e. ovOV[i,a,J,B] == ovOV_ba[J,B,i,a]
        ovOV_ba = numpy.einsum('QJB,Qia->JBia', B_OV, B_ov)
        numpy.testing.assert_allclose(ovOV, ovOV_ba.transpose(2, 3, 0, 1), atol=1e-12)

    def test_B_oo_symmetry(self):
        '''Reconstructed alpha oooo: (ij|kl) == (kl|ij).'''
        B_oo = self.eris.B_oo
        oooo = numpy.einsum('Qij,Qkl->ijkl', B_oo, B_oo)
        numpy.testing.assert_allclose(oooo, oooo.transpose(2, 3, 0, 1), atol=1e-12)

    def test_mo_energy_stored(self):
        numpy.testing.assert_array_equal(self.eris.mo_energy[0], mf.mo_energy[0])
        numpy.testing.assert_array_equal(self.eris.mo_energy[1], mf.mo_energy[1])


# ===========================================================================
# Group 2 — H intermediate unit tests
# ===========================================================================

class TestHIntermediates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfurebws = DFUREBWS2(mf)
        cls.eris = _make_df_eris(cls.dfurebws)
        cls.nocca, cls.noccb = cls.dfurebws.get_nocc()
        nmoa, nmob = cls.dfurebws.get_nmo()
        cls.nvira = nmoa - cls.nocca
        cls.nvirb = nmob - cls.noccb
        cls.eps_oa = cls.eris.mo_energy[0][:cls.nocca]
        cls.eps_ob = cls.eris.mo_energy[1][:cls.noccb]
        _, t2 = init_amps(cls.dfurebws, cls.eris)
        cls.t2aa, cls.t2ab, cls.t2bb = t2
        B_ov = cls.eris.B_ov
        B_OV = cls.eris.B_OV
        cls.Theta_aa = numpy.einsum('iajb,Qjb->Qia', cls.t2aa, B_ov)
        cls.Xi_aa    = numpy.einsum('icka,Qkc->Qia', cls.t2aa, B_ov)
        cls.Phi_ab   = numpy.einsum('iaJB,QJB->Qia', cls.t2ab, B_OV)
        cls.Theta_bb = numpy.einsum('IAJB,QJB->QIA', cls.t2bb, B_OV)
        cls.Xi_bb    = numpy.einsum('ICKA,QKC->QIA', cls.t2bb, B_OV)
        cls.Phi_ba   = numpy.einsum('kaIA,Qka->QIA', cls.t2ab, B_ov)

    def test_H_aa_shape(self):
        H = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                              self.eris.B_ov, self.eps_oa, alpha=1.0)
        self.assertEqual(H.shape, (self.nocca, self.nocca))

    def test_H_bb_shape(self):
        H = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                              self.eris.B_OV, self.eps_ob, alpha=1.0)
        self.assertEqual(H.shape, (self.noccb, self.noccb))

    def test_H_aa_at_alpha_zero(self):
        '''At alpha=0, H_aa == diag(eps_oa) regardless of amplitudes.'''
        H = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                              self.eris.B_ov, self.eps_oa, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_oa), atol=1e-12)

    def test_H_bb_at_alpha_zero(self):
        '''At alpha=0, H_bb == diag(eps_ob) regardless of amplitudes.'''
        H = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                              self.eris.B_OV, self.eps_ob, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_ob), atol=1e-12)

    def test_H_aa_symmetry(self):
        '''H_aa must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                                  self.eris.B_ov, self.eps_oa, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H_aa not symmetric at alpha={alpha}')

    def test_H_bb_symmetry(self):
        '''H_bb must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                                  self.eris.B_OV, self.eps_ob, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H_bb not symmetric at alpha={alpha}')

    def test_H_aa_scales_linearly_with_alpha(self):
        '''H_aa - diag(eps_oa) scales linearly with alpha.'''
        H1 = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                               self.eris.B_ov, self.eps_oa, alpha=1.0)
        H2 = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                               self.eris.B_ov, self.eps_oa, alpha=2.0)
        f_oa = numpy.diag(self.eps_oa)
        numpy.testing.assert_allclose(H2 - f_oa, 2.0 * (H1 - f_oa), atol=1e-11)

    def test_H_bb_scales_linearly_with_alpha(self):
        '''H_bb - diag(eps_ob) scales linearly with alpha.'''
        H1 = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                               self.eris.B_OV, self.eps_ob, alpha=1.0)
        H2 = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                               self.eris.B_OV, self.eps_ob, alpha=2.0)
        f_ob = numpy.diag(self.eps_ob)
        numpy.testing.assert_allclose(H2 - f_ob, 2.0 * (H1 - f_ob), atol=1e-11)


# ===========================================================================
# Group 3 — Residual unit tests
# ===========================================================================

class TestResiduals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfurebws = DFUREBWS2(mf)
        cls.eris = _make_df_eris(cls.dfurebws)
        cls.nocca, cls.noccb = cls.dfurebws.get_nocc()
        nmoa, nmob = cls.dfurebws.get_nmo()
        cls.nvira = nmoa - cls.nocca
        cls.nvirb = nmob - cls.noccb
        cls.eps_oa = cls.eris.mo_energy[0][:cls.nocca]
        cls.eps_va = cls.eris.mo_energy[0][cls.nocca:]
        cls.eps_ob = cls.eris.mo_energy[1][:cls.noccb]
        cls.eps_vb = cls.eris.mo_energy[1][cls.noccb:]
        _, t2 = init_amps(cls.dfurebws, cls.eris)
        cls.t2aa, cls.t2ab, cls.t2bb = t2
        B_ov = cls.eris.B_ov
        B_OV = cls.eris.B_OV
        cls.Theta_aa = numpy.einsum('iajb,Qjb->Qia', cls.t2aa, B_ov)
        cls.Xi_aa    = numpy.einsum('icka,Qkc->Qia', cls.t2aa, B_ov)
        cls.Phi_ab   = numpy.einsum('iaJB,QJB->Qia', cls.t2ab, B_OV)
        cls.Theta_bb = numpy.einsum('IAJB,QJB->QIA', cls.t2bb, B_OV)
        cls.Xi_bb    = numpy.einsum('ICKA,QKC->QIA', cls.t2bb, B_OV)
        cls.Phi_ba   = numpy.einsum('kaIA,Qka->QIA', cls.t2ab, B_ov)

    def _make_H(self, alpha):
        H_aa = _compute_H_aa_df(self.Xi_aa, self.Theta_aa, self.Phi_ab,
                                 self.eris.B_ov, self.eps_oa, alpha)
        H_bb = _compute_H_bb_df(self.Xi_bb, self.Theta_bb, self.Phi_ba,
                                 self.eris.B_OV, self.eps_ob, alpha)
        return H_aa, H_bb

    def test_residual_shapes(self):
        H_aa, H_bb = self._make_H(1.0)
        R_aa = compute_residual_aa_df(self.t2aa, H_aa, self.Theta_aa, self.Phi_ab,
                                      self.eris, self.eps_va, beta=1.0)
        R_ab = compute_residual_ab_df(self.t2ab, H_aa, H_bb,
                                      self.Theta_aa, self.Theta_bb,
                                      self.Phi_ab, self.Phi_ba,
                                      self.eris, self.eps_va, self.eps_vb, beta=1.0)
        R_bb = compute_residual_bb_df(self.t2bb, H_bb, self.Theta_bb, self.Phi_ba,
                                      self.eris, self.eps_vb, beta=1.0)
        self.assertEqual(R_aa.shape, (self.nocca, self.nvira, self.nocca, self.nvira))
        self.assertEqual(R_ab.shape, (self.nocca, self.nvira, self.noccb, self.nvirb))
        self.assertEqual(R_bb.shape, (self.noccb, self.nvirb, self.noccb, self.nvirb))

    def test_residuals_vanish_at_dfump2_alpha_beta_zero(self):
        '''CORE TEST: at alpha=beta=0 all residuals vanish at the DF-UMP2 amplitudes.

        Failure indicates a sign error in the H contractions or driving term.
        '''
        H_aa = numpy.diag(self.eps_oa)
        H_bb = numpy.diag(self.eps_ob)
        Theta_zero_aa = numpy.zeros_like(self.Theta_aa)
        Phi_zero_ab   = numpy.zeros_like(self.Phi_ab)
        Theta_zero_bb = numpy.zeros_like(self.Theta_bb)
        Phi_zero_ba   = numpy.zeros_like(self.Phi_ba)

        R_aa = compute_residual_aa_df(self.t2aa, H_aa, Theta_zero_aa, Phi_zero_ab,
                                      self.eris, self.eps_va, beta=0.0)
        R_ab = compute_residual_ab_df(self.t2ab, H_aa, H_bb,
                                      Theta_zero_aa, Theta_zero_bb,
                                      Phi_zero_ab, Phi_zero_ba,
                                      self.eris, self.eps_va, self.eps_vb, beta=0.0)
        R_bb = compute_residual_bb_df(self.t2bb, H_bb, Theta_zero_bb, Phi_zero_ba,
                                      self.eris, self.eps_vb, beta=0.0)
        numpy.testing.assert_allclose(R_aa, numpy.zeros_like(R_aa), atol=1e-10)
        numpy.testing.assert_allclose(R_ab, numpy.zeros_like(R_ab), atol=1e-10)
        numpy.testing.assert_allclose(R_bb, numpy.zeros_like(R_bb), atol=1e-10)

    def test_driving_term_aa_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_aa = -(ovov - ovov.T).'''
        t2aa_zero = numpy.zeros_like(self.t2aa)
        H_aa = numpy.diag(self.eps_oa)
        Theta_zero = numpy.zeros_like(self.Theta_aa)
        Phi_zero   = numpy.zeros_like(self.Phi_ab)
        R = compute_residual_aa_df(t2aa_zero, H_aa, Theta_zero, Phi_zero,
                                   self.eris, self.eps_va, beta=0.0)
        B_ov = self.eris.B_ov
        ovov = numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)
        expected = -(ovov - ovov.transpose(0, 3, 2, 1))
        numpy.testing.assert_allclose(R, expected, atol=1e-12)

    def test_driving_term_ab_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_ab = -ovOV.'''
        t2ab_zero = numpy.zeros_like(self.t2ab)
        H_aa = numpy.diag(self.eps_oa)
        H_bb = numpy.diag(self.eps_ob)
        Theta_zero_aa = numpy.zeros_like(self.Theta_aa)
        Theta_zero_bb = numpy.zeros_like(self.Theta_bb)
        Phi_zero_ab   = numpy.zeros_like(self.Phi_ab)
        Phi_zero_ba   = numpy.zeros_like(self.Phi_ba)
        R = compute_residual_ab_df(t2ab_zero, H_aa, H_bb,
                                   Theta_zero_aa, Theta_zero_bb,
                                   Phi_zero_ab, Phi_zero_ba,
                                   self.eris, self.eps_va, self.eps_vb, beta=0.0)
        ovOV = numpy.einsum('Qia,QJB->iaJB', self.eris.B_ov, self.eris.B_OV)
        numpy.testing.assert_allclose(R, -ovOV, atol=1e-12)

    def test_driving_term_bb_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_bb = -(OVOV - OVOV.T).'''
        t2bb_zero = numpy.zeros_like(self.t2bb)
        H_bb = numpy.diag(self.eps_ob)
        Theta_zero = numpy.zeros_like(self.Theta_bb)
        Phi_zero   = numpy.zeros_like(self.Phi_ba)
        R = compute_residual_bb_df(t2bb_zero, H_bb, Theta_zero, Phi_zero,
                                   self.eris, self.eps_vb, beta=0.0)
        B_OV = self.eris.B_OV
        OVOV = numpy.einsum('QIA,QJB->IAJB', B_OV, B_OV)
        expected = -(OVOV - OVOV.transpose(0, 3, 2, 1))
        numpy.testing.assert_allclose(R, expected, atol=1e-12)

    def test_beta_linearity_ab(self):
        '''R_ab(beta=0.5) == (R_ab(0) + R_ab(1)) / 2 — beta enters linearly.'''
        H_aa, H_bb = self._make_H(1.0)
        def R_at_beta(b):
            return compute_residual_ab_df(
                self.t2ab, H_aa, H_bb,
                self.Theta_aa, self.Theta_bb, self.Phi_ab, self.Phi_ba,
                self.eris, self.eps_va, self.eps_vb, beta=b)
        R0  = R_at_beta(0.0)
        R1  = R_at_beta(1.0)
        R05 = R_at_beta(0.5)
        numpy.testing.assert_allclose(R05, 0.5 * (R0 + R1), atol=1e-12)

    def test_beta_terms_nontrivial(self):
        '''Beta terms must change R_ab by a non-trivial amount.'''
        H_aa, H_bb = self._make_H(1.0)
        R0 = compute_residual_ab_df(
            self.t2ab, H_aa, H_bb,
            self.Theta_aa, self.Theta_bb, self.Phi_ab, self.Phi_ba,
            self.eris, self.eps_va, self.eps_vb, beta=0.0)
        R1 = compute_residual_ab_df(
            self.t2ab, H_aa, H_bb,
            self.Theta_aa, self.Theta_bb, self.Phi_ab, self.Phi_ba,
            self.eris, self.eps_va, self.eps_vb, beta=1.0)
        self.assertGreater(numpy.max(numpy.abs(R1 - R0)), 1e-10)

    def test_t2aa_antisymmetry(self):
        '''DF-UMP2 t2aa is antisymmetric: t2aa[i,a,j,b] = -t2aa[j,a,i,b].'''
        numpy.testing.assert_allclose(
            self.t2aa, -self.t2aa.transpose(2, 1, 0, 3), atol=1e-12)

    def test_t2bb_antisymmetry(self):
        '''DF-UMP2 t2bb is antisymmetric: t2bb[I,A,J,B] = -t2bb[J,A,I,B].'''
        numpy.testing.assert_allclose(
            self.t2bb, -self.t2bb.transpose(2, 1, 0, 3), atol=1e-12)


# ===========================================================================
# Group 4 — Convergence and DF-UMP2 limit
# ===========================================================================

class TestConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfurebws_ref = DFUREBWS2(mf)
        cls.eris = _make_df_eris(cls.dfurebws_ref)
        cls.e_dfump2, _ = init_amps(cls.dfurebws_ref, cls.eris)

    def test_dfump2_limit_alpha_beta_zero(self):
        '''At alpha=beta=0, DFUREBWS2 must reproduce DF-UMP2 exactly.'''
        dfurebws = DFUREBWS2(mf)
        dfurebws.alpha = 0.0
        dfurebws.beta  = 0.0
        dfurebws.run()
        self.assertAlmostEqual(dfurebws.e_corr, self.e_dfump2, places=9)

    def test_dfump2_limit_converges_in_one_cycle(self):
        '''At alpha=beta=0, residuals vanish at DF-UMP2 start: converges in 1 cycle.'''
        dfurebws = DFUREBWS2(mf)
        dfurebws.alpha     = 0.0
        dfurebws.beta      = 0.0
        dfurebws.max_cycle = 1
        dfurebws.run()
        self.assertTrue(dfurebws.converged)
        self.assertAlmostEqual(dfurebws.e_corr, self.e_dfump2, places=9)

    def test_convergence_default_parameters(self):
        dfurebws = DFUREBWS2(mf)
        dfurebws.run()
        self.assertTrue(dfurebws.converged)

    def test_e_corr_negative_default(self):
        dfurebws = DFUREBWS2(mf)
        dfurebws.run()
        self.assertLess(dfurebws.e_corr, 0.0)

    def test_e_corr_negative_various_params(self):
        for alpha, beta in [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 0.0)]:
            with self.subTest(alpha=alpha, beta=beta):
                dfurebws = DFUREBWS2(mf)
                dfurebws.alpha = alpha
                dfurebws.beta  = beta
                dfurebws.run()
                self.assertLess(dfurebws.e_corr, 0.0,
                                msg=f'e_corr not negative at alpha={alpha}, beta={beta}')

    def test_e_tot_consistent(self):
        dfurebws = DFUREBWS2(mf)
        dfurebws.run()
        self.assertAlmostEqual(dfurebws.e_tot, dfurebws.e_hf + dfurebws.e_corr, 12)

    def test_default_differs_from_dfump2(self):
        '''Full DFUREBWS2 (alpha=beta=1) must differ from DF-UMP2.'''
        dfurebws = DFUREBWS2(mf)
        dfurebws.run()
        self.assertFalse(abs(dfurebws.e_corr - self.e_dfump2) < 1e-9,
                         'DFUREBWS2 (alpha=beta=1) should differ from DF-UMP2')

    def test_uhf_method_registration(self):
        '''mf.DFUREBWS2() shorthand must give the same result as DFUREBWS2(mf).'''
        direct = DFUREBWS2(mf).run()
        via_mf = mf.DFUREBWS2().run()
        self.assertAlmostEqual(direct.e_corr, via_mf.e_corr, 10)


# ===========================================================================
# Group 5 — Parameter sensitivity
# ===========================================================================

class TestParameterSensitivity(unittest.TestCase):

    def test_alpha_beta_zero_recovers_dfump2(self):
        dfurebws_ref = DFUREBWS2(mf)
        eris = _make_df_eris(dfurebws_ref)
        e_dfump2, _ = init_amps(dfurebws_ref, eris)

        dfurebws = DFUREBWS2(mf)
        dfurebws.alpha = 0.0
        dfurebws.beta  = 0.0
        dfurebws.run()
        self.assertAlmostEqual(dfurebws.e_corr, e_dfump2, places=9)

    def test_beta_affects_energy(self):
        dfurebws0 = DFUREBWS2(mf)
        dfurebws0.alpha = 1.0
        dfurebws0.beta  = 0.0
        dfurebws0.run()

        dfurebws1 = DFUREBWS2(mf)
        dfurebws1.alpha = 1.0
        dfurebws1.beta  = 1.0
        dfurebws1.run()

        self.assertFalse(abs(dfurebws0.e_corr - dfurebws1.e_corr) < 1e-9,
                         'beta should affect the energy')

    def test_alpha_affects_energy(self):
        dfurebws05 = DFUREBWS2(mf)
        dfurebws05.alpha = 0.5
        dfurebws05.beta  = 1.0
        dfurebws05.run()

        dfurebws1 = DFUREBWS2(mf)
        dfurebws1.alpha = 1.0
        dfurebws1.beta  = 1.0
        dfurebws1.run()

        self.assertFalse(abs(dfurebws05.e_corr - dfurebws1.e_corr) < 1e-9,
                         'alpha should affect the energy')

    def test_conv_tol_stability(self):
        loose = DFUREBWS2(mf)
        loose.conv_tol = 1e-6
        loose.run()

        tight = DFUREBWS2(mf)
        tight.conv_tol = 1e-10
        tight.run()

        self.assertAlmostEqual(loose.e_corr, tight.e_corr, places=5)


# ===========================================================================
# Group 6 — Physical correctness
# ===========================================================================

class TestPhysics(unittest.TestCase):

    def test_size_consistency(self):
        '''E(A+B separated) == E(A) + E(B) for two H atoms 50 Å apart.'''
        mol_dimer = gto.Mole()
        mol_dimer.verbose = 0
        mol_dimer.output = '/dev/null'
        mol_dimer.atom = 'H 0.0 0.0 0.0; H 50.0 0.0 0.0'
        mol_dimer.basis = 'sto-3g'
        mol_dimer.spin  = 2
        mol_dimer.unit  = 'Angstrom'
        mol_dimer.build()
        mf_dimer = scf.UHF(mol_dimer).run()
        e_dimer = DFUREBWS2(mf_dimer).run().e_corr

        mol_mono = gto.Mole()
        mol_mono.verbose = 0
        mol_mono.output = '/dev/null'
        mol_mono.atom = 'H 0 0 0'
        mol_mono.basis = 'sto-3g'
        mol_mono.spin  = 1
        mol_mono.unit  = 'Angstrom'
        mol_mono.build()
        mf_mono = scf.UHF(mol_mono).run()
        e_mono = DFUREBWS2(mf_mono).run().e_corr

        self.assertAlmostEqual(e_dimer, 2.0 * e_mono, places=6,
                               msg='DFUREBWS2 is not size-consistent')

    def test_open_shell_converges(self):
        '''DFUREBWS2 on OH/sto-3g (doublet): must converge and give negative e_corr.'''
        mol_oh = gto.Mole()
        mol_oh.verbose = 0
        mol_oh.output  = '/dev/null'
        mol_oh.atom    = 'O 0 0 0; H 0 0 1.0'
        mol_oh.basis   = 'sto-3g'
        mol_oh.spin    = 1
        mol_oh.build()
        mf_oh = scf.UHF(mol_oh).run()
        dfurebws = DFUREBWS2(mf_oh).run()
        self.assertTrue(dfurebws.converged)
        self.assertLess(dfurebws.e_corr, 0.0)
        self.assertFalse(numpy.isnan(dfurebws.e_corr))
        self.assertFalse(numpy.isinf(dfurebws.e_corr))

    def test_restricted_limit(self):
        '''DFUREBWS2 on a closed-shell UHF must agree with DFREBWS2 on RHF.'''
        from pyscf.rebws import DFREBWS2
        mol_cs = gto.Mole()
        mol_cs.verbose = 0
        mol_cs.output  = '/dev/null'
        mol_cs.atom    = 'O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161'
        mol_cs.basis   = 'sto-3g'
        mol_cs.spin    = 0
        mol_cs.build()
        mf_rhf = scf.RHF(mol_cs)
        mf_rhf.conv_tol = 1e-12
        mf_rhf.run()
        mf_uhf = scf.UHF(mol_cs)
        mf_uhf.conv_tol = 1e-12
        mf_uhf.run()
        self.assertAlmostEqual(mf_rhf.e_tot, mf_uhf.e_tot, places=9)

        for alpha, beta in [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 0.0)]:
            with self.subTest(alpha=alpha, beta=beta):
                r = DFREBWS2(mf_rhf)
                r.alpha, r.beta = alpha, beta
                r.conv_tol = 1e-10
                r.run()

                u = DFUREBWS2(mf_uhf)
                u.alpha, u.beta = alpha, beta
                u.conv_tol = 1e-10
                u.run()

                self.assertAlmostEqual(u.e_corr, r.e_corr, places=5,
                                       msg=f'DFUREBWS2 != DFREBWS2 at alpha={alpha}, beta={beta}')

    def test_energy_is_real(self):
        dfurebws = DFUREBWS2(mf).run()
        self.assertIsInstance(dfurebws.e_corr, float)

    def test_t2_shapes_after_kernel(self):
        dfurebws = DFUREBWS2(mf).run()
        nocca, noccb = dfurebws.get_nocc()
        nmoa, nmob = dfurebws.get_nmo()
        nvira, nvirb = nmoa - nocca, nmob - noccb
        t2aa, t2ab, t2bb = dfurebws.t2
        self.assertEqual(t2aa.shape, (nocca, nvira, nocca, nvira))
        self.assertEqual(t2ab.shape, (nocca, nvira, noccb, nvirb))
        self.assertEqual(t2bb.shape, (noccb, nvirb, noccb, nvirb))


if __name__ == '__main__':
    unittest.main()
