import unittest
import numpy
from pyscf import gto, scf, mp
from pyscf.rebws import DFREBWS2
from pyscf.rebws.dfrebws2 import (
    _DF_ChemistsERIs, _make_df_eris,
    _compute_H_ij_df, compute_residual_df,
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
    mol.build()
    mf = scf.RHF(mol).run()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_eris(dfrebws_obj):
    return _make_df_eris(dfrebws_obj)


def _make_ov_intermediates(t2, B_ov):
    '''Compute Theta and Xi from t2 and B_ov.'''
    Theta = numpy.einsum('iajb,Qjb->Qia', t2, B_ov)
    Xi    = numpy.einsum('icka,Qkc->Qia', t2, B_ov)
    return Theta, Xi


# ===========================================================================
# Group 0 — Skeleton tests
# ===========================================================================

class TestDFREBWS2Skeleton(unittest.TestCase):

    def test_instantiation(self):
        dfrebws = DFREBWS2(mf)
        self.assertIs(dfrebws.mol, mol)
        self.assertIs(dfrebws._scf, mf)

    def test_attributes_from_mf(self):
        dfrebws = DFREBWS2(mf)
        numpy.testing.assert_array_equal(dfrebws.mo_coeff, mf.mo_coeff)
        numpy.testing.assert_array_equal(dfrebws.mo_energy, mf.mo_energy)
        numpy.testing.assert_array_equal(dfrebws.mo_occ, mf.mo_occ)

    def test_e_tot_without_corr(self):
        dfrebws = DFREBWS2(mf)
        self.assertAlmostEqual(dfrebws.e_tot, mf.e_tot, 12)

    def test_e_tot_with_corr(self):
        dfrebws = DFREBWS2(mf)
        dfrebws.e_corr = -0.1
        self.assertAlmostEqual(dfrebws.e_tot, mf.e_tot - 0.1, 12)

    def test_ao2mo_returns_eris_object(self):
        dfrebws = DFREBWS2(mf)
        eris = dfrebws.ao2mo()
        self.assertIsInstance(eris, _DF_ChemistsERIs)

    def test_with_df_passthrough(self):
        '''If the MF has with_df, DFREBWS2 should inherit it.'''
        mf_df = scf.RHF(mol).density_fit().run()
        dfrebws = DFREBWS2(mf_df)
        self.assertIs(dfrebws.with_df, mf_df.with_df)

    def test_with_df_none_for_conventional_mf(self):
        '''Conventional RHF yields with_df=None; DFREBWS2 must build its own.'''
        dfrebws = DFREBWS2(mf)
        self.assertIsNone(dfrebws.with_df)


# ===========================================================================
# Group 1 — B-tensor block tests
# ===========================================================================

class TestDFChemistsERIs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfrebws = DFREBWS2(mf)
        cls.eris    = _make_df_eris(cls.dfrebws)
        cls.nocc    = cls.dfrebws.nocc
        cls.nvir    = cls.dfrebws.nmo - cls.dfrebws.nocc

    def test_b_tensor_shapes(self):
        nocc, nvir = self.nocc, self.nvir
        naux = self.eris.B_ov.shape[0]
        self.assertEqual(self.eris.B_ov.shape, (naux, nocc, nvir))
        self.assertEqual(self.eris.B_oo.shape, (naux, nocc, nocc))
        self.assertEqual(self.eris.B_vv.shape, (naux, nvir, nvir))

    def test_B_oo_symmetry(self):
        '''B_oo[Q,i,j] must be symmetric in i,j for the symmetric Coulomb metric.'''
        B_oo = self.eris.B_oo
        numpy.testing.assert_allclose(B_oo, B_oo.transpose(0, 2, 1), atol=1e-10)

    def test_B_vv_symmetry(self):
        '''B_vv[Q,a,b] must be symmetric in a,b.'''
        B_vv = self.eris.B_vv
        numpy.testing.assert_allclose(B_vv, B_vv.transpose(0, 2, 1), atol=1e-10)

    def test_ovov_positive_semidefinite(self):
        '''Reconstructed ovov diagonal (ia|ia) must be non-negative.'''
        B_ov = self.eris.B_ov
        ovov_diag = numpy.einsum('Qia,Qia->ia', B_ov, B_ov)
        self.assertTrue(numpy.all(ovov_diag >= 0.0))

    def test_ovov_symmetry_from_b_tensors(self):
        '''(ia|jb) == (jb|ia) from B_ov ⊗ B_ov.'''
        B_ov = self.eris.B_ov
        ovov = numpy.einsum('Qia,Qjb->iajb', B_ov, B_ov)
        numpy.testing.assert_allclose(ovov, ovov.transpose(2, 3, 0, 1), atol=1e-12)

    def test_mo_energy_stored(self):
        numpy.testing.assert_array_equal(self.eris.mo_energy, mf.mo_energy)


# ===========================================================================
# Group 2 — H intermediate tests
# ===========================================================================

class TestHIntermediateDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfrebws = DFREBWS2(mf)
        cls.eris    = _make_df_eris(cls.dfrebws)
        cls.nocc    = cls.dfrebws.nocc
        cls.nvir    = cls.dfrebws.nmo - cls.dfrebws.nocc
        cls.eps_occ = cls.eris.mo_energy[:cls.nocc]
        cls.eps_vir = cls.eris.mo_energy[cls.nocc:]
        _, cls.t2_mp2 = init_amps(cls.dfrebws, cls.eris)
        cls.B_ov = cls.eris.B_ov
        cls.Theta, cls.Xi = _make_ov_intermediates(cls.t2_mp2, cls.B_ov)

    def test_H_at_alpha_zero_equals_fock_diagonal(self):
        '''At alpha=0, H must equal diag(eps_occ) regardless of t2.'''
        H = _compute_H_ij_df(self.Theta, self.Xi, self.B_ov, self.eps_occ, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_at_alpha_zero_zero_amps(self):
        '''diag(eps_occ) even with zero amplitudes.'''
        t2_zero = numpy.zeros_like(self.t2_mp2)
        Th0, Xi0 = _make_ov_intermediates(t2_zero, self.B_ov)
        H = _compute_H_ij_df(Th0, Xi0, self.B_ov, self.eps_occ, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_symmetry(self):
        '''H must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_ij_df(self.Theta, self.Xi, self.B_ov, self.eps_occ, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H not symmetric at alpha={alpha}')

    def test_H_shape(self):
        H = _compute_H_ij_df(self.Theta, self.Xi, self.B_ov, self.eps_occ, alpha=1.0)
        self.assertEqual(H.shape, (self.nocc, self.nocc))

    def test_H_diagonal_contains_eps_occ_at_zero_amps(self):
        '''Even at alpha=1, zero amplitudes give H = diag(eps_occ).'''
        t2_zero = numpy.zeros_like(self.t2_mp2)
        Th0, Xi0 = _make_ov_intermediates(t2_zero, self.B_ov)
        H = _compute_H_ij_df(Th0, Xi0, self.B_ov, self.eps_occ, alpha=1.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_scales_linearly_with_alpha(self):
        '''H - diag(eps_occ) must scale linearly with alpha.'''
        H1 = _compute_H_ij_df(self.Theta, self.Xi, self.B_ov, self.eps_occ, alpha=1.0)
        H2 = _compute_H_ij_df(self.Theta, self.Xi, self.B_ov, self.eps_occ, alpha=2.0)
        f_occ = numpy.diag(self.eps_occ)
        numpy.testing.assert_allclose(
            H2 - f_occ, 2.0 * (H1 - f_occ), atol=1e-11)


# ===========================================================================
# Group 3 — Residual tests
# ===========================================================================

class TestResidualDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfrebws = DFREBWS2(mf)
        cls.eris    = _make_df_eris(cls.dfrebws)
        cls.nocc    = cls.dfrebws.nocc
        cls.nvir    = cls.dfrebws.nmo - cls.dfrebws.nocc
        cls.eps_occ = cls.eris.mo_energy[:cls.nocc]
        cls.eps_vir = cls.eris.mo_energy[cls.nocc:]
        cls.B_ov    = cls.eris.B_ov
        cls.B_oo    = cls.eris.B_oo
        cls.B_vv    = cls.eris.B_vv
        _, cls.t2_mp2 = init_amps(cls.dfrebws, cls.eris)
        cls.Theta, cls.Xi = _make_ov_intermediates(cls.t2_mp2, cls.B_ov)

    def _residual(self, t2, alpha, beta):
        Theta, Xi = _make_ov_intermediates(t2, self.B_ov)
        H = _compute_H_ij_df(Theta, Xi, self.B_ov, self.eps_occ, alpha=alpha)
        return compute_residual_df(t2, H, Theta, Xi, self.B_ov, self.B_oo, self.B_vv,
                                   self.eps_vir, alpha=alpha, beta=beta)

    def test_residual_shape(self):
        R = self._residual(self.t2_mp2, alpha=1.0, beta=1.0)
        self.assertEqual(R.shape, (self.nocc, self.nvir, self.nocc, self.nvir))

    def test_residual_vanishes_at_mp2_alpha_beta_zero(self):
        '''CORE TEST: at alpha=beta=0 with DF-MP2 amplitudes, R must be zero.

        Derivation: R = (eps_i+eps_j-eps_a-eps_b)*t2 - ovov = 0 at t2=DF-MP2.
        Uses DF-computed t2 and ovov consistently, so identity holds exactly.
        '''
        R = self._residual(self.t2_mp2, alpha=0.0, beta=0.0)
        numpy.testing.assert_allclose(R, numpy.zeros_like(R), atol=1e-11)

    def test_residual_structure_alpha_beta_zero(self):
        '''At alpha=beta=0, R = (eps_i+eps_j-eps_a-eps_b)*t2 - ovov.'''
        H_diag = numpy.diag(self.eps_occ)
        Theta0, Xi0 = _make_ov_intermediates(self.t2_mp2, self.B_ov)
        R_act = compute_residual_df(
            self.t2_mp2, H_diag, Theta0, Xi0,
            self.B_ov, self.B_oo, self.B_vv, self.eps_vir,
            alpha=0.0, beta=0.0)
        # Build expected residual: Dij * t2 - ovov, Dij negative (occ - vir)
        eia  = lib.direct_sum('i,a->ia', self.eps_occ, -self.eps_vir)   # negative
        Dij  = lib.direct_sum('ia,jb->iajb', eia, eia)                  # negative
        ovov = numpy.einsum('Qia,Qjb->iajb', self.B_ov, self.B_ov)
        R_exp = Dij * self.t2_mp2 - ovov
        numpy.testing.assert_allclose(R_act, R_exp, atol=1e-11)

    def test_driving_term_is_negative_ovov(self):
        '''L15 = -ovov. At alpha=beta=0 with zero amps, only L15 survives.'''
        t2_zero = numpy.zeros((self.nocc, self.nvir, self.nocc, self.nvir))
        Th0, Xi0 = _make_ov_intermediates(t2_zero, self.B_ov)
        H_diag = numpy.diag(self.eps_occ)
        R = compute_residual_df(
            t2_zero, H_diag, Th0, Xi0,
            self.B_ov, self.B_oo, self.B_vv, self.eps_vir,
            alpha=0.0, beta=0.0)
        ovov = numpy.einsum('Qia,Qjb->iajb', self.B_ov, self.B_ov)
        numpy.testing.assert_allclose(R, -ovov, atol=1e-12)

    def test_beta_terms_absent_at_beta_zero(self):
        '''R(beta=0) and R(beta=1) must differ by the beta terms.'''
        R0 = self._residual(self.t2_mp2, alpha=1.0, beta=0.0)
        R1 = self._residual(self.t2_mp2, alpha=1.0, beta=1.0)
        diff = R1 - R0
        self.assertGreater(numpy.max(numpy.abs(diff)), 1e-10,
                           msg='beta terms should be non-zero')

    def test_residual_scales_linearly_with_beta(self):
        '''R(0.5) = (R(0) + R(1)) / 2 if R is linear in beta.'''
        R0  = self._residual(self.t2_mp2, alpha=1.0, beta=0.0)
        R1  = self._residual(self.t2_mp2, alpha=1.0, beta=1.0)
        R05 = self._residual(self.t2_mp2, alpha=1.0, beta=0.5)
        numpy.testing.assert_allclose(R05, 0.5 * (R0 + R1), atol=1e-12)


# ===========================================================================
# Group 4 — Convergence and MP2 limit
# ===========================================================================

class TestDFConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dfrebws_ref = DFREBWS2(mf)
        cls.eris        = _make_df_eris(cls.dfrebws_ref)
        cls.e_df_mp2, _ = init_amps(cls.dfrebws_ref, cls.eris)

    def test_mp2_limit_alpha_beta_zero(self):
        '''At alpha=beta=0, DF-RE-BWs2 must reproduce DF-MP2 exactly.'''
        dfrebws = DFREBWS2(mf)
        dfrebws.alpha = 0.0
        dfrebws.beta  = 0.0
        dfrebws.run()
        self.assertAlmostEqual(dfrebws.e_corr, self.e_df_mp2, places=9)

    def test_mp2_limit_converges_in_one_cycle(self):
        '''At alpha=beta=0 the residual is zero at the DF-MP2 starting point.'''
        dfrebws = DFREBWS2(mf)
        dfrebws.alpha    = 0.0
        dfrebws.beta     = 0.0
        dfrebws.max_cycle = 1
        dfrebws.run()
        self.assertTrue(dfrebws.converged)
        self.assertAlmostEqual(dfrebws.e_corr, self.e_df_mp2, places=9)

    def test_mp2_limit_matches_pyscf_dfmp2(self):
        '''At alpha=beta=0 with a DF reference, DF-RE-BWs2 must match pyscf MP2.'''
        mf_df = scf.RHF(mol).density_fit().run()
        mp2 = mp.MP2(mf_df).run()

        dfrebws = DFREBWS2(mf_df)
        dfrebws.alpha = 0.0
        dfrebws.beta  = 0.0
        dfrebws.run()
        self.assertAlmostEqual(dfrebws.e_corr, mp2.e_corr, places=9)

    def test_convergence_default_parameters(self):
        dfrebws = DFREBWS2(mf)
        dfrebws.run()
        self.assertTrue(dfrebws.converged)

    def test_e_corr_negative_default(self):
        dfrebws = DFREBWS2(mf)
        dfrebws.run()
        self.assertLess(dfrebws.e_corr, 0.0)

    def test_e_corr_negative_various_alpha_beta(self):
        for alpha, beta in [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)]:
            with self.subTest(alpha=alpha, beta=beta):
                dfrebws = DFREBWS2(mf)
                dfrebws.alpha = alpha
                dfrebws.beta  = beta
                dfrebws.run()
                self.assertLess(dfrebws.e_corr, 0.0,
                                msg=f'e_corr not negative at alpha={alpha}, beta={beta}')

    def test_e_tot_consistent(self):
        dfrebws = DFREBWS2(mf)
        dfrebws.run()
        self.assertAlmostEqual(dfrebws.e_tot, dfrebws.e_hf + dfrebws.e_corr, 12)

    def test_default_differs_from_df_mp2(self):
        '''Full DF-RE-BWs2 (alpha=beta=1) must differ from DF-MP2.'''
        dfrebws = DFREBWS2(mf)
        dfrebws.run()
        self.assertFalse(abs(dfrebws.e_corr - self.e_df_mp2) < 1e-9,
                         'DF-RE-BWs2 (alpha=beta=1) should not equal DF-MP2')

    def test_rhf_method_registration(self):
        '''mf.DFREBWS2() shorthand must work and give the same result as DFREBWS2(mf).'''
        dfrebws_direct = DFREBWS2(mf)
        dfrebws_direct.run()
        dfrebws_method = mf.DFREBWS2().run()
        self.assertAlmostEqual(dfrebws_direct.e_corr, dfrebws_method.e_corr, 10)


# ===========================================================================
# Group 5 — Parameter sensitivity
# ===========================================================================

class TestDFParameterSensitivity(unittest.TestCase):

    def test_alpha_beta_zero_recovers_df_mp2(self):
        '''alpha=0, beta=0 must give DF-MP2 exactly.'''
        dfrebws_ref = DFREBWS2(mf)
        eris        = _make_df_eris(dfrebws_ref)
        e_df_mp2, _ = init_amps(dfrebws_ref, eris)

        dfrebws = DFREBWS2(mf)
        dfrebws.alpha = 0.0
        dfrebws.beta  = 0.0
        dfrebws.run()
        self.assertAlmostEqual(dfrebws.e_corr, e_df_mp2, places=9,
                               msg='alpha=0, beta=0 should give DF-MP2')

    def test_beta_affects_energy(self):
        '''beta>0 must change the energy relative to beta=0.'''
        dfrebws0 = DFREBWS2(mf)
        dfrebws0.alpha = 1.0
        dfrebws0.beta  = 0.0
        dfrebws0.run()

        dfrebws1 = DFREBWS2(mf)
        dfrebws1.alpha = 1.0
        dfrebws1.beta  = 1.0
        dfrebws1.run()

        self.assertFalse(abs(dfrebws0.e_corr - dfrebws1.e_corr) < 1e-9,
                         'beta should affect the energy')

    def test_alpha_affects_energy(self):
        '''alpha>0 must change the energy relative to alpha=0 (compare 0.5 vs 1.0).'''
        dfrebws05 = DFREBWS2(mf)
        dfrebws05.alpha = 0.5
        dfrebws05.beta  = 1.0
        dfrebws05.run()

        dfrebws1 = DFREBWS2(mf)
        dfrebws1.alpha = 1.0
        dfrebws1.beta  = 1.0
        dfrebws1.run()

        self.assertFalse(abs(dfrebws05.e_corr - dfrebws1.e_corr) < 1e-9,
                         'alpha should affect the energy')

    def test_conv_tol_tighter_changes_energy_slightly(self):
        '''Tighter conv_tol should not dramatically change the result.'''
        dfrebws_loose = DFREBWS2(mf)
        dfrebws_loose.conv_tol = 1e-6
        dfrebws_loose.run()

        dfrebws_tight = DFREBWS2(mf)
        dfrebws_tight.conv_tol = 1e-10
        dfrebws_tight.run()

        self.assertAlmostEqual(dfrebws_loose.e_corr, dfrebws_tight.e_corr, places=5)


# ===========================================================================
# Group 6 — Physical correctness
# ===========================================================================

class TestDFPhysics(unittest.TestCase):

    def test_size_consistency(self):
        '''E(A+B separated) must equal E(A) + E(B) within numerical noise.

        Two H2 molecules placed 50 Angstrom apart.
        '''
        mol_dimer = gto.Mole()
        mol_dimer.verbose = 0
        mol_dimer.output = '/dev/null'
        mol_dimer.atom = '''
            H  0.0  0.0  0.00
            H  0.0  0.0  0.74
            H 50.0  0.0  0.00
            H 50.0  0.0  0.74
        '''
        mol_dimer.basis = 'sto-3g'
        mol_dimer.unit  = 'Angstrom'
        mol_dimer.build()
        mf_dimer = scf.RHF(mol_dimer).run()
        e_dimer  = DFREBWS2(mf_dimer).run().e_corr

        mol_mono = gto.Mole()
        mol_mono.verbose = 0
        mol_mono.output = '/dev/null'
        mol_mono.atom = 'H 0 0 0; H 0 0 0.74'
        mol_mono.basis = 'sto-3g'
        mol_mono.unit  = 'Angstrom'
        mol_mono.build()
        mf_mono = scf.RHF(mol_mono).run()
        e_mono  = DFREBWS2(mf_mono).run().e_corr

        self.assertAlmostEqual(e_dimer, 2.0 * e_mono, places=6,
                               msg='DF-RE-BWs2 is not size-consistent')

    def test_hf_molecule_cc_pvdz(self):
        '''DF-RE-BWs2 on HF/cc-pVDZ: must converge and give a negative e_corr.'''
        mol_hf = gto.Mole()
        mol_hf.verbose = 0
        mol_hf.output  = '/dev/null'
        mol_hf.atom    = 'H 0 0 0; F 0 0 1.1'
        mol_hf.basis   = 'cc-pvdz'
        mol_hf.build()
        mf_hf   = scf.RHF(mol_hf).run()
        dfrebws = DFREBWS2(mf_hf).run()

        self.assertTrue(dfrebws.converged)
        self.assertLess(dfrebws.e_corr, 0.0)
        self.assertFalse(numpy.isnan(dfrebws.e_corr))
        self.assertFalse(numpy.isinf(dfrebws.e_corr))

    def test_energy_is_real(self):
        dfrebws = DFREBWS2(mf).run()
        self.assertIsInstance(dfrebws.e_corr, float)

    def test_t2_shape_after_kernel(self):
        dfrebws = DFREBWS2(mf).run()
        nocc    = dfrebws.nocc
        nvir    = dfrebws.nmo - nocc
        self.assertEqual(dfrebws.t2.shape, (nocc, nvir, nocc, nvir))


if __name__ == '__main__':
    unittest.main()
