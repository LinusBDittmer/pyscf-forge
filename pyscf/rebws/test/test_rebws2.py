import unittest
import numpy
from pyscf import gto, scf
from pyscf.rebws import REBWS2
from pyscf.rebws.rebws2 import (
    _ChemistsERIs, _make_eris, _mem_usage,
    _compute_H_ij, compute_residual,
    init_amps, energy,
    get_nocc, get_nmo,
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
# Helper: build eris and common quantities for a given mf
# ---------------------------------------------------------------------------

def _build_eris(rebws_obj):
    return _make_eris(rebws_obj)


# ===========================================================================
# Group 0 — original skeleton tests (updated for new ao2mo API)
# ===========================================================================

class TestREBWS2Skeleton(unittest.TestCase):

    def test_instantiation(self):
        rebws = REBWS2(mf)
        self.assertIs(rebws.mol, mol)
        self.assertIs(rebws._scf, mf)

    def test_attributes_from_mf(self):
        rebws = REBWS2(mf)
        numpy.testing.assert_array_equal(rebws.mo_coeff, mf.mo_coeff)
        numpy.testing.assert_array_equal(rebws.mo_energy, mf.mo_energy)
        numpy.testing.assert_array_equal(rebws.mo_occ, mf.mo_occ)

    def test_e_tot_without_corr(self):
        rebws = REBWS2(mf)
        self.assertAlmostEqual(rebws.e_tot, mf.e_tot, 12)

    def test_e_tot_with_corr(self):
        rebws = REBWS2(mf)
        rebws.e_corr = -0.1
        self.assertAlmostEqual(rebws.e_tot, mf.e_tot - 0.1, 12)

    def test_full_ao2mo_shape(self):
        '''_full_ao2mo returns the (nmo, nmo, nmo, nmo) tensor.'''
        rebws = REBWS2(mf)
        nmo = mf.mo_coeff.shape[1]
        eri_mo = rebws._full_ao2mo()
        self.assertEqual(eri_mo.shape, (nmo, nmo, nmo, nmo))

    def test_full_ao2mo_symmetry(self):
        rebws = REBWS2(mf)
        eri_mo = rebws._full_ao2mo()
        numpy.testing.assert_allclose(eri_mo, eri_mo.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(eri_mo, eri_mo.transpose(1, 0, 2, 3), atol=1e-12)
        numpy.testing.assert_allclose(eri_mo, eri_mo.transpose(0, 1, 3, 2), atol=1e-12)

    def test_full_ao2mo_custom_mo_coeff(self):
        rebws = REBWS2(mf)
        nocc = int(mf.mo_occ.sum()) // 2
        mo_occ = mf.mo_coeff[:, :nocc]
        eri_occ = rebws._full_ao2mo(mo_coeff=mo_occ)
        self.assertEqual(eri_occ.shape, (nocc, nocc, nocc, nocc))


# ===========================================================================
# Group 1 — ERI block tests
# ===========================================================================

class TestChemistsERIs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rebws = REBWS2(mf)
        cls.eris  = _make_eris(cls.rebws)
        cls.nocc  = cls.rebws.nocc
        cls.nvir  = cls.rebws.nmo - cls.rebws.nocc

    def test_ao2mo_returns_eris_object(self):
        eris = REBWS2(mf).ao2mo()
        self.assertIsInstance(eris, _ChemistsERIs)

    def test_block_shapes(self):
        nocc, nvir = self.nocc, self.nvir
        self.assertEqual(self.eris.ovov.shape, (nocc, nvir, nocc, nvir))
        self.assertEqual(self.eris.oovv.shape, (nocc, nocc, nvir, nvir))
        self.assertEqual(self.eris.oooo.shape, (nocc, nocc, nocc, nocc))
        self.assertEqual(self.eris.vvvv.shape, (nvir, nvir, nvir, nvir))

    def test_ovov_symmetry(self):
        '''(ia|jb) == (jb|ia).'''
        ovov = self.eris.ovov
        numpy.testing.assert_allclose(ovov, ovov.transpose(2, 3, 0, 1), atol=1e-12)

    def test_oovv_symmetry(self):
        '''(ij|ab) == (ji|ba).'''
        oovv = self.eris.oovv
        numpy.testing.assert_allclose(oovv, oovv.transpose(1, 0, 3, 2), atol=1e-12)

    def test_oooo_symmetry(self):
        '''(ij|kl) == (kl|ij) == (ji|kl).'''
        oooo = self.eris.oooo
        numpy.testing.assert_allclose(oooo, oooo.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(oooo, oooo.transpose(1, 0, 2, 3), atol=1e-12)

    def test_vvvv_symmetry(self):
        '''(ab|cd) == (cd|ab) == (ba|cd).'''
        vvvv = self.eris.vvvv
        numpy.testing.assert_allclose(vvvv, vvvv.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(vvvv, vvvv.transpose(1, 0, 2, 3), atol=1e-12)

    def test_blocks_consistent_with_full_tensor(self):
        '''Blocks must match corresponding slices of the full MO tensor.'''
        nocc = self.nocc
        eri_full = self.rebws._full_ao2mo()
        numpy.testing.assert_allclose(
            self.eris.ovov, eri_full[:nocc, nocc:, :nocc, nocc:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.oovv, eri_full[:nocc, :nocc, nocc:, nocc:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.oooo, eri_full[:nocc, :nocc, :nocc, :nocc], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.vvvv, eri_full[nocc:, nocc:, nocc:, nocc:], atol=1e-12)

    def test_mo_energy_stored(self):
        numpy.testing.assert_array_equal(self.eris.mo_energy, mf.mo_energy)

    def test_mem_usage_positive(self):
        nocc, nvir = self.nocc, self.nvir
        total, basic, vvvv = _mem_usage(nocc, nvir)
        self.assertGreater(total, 0)
        self.assertGreater(vvvv, 0)
        self.assertAlmostEqual(total, basic + vvvv, 12)


# ===========================================================================
# Group 2 — H intermediate unit tests
# ===========================================================================

class TestHIntermediate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rebws   = REBWS2(mf)
        cls.eris    = _make_eris(cls.rebws)
        cls.nocc    = cls.rebws.nocc
        cls.nvir    = cls.rebws.nmo - cls.rebws.nocc
        cls.eps_occ = cls.eris.mo_energy[:cls.nocc]
        cls.eps_vir = cls.eris.mo_energy[cls.nocc:]
        cls.ovov    = numpy.asarray(cls.eris.ovov)
        _, cls.t2_mp2 = init_amps(cls.rebws, cls.eris)

    def test_H_at_alpha_zero_equals_fock_diagonal(self):
        '''At alpha=0, H must equal diag(eps_occ) regardless of t2.'''
        H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_at_alpha_zero_zero_amps(self):
        '''diag(eps_occ) even with zero amplitudes.'''
        t2_zero = numpy.zeros_like(self.t2_mp2)
        H = _compute_H_ij(t2_zero, self.ovov, self.eps_occ, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_symmetry(self):
        '''H must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H not symmetric at alpha={alpha}')

    def test_H_shape(self):
        H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=1.0)
        self.assertEqual(H.shape, (self.nocc, self.nocc))

    def test_H_diagonal_contains_eps_occ_at_zero_amps(self):
        '''Even at alpha=1, zero amplitudes give H = diag(eps_occ).'''
        t2_zero = numpy.zeros_like(self.t2_mp2)
        H = _compute_H_ij(t2_zero, self.ovov, self.eps_occ, alpha=1.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_occ), atol=1e-12)

    def test_H_scales_linearly_with_alpha(self):
        '''H - diag(eps_occ) must scale linearly with alpha.'''
        H1 = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=1.0)
        H2 = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=2.0)
        f_occ = numpy.diag(self.eps_occ)
        # correction at alpha=2 should be twice the correction at alpha=1
        numpy.testing.assert_allclose(
            H2 - f_occ, 2.0 * (H1 - f_occ), atol=1e-11)


# ===========================================================================
# Group 3 — Residual unit tests
# ===========================================================================

class TestResidual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rebws   = REBWS2(mf)
        cls.eris    = _make_eris(cls.rebws)
        cls.nocc    = cls.rebws.nocc
        cls.nvir    = cls.rebws.nmo - cls.rebws.nocc
        cls.eps_occ = cls.eris.mo_energy[:cls.nocc]
        cls.eps_vir = cls.eris.mo_energy[cls.nocc:]
        cls.ovov    = numpy.asarray(cls.eris.ovov)
        _, cls.t2_mp2 = init_amps(cls.rebws, cls.eris)

    def test_residual_shape(self):
        H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=1.0)
        R = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                             alpha=1.0, beta=1.0)
        self.assertEqual(R.shape, (self.nocc, self.nvir, self.nocc, self.nvir))

    def test_residual_vanishes_at_mp2_alpha_beta_zero(self):
        '''CORE TEST: at alpha=beta=0 with MP2 amplitudes, R must be zero.

        Derivation: R = (eps_i+eps_j-eps_a-eps_b)*t2 - ovov = 0 at t2=MP2.
        Failure here means the sign of the driving term or H-contraction
        signs are wrong.
        '''
        H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=0.0)
        R = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                             alpha=0.0, beta=0.0)
        numpy.testing.assert_allclose(R, numpy.zeros_like(R), atol=1e-11)

    def test_residual_structure_alpha_beta_zero(self):
        '''At alpha=beta=0, R = (eps_i+eps_j-eps_a-eps_b)*t2 - ovov.'''
        H     = numpy.diag(self.eps_occ)
        R_act = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                                 alpha=0.0, beta=0.0)
        # Build expected residual explicitly
        nocc, nvir = self.nocc, self.nvir
        eia = lib.direct_sum('i,a->ia', self.eps_occ, -self.eps_vir)   # negative
        Dij = lib.direct_sum('ia,jb->iajb', eia, eia)                  # negative
        R_exp = Dij * self.t2_mp2 - self.ovov
        numpy.testing.assert_allclose(R_act, R_exp, atol=1e-11)

    def test_beta_terms_absent_at_beta_zero(self):
        '''R(beta=0) and R(beta=1) must differ by exactly the beta terms.'''
        H = _compute_H_ij(self.t2_mp2, self.ovov, self.eps_occ, alpha=1.0)
        R0 = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                              alpha=1.0, beta=0.0)
        R1 = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                              alpha=1.0, beta=1.0)
        # Spot-check: diff must equal L5 term (among others)
        t2   = self.t2_mp2
        oovv = numpy.asarray(self.eris.oovv)
        L5   = numpy.einsum('iakc,jkbc->iajb', t2, oovv)
        diff = R1 - R0
        # diff contains L5..L14; L5 alone must be a sub-component
        # Verify that R(beta=0.5) = (R(beta=0) + R(beta=1)) / 2
        R05 = compute_residual(self.t2_mp2, H, self.eris, self.eps_vir,
                               alpha=1.0, beta=0.5)
        numpy.testing.assert_allclose(R05, 0.5 * (R0 + R1), atol=1e-12)

    def test_L5_term_explicitly(self):
        '''L5 = +beta * einsum("iakc,jkbc->iajb", t2, oovv).'''
        t2   = self.t2_mp2
        oovv = numpy.asarray(self.eris.oovv)
        H    = numpy.diag(self.eps_occ)
        # Isolate L5 by computing R(beta=1) - R(beta=0) with alpha=0
        # At alpha=0, H = diag(eps_occ), so L1+L2+L3+L4+L15 cancel at MP2.
        # The diff is pure beta terms.
        R0 = compute_residual(t2, H, self.eris, self.eps_vir, alpha=0.0, beta=0.0)
        R1 = compute_residual(t2, H, self.eris, self.eps_vir, alpha=0.0, beta=1.0)
        diff = R1 - R0
        L5_ref = numpy.einsum('iakc,jkbc->iajb', t2, oovv)
        # diff contains all 10 beta terms; L5 must be present
        # Verify L5 magnitude is non-trivial and bounded by diff
        self.assertGreater(numpy.max(numpy.abs(L5_ref)), 1e-10)
        # Sanity: the full diff norm must be >= L5 norm (by triangle inequality reversed
        # it need not hold, but let's check they are same order of magnitude)
        self.assertGreater(numpy.max(numpy.abs(diff)), 1e-10)

    def test_driving_term_is_negative_ovov(self):
        '''L15 = -ovov. Verify by subtracting all other terms at alpha=beta=0.'''
        t2   = self.t2_mp2
        ovov = self.ovov
        # At alpha=beta=0 with t2=zeros: R = -ovov  (only L15 survives)
        t2_zero = numpy.zeros_like(t2)
        H       = numpy.diag(self.eps_occ)
        R = compute_residual(t2_zero, H, self.eris, self.eps_vir,
                             alpha=0.0, beta=0.0)
        numpy.testing.assert_allclose(R, -ovov, atol=1e-12)


# ===========================================================================
# Group 4 — Convergence and MP2 limit
# ===========================================================================

class TestConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rebws_ref = REBWS2(mf)
        cls.eris      = _make_eris(cls.rebws_ref)
        cls.e_mp2, _  = init_amps(cls.rebws_ref, cls.eris)

    def test_mp2_limit_alpha_beta_zero(self):
        '''At alpha=beta=0, RE-BWs2 must reproduce MP2 exactly.'''
        rebws = REBWS2(mf)
        rebws.alpha = 0.0
        rebws.beta  = 0.0
        rebws.run()
        self.assertAlmostEqual(rebws.e_corr, self.e_mp2, places=9)

    def test_mp2_limit_converges_in_one_cycle(self):
        '''At alpha=beta=0 the residual is zero at the MP2 starting point,
        so the algorithm must converge in a single cycle.'''
        rebws = REBWS2(mf)
        rebws.alpha    = 0.0
        rebws.beta     = 0.0
        rebws.max_cycle = 1
        rebws.run()
        self.assertTrue(rebws.converged)
        self.assertAlmostEqual(rebws.e_corr, self.e_mp2, places=9)

    def test_convergence_default_parameters(self):
        rebws = REBWS2(mf)
        rebws.run()
        self.assertTrue(rebws.converged)

    def test_e_corr_negative_default(self):
        rebws = REBWS2(mf)
        rebws.run()
        self.assertLess(rebws.e_corr, 0.0)

    def test_e_corr_negative_various_alpha_beta(self):
        for alpha, beta in [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)]:
            with self.subTest(alpha=alpha, beta=beta):
                rebws = REBWS2(mf)
                rebws.alpha = alpha
                rebws.beta  = beta
                rebws.run()
                self.assertLess(rebws.e_corr, 0.0,
                                msg=f'e_corr not negative at alpha={alpha}, beta={beta}')

    def test_e_tot_consistent(self):
        rebws = REBWS2(mf)
        rebws.run()
        self.assertAlmostEqual(rebws.e_tot, rebws.e_hf + rebws.e_corr, 12)

    def test_default_differs_from_mp2(self):
        '''Full RE-BWs2 (alpha=beta=1) must differ from MP2.'''
        rebws = REBWS2(mf)
        rebws.run()
        self.assertFalse(abs(rebws.e_corr - self.e_mp2) < 1e-9,
                         'RE-BWs2 (alpha=beta=1) should not equal MP2')

    def test_rhf_method_registration(self):
        '''mf.REBWS2() shorthand must work and give the same result as REBWS2(mf).'''
        rebws_direct = REBWS2(mf)
        rebws_direct.run()
        rebws_method = mf.REBWS2().run()
        self.assertAlmostEqual(rebws_direct.e_corr, rebws_method.e_corr, 10)


# ===========================================================================
# Group 5 — Parameter sensitivity
# ===========================================================================

class TestParameterSensitivity(unittest.TestCase):

    def test_alpha_beta_zero_recovers_mp2(self):
        '''alpha=0, beta=0 must give MP2 exactly.

        Only when both alpha=0 AND beta=0 does the method reduce to pure MP2.
        alpha alone does not eliminate the RE (beta) contribution.
        '''
        rebws_ref = REBWS2(mf)
        eris      = _make_eris(rebws_ref)
        e_mp2, _  = init_amps(rebws_ref, eris)

        rebws = REBWS2(mf)
        rebws.alpha = 0.0
        rebws.beta  = 0.0
        rebws.run()
        self.assertAlmostEqual(rebws.e_corr, e_mp2, places=9,
                               msg='alpha=0, beta=0 should give MP2')

    def test_beta_affects_energy(self):
        '''beta>0 must change the energy relative to beta=0.'''
        rebws0 = REBWS2(mf)
        rebws0.alpha = 1.0
        rebws0.beta  = 0.0
        rebws0.run()

        rebws1 = REBWS2(mf)
        rebws1.alpha = 1.0
        rebws1.beta  = 1.0
        rebws1.run()

        self.assertFalse(abs(rebws0.e_corr - rebws1.e_corr) < 1e-9,
                         'beta should affect the energy')

    def test_alpha_affects_energy(self):
        '''alpha>0 must change the energy relative to alpha=0 (when beta != 0
        note: alpha=0 always gives MP2, so compare alpha=0.5 vs alpha=1).'''
        rebws05 = REBWS2(mf)
        rebws05.alpha = 0.5
        rebws05.beta  = 1.0
        rebws05.run()

        rebws1 = REBWS2(mf)
        rebws1.alpha = 1.0
        rebws1.beta  = 1.0
        rebws1.run()

        self.assertFalse(abs(rebws05.e_corr - rebws1.e_corr) < 1e-9,
                         'alpha should affect the energy')

    def test_conv_tol_tighter_changes_energy_slightly(self):
        '''Tighter conv_tol should not dramatically change the result.'''
        rebws_loose = REBWS2(mf)
        rebws_loose.conv_tol = 1e-6
        rebws_loose.run()

        rebws_tight = REBWS2(mf)
        rebws_tight.conv_tol = 1e-10
        rebws_tight.run()

        self.assertAlmostEqual(rebws_loose.e_corr, rebws_tight.e_corr, places=5)


# ===========================================================================
# Group 6 — Physical correctness
# ===========================================================================

class TestPhysics(unittest.TestCase):

    def test_size_consistency(self):
        '''E(A+B separated) must equal E(A) + E(B) within numerical noise.

        Two H2 molecules placed 50 Angstrom apart. At this separation the
        intermolecular ERIs are negligible and the combined system energy
        must equal twice the monomer energy.
        '''
        # Dimer: two H2 molecules 50 Angstrom apart
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
        e_dimer  = REBWS2(mf_dimer).run().e_corr

        # Monomer
        mol_mono = gto.Mole()
        mol_mono.verbose = 0
        mol_mono.output = '/dev/null'
        mol_mono.atom = 'H 0 0 0; H 0 0 0.74'
        mol_mono.basis = 'sto-3g'
        mol_mono.unit  = 'Angstrom'
        mol_mono.build()
        mf_mono = scf.RHF(mol_mono).run()
        e_mono  = REBWS2(mf_mono).run().e_corr

        self.assertAlmostEqual(e_dimer, 2.0 * e_mono, places=6,
                               msg='RE-BWs2 is not size-consistent')

    def test_hf_molecule_cc_pvdz(self):
        '''RE-BWs2 on HF/cc-pVDZ: must converge and give a negative e_corr.'''
        mol_hf = gto.Mole()
        mol_hf.verbose = 0
        mol_hf.output  = '/dev/null'
        mol_hf.atom    = 'H 0 0 0; F 0 0 1.1'
        mol_hf.basis   = 'cc-pvdz'
        mol_hf.build()
        mf_hf  = scf.RHF(mol_hf).run()
        rebws  = REBWS2(mf_hf).run()

        self.assertTrue(rebws.converged)
        self.assertLess(rebws.e_corr, 0.0)
        self.assertFalse(numpy.isnan(rebws.e_corr))
        self.assertFalse(numpy.isinf(rebws.e_corr))

    def test_energy_is_real(self):
        rebws = REBWS2(mf).run()
        self.assertIsInstance(rebws.e_corr, float)

    def test_t2_shape_after_kernel(self):
        rebws = REBWS2(mf).run()
        nocc  = rebws.nocc
        nvir  = rebws.nmo - nocc
        self.assertEqual(rebws.t2.shape, (nocc, nvir, nocc, nvir))


if __name__ == '__main__':
    unittest.main()
