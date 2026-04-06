import unittest
import numpy
from pyscf import gto, scf
from pyscf.rebws import UREBWS2
from pyscf.rebws.urebws2 import (
    _ChemistsERIs, _make_eris, _mem_usage,
    _compute_H_aa, _compute_H_bb,
    compute_residual_aa, compute_residual_ab, compute_residual_bb,
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

class TestUREBWS2Skeleton(unittest.TestCase):

    def test_instantiation(self):
        urebws = UREBWS2(mf)
        self.assertIs(urebws.mol, mol)
        self.assertIs(urebws._scf, mf)

    def test_attributes_from_mf(self):
        urebws = UREBWS2(mf)
        numpy.testing.assert_array_equal(urebws.mo_coeff[0], mf.mo_coeff[0])
        numpy.testing.assert_array_equal(urebws.mo_coeff[1], mf.mo_coeff[1])
        numpy.testing.assert_array_equal(urebws.mo_energy[0], mf.mo_energy[0])
        numpy.testing.assert_array_equal(urebws.mo_energy[1], mf.mo_energy[1])
        numpy.testing.assert_array_equal(urebws.mo_occ[0], mf.mo_occ[0])
        numpy.testing.assert_array_equal(urebws.mo_occ[1], mf.mo_occ[1])

    def test_e_tot_without_corr(self):
        urebws = UREBWS2(mf)
        self.assertAlmostEqual(urebws.e_tot, mf.e_tot, 12)

    def test_e_tot_with_corr(self):
        urebws = UREBWS2(mf)
        urebws.e_corr = -0.1
        self.assertAlmostEqual(urebws.e_tot, mf.e_tot - 0.1, 12)

    def test_full_ao2mo_shapes(self):
        '''_full_ao2mo returns three tensors of the correct shape.'''
        urebws = UREBWS2(mf)
        mo_a, mo_b = mf.mo_coeff
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        eri_aa, eri_ab, eri_bb = urebws._full_ao2mo()
        self.assertEqual(eri_aa.shape, (nmoa, nmoa, nmoa, nmoa))
        self.assertEqual(eri_ab.shape, (nmoa, nmoa, nmob, nmob))
        self.assertEqual(eri_bb.shape, (nmob, nmob, nmob, nmob))

    def test_full_ao2mo_aa_symmetry(self):
        urebws = UREBWS2(mf)
        eri_aa, _, _ = urebws._full_ao2mo()
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(1, 0, 2, 3), atol=1e-12)
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(0, 1, 3, 2), atol=1e-12)

    def test_full_ao2mo_bb_symmetry(self):
        urebws = UREBWS2(mf)
        _, _, eri_bb = urebws._full_ao2mo()
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(1, 0, 2, 3), atol=1e-12)
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(0, 1, 3, 2), atol=1e-12)

    def test_full_ao2mo_ab_index_symmetry(self):
        '''(pq|PQ) = (qp|PQ) and (pq|PQ) = (pq|QP) from 8-fold AO symmetry.'''
        urebws = UREBWS2(mf)
        _, eri_ab, _ = urebws._full_ao2mo()
        numpy.testing.assert_allclose(eri_ab, eri_ab.transpose(1, 0, 2, 3), atol=1e-12)
        numpy.testing.assert_allclose(eri_ab, eri_ab.transpose(0, 1, 3, 2), atol=1e-12)

    def test_full_ao2mo_custom_mo_coeff(self):
        urebws = UREBWS2(mf)
        nocca, noccb = urebws.get_nocc()
        mo_occ_a = mf.mo_coeff[0][:, :nocca]
        mo_occ_b = mf.mo_coeff[1][:, :noccb]
        eri_aa, eri_ab, eri_bb = urebws._full_ao2mo(mo_coeff=(mo_occ_a, mo_occ_b))
        self.assertEqual(eri_aa.shape, (nocca, nocca, nocca, nocca))
        self.assertEqual(eri_ab.shape, (nocca, nocca, noccb, noccb))
        self.assertEqual(eri_bb.shape, (noccb, noccb, noccb, noccb))


# ===========================================================================
# Group 1 — ERI block tests
# ===========================================================================

class TestChemistsERIs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.urebws = UREBWS2(mf)
        cls.eris   = _make_eris(cls.urebws)
        cls.nocca, cls.noccb = cls.urebws.get_nocc()
        nmoa, nmob = cls.urebws.get_nmo()
        cls.nvira  = nmoa - cls.nocca
        cls.nvirb  = nmob - cls.noccb

    def test_ao2mo_returns_eris_object(self):
        eris = UREBWS2(mf).ao2mo()
        self.assertIsInstance(eris, _ChemistsERIs)

    def test_block_shapes(self):
        nocca, noccb = self.nocca, self.noccb
        nvira, nvirb = self.nvira, self.nvirb
        self.assertEqual(self.eris.ovov.shape, (nocca, nvira, nocca, nvira))
        self.assertEqual(self.eris.oovv.shape, (nocca, nocca, nvira, nvira))
        self.assertEqual(self.eris.oooo.shape, (nocca, nocca, nocca, nocca))
        self.assertEqual(self.eris.vvvv.shape, (nvira, nvira, nvira, nvira))
        self.assertEqual(self.eris.OVOV.shape, (noccb, nvirb, noccb, nvirb))
        self.assertEqual(self.eris.OOVV.shape, (noccb, noccb, nvirb, nvirb))
        self.assertEqual(self.eris.OOOO.shape, (noccb, noccb, noccb, noccb))
        self.assertEqual(self.eris.VVVV.shape, (nvirb, nvirb, nvirb, nvirb))
        self.assertEqual(self.eris.ovOV.shape, (nocca, nvira, noccb, nvirb))
        self.assertEqual(self.eris.ooVV.shape, (nocca, nocca, nvirb, nvirb))
        self.assertEqual(self.eris.OOvv.shape, (noccb, noccb, nvira, nvira))
        self.assertEqual(self.eris.ooOO.shape, (nocca, nocca, noccb, noccb))
        self.assertEqual(self.eris.vvVV.shape, (nvira, nvira, nvirb, nvirb))

    def test_ovov_symmetry(self):
        '''(ia|jb) == (jb|ia).'''
        ovov = self.eris.ovov
        numpy.testing.assert_allclose(ovov, ovov.transpose(2, 3, 0, 1), atol=1e-12)

    def test_OVOV_symmetry(self):
        '''(IA|JB) == (JB|IA).'''
        OVOV = self.eris.OVOV
        numpy.testing.assert_allclose(OVOV, OVOV.transpose(2, 3, 0, 1), atol=1e-12)

    def test_oovv_symmetry(self):
        '''(ij|ab) == (ji|ba).'''
        oovv = self.eris.oovv
        numpy.testing.assert_allclose(oovv, oovv.transpose(1, 0, 3, 2), atol=1e-12)

    def test_OOVV_symmetry(self):
        '''(IJ|AB) == (JI|BA).'''
        OOVV = self.eris.OOVV
        numpy.testing.assert_allclose(OOVV, OOVV.transpose(1, 0, 3, 2), atol=1e-12)

    def test_oooo_symmetry(self):
        '''(ij|kl) == (kl|ij) == (ji|kl).'''
        oooo = self.eris.oooo
        numpy.testing.assert_allclose(oooo, oooo.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(oooo, oooo.transpose(1, 0, 2, 3), atol=1e-12)

    def test_OOOO_symmetry(self):
        '''(IJ|KL) == (KL|IJ) == (JI|KL).'''
        OOOO = self.eris.OOOO
        numpy.testing.assert_allclose(OOOO, OOOO.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(OOOO, OOOO.transpose(1, 0, 2, 3), atol=1e-12)

    def test_vvvv_symmetry(self):
        '''(ab|cd) == (cd|ab) == (ba|cd).'''
        vvvv = self.eris.vvvv
        numpy.testing.assert_allclose(vvvv, vvvv.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(vvvv, vvvv.transpose(1, 0, 2, 3), atol=1e-12)

    def test_VVVV_symmetry(self):
        '''(AB|CD) == (CD|AB) == (BA|CD).'''
        VVVV = self.eris.VVVV
        numpy.testing.assert_allclose(VVVV, VVVV.transpose(2, 3, 0, 1), atol=1e-12)
        numpy.testing.assert_allclose(VVVV, VVVV.transpose(1, 0, 2, 3), atol=1e-12)

    def test_blocks_consistent_with_full_tensor(self):
        '''All 13 ERI blocks must match slices of the full MO tensors.'''
        nocca, noccb = self.nocca, self.noccb
        eri_aa, eri_ab, eri_bb = self.urebws._full_ao2mo()
        # α-α blocks
        numpy.testing.assert_allclose(
            self.eris.ovov, eri_aa[:nocca, nocca:, :nocca, nocca:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.oovv, eri_aa[:nocca, :nocca, nocca:, nocca:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.oooo, eri_aa[:nocca, :nocca, :nocca, :nocca], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.vvvv, eri_aa[nocca:, nocca:, nocca:, nocca:], atol=1e-12)
        # β-β blocks
        numpy.testing.assert_allclose(
            self.eris.OVOV, eri_bb[:noccb, noccb:, :noccb, noccb:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.OOVV, eri_bb[:noccb, :noccb, noccb:, noccb:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.OOOO, eri_bb[:noccb, :noccb, :noccb, :noccb], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.VVVV, eri_bb[noccb:, noccb:, noccb:, noccb:], atol=1e-12)
        # mixed α-β blocks
        numpy.testing.assert_allclose(
            self.eris.ovOV, eri_ab[:nocca, nocca:, :noccb, noccb:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.ooVV, eri_ab[:nocca, :nocca, noccb:, noccb:], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.ooOO, eri_ab[:nocca, :nocca, :noccb, :noccb], atol=1e-12)
        numpy.testing.assert_allclose(
            self.eris.vvVV, eri_ab[nocca:, nocca:, noccb:, noccb:], atol=1e-12)
        # OOvv[I,J,a,b] = (IJ|ab) = (ab|IJ) = eri_ab[a,b,I,J]
        numpy.testing.assert_allclose(
            self.eris.OOvv,
            eri_ab[nocca:, nocca:, :noccb, :noccb].transpose(2, 3, 0, 1),
            atol=1e-12)

    def test_mo_energy_stored(self):
        numpy.testing.assert_array_equal(self.eris.mo_energy[0], mf.mo_energy[0])
        numpy.testing.assert_array_equal(self.eris.mo_energy[1], mf.mo_energy[1])

    def test_mem_usage_positive(self):
        nocca, noccb = self.nocca, self.noccb
        nvira, nvirb = self.nvira, self.nvirb
        total = _mem_usage(nocca, nvira, noccb, nvirb)
        self.assertGreater(total, 0)


# ===========================================================================
# Group 2 — H intermediate unit tests
# ===========================================================================

class TestHIntermediates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.urebws = UREBWS2(mf)
        cls.eris   = _make_eris(cls.urebws)
        cls.nocca, cls.noccb = cls.urebws.get_nocc()
        nmoa, nmob = cls.urebws.get_nmo()
        cls.nvira  = nmoa - cls.nocca
        cls.nvirb  = nmob - cls.noccb
        cls.eps_oa = cls.eris.mo_energy[0][:cls.nocca]
        cls.eps_va = cls.eris.mo_energy[0][cls.nocca:]
        cls.eps_ob = cls.eris.mo_energy[1][:cls.noccb]
        cls.eps_vb = cls.eris.mo_energy[1][cls.noccb:]
        cls.ovov   = numpy.asarray(cls.eris.ovov)
        cls.ovOV   = numpy.asarray(cls.eris.ovOV)
        cls.OVOV   = numpy.asarray(cls.eris.OVOV)
        _, t2 = init_amps(cls.urebws, cls.eris)
        cls.t2aa, cls.t2ab, cls.t2bb = t2

    def test_H_aa_shape(self):
        H = _compute_H_aa(self.t2aa, self.t2ab, self.ovov, self.ovOV,
                          self.eps_oa, alpha=1.0)
        self.assertEqual(H.shape, (self.nocca, self.nocca))

    def test_H_bb_shape(self):
        H = _compute_H_bb(self.t2bb, self.t2ab, self.OVOV, self.ovOV,
                          self.eps_ob, alpha=1.0)
        self.assertEqual(H.shape, (self.noccb, self.noccb))

    def test_H_aa_at_alpha_zero(self):
        '''At alpha=0 H_aa must equal diag(eps_oa) regardless of amplitudes.'''
        H = _compute_H_aa(self.t2aa, self.t2ab, self.ovov, self.ovOV,
                          self.eps_oa, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_oa), atol=1e-12)

    def test_H_bb_at_alpha_zero(self):
        '''At alpha=0 H_bb must equal diag(eps_ob) regardless of amplitudes.'''
        H = _compute_H_bb(self.t2bb, self.t2ab, self.OVOV, self.ovOV,
                          self.eps_ob, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_ob), atol=1e-12)

    def test_H_aa_at_alpha_zero_zero_amps(self):
        t2aa_zero = numpy.zeros_like(self.t2aa)
        t2ab_zero = numpy.zeros_like(self.t2ab)
        H = _compute_H_aa(t2aa_zero, t2ab_zero, self.ovov, self.ovOV,
                          self.eps_oa, alpha=0.0)
        numpy.testing.assert_allclose(H, numpy.diag(self.eps_oa), atol=1e-12)

    def test_H_aa_symmetry(self):
        '''H_aa must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_aa(self.t2aa, self.t2ab, self.ovov, self.ovOV,
                               self.eps_oa, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H_aa not symmetric at alpha={alpha}')

    def test_H_bb_symmetry(self):
        '''H_bb must be symmetric for any alpha.'''
        for alpha in (0.0, 0.5, 1.0):
            H = _compute_H_bb(self.t2bb, self.t2ab, self.OVOV, self.ovOV,
                               self.eps_ob, alpha=alpha)
            numpy.testing.assert_allclose(H, H.T, atol=1e-11,
                                          err_msg=f'H_bb not symmetric at alpha={alpha}')

    def test_H_aa_scales_linearly_with_alpha(self):
        '''H_aa - diag(eps_oa) scales linearly with alpha.'''
        H1 = _compute_H_aa(self.t2aa, self.t2ab, self.ovov, self.ovOV,
                            self.eps_oa, alpha=1.0)
        H2 = _compute_H_aa(self.t2aa, self.t2ab, self.ovov, self.ovOV,
                            self.eps_oa, alpha=2.0)
        f_oa = numpy.diag(self.eps_oa)
        numpy.testing.assert_allclose(H2 - f_oa, 2.0 * (H1 - f_oa), atol=1e-11)

    def test_H_bb_scales_linearly_with_alpha(self):
        '''H_bb - diag(eps_ob) scales linearly with alpha.'''
        H1 = _compute_H_bb(self.t2bb, self.t2ab, self.OVOV, self.ovOV,
                            self.eps_ob, alpha=1.0)
        H2 = _compute_H_bb(self.t2bb, self.t2ab, self.OVOV, self.ovOV,
                            self.eps_ob, alpha=2.0)
        f_ob = numpy.diag(self.eps_ob)
        numpy.testing.assert_allclose(H2 - f_ob, 2.0 * (H1 - f_ob), atol=1e-11)


# ===========================================================================
# Group 3 — Residual unit tests
# ===========================================================================

class TestResiduals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.urebws = UREBWS2(mf)
        cls.eris   = _make_eris(cls.urebws)
        cls.nocca, cls.noccb = cls.urebws.get_nocc()
        nmoa, nmob = cls.urebws.get_nmo()
        cls.nvira  = nmoa - cls.nocca
        cls.nvirb  = nmob - cls.noccb
        cls.eps_oa = cls.eris.mo_energy[0][:cls.nocca]
        cls.eps_va = cls.eris.mo_energy[0][cls.nocca:]
        cls.eps_ob = cls.eris.mo_energy[1][:cls.noccb]
        cls.eps_vb = cls.eris.mo_energy[1][cls.noccb:]
        _, t2 = init_amps(cls.urebws, cls.eris)
        cls.t2aa, cls.t2ab, cls.t2bb = t2

    def _make_H(self, alpha):
        ovov = numpy.asarray(self.eris.ovov)
        ovOV = numpy.asarray(self.eris.ovOV)
        OVOV = numpy.asarray(self.eris.OVOV)
        H_aa = _compute_H_aa(self.t2aa, self.t2ab, ovov, ovOV,
                              self.eps_oa, alpha)
        H_bb = _compute_H_bb(self.t2bb, self.t2ab, OVOV, ovOV,
                              self.eps_ob, alpha)
        return H_aa, H_bb

    def test_residual_shapes(self):
        H_aa, H_bb = self._make_H(1.0)
        R_aa = compute_residual_aa(self.t2aa, self.t2ab, H_aa,
                                   self.eris, self.eps_va, beta=1.0)
        R_ab = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                   H_aa, H_bb, self.eris,
                                   self.eps_va, self.eps_vb, beta=1.0)
        R_bb = compute_residual_bb(self.t2bb, self.t2ab, H_bb,
                                   self.eris, self.eps_vb, beta=1.0)
        self.assertEqual(R_aa.shape, (self.nocca, self.nvira, self.nocca, self.nvira))
        self.assertEqual(R_ab.shape, (self.nocca, self.nvira, self.noccb, self.nvirb))
        self.assertEqual(R_bb.shape, (self.noccb, self.nvirb, self.noccb, self.nvirb))

    def test_residuals_vanish_at_ump2_alpha_beta_zero(self):
        '''CORE TEST: at alpha=beta=0 with UMP2 amplitudes all residuals are zero.

        R = (eps_i+eps_j-eps_a-eps_b)*t2 - driving = 0 at t2=UMP2.
        Failure indicates a sign error in the H-contractions or driving term.
        '''
        H_aa = numpy.diag(self.eps_oa)
        H_bb = numpy.diag(self.eps_ob)
        R_aa = compute_residual_aa(self.t2aa, self.t2ab, H_aa,
                                   self.eris, self.eps_va, beta=0.0)
        R_ab = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                   H_aa, H_bb, self.eris,
                                   self.eps_va, self.eps_vb, beta=0.0)
        R_bb = compute_residual_bb(self.t2bb, self.t2ab, H_bb,
                                   self.eris, self.eps_vb, beta=0.0)
        numpy.testing.assert_allclose(R_aa, numpy.zeros_like(R_aa), atol=1e-11)
        numpy.testing.assert_allclose(R_ab, numpy.zeros_like(R_ab), atol=1e-11)
        numpy.testing.assert_allclose(R_bb, numpy.zeros_like(R_bb), atol=1e-11)

    def test_driving_term_aa_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_aa = (ib|ja) - (ia|jb) (sign-corrected).'''
        t2aa_zero = numpy.zeros_like(self.t2aa)
        t2ab_zero = numpy.zeros_like(self.t2ab)
        H_aa = numpy.diag(self.eps_oa)
        R = compute_residual_aa(t2aa_zero, t2ab_zero, H_aa,
                                self.eris, self.eps_va, beta=0.0)
        ovov = numpy.asarray(self.eris.ovov)
        expected = -(ovov - ovov.transpose(0, 3, 2, 1))
        numpy.testing.assert_allclose(R, expected, atol=1e-12)

    def test_driving_term_ab_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_ab = -ovOV.'''
        t2aa_zero = numpy.zeros_like(self.t2aa)
        t2ab_zero = numpy.zeros_like(self.t2ab)
        t2bb_zero = numpy.zeros_like(self.t2bb)
        H_aa = numpy.diag(self.eps_oa)
        H_bb = numpy.diag(self.eps_ob)
        R = compute_residual_ab(t2aa_zero, t2ab_zero, t2bb_zero,
                                H_aa, H_bb, self.eris,
                                self.eps_va, self.eps_vb, beta=0.0)
        ovOV = numpy.asarray(self.eris.ovOV)
        numpy.testing.assert_allclose(R, -ovOV, atol=1e-12)

    def test_driving_term_bb_at_zero_amps(self):
        '''At zero amplitudes with alpha=beta=0: R_bb = (IB|JA) - (IA|JB) (sign-corrected).'''
        t2bb_zero = numpy.zeros_like(self.t2bb)
        t2ab_zero = numpy.zeros_like(self.t2ab)
        H_bb = numpy.diag(self.eps_ob)
        R = compute_residual_bb(t2bb_zero, t2ab_zero, H_bb,
                                self.eris, self.eps_vb, beta=0.0)
        OVOV = numpy.asarray(self.eris.OVOV)
        expected = -(OVOV - OVOV.transpose(0, 3, 2, 1))
        numpy.testing.assert_allclose(R, expected, atol=1e-12)

    def test_beta_terms_linearity_ab(self):
        '''R_ab(beta=0.5) = (R_ab(0) + R_ab(1)) / 2 — beta enters linearly.'''
        H_aa, H_bb = self._make_H(1.0)
        R0 = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                 H_aa, H_bb, self.eris,
                                 self.eps_va, self.eps_vb, beta=0.0)
        R1 = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                 H_aa, H_bb, self.eris,
                                 self.eps_va, self.eps_vb, beta=1.0)
        R05 = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                  H_aa, H_bb, self.eris,
                                  self.eps_va, self.eps_vb, beta=0.5)
        numpy.testing.assert_allclose(R05, 0.5 * (R0 + R1), atol=1e-12)

    def test_beta_terms_nontrivial_ab(self):
        '''Beta terms must change R_ab by a non-trivial amount.'''
        H_aa, H_bb = self._make_H(1.0)
        R0 = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                 H_aa, H_bb, self.eris,
                                 self.eps_va, self.eps_vb, beta=0.0)
        R1 = compute_residual_ab(self.t2aa, self.t2ab, self.t2bb,
                                 H_aa, H_bb, self.eris,
                                 self.eps_va, self.eps_vb, beta=1.0)
        diff = R1 - R0
        self.assertGreater(numpy.max(numpy.abs(diff)), 1e-10)

    def test_t2aa_antisymmetry_preserved(self):
        '''UMP2 t2aa must be antisymmetric: t2aa[i,a,j,b] = -t2aa[j,a,i,b].'''
        numpy.testing.assert_allclose(
            self.t2aa, -self.t2aa.transpose(2, 1, 0, 3), atol=1e-12)

    def test_t2bb_antisymmetry_preserved(self):
        '''UMP2 t2bb must be antisymmetric: t2bb[I,A,J,B] = -t2bb[J,A,I,B].'''
        numpy.testing.assert_allclose(
            self.t2bb, -self.t2bb.transpose(2, 1, 0, 3), atol=1e-12)


# ===========================================================================
# Group 4 — Convergence and UMP2 limit
# ===========================================================================

class TestConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.urebws_ref = UREBWS2(mf)
        cls.eris       = _make_eris(cls.urebws_ref)
        cls.e_ump2, _  = init_amps(cls.urebws_ref, cls.eris)

    def test_ump2_limit_alpha_beta_zero(self):
        '''At alpha=beta=0, UREBWS2 must reproduce UMP2 exactly.'''
        urebws = UREBWS2(mf)
        urebws.alpha = 0.0
        urebws.beta  = 0.0
        urebws.run()
        self.assertAlmostEqual(urebws.e_corr, self.e_ump2, places=9)

    def test_ump2_limit_converges_in_one_cycle(self):
        '''At alpha=beta=0, residuals vanish at UMP2 start: convergence in 1 cycle.'''
        urebws = UREBWS2(mf)
        urebws.alpha     = 0.0
        urebws.beta      = 0.0
        urebws.max_cycle = 1
        urebws.run()
        self.assertTrue(urebws.converged)
        self.assertAlmostEqual(urebws.e_corr, self.e_ump2, places=9)

    def test_convergence_default_parameters(self):
        urebws = UREBWS2(mf)
        urebws.run()
        self.assertTrue(urebws.converged)

    def test_e_corr_negative_default(self):
        urebws = UREBWS2(mf)
        urebws.run()
        self.assertLess(urebws.e_corr, 0.0)

    def test_e_corr_negative_various_alpha_beta(self):
        for alpha, beta in [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 0.0)]:
            with self.subTest(alpha=alpha, beta=beta):
                urebws = UREBWS2(mf)
                urebws.alpha = alpha
                urebws.beta  = beta
                urebws.run()
                self.assertLess(urebws.e_corr, 0.0,
                                msg=f'e_corr not negative at alpha={alpha}, beta={beta}')

    def test_e_tot_consistent(self):
        urebws = UREBWS2(mf)
        urebws.run()
        self.assertAlmostEqual(urebws.e_tot, urebws.e_hf + urebws.e_corr, 12)

    def test_default_differs_from_ump2(self):
        '''Full UREBWS2 (alpha=beta=1) must differ from UMP2.'''
        urebws = UREBWS2(mf)
        urebws.run()
        self.assertFalse(abs(urebws.e_corr - self.e_ump2) < 1e-9,
                         'UREBWS2 (alpha=beta=1) should not equal UMP2')

    def test_uhf_method_registration(self):
        '''mf.UREBWS2() shorthand must give the same result as UREBWS2(mf).'''
        urebws_direct = UREBWS2(mf)
        urebws_direct.run()
        urebws_method = mf.UREBWS2().run()
        self.assertAlmostEqual(urebws_direct.e_corr, urebws_method.e_corr, 10)


# ===========================================================================
# Group 5 — Parameter sensitivity
# ===========================================================================

class TestParameterSensitivity(unittest.TestCase):

    def test_alpha_beta_zero_recovers_ump2(self):
        urebws_ref = UREBWS2(mf)
        eris       = _make_eris(urebws_ref)
        e_ump2, _  = init_amps(urebws_ref, eris)

        urebws = UREBWS2(mf)
        urebws.alpha = 0.0
        urebws.beta  = 0.0
        urebws.run()
        self.assertAlmostEqual(urebws.e_corr, e_ump2, places=9,
                               msg='alpha=0, beta=0 should give UMP2')

    def test_beta_affects_energy(self):
        '''beta>0 must change the energy relative to beta=0.'''
        urebws0 = UREBWS2(mf)
        urebws0.alpha = 1.0
        urebws0.beta  = 0.0
        urebws0.run()

        urebws1 = UREBWS2(mf)
        urebws1.alpha = 1.0
        urebws1.beta  = 1.0
        urebws1.run()

        self.assertFalse(abs(urebws0.e_corr - urebws1.e_corr) < 1e-9,
                         'beta should affect the energy')

    def test_alpha_affects_energy(self):
        '''alpha=0.5 must give a different energy than alpha=1.0 when beta!=0.'''
        urebws05 = UREBWS2(mf)
        urebws05.alpha = 0.5
        urebws05.beta  = 1.0
        urebws05.run()

        urebws1 = UREBWS2(mf)
        urebws1.alpha = 1.0
        urebws1.beta  = 1.0
        urebws1.run()

        self.assertFalse(abs(urebws05.e_corr - urebws1.e_corr) < 1e-9,
                         'alpha should affect the energy')

    def test_conv_tol_tighter_changes_energy_slightly(self):
        urebws_loose = UREBWS2(mf)
        urebws_loose.conv_tol = 1e-6
        urebws_loose.run()

        urebws_tight = UREBWS2(mf)
        urebws_tight.conv_tol = 1e-10
        urebws_tight.run()

        self.assertAlmostEqual(urebws_loose.e_corr, urebws_tight.e_corr, places=5)


# ===========================================================================
# Group 6 — Physical correctness
# ===========================================================================

class TestPhysics(unittest.TestCase):

    def test_size_consistency(self):
        '''E(A+B separated) == E(A) + E(B) for two H atoms 50 Å apart.

        Each H atom is a doublet (spin=1). The dimer at large separation
        has spin=2 (both ms=+1/2 aligned). The UREBWS2 energies must be
        size-consistent, i.e. E_dimer = 2 * E_monomer.
        '''
        mol_dimer = gto.Mole()
        mol_dimer.verbose = 0
        mol_dimer.output = '/dev/null'
        mol_dimer.atom = 'H 0.0 0.0 0.0; H 50.0 0.0 0.0'
        mol_dimer.basis = 'sto-3g'
        mol_dimer.spin  = 2
        mol_dimer.unit  = 'Angstrom'
        mol_dimer.build()
        mf_dimer = scf.UHF(mol_dimer).run()
        e_dimer  = UREBWS2(mf_dimer).run().e_corr

        mol_mono = gto.Mole()
        mol_mono.verbose = 0
        mol_mono.output = '/dev/null'
        mol_mono.atom = 'H 0 0 0'
        mol_mono.basis = 'sto-3g'
        mol_mono.spin  = 1
        mol_mono.unit  = 'Angstrom'
        mol_mono.build()
        mf_mono = scf.UHF(mol_mono).run()
        e_mono  = UREBWS2(mf_mono).run().e_corr

        self.assertAlmostEqual(e_dimer, 2.0 * e_mono, places=6,
                               msg='UREBWS2 is not size-consistent')

    def test_open_shell_molecule_converges(self):
        '''UREBWS2 on OH/sto-3g (doublet): must converge and give negative e_corr.'''
        mol_oh = gto.Mole()
        mol_oh.verbose = 0
        mol_oh.output  = '/dev/null'
        mol_oh.atom    = 'O 0 0 0; H 0 0 1.0'
        mol_oh.basis   = 'sto-3g'
        mol_oh.spin    = 1
        mol_oh.build()
        mf_oh  = scf.UHF(mol_oh).run()
        urebws = UREBWS2(mf_oh).run()

        self.assertTrue(urebws.converged)
        self.assertLess(urebws.e_corr, 0.0)
        self.assertFalse(numpy.isnan(urebws.e_corr))
        self.assertFalse(numpy.isinf(urebws.e_corr))

    def test_restricted_limit(self):
        '''UREBWS2 on a closed-shell molecule must give the same e_corr as REBWS2.

        For a singlet reference UHF == RHF, so the two methods must agree to
        numerical precision on every (alpha, beta) combination.  Tight SCF
        conv_tol ensures mo_coeff_alpha == mo_coeff_beta to high accuracy.
        '''
        from pyscf.rebws import REBWS2
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
                r = REBWS2(mf_rhf)
                r.alpha, r.beta = alpha, beta
                r.conv_tol = 1e-10
                r.run()

                u = UREBWS2(mf_uhf)
                u.alpha, u.beta = alpha, beta
                u.conv_tol = 1e-10
                u.run()

                self.assertAlmostEqual(u.e_corr, r.e_corr, places=7,
                                       msg=f'UREBWS2 != REBWS2 at alpha={alpha}, beta={beta}')

    def test_energy_is_real(self):
        urebws = UREBWS2(mf).run()
        self.assertIsInstance(urebws.e_corr, float)

    def test_t2_shapes_after_kernel(self):
        urebws = UREBWS2(mf).run()
        nocca, noccb = urebws.get_nocc()
        nmoa, nmob = urebws.get_nmo()
        nvira, nvirb = nmoa - nocca, nmob - noccb
        t2aa, t2ab, t2bb = urebws.t2
        self.assertEqual(t2aa.shape, (nocca, nvira, nocca, nvira))
        self.assertEqual(t2ab.shape, (nocca, nvira, noccb, nvirb))
        self.assertEqual(t2bb.shape, (noccb, nvirb, noccb, nvirb))


if __name__ == '__main__':
    unittest.main()
