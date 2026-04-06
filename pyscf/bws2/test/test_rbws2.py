import unittest
import numpy
from pyscf import gto, scf, mp
from pyscf.bws2 import RBWS2
from pyscf.bws2.rbws2 import _compute_W, _rotate_ovov, energy, init_amps


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


class TestRBWS2Instantiation(unittest.TestCase):

    def test_attributes(self):
        bws = RBWS2(mf)
        self.assertIs(bws.mol, mol)
        self.assertIs(bws._scf, mf)
        self.assertEqual(bws.nocc, numpy.count_nonzero(mf.mo_occ > 0))
        self.assertEqual(bws.nmo, len(mf.mo_occ))

    def test_e_tot_after_kernel(self):
        bws = RBWS2(mf)
        bws.kernel()
        self.assertAlmostEqual(bws.e_tot, bws.e_hf + bws.e_corr, 12)


class TestRBWS2MP2Limit(unittest.TestCase):
    '''With max_cycle=1, BW-s2 reduces exactly to MP2 (W=0 => U=I).'''

    def test_mp2_limit(self):
        # Reference: PySCF MP2
        mp2_ref = mp.MP2(mf)
        e_mp2, _ = mp2_ref.kernel()

        bws = RBWS2(mf)
        bws.max_cycle = 0
        e_bws, _ = bws.kernel()

        self.assertAlmostEqual(e_bws, e_mp2, 10)


class TestRBWS2WMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bws = RBWS2(mf)
        eris = bws.ao2mo()
        nocc = bws.nocc
        nvir = bws.nmo - nocc
        ovov = numpy.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
        eia  = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:]
        from pyscf.lib import direct_sum
        Dij  = direct_sum('ia,jb->iajb', eia, eia)
        cls.t2   = ovov / Dij
        cls.ovov = ovov
        cls.e_mp2 = energy(bws, cls.t2, cls.ovov)

    def test_W_symmetry(self):
        W = _compute_W(self.t2, self.ovov)
        numpy.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_W_trace_equals_e_corr(self):
        '''tr(W) must equal the MP2 correlation energy (size-consistency condition).'''
        W = _compute_W(self.t2, self.ovov)
        self.assertAlmostEqual(numpy.trace(W), self.e_mp2, 10)


class TestRBWS2OvovRotation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bws = RBWS2(mf)
        eris = bws.ao2mo()
        nocc = bws.nocc
        nvir = bws.nmo - nocc
        cls.ovov_can = numpy.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
        cls.nocc = nocc

    def test_identity_rotation(self):
        '''Identity U_cum must leave ovov unchanged.'''
        U = numpy.eye(self.nocc)
        ovov_rot = _rotate_ovov(U, self.ovov_can)
        numpy.testing.assert_allclose(ovov_rot, self.ovov_can, atol=1e-12)

    def test_random_unitary_rotation(self):
        '''Rotating by a random unitary then its inverse must recover canonical ovov.'''
        rng = numpy.random.default_rng(42)
        A   = rng.standard_normal((self.nocc, self.nocc))
        U, _ = numpy.linalg.qr(A)   # random unitary

        ovov_rot     = _rotate_ovov(U, self.ovov_can)
        ovov_recover = _rotate_ovov(U.T, ovov_rot)   # U^{-1} = U^T for unitary

        numpy.testing.assert_allclose(ovov_recover, self.ovov_can, atol=1e-10)


class TestRBWS2SizeConsistency(unittest.TestCase):
    '''E(A+B at large separation) == E(A) + E(B).'''

    def test_size_consistency(self):
        # Two He atoms 100 Å apart — interaction energy must be ~0.
        mol_he = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0,
                       output='/dev/null')
        mf_he  = scf.RHF(mol_he).run()
        e_he   = RBWS2(mf_he).run().e_tot

        mol_pair = gto.M(atom='He 0 0 0; He 0 0 100', basis='sto-3g',
                         verbose=0, output='/dev/null')
        mf_pair  = scf.RHF(mol_pair).run()
        e_pair   = RBWS2(mf_pair).run().e_tot

        self.assertAlmostEqual(e_pair - 2 * e_he, 0.0, 8)


class TestRBWS2Convergence(unittest.TestCase):

    def test_converged_flag(self):
        bws = RBWS2(mf)
        bws.kernel()
        self.assertTrue(bws.converged)

    def test_energy_lower_than_mp2(self):
        '''BW-s2 regularises the denominator, so |E_corr(BW-s2)| <= |E_corr(MP2)|
        for a well-behaved system.'''
        e_mp2 = mp.MP2(mf).kernel()[0]
        bws   = RBWS2(mf)
        bws.kernel()
        # BW-s2 increases the denominator gap, reducing the magnitude of E_corr.
        self.assertGreaterEqual(abs(e_mp2), abs(bws.e_corr))

    def test_mf_method_attachment(self):
        '''scf.RHF.BWS2 convenience method must work.'''
        bws = mf.BWS2()
        bws.kernel()
        self.assertIsNotNone(bws.e_corr)


class TestRBWS2Alpha(unittest.TestCase):

    def test_alpha_default(self):
        bws = RBWS2(mf)
        self.assertEqual(bws.alpha, 1.0)

    def test_alpha_zero_equals_mp2(self):
        '''alpha=0 collapses to MP2: W is built but zeroed out before diagonalisation.'''
        e_mp2 = mp.MP2(mf).kernel()[0]
        bws = RBWS2(mf)
        bws.alpha = 0.0
        e_bws, _ = bws.kernel()
        self.assertAlmostEqual(e_bws, e_mp2, 10)

    def test_alpha_one_equals_default_bws2(self):
        '''alpha=1.0 must reproduce the default BW-s2 result.'''
        bws_default = RBWS2(mf)
        bws_default.kernel()

        bws_alpha1 = RBWS2(mf)
        bws_alpha1.alpha = 1.0
        bws_alpha1.kernel()

        self.assertAlmostEqual(bws_alpha1.e_corr, bws_default.e_corr, 12)

    def test_alpha_interpolates(self):
        '''Larger alpha -> larger denominator dressing -> smaller |e_corr|.'''
        e_mp2 = mp.MP2(mf).kernel()[0]

        bws_half = RBWS2(mf)
        bws_half.alpha = 0.5
        bws_half.kernel()

        bws_full = RBWS2(mf)
        bws_full.kernel()

        # All energies are negative; alpha=0 gives MP2, alpha=1 gives BW-s2
        self.assertGreater(abs(e_mp2), abs(bws_half.e_corr))
        self.assertGreater(abs(bws_half.e_corr), abs(bws_full.e_corr))

    def test_size_consistency_nonunit_alpha(self):
        '''Size-consistency must hold for alpha != 1.'''
        mol_he = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0,
                       output='/dev/null')
        mf_he = scf.RHF(mol_he).run()
        bws_he = RBWS2(mf_he)
        bws_he.alpha = 0.5
        e_he = bws_he.run().e_tot

        mol_pair = gto.M(atom='He 0 0 0; He 0 0 100', basis='sto-3g',
                         verbose=0, output='/dev/null')
        mf_pair = scf.RHF(mol_pair).run()
        bws_pair = RBWS2(mf_pair)
        bws_pair.alpha = 0.5
        e_pair = bws_pair.run().e_tot

        self.assertAlmostEqual(e_pair - 2 * e_he, 0.0, 8)


class TestRBWS2DissociationLimit(unittest.TestCase):
    '''RBWS2(alpha=1) drives E_tot → 2 × E(H atom) at H2 dissociation.

    With symmetric (delocalized) RHF MOs the Brillouin-Wigner correlation
    energy exactly cancels the ionic contamination in E_RHF: as R → ∞,
    E_RHF → 2·E_H + K  (K = exchange integral ≈ J_AA/2)
    E_corr → gap − √(gap² + K²) → −K
    so E_RBWS2 → 2·E_H.

    Convergence is algebraic: gap ~ 1/R (Coulomb correction), giving
    |E_RBWS2 − 2·E_H| ≈ gap ≈ 5×10⁻⁴ Ha at R = 1000 Å.  mol.symmetry=True
    forces the bonding/antibonding MOs so that RHF does not break symmetry
    to the H⁻/H⁺ minimum.  The amplitude-damping default (t2_damp=0.5)
    stabilises the near-degenerate iteration.
    '''

    @classmethod
    def setUpClass(cls):
        mol_h = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1,
                      verbose=0, output='/dev/null')
        cls.e_h = scf.UHF(mol_h).run().e_tot

    def test_dissociation_limit(self):
        mol_h2 = gto.M(atom='H 0 0 0; H 0 0 100000', basis='sto-3g',
                       unit='Angstrom', symmetry=True,
                       verbose=0, output='/dev/null')
        mf_h2 = scf.RHF(mol_h2)
        mf_h2.conv_tol = 1e-12
        mf_h2.run()
        bws = RBWS2(mf_h2)
        bws.max_cycle = int(1e7)
        bws.run()
        # Analytic limit: |E_RBWS2 − 2·E_H| ≈ gap(R) ≈ 5×10⁻⁶ Ha at 1e5 Å.
        # Convergence is slow (|J| → 1 as gap → 0); ~6×10⁵ cycles needed.
        self.assertAlmostEqual(bws.e_tot, 2.0 * self.e_h, places=3)


if __name__ == '__main__':
    unittest.main()
