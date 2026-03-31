import unittest
import numpy
from pyscf import gto, scf, mp
from pyscf.bws2 import RBWS2, UBWS2
from pyscf.bws2.ubws2 import _compute_W, _rotate_ovOV, energy, init_amps


def setUpModule():
    global mol, mf
    # O atom, triplet: nocca=5, noccb=3, sto-3g
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.atom = 'O 0 0 0'
    mol.basis = 'sto-3g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class TestUBWS2Instantiation(unittest.TestCase):

    def test_attributes(self):
        bws = UBWS2(mf)
        self.assertIs(bws.mol, mol)
        self.assertIs(bws._scf, mf)
        nocca, noccb = bws.get_nocc()
        self.assertEqual(nocca, numpy.count_nonzero(mf.mo_occ[0] > 0))
        self.assertEqual(noccb, numpy.count_nonzero(mf.mo_occ[1] > 0))
        self.assertEqual(bws.alpha, 1.0)

    def test_e_tot_after_kernel(self):
        bws = UBWS2(mf)
        bws.kernel()
        self.assertAlmostEqual(bws.e_tot, bws.e_hf + bws.e_corr, 12)

    def test_uhf_method_attachment(self):
        bws = mf.BWS2()
        bws.kernel()
        self.assertIsNotNone(bws.e_corr)


class TestUBWS2MP2Limit(unittest.TestCase):

    def test_mp2_limit_max_cycle_zero(self):
        '''max_cycle=0 must reproduce UMP2 energy exactly.'''
        e_ump2 = mp.UMP2(mf).kernel()[0]
        bws = UBWS2(mf)
        bws.max_cycle = 0
        e_bws, _ = bws.kernel()
        self.assertAlmostEqual(e_bws, e_ump2, 10)

    def test_alpha_zero_equals_ump2(self):
        '''alpha=0 collapses to UMP2 (W computed but zeroed out).'''
        e_ump2 = mp.UMP2(mf).kernel()[0]
        bws = UBWS2(mf)
        bws.alpha = 0.0
        e_bws, _ = bws.kernel()
        self.assertAlmostEqual(e_bws, e_ump2, 10)


class TestUBWS2WMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bws  = UBWS2(mf)
        eris = bws.ao2mo()
        nocca, noccb = bws.get_nocc()
        nmoa,  nmob  = bws.get_nmo()
        nvira, nvirb = nmoa - nocca, nmob - noccb

        ovov = numpy.asarray(eris.ovov).reshape(nocca, nvira, nocca, nvira)
        ovOV = numpy.asarray(eris.ovOV).reshape(nocca, nvira, noccb, nvirb)
        OVOV = numpy.asarray(eris.OVOV).reshape(noccb, nvirb, noccb, nvirb)

        mo_ea, mo_eb = eris.mo_energy
        from pyscf.lib import direct_sum
        Daa = direct_sum('ia,jb->iajb', mo_ea[:nocca, None] - mo_ea[None, nocca:],
                         mo_ea[:nocca, None] - mo_ea[None, nocca:])
        Dab = direct_sum('ia,JB->iaJB', mo_ea[:nocca, None] - mo_ea[None, nocca:],
                         mo_eb[:noccb, None] - mo_eb[None, noccb:])
        Dbb = direct_sum('IA,JB->IAJB', mo_eb[:noccb, None] - mo_eb[None, noccb:],
                         mo_eb[:noccb, None] - mo_eb[None, noccb:])

        cls.t2aa = ovov / Daa
        cls.t2ab = ovOV / Dab
        cls.t2bb = OVOV / Dbb
        cls.ovov = ovov
        cls.ovOV = ovOV
        cls.OVOV = OVOV
        cls.bws  = bws
        cls.e_mp2 = energy(bws, (cls.t2aa, cls.t2ab, cls.t2bb),
                           (ovov, ovOV, OVOV))

    def test_W_aa_symmetry(self):
        W_aa = _compute_W(self.t2aa, self.t2ab, self.ovov, self.ovOV)
        numpy.testing.assert_allclose(W_aa, W_aa.T, atol=1e-12)

    def test_W_bb_symmetry(self):
        W_bb = _compute_W(self.t2bb, self.t2ab.transpose(2, 3, 0, 1),
                          self.OVOV, self.ovOV.transpose(2, 3, 0, 1))
        numpy.testing.assert_allclose(W_bb, W_bb.T, atol=1e-12)

    def test_W_trace_property(self):
        '''tr(W_aa) + tr(W_bb) == E_corr (size-consistency condition at UMP2 amps).'''
        W_aa = _compute_W(self.t2aa, self.t2ab, self.ovov, self.ovOV)
        W_bb = _compute_W(self.t2bb, self.t2ab.transpose(2, 3, 0, 1),
                          self.OVOV, self.ovOV.transpose(2, 3, 0, 1))
        trace_sum = numpy.trace(W_aa) + numpy.trace(W_bb)
        self.assertAlmostEqual(trace_sum, self.e_mp2, 10)


class TestUBWS2OvOVRotation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bws  = UBWS2(mf)
        eris = bws.ao2mo()
        nocca, noccb = bws.get_nocc()
        nmoa,  nmob  = bws.get_nmo()
        nvira, nvirb = nmoa - nocca, nmob - noccb
        cls.ovOV_can = numpy.asarray(eris.ovOV).reshape(nocca, nvira, noccb, nvirb)
        cls.nocca = nocca
        cls.noccb = noccb

    def test_identity_rotation(self):
        U_a = numpy.eye(self.nocca)
        U_b = numpy.eye(self.noccb)
        ovOV_rot = _rotate_ovOV(U_a, U_b, self.ovOV_can)
        numpy.testing.assert_allclose(ovOV_rot, self.ovOV_can, atol=1e-12)

    def test_random_unitary_rotation(self):
        rng = numpy.random.default_rng(7)
        A_a, _ = numpy.linalg.qr(rng.standard_normal((self.nocca, self.nocca)))
        A_b, _ = numpy.linalg.qr(rng.standard_normal((self.noccb, self.noccb)))
        ovOV_rot     = _rotate_ovOV(A_a,   A_b,   self.ovOV_can)
        ovOV_recover = _rotate_ovOV(A_a.T, A_b.T, ovOV_rot)
        numpy.testing.assert_allclose(ovOV_recover, self.ovOV_can, atol=1e-10)


class TestUBWS2Convergence(unittest.TestCase):

    def test_converged_flag(self):
        bws = UBWS2(mf)
        bws.kernel()
        self.assertTrue(bws.converged)

    def test_energy_lower_than_ump2(self):
        '''|E_corr(UBWS2)| <= |E_corr(UMP2)| for a well-behaved system.'''
        e_ump2 = mp.UMP2(mf).kernel()[0]
        bws    = UBWS2(mf)
        bws.kernel()
        self.assertGreaterEqual(abs(e_ump2), abs(bws.e_corr))

    def test_alpha_one_equals_default(self):
        bws_def   = UBWS2(mf)
        bws_def.kernel()
        bws_alpha = UBWS2(mf)
        bws_alpha.alpha = 1.0
        bws_alpha.kernel()
        self.assertAlmostEqual(bws_alpha.e_corr, bws_def.e_corr, 12)

    def test_alpha_interpolates(self):
        '''|e_corr(alpha=0.5)| lies strictly between UMP2 and UBWS2.
        Uses O/cc-pVDZ spin=2 to ensure nvira > 0 (sto-3g fills all alpha MOs).
        '''
        mol2 = gto.M(atom='O', basis='cc-pvdz', spin=2, verbose=0,
                     output='/dev/null')
        mf2  = scf.UHF(mol2).run()

        e_ump2   = mp.UMP2(mf2).kernel()[0]
        bws_half = UBWS2(mf2)
        bws_half.alpha = 0.5
        bws_half.kernel()
        bws_full = UBWS2(mf2)
        bws_full.kernel()
        self.assertGreater(abs(e_ump2),          abs(bws_half.e_corr))
        self.assertGreater(abs(bws_half.e_corr), abs(bws_full.e_corr))


class TestUBWS2SizeConsistency(unittest.TestCase):
    '''E(H + H at 100 Å) == 2 * E(H).'''

    def test_size_consistency(self):
        mol_h = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1,
                      verbose=0, output='/dev/null')
        mf_h  = scf.UHF(mol_h).run()
        e_h   = UBWS2(mf_h).run().e_tot

        mol_pair = gto.M(atom='H 0 0 0; H 0 0 100', basis='sto-3g', spin=2,
                         verbose=0, output='/dev/null')
        mf_pair  = scf.UHF(mol_pair).run()
        e_pair   = UBWS2(mf_pair).run().e_tot

        self.assertAlmostEqual(e_pair - 2 * e_h, 0.0, 8)


class TestUBWS2RestrictedLimit(unittest.TestCase):
    '''For a closed-shell molecule, UBWS2(UHF) should agree with RBWS2(RHF).'''

    def test_restricted_limit(self):
        mol_cs = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                       basis='sto-3g', spin=0, verbose=0, output='/dev/null')
        mf_rhf = scf.RHF(mol_cs).run()
        mf_uhf = scf.UHF(mol_cs).run()

        e_rbws2 = RBWS2(mf_rhf).run().e_corr
        e_ubws2 = UBWS2(mf_uhf).run().e_corr

        self.assertAlmostEqual(e_ubws2, e_rbws2, 7)


if __name__ == '__main__':
    unittest.main()
