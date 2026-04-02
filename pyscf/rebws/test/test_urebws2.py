import unittest
import numpy
from pyscf import gto, scf
from pyscf.rebws import UREBWS2


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


class TestUREBWS2(unittest.TestCase):

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

    def test_ao2mo_shapes(self):
        urebws = UREBWS2(mf)
        mo_a, mo_b = mf.mo_coeff
        nmo_a = mo_a.shape[1]
        nmo_b = mo_b.shape[1]
        eri_aa, eri_ab, eri_bb = urebws.ao2mo()
        self.assertEqual(eri_aa.shape, (nmo_a, nmo_a, nmo_a, nmo_a))
        self.assertEqual(eri_ab.shape, (nmo_a, nmo_a, nmo_b, nmo_b))
        self.assertEqual(eri_bb.shape, (nmo_b, nmo_b, nmo_b, nmo_b))

    def test_ao2mo_aa_symmetry(self):
        urebws = UREBWS2(mf)
        eri_aa, _, _ = urebws.ao2mo()
        # (pq|rs) == (rs|pq)
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(2, 3, 0, 1),
                                      atol=1e-12)
        # (pq|rs) == (qp|rs)
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(1, 0, 2, 3),
                                      atol=1e-12)
        # (pq|rs) == (pq|sr)
        numpy.testing.assert_allclose(eri_aa, eri_aa.transpose(0, 1, 3, 2),
                                      atol=1e-12)

    def test_ao2mo_bb_symmetry(self):
        urebws = UREBWS2(mf)
        _, _, eri_bb = urebws.ao2mo()
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(2, 3, 0, 1),
                                      atol=1e-12)
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(1, 0, 2, 3),
                                      atol=1e-12)
        numpy.testing.assert_allclose(eri_bb, eri_bb.transpose(0, 1, 3, 2),
                                      atol=1e-12)

    def test_ao2mo_ab_symmetry(self):
        urebws = UREBWS2(mf)
        eri_aa, eri_ab, eri_bb = urebws.ao2mo()
        # (pq|rs)_ab == (rs|pq)_ba, i.e. transpose gives the ba block
        # which equals the ab block since the Coulomb operator is symmetric
        numpy.testing.assert_allclose(eri_ab, eri_ab.transpose(1, 0, 2, 3),
                                      atol=1e-12)
        numpy.testing.assert_allclose(eri_ab, eri_ab.transpose(0, 1, 3, 2),
                                      atol=1e-12)

    def test_ao2mo_custom_mo_coeff(self):
        urebws = UREBWS2(mf)
        nocc_a = int(mf.mo_occ[0].sum())
        nocc_b = int(mf.mo_occ[1].sum())
        mo_occ_a = mf.mo_coeff[0][:, :nocc_a]
        mo_occ_b = mf.mo_coeff[1][:, :nocc_b]
        eri_aa, eri_ab, eri_bb = urebws.ao2mo(mo_coeff=(mo_occ_a, mo_occ_b))
        self.assertEqual(eri_aa.shape, (nocc_a, nocc_a, nocc_a, nocc_a))
        self.assertEqual(eri_ab.shape, (nocc_a, nocc_a, nocc_b, nocc_b))
        self.assertEqual(eri_bb.shape, (nocc_b, nocc_b, nocc_b, nocc_b))


if __name__ == '__main__':
    unittest.main()
