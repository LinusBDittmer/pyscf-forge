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

if __name__ == "__main__":
    print("Full Tests for SF-XTDA")
    unittest.main()
