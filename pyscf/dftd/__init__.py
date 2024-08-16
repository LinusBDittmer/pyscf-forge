'''

Empirical dispersion models from Grimme's DFT-D suite

'''

import pyscf.dftd.d4s as d4s_main

def d4s(scf_method, **kwargs):
    return d4s_main.d4s(scf_method, **kwargs)
