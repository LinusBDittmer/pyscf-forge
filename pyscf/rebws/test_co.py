from pyscf import scf, gto
from pyscf.rebws import REBWS2

mol = gto.M(atom='C 0 0 0; O 1.14 0 0', basis='def2-svp', verbose=4)
rhf = scf.RHF(mol)
rhf.kernel()

rebws2 = REBWS2(rhf)
rebws2.kernel()
