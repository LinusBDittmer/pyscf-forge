'''

Empirical dispersion models from Grimme's DFT-D suite

'''

from dftd4.interface import DispersionModel
from dftd4.parameters import get_damping_param
import numpy

from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.dftd.d4s_driver import initialise, d4s_driver

FUNC_CODE = {
    # mf.xc          name in dftd3 library      dftd3 versions
    'BLYP'         : ('b-lyp',                  (2,3,4,5,6)),
    'B88,LYP'      : ('b-lyp',                  (2,3,4,5,6)),
    'BP86'         : ('b-p',                    (2,3,4,5,6)),
    'B88,P86'      : ('b-p',                    (2,3,4,5,6)),
    'B88B95'       : ('b1b95',                  (3,4)),
    'B1B95'        : ('b1b95',                  (3,4)),
    'B3LYP'        : ('b3-lyp',                 (2,3,4,5,6)),
    'B3LYP/631GD'  : ('b3-lyp/6-31gd',          (4,)),
    'B3LYPG'       : ('b3-lyp',                 (2,3,4,5,6)),
    'B3PW91'       : ('b3pw91',                 (3,4)),
    'B97-D'        : ('b97-d',                  (2,3,4,5,6)),
    'BHANDHLYP'    : ('bh-lyp',                 (3,4)),
    'BMK,BMK'      : ('bmk',                    (3,4)),
    'BOP'          : ('bop',                    (3,4)),
    'B88,OP_B88'   : ('bop',                    (3,4)),
    'BPBE'         : ('bpbe',                   (3,4)),
    'B88,PBE'      : ('bpbe',                   (3,4)),
    'CAMB3LYP'     : ('cam-b3lyp',              (3,4)),
    'CAM_B3LYP'    : ('cam-b3lyp',              (3,4)),
    #''             : ('dsd-blyp',               (2,4)),
    #''             : ('dsd-blyp-fc',            (4,)),
    'HCTH-120'     : ('hcth120',                (3,4)),
    'HF'           : ('hf',                     (3,4)),
    #''             : ('hf/minis',               (4,)),
    #''             : ('hf/mixed',               (4,)),
    'HF/SV'        : ('hf/sv',                  (4,)),
    #''             : ('hf3c',                   (4,)),
    #''             : ('hf3cv',                  (4,)),
    'HSE06'        : ('hse06',                  (3,4)),
    'HSE_SOL'      : ('hsesol',                 (4,)),
    'LRC-WPBE'     : ('lc-wpbe',                (3,4,5,6)),
    'LRC-WPBEH'    : ('lc-wpbe',                (3,4,5,6)),
    'M05'          : ('m05',                    (3,)),
    'M05,M05'      : ('m05',                    (3,)),
    'M05-2X'       : ('m052x',                  (3,)),
    'M06'          : ('m06',                    (3,)),
    'M06,M06'      : ('m06',                    (3,)),
    'M06-2X'       : ('m062x',                  (3,)),
    'M06_HF'       : ('m06hf',                  (3,)),
    'M06-L'        : ('m06l',                   (3,)),
    #''             : ('mpw1b95',                (3,4)),
    #''             : ('mpwb1k',                 (3,4)),
    #''             : ('mpwlyp',                 (3,4)),
    'OLYP'         : ('o-lyp',                  (3,4)),
    'OPBE'         : ('opbe',                   (3,4)),
    'OTPSS_D'      : ('otpss',                  (3,4)),
    'OTPSS-D'      : ('otpss',                  (3,4)),
    'PBE'          : ('pbe',                    (2,3,4,5,6)),
    'PBE,PBE'      : ('pbe',                    (2,3,4,5,6)),
    'PBE0'         : ('pbe0',                   (2,3,4,5,6)),
    'PBEH'         : ('pbe0',                   (2,3,4,5,6)),
    #''             : ('pbeh3c',                 (4,)),
    #''             : ('pbeh-3c',                (4,)),
    'PBESOL'       : ('pbesol',                 (3,4)),
    #''             : ('ptpss',                  (3,4)),
    #''             : ('pw1pw',                  (4,)),
    #''             : ('pw6b95',                 (2,3,4)),
    #''             : ('pwb6k',                  (4,)),
    #''             : ('pwgga',                  (4,)),
    #''             : ('pwpb95',                 (3,4)),
    'REVPBE'       : ('revpbe',                 (2,3,4)),
    'REVPBE0'      : ('revpbe0',                (3,4)),
    #''             : ('revpbe38',               (3,4)),
    #''             : ('revssb',                 (3,4)),
    'RPBE'         : ('rpbe',                   (3,4)),
    'RPBE,RPBE'    : ('rpbe',                   (3,4)),
    'RPW86,PBE'    : ('rpw86-pbe',              (3,4)),
    'SLATER'       : ('slater-dirac-exchange',  (3,)),
    'XALPHA'       : ('slater-dirac-exchange',  (3,)),
    'SSB,PBE'      : ('ssb',                    (3,4)),
    'TPSS'         : ('tpss',                   (2,3,4)),
    'TPSS0'        : ('tpss0',                  (3,4)),
    'TPSSH'        : ('tpssh',                  (3,4)),
    #''             : ('dftb3',                  (4,)),
}

def d4s(scf_method, s6=None, s8=None, alpha1=None, alpha2=None, beta2=None, s_3body=None):
    # Create the object of dftd3 interface wrapper
    xc = getattr(scf_method, 'xc', 'HF').upper().replace(' ', '')
    with_dftd4s = DFTD4S_Model(scf_method.mol, xc, s6, s8, alpha1, alpha2, beta2, s_3body)

    # DFT-D3 has been initialized, avoid to create the derived classes twice.
    if isinstance(scf_method, _DFTD4S):
        scf_method.with_dftd4s = with_dftd4s
        return scf_method

    method_class = scf_method.__class__

    # A DFTD3 extension class is defined because other extensions are applied
    # based on the dynamic class. If DFT-D3 correction was applied by patching
    # the functions of object scf_method, these patches may not be realized by
    # other extensions.
    class DFTD4S(_DFTD4S, method_class):
        def __init__(self, method, with_dftd4s):
            self.__dict__.update(method.__dict__)
            self.with_dftd4s = with_dftd4s
            self._keys.update(['with_dftd4s'])

        def dump_flags(self, verbose=None):
            method_class.dump_flags(self, verbose)
            if self.with_dftd4s:
                self.with_dftd4s.dump_flags(verbose)
            return self

        def energy_nuc(self):
            # Adding DFT D4S correction to nuclear part because it is computed
            # based on nuclear coordinates only.  It does not depend on
            # quantum effects.
            enuc = method_class.energy_nuc(self)
            if self.with_dftd4s:
                enuc += self.with_dftd4s.kernel()[0]
            return enuc
        
        def energy_disp(self):
            # Just the dispersion energy:
            edisp = 0.0
            if self.with_dftd4s:
                edisp += self.with_dftd4s.kernel()[0]
            return edisp

        def reset(self, mol=None):
            self.with_dftd4s.reset(mol)
            return method_class.reset(self, mol)

        # def nuc_grad_method(self):
        #     scf_grad = method_class.nuc_grad_method(self)
        #     return grad(scf_grad)
        # Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    return DFTD4S(scf_method, with_dftd4s)


class _DFTD4S:
    pass

class DFTD4S_Model(lib.StreamObject):
    def __init__(self, mol, xc='hf', s6=None, s8=None, 
                 alpha1=None, alpha2=None, beta2=None, s_3body=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.xc = xc
        self.version = 4
        self.edisp = None
        self.grads = None
        self.data = None
        self.sec_data = None
        self.disp_driver = d4s_driver

        # Translate XC functional to DFTD version
        basis_type = _get_basis_type(mol)
        if self.xc in FUNC_CODE:
            func, supported_versions = FUNC_CODE[self.xc]
            self.xc_dftd = func
            if func == 'b3lyp' and basis_type == '6-31gd':
                func, supported_versions = FUNC_CODE['B3LYP/631GD']
            elif func == 'hf' and basis_type == 'sv':
                func, supported_versions = FUNC_CODE['HF/SV']
        else:
            raise RuntimeError('Functional %s not found' % self.xc)
        # assert(self.version in supported_versions)

        # Standard D4 params
        try:
            std_params = get_damping_param(self.xc_dftd)
        except:
            std_params = get_damping_param(self.xc)
        self.__dict__.update(std_params)

        if s6 is not None:
            self.s6 = s6
        if s8 is not None:
            self.s8 = s8
        if alpha1 is not None:
            self.a1 = alpha1
        if alpha2 is not None:
            self.a2 = alpha2
        if beta2 is not None:
            self.b2 = beta2
        else:
            self.b2 = 6.0
        if s_3body is not None:
            self.s_3body = s_3body
        else:
            self.s_3body = 1.0

        # Gather all Driver required data
        initialise(self)

    def dump_flags(self, verbose=None):
        logger.info(self, '** DFTD4S parameter **')
        logger.info(self, 'func %s', self.xc)
        logger.info(self, 'version %s', self.version)
        return self

    def kernel(self):
        mol = self.mol        
        basis_type = _get_basis_type(mol)

        # dft-d3 has special treatment for def2-TZ basis
        tz = (basis_type == 'def2-TZ')

        coords = mol.atom_coords(unit='Bohr')

        nuc_types = numpy.asarray([gto.charge(mol.atom_symbol(ia)) 
                                    for ia in range(mol.natm)], dtype=int)

        edisp = 0.0 
        grads = numpy.zeros((mol.natm,3))

        edisp, grads = self.disp_driver(nuc_types, coords, self.s6, self.s8, 
                                        self.a1, self.a2, self.b2, 
                                        self.s_3body, self.data, self.sec_data, self.r4r2)

        self.edisp = edisp
        self.grads = grads
        return edisp, grads

    def reset(self, mol):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        self.mol = mol
        return self


def _get_basis_type(mol):
    def classify(mol_basis):
        basis_type = 'other'
        if isinstance(mol_basis, str):
            mol_basis = gto.basis._format_basis_name(mol_basis)
            if mol_basis[:6] == 'def2tz':
                basis_type = 'def2-TZ'
            elif mol_basis[:6] == 'def2sv':
                basis_type = 'sv'
            elif mol_basis[:5] == '631g*':
                basis_type = '6-31gd'
            elif mol_basis[:4] == '631g' and 'd' in mol_basis:
                basis_type = '6-31gd'
        return basis_type

    if isinstance(mol.basis, dict):
        basis_types = [classify(b) for b in mol.basis.values()]
        basis_type = 'other'
        for bt in basis_types:
            if bt != 'other':
                basis_type = bt
                break
        if (len(basis_types) > 1 and
            all(b == basis_type for b in basis_types)):
            logger.warn(mol, 'Mutliple types of basis found in mol.basis. '
                        'Type %s is applied\n', basis_type)
    else:
        basis_type = classify(mol.basis)
    return basis_type