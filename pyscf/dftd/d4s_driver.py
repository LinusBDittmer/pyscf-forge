'''
Mathematical details for D4S corrections
'''

import os.path
import json
import numpy

from dftd4.interface import DispersionModel

#REFERENCE DATA READING PREPARATION AND CURATION
PERIODIC_TABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}
BETA_1 = 3
GC = 2
MAX_ELEM = 118

"""

INITIALISATION FUNCTIONS

"""


def determine_Ns(atomic_number, data):
    #Determines Ns number that comes to eq 8
    max_cn = 19
    cnc = numpy.zeros(max_cn)
    cnc[0] = 1
    ref = data['refn'][atomic_number]
    ngw = numpy.ones(ref)
    for ir in range(ref):
        icn = numpy.min((round(data['refcovcn'][ir][atomic_number]), max_cn))
        cnc[icn] = cnc[icn] + 1
    for ir in range(ref):
        icn = cnc[numpy.min((round(data['refcovcn'][ir][atomic_number]), max_cn))]
        ngw[ir] = icn*(icn+1)/2
    return ngw

def initialise(d4s):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load SEC data
    with open(os.path.join(current_dir, "sec_d4_data.json"),"r") as f:
        sec_data = json.load(f)
    sec_data['sec_atoms'] = numpy.array([0, 0, 0, 0, 0, 5, 6, 7, 7, 6, 8, 0, 0, 0, 0, 0, 16])

    # Load all further data
    with open(os.path.join(current_dir, "d4_data.json"), "r") as f:
        data = json.load(f)

    data = {k: numpy.array(v) for k, v in data.items()}
    sec_data = {k: numpy.array(v) for k, v in sec_data.items()}

    data['Ns_values'] = numpy.zeros((7, MAX_ELEM))
    for atom in range(MAX_ELEM):
        ref = data['refn'][atom]
        data['Ns_values'][:ref,atom] = determine_Ns(atom, data)

    data['chemical_hardness'] = numpy.array([0.47259288, 0.92203391, 0.17452888, 0.25700733, 0.33949086,
                                      0.42195412, 0.50438193, 0.58691863, 0.66931351, 0.75191607,
                                      0.17964105, 0.22157276, 0.26348578, 0.30539645, 0.34734014,
                                      0.38924725, 0.43115670, 0.47308269, 0.17105469, 0.20276244,
                                      0.21007322, 0.21739647, 0.22471039, 0.23201501, 0.23933969,
                                      0.24665638, 0.25398255, 0.26128863, 0.26859476, 0.27592565,
                                      0.30762999, 0.33931580, 0.37235985, 0.40273549, 0.43445776,
                                      0.46611708, 0.15585079, 0.18649324, 0.19356210, 0.20063311,
                                      0.20770522, 0.21477254, 0.22184614, 0.22891872, 0.23598621,
                                      0.24305612, 0.25013018, 0.25719937, 0.28784780, 0.31848673,
                                      0.34912431, 0.37976593, 0.41040808, 0.44105777, 0.05019332,
                                      0.06762570, 0.08504445, 0.10247736, 0.11991105, 0.13732772,
                                      0.15476297, 0.17218265, 0.18961288, 0.20704760, 0.22446752,
                                      0.24189645, 0.25932503, 0.27676094, 0.29418231, 0.31159587,
                                      0.32902274, 0.34592298, 0.36388048, 0.38130586, 0.39877476,
                                      0.41614298, 0.43364510, 0.45104014, 0.46848986, 0.48584550,
                                      0.12526730, 0.14268677, 0.16011615, 0.17755889, 0.19497557,
                                      0.21240778, 0.07263525, 0.09422158, 0.09920295, 0.10418621,
                                      0.14235633, 0.16394294, 0.18551941, 0.22370139, 0.25110000,
                                      0.25030000, 0.28840000, 0.31000000, 0.33160000, 0.35320000,
                                      0.36820000, 0.39630000, 0.40140000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                      0.00000000, 0.00000000, 0.00000000])
    data['Zeff'] = numpy.array([ 1,                                                 2,  # H-He
                            3, 4,                               5, 6, 7, 8, 9,10,  # Li-Ne
                            11,12,                              13,14,15,16,17,18,  # Na-Ar
                            19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,  # K-Kr
                            9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,  # Rb-Xe
                            9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,  # Cs-Lu
                            12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, # Hf-Rn
                            9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,  # Fr-Lr
                            12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]) # Rf-Og

    for atomic_number in range(MAX_ELEM):
        n_ref = data['refn'][atomic_number]
        for ref_i in range(n_ref):
            _is = int(data['refsys'][ref_i][atomic_number])-1
            iz = data['Zeff'][_is]
            aiw = (sec_data['sscale'][_is] 
                   * sec_data['secaiw'][:,_is] 
                   * (numpy.exp(BETA_1 * 
                                (1-numpy.exp(data['chemical_hardness'][_is] * GC 
                                             * (1-iz/(iz+data['clsh'][ref_i][atomic_number])))))))
            alpha = numpy.max((data['ascale'][ref_i][atomic_number] * 
                               (data['alphaiw'][:,ref_i,atomic_number]-data['hcount'][ref_i][atomic_number] * aiw), 
                               numpy.zeros(23)), axis=0)
            data['alphaiw'][:,ref_i,atomic_number] = alpha

    # Load R2/R4 coeffs
    r4r2 = numpy.genfromtxt(os.path.join(current_dir, "R2R4.txt"), delimiter=',')
    r4r2 = r4r2[~numpy.isnan(r4r2)]

    d4s.sec_data = sec_data
    d4s.data = data
    d4s.r4r2 = r4r2

def get_cn_and_charges(nuc_types, coords):
    model = DispersionModel(nuc_types, coords)
    res = model.get_properties()
    return res['partial charges'], res['coordination numbers']


"""

FUNCTIONS FOR CALCULATING C6

"""

def calculate_ref_C6(atomic_number_i, ref_i, atomic_number_j, ref_j, data):
    #Calculates reference C6_AB. Reference C6_AB could be seen from the expansion of the sum described in eq 9.
    freq = numpy.array([0.000001, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.200000, 1.400000, 1.600000, 1.800000, 2.000000, 2.500000,
            3.000000, 4.000000, 5.000000, 7.500000, 10.00000])
    weights = 0.5 * numpy.array([ ( freq[1] - freq[0] ), 
                              (freq[1] - freq[0]) + (freq[2] - freq[1]),
                              (freq[2] - freq[1]) + (freq[3] - freq[2]),
                              (freq[3] - freq[2]) + (freq[4] - freq[3]),
                              (freq[4] - freq[3]) + (freq[5] - freq[4]),
                              (freq[5] - freq[4]) + (freq[6] - freq[5]),
                              (freq[6] - freq[5]) + (freq[7] - freq[6]),
                              (freq[7] - freq[6]) + (freq[8] - freq[7]),
                              (freq[8] - freq[7]) + (freq[9] - freq[8]),
                              (freq[9] - freq[8]) + (freq[10] - freq[9]),
                              (freq[10] - freq[9]) + (freq[11] - freq[10]),
                              (freq[11] - freq[10]) + (freq[12] - freq[11]),
                              (freq[12] - freq[11]) + (freq[13] - freq[12]),
                              (freq[13] - freq[12]) + (freq[14] - freq[13]),
                              (freq[14] - freq[13]) + (freq[15] - freq[14]),
                              (freq[15] - freq[14]) + (freq[16] - freq[15]),
                              (freq[16] - freq[15]) + (freq[17] - freq[16]),
                              (freq[17] - freq[16]) + (freq[18] - freq[17]),
                              (freq[18] - freq[17]) + (freq[19] - freq[18]),
                              (freq[19] - freq[18]) + (freq[20] - freq[19]),
                              (freq[20] - freq[19]) + (freq[21] - freq[20]),
                              (freq[21] - freq[20]) + (freq[22] - freq[21]),
                              (freq[22] - freq[21])])
    thopi = 3.0/numpy.pi
    alpha_product = data['alphaiw'][:,ref_i,atomic_number_i]*data['alphaiw'][:,ref_j,atomic_number_j]
    c6 = thopi*numpy.sum(alpha_product*weights)
    return c6

def weight(CN_A, atomic_number, ref_number, beta_2, data, weight_method='gaussian', gaussian_window=None):
    if weight_method == 'gaussian':
        return gaussian_weight(CN_A, atomic_number, ref_number, beta_2, data)
    elif weight_method == 'soft_bilinear':
        return soft_bilinear_weight(CN_A, atomic_number, ref_number, beta_2, data, gaussian_window)

def soft_bilinear_weight(CN_A, atomic_number, ref_number, beta_2, data, gaussian_window):
    return 0.0

def gaussian_weight(CN_A, atomic_number, ref_number, beta_2, data):
    #Calculates gaussian weights from eq 8 for a particular reference
    N_A_ref = data['refn'][atomic_number]
    Ns = data['Ns_values'][:N_A_ref,atomic_number]
    CN_A_ref = data['refcovcn'][ref_number][atomic_number]
    numerator = 0
    for j in range(1,int(Ns[ref_number]+1)):
        numerator += numpy.exp(-beta_2*j*(CN_A - CN_A_ref)**2)
    denominator = 0
    for A_ref in range(N_A_ref):
        CN_A_ref = data['refcovcn'][A_ref][atomic_number]
        for j in range(1,int(Ns[A_ref]+1)):
            denominator += numpy.exp(-beta_2*j*(CN_A - CN_A_ref)**2)
    return numerator/denominator

def zeta(q_A, atomic_number, ref_number, data):
    #Zeta function from eq 2
    ga = 3 #Charge scaling height (beta_1)
    gc = 2 #Charge scaling steepness
    z_A_ref = data['clsq'][ref_number][atomic_number]+data['Zeff'][atomic_number]
    z_A = q_A+data['Zeff'][atomic_number]
    if z_A<0:
        zeta = numpy.exp(ga)
    else:
        zeta = numpy.exp(ga*(1-numpy.exp(gc*data['chemical_hardness'][atomic_number]*(1-z_A_ref/z_A))))
    return zeta

def get_weight_factors(atomic_number, CN_A, beta_2, data):
    n_window = 48
    # Window extent: 2 sigma
    x_window = 2.0 / numpy.sqrt(beta_2)
    n_refdata = data['refn'][atomic_number]
    linear_window = numpy.linspace(-x_window, x_window, num=n_window+1)
    gaussian_window = numpy.exp(-beta_2 * linear_window * linear_window)
    gaussian_window /= numpy.sum(gaussian_window)
    CN_A_ref = numpy.array([data['refcovcn'][ref][atomic_number] for ref in range(n_refdata)])
   
    def weight_factors(cn):
        weights = numpy.zeros(CN_A_ref.shape, dtype=float)
        if cn <= numpy.min(CN_A_ref):
            weights[numpy.argmin(CN_A_ref)] = 1
            return weights
        if cn >= numpy.max(CN_A_ref):
            weights[numpy.argmax(CN_A_ref)] = 1
            return weights
        
        # Find the left neighbour, i. e. biggest value smaller than cn
        mask_left = CN_A_ref < cn
        left_cn = numpy.max(CN_A_ref[mask_left])
        index_left = numpy.where(CN_A_ref == left_cn)[0][0]

        # Find the right neighbour, i. e. smallest value bigger than cn
        mask_right = CN_A_ref > cn
        right_cn = numpy.min(CN_A_ref[mask_right])
        index_right = numpy.where(CN_A_ref == right_cn)[0][0]

        # Number between 0 and 1 that encodes where cn is in the interval
        cn_int = (cn - left_cn) / (right_cn - left_cn)

        weights[index_right] = cn_int
        weights[index_left] = 1 - cn_int
            
        return weights

    total_weights = numpy.zeros(CN_A_ref.shape, dtype=float)

    for i, cn_i in enumerate(linear_window):
        gweight = weight_factors(cn_i + CN_A)
        total_weights += gweight * gaussian_window[i]

    return total_weights

    total_weights = numpy.zeros(CN_A_ref.shape, dtype=float)
    for i, cn_i in enumerate(linear_window):
        gweight = weight_factors(cn_i + CN_A)
        total_weights += gweight * gaussian_window[i]

    return total_weights


def calculate_C6(atomic_number_A, CN_A, q_A, atomic_number_B, CN_B, q_B, beta_2, data, weight_method='gaussian'):
    # Escape concurrent evaluation of many datapoints
    if isinstance(CN_A, tuple):
        return tuple(calculate_C6(atomic_number_A, CN, q_A, atomic_number_B, CN_B, 
                                  q_B, beta_2, data, weight_method) for CN in CN_A)
    elif isinstance(CN_A, list):
        return [calculate_C6(atomic_number_A, CN, q_A, atomic_number_B, CN_B, 
                             q_B, beta_2, data, weight_method) for CN in CN_A]
    elif isinstance(CN_A, numpy.ndarray):
        return numpy.array([calculate_C6(atomic_number_A, CN, q_A, atomic_number_B, CN_B, 
                                         q_B, beta_2, data, weight_method) for CN in CN_A])

    #Computes C6_AB coefficient
    N_A_ref = data['refn'][atomic_number_A]
    N_B_ref = data['refn'][atomic_number_B]
    C6 = 0
    if weight_method == 'gaussian':
        for ref_i in range(N_A_ref):
            W_A = weight(CN_A, atomic_number_A, ref_i, beta_2, data, weight_method)
            zetta_A = zeta(q_A, atomic_number_A, ref_i, data)
            for ref_j in range(N_B_ref):
                W_B = weight(CN_B, atomic_number_B, ref_j, beta_2, data, weight_method)
                zetta_B = zeta(q_B, atomic_number_B, ref_j, data)
                ref_c6 = calculate_ref_C6(atomic_number_A, ref_i, atomic_number_B, ref_j, data)
                C6 += W_A*zetta_A*W_B*zetta_B*ref_c6

    elif weight_method == 'soft_bilinear':
        weight_factors_A = get_weight_factors(atomic_number_A, CN_A, beta_2, data)
        weight_factors_B = get_weight_factors(atomic_number_B, CN_B, beta_2, data)

        for ref_i in range(N_A_ref):
            zetta_A = zeta(q_A, atomic_number_A, ref_i, data)
            for ref_j in range(N_B_ref):
                ref_c6 = calculate_ref_C6(atomic_number_A, ref_i, atomic_number_B, ref_j, data)
                zetta_B = zeta(q_B, atomic_number_B, ref_j, data)
                C6 += weight_factors_A[ref_i] * weight_factors_B[ref_j] * zetta_A * zetta_B * ref_c6
        
    return C6

def becke_johnson_damping(coords_1, coords_2, ALPHA_1, ALPHA_2, C6, C8, N):
    R_IJ_cutoff = numpy.sqrt(C8/C6)
    R_IJ = numpy.linalg.norm(coords_1-coords_2)
    return R_IJ**N/(R_IJ**N + (ALPHA_1*R_IJ_cutoff + ALPHA_2)**N)

"""

FUNCTIONS FOR CALCULATING C8

"""

def calculate_C8(C6_AB,I,J,R4R2):
    return 3*C6_AB*numpy.sqrt(numpy.sqrt(I+1)*numpy.sqrt(J+1)*R4R2[I]*R4R2[J]/4)

"""

FUNCTIONS FOR 3-BODY TERMS

"""

def find_cos_product(a,b,c):
    A = a**2+b**2-c**2
    B = b**2+c**2-a**2
    C = c**2+a**2-b**2
    return A*B*C/(8*a**2*b**2*c**2)

def damping_3body(R_damp):
    return 1/(1+6*R_damp**(-16))

"""

THE DRIVER FUNCTION

"""

def d4s_driver(nuc_types, coords, s6, s8, alpha_1, alpha_2, beta_2, s_3body, data, r4r2, weight_method):
    if len(nuc_types) == 1:
        return 0, None

    energy_disp_e6 = 0
    energy_disp_e8 = 0
    energy_disp_3body = 0
    # Coordination numbers and effective charges
    effective_charges, coordination_numbers = get_cn_and_charges(nuc_types, coords)

    # List of C6 and C8 coefficients
    # These are only used for 3 body terms and
    # therefore differ from the actual C6 and C8 values
    c6_3body_list = numpy.zeros((len(coords),len(coords)))
    c8_3body_list = numpy.zeros((len(coords),len(coords)))

    # Calculation of two-body terms
    for d1 in range(len(nuc_types)-1):
        nuc_charge_1 = nuc_types[d1]
        atom_coords_1 = coords[d1]
        for _d2 in range(len(nuc_types[d1+1:])):
            # d2 = _d2 + d1 + 1
            d2 = _d2
            nuc_charge_2 = nuc_types[d2+d1+1]
            atom_coords_2 = coords[d2+d1+1]

            # Distance between Atom 1 and Atom 2
            dist_12 = numpy.linalg.norm(atom_coords_1 - atom_coords_2)
            atom1_idx = int(nuc_charge_1-1)
            atom2_idx = int(nuc_charge_2-1)

            # Pair C6 coefficient between Atoms 1 and 2
            c6_12 = calculate_C6(atom1_idx, coordination_numbers[d1], effective_charges[d1], 
                                 atom2_idx, coordination_numbers[d2+d1+1], effective_charges[d2+d1+1], 
                                 beta_2, data, weight_method)
            
            # Pair C8 coefficient between Atoms 1 and 2
            c8_12 = calculate_C8(c6_12, atom1_idx, atom2_idx, r4r2)

            # Pair C6 coefficient for 3-body calculation
            c6_3body_list[d1][d2+d1+1] = calculate_C6(atom1_idx, coordination_numbers[d1], 0, 
                                                      atom2_idx, coordination_numbers[d2+d1+1], 0, 
                                                      beta_2, data, weight_method)
            # Pair C8 coefficient for 3-body calculation
            c8_3body_list[d1][d2+d1+1] = calculate_C8(c6_3body_list[d1][d2+d1+1], atom1_idx, atom2_idx, r4r2)
            energy_disp_e6 += s6*becke_johnson_damping(atom_coords_1, atom_coords_2, alpha_1, alpha_2, c6_12, 
                                                       c8_12, 6)*c6_12/dist_12**6
            energy_disp_e8 += s8*becke_johnson_damping(atom_coords_1, atom_coords_2, alpha_1, alpha_2, c6_12, 
                                                       c8_12, 8)*c8_12/dist_12**8

    # Symmetrising the list of C6 coefficients
    c6_3body_list += c6_3body_list.T
    c8_3body_list += c8_3body_list.T

    if len(nuc_types) == 2:
        return - (energy_disp_e8 + energy_disp_e6), None

    # Calculation of 3 body terms
    for d1 in range(len(nuc_types)-2):
        nuc_charge_1 = nuc_types[d1]
        atom_coords_1 = coords[d1]
        for _d2 in range(len(nuc_types[d1+1:-1])):
            # d2 = _d2 + d1 + 1
            d2 = _d2
            nuc_charge_2 = nuc_types[d2+d1+1]
            atom_coords_2 = coords[d2+d1+1]
            for _d3 in range(len(nuc_types[d1+d2+2:])):
                # d3 = _d3 + d1 + d2 + 2
                d3 = _d3
                nuc_charge_3 = nuc_types[d3+d2+d1+2]
                atom_coords_3 = coords[d3+d2+d1+2]

                # 3 Body distances
                dist_12 = numpy.linalg.norm(atom_coords_1 - atom_coords_2)
                dist_23 = numpy.linalg.norm(atom_coords_2 - atom_coords_3)
                dist_31 = numpy.linalg.norm(atom_coords_3 - atom_coords_1)

                # Finding the orientation dependent factor
                cos_factor = find_cos_product(dist_12, dist_23, dist_31)

                C9 = numpy.sqrt(c6_3body_list[d1][d1+d2+1]
                                * c6_3body_list[d1+d2+1][d1+d2+d3+2]
                                * c6_3body_list[d1+d2+d3+2][d1])
                R_damp_ij = (alpha_1 * numpy.sqrt(c8_3body_list[d1][d1+d2+1] / c6_3body_list[d1][d1+d2+1]) + alpha_2)
                R_damp_jk = (alpha_1 * numpy.sqrt(c8_3body_list[d1+d2+1][d1+d2+d3+2] / c6_3body_list[d1+d2+1][d1+d2+d3+2]) + alpha_2)
                R_damp_ki = (alpha_1 * numpy.sqrt(c8_3body_list[d1+d2+d3+2][d1] / c6_3body_list[d1+d2+d3+2][d1]) + alpha_2)
                R_damp = ((dist_12*dist_23*dist_31)
                          / (R_damp_ij*R_damp_jk*R_damp_ki))**(1/3)
                
                energy_disp_3body += (s_3body * damping_3body(R_damp) * C9 
                                      * (3*cos_factor+1) / (dist_12*dist_23*dist_31)**3)
    
    return -(energy_disp_e8 + energy_disp_e6 - energy_disp_3body), None #, c6_3body_list, energy_disp_e6, energy_disp_e8