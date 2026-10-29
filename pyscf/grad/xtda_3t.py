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
#
# Author: Linus B. Dittmer <linus.dittmer@stud.uni-heidelberg.de>
#

'''
Analytic nuclear gradients of one XTDA-3T root, E = E_ROKS + omega.

omega = x.(H - E0).x is a Rayleigh quotient, so the root vector x needs no response.  H - E0 is linear in
the spin-averaged Fock matrix Fbar and in the integrals [pq|rs] = sum_t c_t (pq|rs)_t of the kernel w:

    omega = tr(P Fbar) + sum Gamma_pqrs [pq|rs],   P = gamma^{kk} - gamma0   (gamma0 = 2 on C, 1 on O).

The two-body part is contracted with own DF fits (pyscf.xtda_3t._rdm2), T_t = (P|pq)_t, d_t = j2c_t^-1 T_t:

    Ghat_t = 2 c_t Y[d_t],   omega_2e = 1/2 sum_t <Ghat_t, T_t>,   M_t = sum_pq Ghat_t^P d_t^Q.

Orbital relaxation: with spin densities Q_t (MO) and C -> C(1 + U)

    X_rp = sum_t [2 (F_t Q_t)_rp + 2 V_t[Q]_rp n_t,p] + 2 sum_t sum_P (T_t^P Ghat_t^P)_rp,

Q_t = P/2 for omega alone.  A Z-vector equation with the exact ROHF/ROKS orbital Hessian on the
non-redundant pairs (VC, VO, OC) gives Z_t; Q_t = P/2 + Z_t, W = (X^L + X^L^T)/4 and

    dE/dA = dE_ROKS/dA + d/dA sum_t tr(Q_t F_t[D])|skeleton + own DF derivative - tr(W_AO S^(A)).

Damping (td.damping = (p2, n)): omega = x.Htilde.x with Htilde_KL = f_KL H_KL on pairs touching a small CSF,
f_KL = f(d_K - d_L) = 1/(1 + y arctan y), y = (p2 |d_K - d_L|)^n, d_K = H_KK.  Then d omega = sum W_KL dH_KL with
W = x x^T + dW,

    dW_ss = x_s x_s^T o (f_ss - 1) + diag(g_s),   dW_sl = x_s x_l^T o (f_sl - 1),   dW_ll = diag(g_l),
    g_K = 2 sum_L a_KL,   a_KL = x_K x_L H_KL f'(d_K - d_L),

contracted into P and Ghat_t by _tdm.weighted_dm and _rdm2.contract_weighted.  The d_K are not invariant under
rotations inside C, O and V; td.mo_coeff fixes them by Fbar_rp = 0, whose multipliers enter P as
Pz_rp = -A_rp / (2 (eps_r - eps_p)), A = X - X^T (pairs degenerate within 1e-6 are dropped, with a warning if
A_rp is not negligible).  Grid response of the XC terms is omitted (as in pyscf.grad.tduks).

The densities are bilinear in two roots (state2), and _relaxed takes an extra antisymmetric Lagrangian term:
pyscf.nac.xtda_3t reuses both.
'''

import types
import numpy
from pyscf import lib, gto, df, dft
from pyscf.lib import logger
from pyscf.df.grad import rhf as df_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.xtda_3t import xtda_3t, _rdm2, _tdm


def _occupations(td):
    core, somo, _ = td._orbs
    nmo = td._scf.mo_coeff.shape[1]
    na, nb = numpy.zeros(nmo), numpy.zeros(nmo)
    na[core] = na[somo] = nb[core] = 1
    return na, nb


def _fock_mo(mf, mo):
    f = mf.get_fock()
    return numpy.array([mo.T @ f.focka @ mo, mo.T @ f.fockb @ mo])


def _uks_response(mf, mo, na, nb):
    # ROKS gen_response feeds the 2D ROHF mo_coeff/mo_occ to the UKS XC kernel (wrong density): pass UKS-shaped ones
    return mf.gen_response(mo_coeff=(mo, mo), mo_occ=(na, nb), hermi=1)


def _df_terms(mol, auxmol, mo_coeff, terms):
    '''Own DF fit per kernel term: [(c, omega, T (naux, nmo, nmo), d = j2c^-1 T)].'''
    out = []
    for c, omega in terms:
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1')
            j2c = auxmol.intor('int2c2e', hermi=1)
        t = lib.einsum('uvP,up,vq->Ppq', int3c, mo_coeff, mo_coeff)
        # same metric solver as pyscf.df (Cholesky, else eigen pseudo-inverse with LINEAR_DEP_THR): the fit then
        # equals the one in the energy; long-range metrics are often numerically singular
        solve = df_grad._gen_metric_solver(j2c, 'CD', df.incore.LINEAR_DEP_THR)
        d = solve(t.reshape(len(t), -1)).reshape(t.shape)
        out.append((c, omega, t, d))
    return out


def _two_body(td, x, spin, dfterms, x2=None):
    core, somo, virt = td._orbs
    blocks = xtda_3t.csf_list(spin, len(core), len(virt))
    return [2 * c * _rdm2.contract(x, blocks, d, core, somo, virt, x2) for c, _, _, d in dfterms]


def _damping_weights(x, hss, hsl, d_small, d_large, p2, n=5, x2=None):
    '''dW of the damped bilinear form x.Htilde.x2 (undamped hss, hsl): d(x.Htilde.x2) = sum (S + dW) dH with the
    symmetric S = (x x2^T + x2 x^T)/2 (x2 = None: x2 = x, the damped Rayleigh quotient).

    Returns (wss, wsl, gs, gl) with dW_ss = wss + diag(gs), dW_sl = wsl, dW_ll = diag(gl).
    '''
    ns = hss.shape[0]
    if x2 is None:
        x2 = x
    (fss, dfss), (fsl, dfsl) = xtda_3t.damping_factors(d_small, d_large, p2, n, deriv=True)
    # ponytail: about six (ns, nl) arrays live at once, the size of hsl; block over large columns if memory matters
    oss = .5 * (numpy.outer(x[:ns], x2[:ns]) + numpy.outer(x2[:ns], x[:ns]))
    osl = .5 * (numpy.outer(x[:ns], x2[ns:]) + numpy.outer(x2[:ns], x[ns:]))
    a_ss = oss * hss * dfss
    a_sl = osl * hsl * dfsl
    gs = 2 * (a_ss.sum(1) + a_sl.sum(1))
    gl = -2 * a_sl.sum(0)
    return oss * (fss - 1), osl * (fsl - 1), gs, gl


def _state_densities(td, spin, state, dfterms, state2=None):
    '''P (MO) and Ghat_t of the symmetric weight S = (x x2^T + x2 x^T)/2 of roots state and state2 (default: state),
    plus the damping weight dW if td.damping is set.  P = sym(gamma^{x x2}) - (x.x2) gamma0: the state density
    difference for one root, the symmetric transition density for two.'''
    x = td.xs[spin][state]
    x2 = None if state2 is None else td.xs[spin][state2]
    core, somo, virt = td._orbs
    blocks = xtda_3t.csf_list(spin, len(core), len(virt))
    y = x if x2 is None else x2
    dm = _tdm.transition_dm(x, y, blocks, core, somo, virt)
    g0 = numpy.zeros(len(dm))
    g0[core], g0[somo] = 2, 1
    p = .5 * (dm + dm.T) - (x @ y) * numpy.diag(g0)
    ghats = _two_body(td, x, spin, dfterms, x2)
    if td.damping is None:
        return p, ghats
    ints = td._ints
    hss, hsl = xtda_3t.build_small(ints, spin)
    hdl = xtda_3t.hdiag_large(ints, *xtda_3t.cvcv_operator(ints, spin))
    wss, wsl, gs, gl = _damping_weights(x, hss, hsl, hss.diagonal().copy(), hdl.ravel(), *td.damping, x2=x2)
    wss = wss + numpy.diag(gs)
    p = p + _tdm.weighted_dm(wss, wsl, gl, blocks, core, somo, virt)
    ghats = [g + 2 * c * _rdm2.contract_weighted(wss, wsl, gl, blocks, d, core, somo, virt)
             for g, (c, _, _, d) in zip(ghats, dfterms)]
    return p, ghats


def _orbital_lagrangian(fock, vresp, mo, occs, q, dfterms, ghats):
    '''X_rp = d omega / d U_rp for spin densities q = (Q_alpha, Q_beta) (MO); Fock part only without DF terms.'''
    v = vresp(numpy.array([mo @ qt @ mo.T for qt in q]))
    x = 0
    for t in range(2):
        x = x + 2 * (fock[t] @ q[t] + (mo.T @ v[t] @ mo) * occs[t][None, :])
    for (_, _, tt, _), g in zip(dfterms, ghats):
        x = x + 2 * lib.einsum('Prs,Psp->rp', tt, g)
    return x


def _canonical_pz(xo, eps, spaces, log, thresh=1e-6):
    '''Multiplier density of the canonical condition Fbar_rp = 0 (r != p in one space):
    A_rp + 2 (eps_r - eps_p) Pz_rp = 0 with A = xo - xo^T.

    Pairs with |eps_r - eps_p| < thresh get 0; a warning is issued if such a pair has |A_rp| > 1e-6
    (the damped energy then depends on the arbitrary rotation inside the degenerate set).
    '''
    a = xo - xo.T
    pz = numpy.zeros_like(xo)
    for sp in spaces:
        de = eps[sp, None] - eps[None, sp]
        asp = a[numpy.ix_(sp, sp)]
        deg = abs(de) < thresh
        numpy.fill_diagonal(deg, False)
        ok = ~deg & ~numpy.eye(len(sp), dtype=bool)
        blk = numpy.zeros_like(asp)
        blk[ok] = -asp[ok] / (2 * de[ok])
        pz[numpy.ix_(sp, sp)] = blk
        for r, q in zip(*numpy.nonzero(numpy.triu(deg & (abs(asp) > 1e-6)))):
            log.warn('XTDA-3T damped model: canonical orbitals %d and %d are degenerate (|de| < %g) but carry a '
                     'Lagrangian asymmetry of %.2g; the damped roots depend on their arbitrary rotation, the '
                     'derivative is approximate', sp[r], sp[q], thresh, asp[r, q])
    return pz


def _rohf_hessian(fock, vresp, mo, occs):
    '''Exact ROHF/ROKS orbital Jacobian on the non-redundant pairs mask[r, p] = occ[p] > occ[r].'''
    occ = occs[0] + occs[1]
    mask = occ[None, :] > occ[:, None]

    def hop(z):
        kap = numpy.zeros(mask.shape)
        kap[mask] = z
        kap -= kap.T
        dd = numpy.array([mo @ (kap * n[None, :] - n[:, None] * kap) @ mo.T for n in occs])
        v = vresp(dd)
        x = 0
        for f, vt, n in zip(fock, v, occs):
            x = x + 2 * (f @ kap - kap @ f + mo.T @ vt @ mo) * n[None, :]
        return (x - x.T)[mask]

    hd = 0
    for f, n in zip(fock, occs):
        e = f.diagonal()
        hd = hd + 2 * (e[:, None] - e[None, :]) * (n[None, :] - n[:, None])
    return mask, hop, hd[mask]


def solve_z(hop, hdiag, rhs, tol, max_cycle, log):
    '''Solve J z = rhs with lib.krylov, diagonally preconditioned.

    lib.krylov stops on an absolute norm of its (unnormalised, once-orthogonalised) Krylov vectors, at a
    residual far above tol: solve for the normalised residual and restart from it.
    '''
    hd = hdiag.copy()
    hd[abs(hd) < 1e-8] = 1e-8
    z = numpy.zeros_like(rhs)
    r = rhs
    thresh = tol * max(1, numpy.linalg.norm(rhs))
    for _ in range(5):
        if numpy.linalg.norm(r) < thresh:
            break
        b = r / hd
        nb = numpy.linalg.norm(b)
        z = z + nb * lib.krylov(lambda xs: numpy.array([(hop(x) - hd * x) / hd for x in xs]), b / nb,
                                tol=tol, max_cycle=max_cycle)
        r = rhs - hop(z)
    res = numpy.linalg.norm(r)
    if res > 10 * tol * max(1, numpy.linalg.norm(rhs)):
        log.warn('Z-vector not converged: |J z - rhs| = %.3g', res)
    return z


def _z_densities(z, mask, occs):
    '''Spin densities Z_t (MO) of the ROKS gradient term z.g = sum_t tr(Z_t F_t).'''
    out = numpy.zeros((2,) + mask.shape)
    for zt, n in zip(out, occs):
        zt[mask] = 2 * z * (n[None, :] - n[:, None])[mask]
    return .5 * (out + out.transpose(0, 2, 1))


def _fock_skeleton(mf, xc_coeffs, q_ao, d_ao, atmlst=None, max_memory=2000):
    '''d/dA sum_t tr(Q_t F_t[D]) at fixed AO matrices (h, J, K, XC without grid response, DF aux response).'''
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    mf_grad = mf.nuc_grad_method()
    aux = getattr(mf_grad, 'auxbasis_response', False)
    dms = numpy.vstack([d_ao, q_ao])                      # D_alpha, D_beta, Q_alpha, Q_beta
    vj = mf_grad.get_j(mol, dms)
    # (3, nao, nao) bra-derivative matrices; the atom contribution is 2 sum_{mu in A} g . (symmetric density)
    g = (vj[0] + vj[1]) * (q_ao[0] + q_ao[1]) + (vj[2] + vj[3]) * (d_ao[0] + d_ao[1])
    de_aux = 0
    if aux:
        de_aux = sum(vj.aux[2 + t, s] + vj.aux[s, 2 + t] for s in range(2) for t in range(2))
    for c, omega in xtda_3t._kernel_terms(*xc_coeffs):
        vk = mf_grad.get_k(mol, dms, omega=(omega or None))
        g = g - c * sum(vk[t] * q_ao[t] + vk[2 + t] * d_ao[t] for t in range(2))
        if aux:
            de_aux = de_aux - c * sum(vk.aux[2 + t, t] + vk.aux[t, 2 + t] for t in range(2))
    if isinstance(mf, dft.rks.KohnShamDFT):
        shim = types.SimpleNamespace(mol=mol, base=types.SimpleNamespace(_scf=mf.to_uks()))
        _, f1oo, v1ao, _ = tduks_grad._contract_xc_kernel(shim, mf.xc, numpy.zeros_like(q_ao), q_ao,
                                                          with_vxc=True, with_kxc=False, max_memory=max_memory)
        g = g + sum(v1ao[t, 1:] * q_ao[t] + f1oo[t, 1:] * d_ao[t] for t in range(2))
    hcore_deriv = mf_grad.hcore_generator(mol)
    qtot = q_ao[0] + q_ao[1]
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = numpy.einsum('xij,ij->x', hcore_deriv(ia), qtot) + 2 * g[:, p0:p1].sum(axis=(1, 2))
        if aux:
            de[k] += de_aux[ia]
    return de


def _df_skeleton(mol, auxmol, mo_coeff, dfterms, ghats, atmlst=None):
    '''Derivative of the own DF fits: sum_t [Ghat_t . (P|pq)_t^(A) - 1/2 M_t . j2c_t^(A)].'''
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()
    de = numpy.zeros((mol.natm, 3))
    for (c, omega, t, d), ghat in zip(dfterms, ghats):
        gao = lib.einsum('Ppq,up,vq->uvP', ghat, mo_coeff, mo_coeff)
        m = lib.einsum('Ppq,Qpq->PQ', ghat, d)
        m = .5 * (m + m.T)
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            # ponytail: full (3, nao, nao, naux) derivative integrals, 3 nao^2 naux memory; block over aux
            # shells with shls_slice if that matters
            ip1 = df.incore.aux_e2(mol, auxmol, 'int3c2e_ip1', aosym='s1', comp=3)
            ip2 = df.incore.aux_e2(mol, auxmol, 'int3c2e_ip2', aosym='s1', comp=3)
            j2c_ip1 = auxmol.intor('int2c2e_ip1', comp=3)
        for ia in range(mol.natm):
            p0, p1 = aoslices[ia, 2:]
            q0, q1 = auxslices[ia, 2:]
            de[ia] -= 2 * numpy.einsum('xuvP,uvP->x', ip1[:, p0:p1], gao[p0:p1])
            de[ia] -= numpy.einsum('xuvP,uvP->x', ip2[..., q0:q1], gao[..., q0:q1])
            de[ia] += numpy.einsum('xPQ,PQ->x', j2c_ip1[:, q0:q1], m[q0:q1])
    return de[list(atmlst)]


def _auxmol(td):
    auxmol = td._with_df.auxmol
    if auxmol is None:
        # a td-owned DF used only through range_coulomb copies (no full-range term) is never built itself
        auxmol = df.addons.make_auxmol(td.mol, td._with_df.auxbasis)
    return auxmol


def _relaxed(td_grad, p, ghats, auxmol, dfterms, xa=0, atmlst=None):
    '''Orbital-relaxed derivative of tr(P Fbar) + sum Gamma [pq|rs] (P, Ghat_t of _state_densities) plus sum xa.U
    (C -> C(1 + U)) for an antisymmetric MO matrix xa; no dE_ROKS/dA.  The canonical constraint is added if
    td.damping is set (its multipliers see xa).'''
    log = logger.new_logger(td_grad)
    td = td_grad.base
    mf = td._scf
    mol = td_grad.mol
    mo = td.mo_coeff
    if atmlst is None:
        atmlst = range(mol.natm)
    occs = _occupations(td)
    fock = _fock_mo(mf, mo)
    vresp = _uks_response(mf, mo, *occs)
    xo = _orbital_lagrangian(fock, vresp, mo, occs, (p / 2, p / 2), dfterms, ghats) + xa
    if td.damping is not None:
        pz = _canonical_pz(xo, .5 * (fock[0] + fock[1]).diagonal(), td._orbs, log)
        p = p + pz
        xo = xo + _orbital_lagrangian(fock, vresp, mo, occs, (pz / 2, pz / 2), [], [])
    mask, hop, hdiag = _rohf_hessian(fock, vresp, mo, occs)
    z = solve_z(hop, hdiag, -(xo - xo.T)[mask], td_grad.cphf_conv_tol, td_grad.cphf_max_cycle, log)
    zz = _z_densities(z, mask, occs)
    xl = xo + _orbital_lagrangian(fock, vresp, mo, occs, zz, [], [])
    # non-redundant pairs: Z-vector residual; intra-space: a check for gradients, expected for transition quantities
    asym = abs(xl - xl.T)
    log.debug('max|X^L - X^L^T| = %.3g on non-redundant pairs, %.3g in all', asym[mask].max(), asym.max())
    w_ao = mo @ ((xl + xl.T) / 4) @ mo.T
    q_ao = lib.einsum('up,tpq,vq->tuv', mo, p / 2 + zz, mo)
    de = _fock_skeleton(mf, td.xc_coeffs, q_ao, numpy.asarray(mf.make_rdm1()), atmlst, td_grad.max_memory)
    if dfterms:
        de += _df_skeleton(mol, auxmol, mo, dfterms, ghats, atmlst)
    s1 = rhf_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] -= 2 * numpy.einsum('xij,ij->x', s1[:, p0:p1], w_ao[p0:p1])
    return de


def grad_elec(td_grad, spin, state, atmlst=None):
    '''omega part of the gradient of root `state` of spin `spin` (Lagrangian assembly; no dE_ROKS/dA).'''
    td = td_grad.base
    auxmol = _auxmol(td)
    dfterms = _df_terms(td_grad.mol, auxmol, td.mo_coeff, xtda_3t._kernel_terms(*td.xc_coeffs))
    p, ghats = _state_densities(td, spin, state, dfterms)
    return _relaxed(td_grad, p, ghats, auxmol, dfterms, atmlst=atmlst)


def _check_root(td, spin, state, log):
    '''Refuse a missing or collapsed root, warn on an unconverged one.'''
    xs = td._roots(spin)
    if not 0 <= state < len(xs):
        raise ValueError(f'state {state} out of range: S={spin} has {len(xs)} roots')
    if abs(numpy.linalg.norm(xs[state]) - 1) > 1e-6:
        raise RuntimeError(f'S={spin} root {state} has a non-normalised vector (Davidson subspace collapse)')
    if not td.converged[spin][state]:
        log.warn('XTDA-3T S=%d root %d not converged', spin, state)


class Gradients(rhf_grad.GradientsBase):
    '''Analytic nuclear gradients of one XTDA-3T root (total energy E_ROKS + omega).

    Attributes:
        state : int
            0-based root index within the spin block (as in the XTDA_3T log and transition_dm).
        spin : int
            Total spin S of the root (default: first entry of td.spins).
        cphf_conv_tol, cphf_max_cycle :
            Z-vector convergence settings.
    '''
    cphf_conv_tol = 1e-8
    cphf_max_cycle = 50
    _keys = {'state', 'spin', 'cphf_conv_tol', 'cphf_max_cycle'}

    def __init__(self, td):
        rhf_grad.GradientsBase.__init__(self, td)
        self.state = 0
        self.spin = td.spins[0]

    def dump_flags(self, verbose=None):
        rhf_grad.GradientsBase.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('S = %d  root %d  cphf_conv_tol = %g  cphf_max_cycle = %d',
                 self.spin, self.state, self.cphf_conv_tol, self.cphf_max_cycle)
        return self

    def grad_elec(self, spin=None, state=None, atmlst=None):
        return grad_elec(self, self.spin if spin is None else spin, self.state if state is None else state, atmlst)

    def kernel(self, state=None, spin=None, atmlst=None):
        td = self.base
        mf = td._scf
        if state is not None:
            self.state = state
        if spin is not None:
            self.spin = spin
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        state, spin = self.state, self.spin
        if isinstance(mf, dft.rks.KohnShamDFT) and mf.do_nlc():
            raise NotImplementedError('XTDA-3T gradients with NLC')
        if td.xs is None:
            td.kernel()
        log = logger.new_logger(self)
        _check_root(td, spin, state, log)
        w = td.omega[spin]
        near = [j for j in (state - 1, state + 1) if 0 <= j < len(w) and abs(w[j] - w[state]) < 1e-4]
        if near:
            log.warn('XTDA-3T S=%d root %d near-degenerate with root(s) %s (< 1e-4 Ha): the gradient of a single '
                     'root is ill-defined', spin, state, near)
        if self.verbose >= logger.INFO:
            self.dump_flags()
        mf_grad = mf.nuc_grad_method()
        mf_grad.verbose = min(self.verbose, logger.WARN)
        de = mf_grad.kernel(atmlst=atmlst) + self.grad_elec(spin, state, atmlst)
        if self.mol.symmetry:
            de = self.symmetrize(de, atmlst)
        self.de = de
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- XTDA-3T gradients for S=%d root %d ----------', self.spin, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def as_scanner(self, state=None, spin=None):
        '''Scanner: scanner(mol_or_geom) returns (e_tot, de) of the root (spin, state), tracked by index.'''
        if isinstance(self, lib.GradScanner):
            return self
        logger.info(self, 'Create scanner for %s', self.__class__)
        name = self.__class__.__name__ + XTDA_3T_GradScanner.__name_mixin__
        return lib.set_class(XTDA_3T_GradScanner(self, state, spin), (XTDA_3T_GradScanner, self.__class__), name)


class XTDA_3T_GradScanner(lib.GradScanner):
    def __init__(self, g, state=None, spin=None):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state
        if spin is not None:
            self.spin = spin

    def __call__(self, mol_or_geom, state=None, spin=None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        self.base(mol)
        de = self.kernel(state=state, spin=spin, **kwargs)
        return self.base.e_tot[self.spin][self.state], de

    @property
    def converged(self):
        td = self.base
        return bool(td._scf.converged and td.converged[self.spin][self.state])


Grad = Gradients
