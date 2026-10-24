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
Analytic nuclear gradients of one SF-XTDA root, E = E_ROKS + omega.

omega = x.(H - E0).x is a Rayleigh quotient, so the root vector x needs no response.  H - E0 is linear in
the spin-averaged Fock matrix Fbar and in the integrals [pq|rs] = sum_t c_t (pq|rs)_t of the kernel w:

    omega = tr(P Fbar) + sum Gamma_pqrs [pq|rs],   P = gamma^{kk} - gamma0   (gamma0 = 2 on C, 1 on O).

The two-body part is contracted with own DF fits (pyscf.sfxtda._rdm2), T_t = (P|pq)_t, d_t = j2c_t^-1 T_t:

    Ghat_t = 2 c_t Y[d_t],   omega_2e = 1/2 sum_t <Ghat_t, T_t>,   M_t = sum_pq Ghat_t^P d_t^Q.

Orbital relaxation: with spin densities Q_t (MO) and C -> C(1 + U)

    X_rp = sum_t [2 (F_t Q_t)_rp + 2 V_t[Q]_rp n_t,p] + 2 sum_t sum_P (T_t^P Ghat_t^P)_rp,

Q_t = P/2 for omega alone.  A Z-vector equation with the exact ROHF/ROKS orbital Hessian on the
non-redundant pairs (VC, VO, OC) gives Z_t; Q_t = P/2 + Z_t, W = (X^L + X^L^T)/4 and

    dE/dA = dE_ROKS/dA + d/dA sum_t tr(Q_t F_t[D])|skeleton + own DF derivative - tr(W_AO S^(A)).

Only the undamped model is supported; grid response of the XC terms is omitted (as in pyscf.grad.tduks).
'''

import numpy
from pyscf import lib, df
from pyscf.lib import logger
from pyscf.df.grad import rhf as df_grad
from pyscf.sfxtda import sfxtda, _rdm2


def _occupations(td):
    core, somo, _ = td._orbs
    nmo = td._scf.mo_coeff.shape[1]
    na, nb = numpy.zeros(nmo), numpy.zeros(nmo)
    na[core] = na[somo] = nb[core] = 1
    return na, nb


def _fock_mo(mf):
    mo = mf.mo_coeff
    f = mf.get_fock()
    return numpy.array([mo.T @ f.focka @ mo, mo.T @ f.fockb @ mo])


def _uks_response(mf, na, nb):
    # ROKS gen_response feeds the 2D ROHF mo_coeff/mo_occ to the UKS XC kernel (wrong density): pass UKS-shaped ones
    mo = mf.mo_coeff
    return mf.gen_response(mo_coeff=(mo, mo), mo_occ=(na, nb), hermi=1)


def _pdiff(td, spin, state):
    core, somo, _ = td._orbs
    g0 = numpy.zeros(td._scf.mo_coeff.shape[1])
    g0[core], g0[somo] = 2, 1
    return td.transition_dm(spin, state, state) - numpy.diag(g0)


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


def _two_body(td, x, spin, dfterms):
    core, somo, virt = td._orbs
    blocks = sfxtda.csf_list(spin, len(core), len(virt))
    return [2 * c * _rdm2.contract(x, blocks, d, core, somo, virt) for c, _, _, d in dfterms]


def _orbital_lagrangian(fock, vresp, mo, occs, q, dfterms, ghats):
    '''X_rp = d omega / d U_rp for spin densities q = (Q_alpha, Q_beta) (MO); Fock part only without DF terms.'''
    v = vresp(numpy.array([mo @ qt @ mo.T for qt in q]))
    x = 0
    for t in range(2):
        x = x + 2 * (fock[t] @ q[t] + (mo.T @ v[t] @ mo) * occs[t][None, :])
    for (_, _, tt, _), g in zip(dfterms, ghats):
        x = x + 2 * lib.einsum('Prs,Psp->rp', tt, g)
    return x
