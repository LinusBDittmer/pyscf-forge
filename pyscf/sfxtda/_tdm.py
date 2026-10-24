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
One-particle (transition) density matrices of SF-XTDA roots in the CSF basis.

For two roots |I> = sum_mu x^I_mu |mu>, |J> of the same spin S (E_rs = sum_sigma a+_rs a_ss is a spin
scalar, so different S never couple and the M_S = 0 components suffice)

    gamma^IJ_rs = <I|E_rs|J> = (x^I . x^J) gamma0_rs + sum_{mu nu} x^I_mu x^J_nu Delta_{mu nu, rs},
    Delta_{mu nu, rs} = <mu|E_rs|nu> - delta_{mu nu} gamma0_rs,   gamma0 = 2 on C, 1 on O1, O2.

Delta is the coefficient of F_pq in the _blocks.py formula <mu|H-E0|nu> (set every integral to zero:
F = h' and <mu|H-E0|nu> = sum_rs h'_rs Delta_{mu nu, rs}); F is symmetric, and of E_pq, E_qp only the
one with r = orbital more occupied in the bra survives.  COUPLINGS lists every nonzero Delta per pair of
CSF types (same key order as BLOCKS, the reverse pair follows from <nu|E_sr|mu> = <mu|E_rs|nu>) as
(c, r, s, deltas): Delta = c * deltas.  Labels i, a: bra hole/particle; j, b: ket; O1, O2: SOMOs.
'''

import numpy

R2, H, T = numpy.sqrt(2.), numpy.sqrt(.5), numpy.sqrt(1.5)

COUPLINGS = {
    # OO - OO
    ('OO', 'G', 'OO', 'G'): [(1, 'O1', 'O1', ''), (-1, 'O2', 'O2', '')],
    ('OO', 'G', 'OO', '1OO'): [(R2, 'O1', 'O2', '')],
    ('OO', 'D', 'OO', 'D'): [(-1, 'O1', 'O1', ''), (1, 'O2', 'O2', '')],
    ('OO', 'D', 'OO', '1OO'): [(R2, 'O2', 'O1', '')],
    # OO - CO
    ('OO', 'G', 'CO', '1CO:O1^2'): [(R2, 'j', 'O2', '')],
    ('OO', 'D', 'CO', '1CO:O2^2'): [(R2, 'j', 'O1', '')],
    ('OO', '1OO', 'CO', '1CO:O1^2'): [(-1, 'j', 'O1', '')],
    ('OO', '1OO', 'CO', '1CO:O2^2'): [(-1, 'j', 'O2', '')],
    ('OO', '3OO', 'CO', '3CO:O1^2'): [(-1, 'j', 'O1', '')],
    ('OO', '3OO', 'CO', '3CO:O2^2'): [(1, 'j', 'O2', '')],
    # OO - OV
    ('OO', 'G', 'OV', '1OV:O1^1'): [(R2, 'O1', 'b', '')],
    ('OO', 'D', 'OV', '1OV:O2^1'): [(R2, 'O2', 'b', '')],
    ('OO', '1OO', 'OV', '1OV:O1^1'): [(1, 'O2', 'b', '')],
    ('OO', '1OO', 'OV', '1OV:O2^1'): [(1, 'O1', 'b', '')],
    ('OO', '3OO', 'OV', '3OV:O1^1'): [(1, 'O2', 'b', '')],
    ('OO', '3OO', 'OV', '3OV:O2^1'): [(-1, 'O1', 'b', '')],
    # OO - CV
    ('OO', 'G', 'CV', '1CV:O1^2'): [(R2, 'j', 'b', '')],
    ('OO', 'D', 'CV', '1CV:O2^2'): [(R2, 'j', 'b', '')],
    ('OO', '1OO', 'CV', '[00]0'): [(R2, 'j', 'b', '')],
    ('OO', '3OO', 'CV', '[01]1'): [(R2, 'j', 'b', '')],
    # CO - CO
    ('CO', '1CO:O1^2', 'CO', '1CO:O1^2'): [(1, 'O1', 'O1', 'ij'), (-1, 'j', 'i', '')],
    ('CO', '1CO:O1^2', 'CO', '1CO:O2^2'): [(-1, 'O1', 'O2', 'ij')],
    ('CO', '3CO:O1^2', 'CO', '3CO:O1^2'): [(1, 'O1', 'O1', 'ij'), (-1, 'j', 'i', '')],
    ('CO', '3CO:O1^2', 'CO', '3CO:O2^2'): [(-1, 'O1', 'O2', 'ij')],
    ('CO', '1CO:O2^2', 'CO', '1CO:O2^2'): [(1, 'O2', 'O2', 'ij'), (-1, 'j', 'i', '')],
    ('CO', '3CO:O2^2', 'CO', '3CO:O2^2'): [(1, 'O2', 'O2', 'ij'), (-1, 'j', 'i', '')],
    # CO - CV
    ('CO', '1CO:O1^2', 'CV', '[00]0'): [(-H, 'O1', 'b', 'ij')],
    ('CO', '1CO:O1^2', 'CV', '[11]0'): [(T, 'O1', 'b', 'ij')],
    ('CO', '1CO:O1^2', 'CV', '1CV:O1^2'): [(1, 'O2', 'b', 'ij')],
    ('CO', '3CO:O1^2', 'CV', '[11]1'): [(1, 'O1', 'b', 'ij')],
    ('CO', '3CO:O1^2', 'CV', '[01]1'): [(-H, 'O1', 'b', 'ij')],
    ('CO', '3CO:O1^2', 'CV', '[10]1'): [(-H, 'O1', 'b', 'ij')],
    ('CO', '3CO:O1^2', 'CV', '3CV:O1^2'): [(1, 'O2', 'b', 'ij')],
    ('CO', '1CO:O2^2', 'CV', '[00]0'): [(-H, 'O2', 'b', 'ij')],
    ('CO', '1CO:O2^2', 'CV', '[11]0'): [(-T, 'O2', 'b', 'ij')],
    ('CO', '1CO:O2^2', 'CV', '1CV:O2^2'): [(1, 'O1', 'b', 'ij')],
    ('CO', '3CO:O2^2', 'CV', '[11]1'): [(-1, 'O2', 'b', 'ij')],
    ('CO', '3CO:O2^2', 'CV', '[01]1'): [(H, 'O2', 'b', 'ij')],
    ('CO', '3CO:O2^2', 'CV', '[10]1'): [(-H, 'O2', 'b', 'ij')],
    ('CO', '3CO:O2^2', 'CV', '3CV:O2^2'): [(1, 'O1', 'b', 'ij')],
    # OV - OV
    ('OV', '1OV:O1^1', 'OV', '1OV:O1^1'): [(-1, 'O2', 'O2', 'ab'), (1, 'a', 'b', '')],
    ('OV', '1OV:O1^1', 'OV', '1OV:O2^1'): [(1, 'O1', 'O2', 'ab')],
    ('OV', '3OV:O1^1', 'OV', '3OV:O1^1'): [(-1, 'O2', 'O2', 'ab'), (1, 'a', 'b', '')],
    ('OV', '3OV:O1^1', 'OV', '3OV:O2^1'): [(1, 'O1', 'O2', 'ab')],
    ('OV', '1OV:O2^1', 'OV', '1OV:O2^1'): [(-1, 'O1', 'O1', 'ab'), (1, 'a', 'b', '')],
    ('OV', '3OV:O2^1', 'OV', '3OV:O2^1'): [(-1, 'O1', 'O1', 'ab'), (1, 'a', 'b', '')],
    # OV - CV
    ('OV', '1OV:O1^1', 'CV', '[00]0'): [(-H, 'j', 'O2', 'ab')],
    ('OV', '1OV:O1^1', 'CV', '[11]0'): [(T, 'j', 'O2', 'ab')],
    ('OV', '1OV:O1^1', 'CV', '1CV:O1^2'): [(-1, 'j', 'O1', 'ab')],
    ('OV', '3OV:O1^1', 'CV', '[11]1'): [(-1, 'j', 'O2', 'ab')],
    ('OV', '3OV:O1^1', 'CV', '[01]1'): [(-H, 'j', 'O2', 'ab')],
    ('OV', '3OV:O1^1', 'CV', '[10]1'): [(-H, 'j', 'O2', 'ab')],
    ('OV', '3OV:O1^1', 'CV', '3CV:O1^2'): [(-1, 'j', 'O1', 'ab')],
    ('OV', '1OV:O2^1', 'CV', '[00]0'): [(-H, 'j', 'O1', 'ab')],
    ('OV', '1OV:O2^1', 'CV', '[11]0'): [(-T, 'j', 'O1', 'ab')],
    ('OV', '1OV:O2^1', 'CV', '1CV:O2^2'): [(-1, 'j', 'O2', 'ab')],
    ('OV', '3OV:O2^1', 'CV', '[11]1'): [(1, 'j', 'O1', 'ab')],
    ('OV', '3OV:O2^1', 'CV', '[01]1'): [(H, 'j', 'O1', 'ab')],
    ('OV', '3OV:O2^1', 'CV', '[10]1'): [(-H, 'j', 'O1', 'ab')],
    ('OV', '3OV:O2^1', 'CV', '3CV:O2^2'): [(-1, 'j', 'O2', 'ab')],
    # CV - CV: hole/particle hopping on every diagonal type, SOMO shift on the Ox^2 types
    ('CV', '[00]0', 'CV', '1CV:O1^2'): [(R2, 'O2', 'O1', 'ijab')],
    ('CV', '[00]0', 'CV', '1CV:O2^2'): [(R2, 'O1', 'O2', 'ijab')],
    ('CV', '[10]1', 'CV', '3CV:O1^2'): [(R2, 'O2', 'O1', 'ijab')],
    ('CV', '[10]1', 'CV', '3CV:O2^2'): [(R2, 'O1', 'O2', 'ijab')],
}
_HOP = [(1, 'a', 'b', 'ij'), (-1, 'j', 'i', 'ab')]
for _nm in ('[00]0', '[11]0', '[11]1', '[01]1', '[10]1', '[11]2'):
    COUPLINGS[('CV', _nm, 'CV', _nm)] = _HOP
for _nm, _x, _y in (('1CV:O1^2', 'O1', 'O2'), ('3CV:O1^2', 'O1', 'O2'),
                    ('1CV:O2^2', 'O2', 'O1'), ('3CV:O2^2', 'O2', 'O1')):
    COUPLINGS[('CV', _nm, 'CV', _nm)] = _HOP + [(1, _x, _x, 'ijab'), (-1, _y, _y, 'ijab')]

# einsum subscripts of a CSF-type block on the bra and the ket side
_BRA = {'OO': '', 'CO': 'i', 'OV': 'a', 'CV': 'ia'}
_KET = {'OO': '', 'CO': 'j', 'OV': 'b', 'CV': 'jb'}


def _add(g, u, v, A, B, terms, orb):
    '''g[r, s] += sum c * u_(bra indices) v_(ket indices) for the terms of one CSF-type pair.'''
    for c, r, s, d in terms:
        ren = {'j': 'i'} if 'ij' in d else {}
        if 'ab' in d:
            ren['b'] = 'a'
        ket = ''.join(ren.get(k, k) for k in _KET[B])
        out = ''.join(ren.get(k, k) for k in (r, s) if k in 'ijab')
        blk = numpy.einsum(f'{_BRA[A]},{ket}->{out}', u, v)
        g[numpy.ix_(orb[r], orb[s])] += c * blk.reshape(len(orb[r]), len(orb[s]))


def transition_dm(xbra, xket, blocks, core, somo, virt):
    '''dm[p,q] = <bra|a+_q a_p|ket> (MO basis, PySCF fci convention) of two CSF vectors of one spin.

    blocks is csf_list(spin, nC, nV); bra == ket gives the state density.
    '''
    nC, nV = len(core), len(virt)
    nmo = nC + len(somo) + nV
    orb = {'i': core, 'j': core, 'a': virt, 'b': virt, 'O1': somo[:1], 'O2': somo[1:]}
    shape = {'OO': (), 'CO': (nC,), 'OV': (nV,), 'CV': (nC, nV)}
    off = numpy.cumsum([0] + [r * c for _, _, (r, c) in blocks])
    ub = [xbra[off[k]:off[k+1]].reshape(shape[cl]) for k, (cl, _, _) in enumerate(blocks)]
    uk = [xket[off[k]:off[k+1]].reshape(shape[cl]) for k, (cl, _, _) in enumerate(blocks)]
    g = numpy.zeros((nmo, nmo))
    gt = numpy.zeros((nmo, nmo))
    for p, (A, na, _) in enumerate(blocks):
        for q, (B, nb, _) in enumerate(blocks):
            terms = COUPLINGS.get((A, na, B, nb))
            if terms is None:
                continue
            _add(g, ub[p], uk[q], A, B, terms, orb)
            if p != q:
                # bra of type B, ket of type A: <nu|E_sr|mu> = <mu|E_rs|nu>
                _add(gt, uk[p], ub[q], A, B, terms, orb)
    g += gt.T
    ov = xbra @ xket
    g[core, core] += 2 * ov
    g[somo, somo] += ov
    return g.T
