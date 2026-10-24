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

weighted_dm contracts Delta with a general symmetric CSF weight W instead of x^I x^J (the damped-model gradient):
dense blocks between small (OO, CO, OV) CSFs and from small to CV CSFs, and a diagonal on each CV-CV block, for
which every ket label is renamed to the bra label (j -> i, b -> a).
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


def _ren(kind, deltas):
    '''Ket -> bra label renaming of one term: by its deltas, or all labels for a diagonal weight.'''
    if kind == 'diag':
        return {'j': 'i', 'b': 'a'}
    ren = {'j': 'i'} if 'ij' in deltas else {}
    if 'ab' in deltas:
        ren['b'] = 'a'
    return ren


def _wsub(kind, A, ket):
    '''einsum subscripts of the CSF weight: outer product (u, v), dense block (w,) or CV diagonal (g,).'''
    return {'outer': f'{_BRA[A]},{ket}', 'dense': _BRA[A] + ket, 'diag': _BRA[A]}[kind]


def _add(g, kind, ops, A, B, terms, orb):
    '''g[r, s] += sum c * W_(bra indices, ket indices) for the terms of one CSF-type pair (weight see _wsub).'''
    for c, r, s, d in terms:
        ren = _ren(kind, d)
        ket = ''.join(ren.get(k, k) for k in _KET[B])
        out = ''.join(ren.get(k, k) for k in (r, s) if k in 'ijab')
        if len(out) == 2 and out[0] == out[1]:
            # repeated output label (diagonal weight only): numpy einsum cannot write a diagonal
            idx = orb[r]
            g[idx, idx] += c * numpy.einsum(f'{_wsub(kind, A, ket)}->{out[0]}', *ops)
            continue
        blk = numpy.einsum(f'{_wsub(kind, A, ket)}->{out}', *ops)
        g[numpy.ix_(orb[r], orb[s])] += c * blk.reshape(len(orb[r]), len(orb[s]))


def _layout(blocks, core, somo, virt):
    nC, nV = len(core), len(virt)
    orb = {'i': core, 'j': core, 'a': virt, 'b': virt, 'O1': somo[:1], 'O2': somo[1:]}
    shape = {'OO': (), 'CO': (nC,), 'OV': (nV,), 'CV': (nC, nV)}
    off = numpy.cumsum([0] + [r * c for _, _, (r, c) in blocks])
    return orb, shape, off


def weight_blocks(wss, wsl, gl, blocks, core, somo, virt):
    '''(p, q, kind, ops) of a symmetric CSF weight W given by its small-small block wss, small-large block wsl and
    the diagonal gl of the large-large block (layout of hss, hsl and hdiag_large); pairs in csf_list order.'''
    _, shape, off = _layout(blocks, core, somo, virt)
    ns = sum(r * c for cl, _, (r, c) in blocks if cl != 'CV')
    out = []
    for p, (A, _, _) in enumerate(blocks):
        for q, (B, _, _) in enumerate(blocks):
            if A == 'CV' and B == 'CV':
                if p == q:
                    out.append((p, q, 'diag', (gl[off[p]-ns:off[p+1]-ns].reshape(shape[A]),)))
            elif A != 'CV':
                w = wss[off[p]:off[p+1], off[q]:off[q+1]] if B != 'CV' else wsl[off[p]:off[p+1], off[q]-ns:off[q+1]-ns]
                out.append((p, q, 'dense', (w.reshape(shape[A] + shape[B]),)))
    return out


def weighted_dm(wss, wsl, gl, blocks, core, somo, virt):
    '''sum_{mu nu} W_{mu nu} Delta_{mu nu, rs} of a symmetric CSF weight W (see weight_blocks); symmetric, no
    tr(W) gamma0 term.'''
    orb, _, _ = _layout(blocks, core, somo, virt)
    nmo = len(core) + len(somo) + len(virt)
    g = numpy.zeros((nmo, nmo))
    for p, q, kind, ops in weight_blocks(wss, wsl, gl, blocks, core, somo, virt):
        terms = COUPLINGS.get(blocks[p][:2] + blocks[q][:2])
        if terms is None:
            continue
        gpq = numpy.zeros((nmo, nmo))
        _add(gpq, kind, ops, blocks[p][0], blocks[q][0], terms, orb)
        # W symmetric: the reverse pair <nu|E_rs|mu> W_nu,mu adds the transpose
        g += gpq if p == q else gpq + gpq.T
    return g


def transition_dm(xbra, xket, blocks, core, somo, virt):
    '''dm[p,q] = <bra|a+_q a_p|ket> (MO basis, PySCF fci convention) of two CSF vectors of one spin.

    blocks is csf_list(spin, nC, nV); bra == ket gives the state density.
    '''
    orb, shape, off = _layout(blocks, core, somo, virt)
    nmo = len(core) + len(somo) + len(virt)
    ub = [xbra[off[k]:off[k+1]].reshape(shape[cl]) for k, (cl, _, _) in enumerate(blocks)]
    uk = [xket[off[k]:off[k+1]].reshape(shape[cl]) for k, (cl, _, _) in enumerate(blocks)]
    g = numpy.zeros((nmo, nmo))
    gt = numpy.zeros((nmo, nmo))
    for p, (A, na, _) in enumerate(blocks):
        for q, (B, nb, _) in enumerate(blocks):
            terms = COUPLINGS.get((A, na, B, nb))
            if terms is None:
                continue
            _add(g, 'outer', (ub[p], uk[q]), A, B, terms, orb)
            if p != q:
                # bra of type B, ket of type A: <nu|E_sr|mu> = <mu|E_rs|nu>
                _add(gt, 'outer', (uk[p], ub[q]), A, B, terms, orb)
    g += gt.T
    ov = xbra @ xket
    g[core, core] += 2 * ov
    g[somo, somo] += ov
    return g.T
