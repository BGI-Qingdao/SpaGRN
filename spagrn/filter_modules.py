#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 14 Aug 2023 07:48
# @Author: Yao LI
# @File: spagrn/filter_modules.py
import sys
import scanpy as sc
import json
import pandas as pd
from pyscenic.utils import modules_from_adjacencies

'''
python filter_modules.py adj.csv data.h5ad regulons.json
'''


def intersection_ci(iterableA, iterableB, key=lambda x: x):
    """Return the intersection of two iterables with respect to `key` function.
    ci: case insensitive
    """

    def unify(iterable):
        d = {}
        for item in iterable:
            d.setdefault(key(item), []).append(item)
        return d

    A, B = unify(iterableA), unify(iterableB)
    # return [(A[k], B[k]) for k in A if k in B]
    matched = []
    for k in A:
        if k in B:
            matched.append(B[k][0])
    return matched


adjacencies = pd.read_csv(sys.argv[1])
# fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/L3_pca.h5ad'
adata = sc.read_h5ad(sys.argv[2])
matrix = adata.to_df()
regulons = json.load(open(sys.argv[3]))
prefix = sys.argv[4]

modules = list(
    modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=False)
)

module_tf = []
for i in modules:
    module_tf.append(i.transcription_factor)

final_tf = [i.strip('(+)') for i in list(regulons.keys())]
com = set(final_tf).intersection(set(module_tf))

before_tf = {}
for tf in com:
    before_tf[tf] = []
    for i in modules:
        if tf == i.transcription_factor:
            before_tf[tf] += list(i.genes)  # .remove(tf)

filtered = {}
for tf in com:
    final_targets = regulons[f'{tf}(+)']
    before_targets = set(before_tf[tf])
    filtered_targets = before_targets - set(final_targets)
    print(tf)
    print(len(filtered_targets))
    if tf in filtered_targets:
        filtered_targets.remove(tf)
    filtered[tf] = list(filtered_targets)

with open(f'{prefix}_filtered_targets.json', 'w') as fp:
    json.dump(filtered, fp)

niche_human = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_human.csv')
niche_mouse = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_mouse.csv')

receptor_tf = {}
total_receptor = set()
for tf, targets in filtered.items():
    # trg = [i.upper() for i in targets]
    # rtf1 = set(niche_human['to']).intersection(set(trg))
    # rtf2 = set(niche_mouse['to']).intersection(set(targets))
    # rtf1 = set([word[0] + word[1:].lower() for word in rtf1])
    rtf1 = intersection_ci(set(niche_human['to']), set(targets), key=str.lower)
    rtf2 = intersection_ci(set(niche_mouse['to']), set(targets), key=str.lower)
    rtf = set(rtf1) | set(rtf2)
    if len(rtf) > 0:
        receptor_tf[tf] = list(rtf)
        total_receptor = total_receptor | rtf
print(total_receptor)

with open(f'{prefix}_filtered_targets_receptor.json', 'w') as fp:
    json.dump(receptor_tf, fp)

with open(f'{prefix}_filtered_targets_receptor_total.txt', 'w') as f:
    f.writelines('\n'.join(list(total_receptor)))


class StrIgnoreCase:
    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, StrIgnoreCase):
            return False

        return self.val.lower() == other.val.lower()

    def __hash__(self):
        return hash(self.val.lower())
