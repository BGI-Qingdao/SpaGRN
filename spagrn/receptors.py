#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 06 Nov 2023 16:29
# @Author: Yao LI
# @File: spagrn/receptors.py

from .regulatory_network import InferNetwork

"""
Calculate receptor genes enrichment in cells.

Assess the AUCell implementation.
"""
import scanpy as sc
import frozendict
from ctxcore.genesig import GeneSignature, Regulon
from pyscenic.aucell import aucell, derive_auc_threshold, create_rankings


# --------------------------------------------------------------------------------------------------------------
# A module from GeneSigDB (C6)
# GMT_FNAME = 'signatures.hgnc.gmt'
# signatures = GeneSignature.from_gmt(GMT_FNAME, 'HGNC', field_separator='\t', gene_separator='\t')
# len(signatures)
#
# # An expression matrix from GEO
# EXPRESSION_MTX_FNAME = 'GSE103322.mtx.tsv'  # Gene expression as (cell, gene) - matrix.
#
# ex_matrix = pd.read_csv(EXPRESSION_MTX_FNAME, sep='\t', header=0, index_col=0).T
# print(ex_matrix.shape)  # (5902, 20684)
#
# df_rnk = create_rankings(ex_matrix)
# df_rnk.head()
#
# percentiles = derive_auc_threshold(ex_matrix)
# print(percentiles)
#
# aucs_mtx = aucell(ex_matrix, signatures, auc_threshold=percentiles[0.01], num_cores=8)
# print(aucs_mtx.head())

# Example 1: A single gene signature in a custom format
# --------------------------------------------------------------------------------------------------------------

# Receptors
# 1. get receptors
# modules = []
# receptor_genes = []  # ?
#
# receptor_signatures = list(
#     map(
#         lambda x: GeneSignature(
#             name=x.name,
#             gene2weight=x.gene2weight  # minus
#         ),
#         modules,
#     )
# )
#
# regulons = [Regulon(
#     name='TP53 regulon',
#     gene2weight={'TP53': 0.8, 'SOX4': 0.75},
#     transcription_factor="TP53",
#     gene2occurrence={"TP53": 1},
# )
# ]
# receptor_signatures = list(
#     map(
#         lambda x: GeneSignature(  # gs1 = GeneSignature(name="test1", gene2weight=['TP53', 'SOX4'])
#             name=x.name,
#             gene2weight=['TP53', 'SOX4']
#         ),
#         regulons,
#     )
# )


def frozen2regular(f_dir: frozendict.frozendict):
    regular_dict = {}  # {} and dir(), what's the big difference?
    for k, v in f_dir.items():
        regular_dict[k] = v
    return regular_dict


def zero_ratio(df) -> float:
    import pandas as pd
    d = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    m = d.eq(0) | d.isna()
    s = m.stack()
    indices = s[s].index.tolist()
    n = len(indices)
    return n


def zero_ratio2(df) -> float:
    n = sum(df.apply(lambda s: s.value_counts().get(key=0, default=0), axis=1))
    return n


def intersection_ci(iterableA, iterableB, key=lambda x: x) -> list:
    """
    Return the intersection of two iterables with respect to `key` function.
    (ci: case insensitive)
    :param iterableA: list no.1
    :param iterableB: list no.2
    :param key:
    :return:
    """

    def unify(iterable):
        d = {}
        for item in iterable:
            d.setdefault(key(item), []).append(item)
        return d

    A, B = unify(iterableA), unify(iterableB)
    matched = []
    for k in A:
        if k in B:
            matched.append(B[k][0])
    return matched


class DetectReceptor(InferNetwork):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    # adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/hetero.h5ad')
    # ex_matrix = adata.to_df()
    # modules = pickle.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/hotspot_danb/hotspot_regulons.pkl', 'rb'))
    # receptor_signatures = list(
    #     map(
    #         lambda x: GeneSignature(
    #             name=x.name,
    #             gene2weight=['TP53', 'SOX4']
    #         ),
    #         modules,
    #     )
    # )
    # receptor_auc_mtx = aucell(
    #     ex_matrix, receptor_signatures, num_workers=20
    # )  # (n_cells x n_regulons)
    # receptor_auc_mtx = receptor_auc_mtx.loc[ex_matrix.index]

    # receptors results
    fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/09.fly_rerun/E16-18h_pca/hotspot/hotspot_spagrn.h5ad'
    # fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver9/data2/hotspot/hotspot_spagrn.h5ad'
    adata = sc.read_h5ad(fn)
    # recetors = adata.uns['receptors']
    r_dir = adata.uns['receptor_dict']
    # modules = pickle.load(open('hotspot_modules.pkl', 'rb'))  # list of Regulons
    # 1. create new modules
    receptor_modules = list(
        map(
            lambda x: GeneSignature(
                name=x,
                gene2weight=r_dir[x],
            ),
            r_dir,
        )
    )
    ex_matrix = adata.to_df()
    percentiles = derive_auc_threshold(ex_matrix)
    receptor_auc_mtx = aucell(ex_matrix, receptor_modules, auc_threshold=percentiles[0.01], num_workers=20)
    # 0.9906666666666667 都是0
    print(receptor_auc_mtx)
    print(zero_ratio(receptor_auc_mtx))
