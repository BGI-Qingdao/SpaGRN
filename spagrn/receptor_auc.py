#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 19 Sep 2023 15:13
# @Author: Yao LI
# @File: spagrn/receptor_auc.py

"""
Calculate receptor genes enrichment in cells.

Assess the AUCell implementation.
"""

import pandas as pd
import pickle
import json
from ctxcore.genesig import GeneSignature, Regulon
from pyscenic.aucell import aucell, derive_auc_threshold, create_rankings
import scanpy as sc

#--------------------------------------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------------------------------

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

if __name__ == '__main__':
    adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/hetero.h5ad')
    ex_matrix = adata.to_df()

    modules = pickle.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/hotspot_danb/hotspot_regulons.pkl', 'rb'))
    receptors = json.load(open(''))

    receptor_signatures = list(
        map(
            lambda x: GeneSignature(
                name=x.name,
                gene2weight=['TP53', 'SOX4']
            ),
            modules,
        )
    )

    receptor_auc_mtx = aucell(
        ex_matrix, receptor_signatures, num_workers=20
    )  # (n_cells x n_regulons)
    receptor_auc_mtx = receptor_auc_mtx.loc[ex_matrix.index]
    print(receptor_auc_mtx)

