#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 15 Aug 2023 15:58
# @Author: Yao LI
# @File: spagrn/ts.py

import scanpy as sc
import json
import pandas as pd
import numpy as np
import sys
import anndata as ad

auc_mtx = pd.read_csv(sys.argv[1], index_col=0)
adata = sc.read_h5ad(sys.argv[2])
receptors = json.load(open(sys.argv[3]))

matrix = adata.to_df()

# drop regulons do not have receptors
receptor_tf = [f'{i}(+)' for i in list(receptors.keys())]
rtf = auc_mtx[list(receptor_tf)].copy()

# multiply sum of the receptor exp values to regulon auc value
sub_matrix = matrix.loc[auc_mtx.index]
for regulon in receptors.keys():
    rep_sum = sub_matrix[receptors[regulon]].sum(axis='columns')  # regulon string does not contain (+)
    rtf[f'{regulon}(+)'] = rep_sum * rtf[f'{regulon}(+)']

# rtf.to_csv('')

regulons = json.load(open('hotspot_regulons.json'))


def to_h5ad(Xdata, regulons, raw_adata, spatialkey='spatial', annokey='annotation'):
    grn_adata = ad.AnnData(X=Xdata.to_numpy(), dtype=float)
    grn_adata.obs_names = Xdata.index.to_list()
    grn_adata.var_names = Xdata.columns
    for reg in regulons:
        grn_adata.uns[reg] = regulons[reg]
    raw_adata = raw_adata[grn_adata.obs_names, :]
    grn_adata.obsm['spatial'] = raw_adata.obsm[spatialkey].copy()  # xyz
    grn_adata.obs['annotation'] = raw_adata.obs[annokey]
    # adata.write(f'{prefix}.h5ad',compression='gzip')
    return grn_adata


grn_adata = to_h5ad(rtf, regulons, adata)
sc.tl.pca(grn_adata, svd_solver='arpack')
sc.pp.neighbors(grn_adata)
sc.tl.leiden(grn_adata)

sc.tl.umap(grn_adata)
sc.pl.umap(grn_adata, color='leiden')
