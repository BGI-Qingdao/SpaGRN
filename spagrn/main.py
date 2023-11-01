#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: test.py
@time: 2023/Mar/09
@description: test file for inference gene regulatory networks module
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

import os
import sys
sys.path.append('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/')
import argparse
import pandas as pd
from multiprocessing import cpu_count
from spagrn_debug.regulatory_network import InferNetwork as irn
import spagrn_debug.plot as prn
import scanpy as sc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spaGRN tester')
    parser.add_argument("--data", '-i', type=str, help='experiment data file, in h5ad/loom format')
    parser.add_argument("--tf", '-t', type=str, help='TF list file')
    parser.add_argument("--database", '-d', type=str, help='ranked motifs database file, in feather format')
    parser.add_argument("--motif_anno", '-m', type=str, help='motifs annotation file, in tbl format')
    parser.add_argument("--method", type=str, default='grnboost', choices=['grnboost', 'hotspot', 'scoexp'], help='method to calculate TF-gene similarity')
    parser.add_argument("--output", '-o', type=str, help='output directory')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    fn = args.data
    tfs_fn = args.tf
    database_fn = args.database
    motif_anno_fn = args.motif_anno
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    method = args.method
    prefix = os.path.join(out_dir, method)

    # load data
    data = irn.read_file(fn)
    data = irn.preprocess(data)
    sc.tl.pca(data)

    # create grn
    grn = irn(data)

    # set parameters
    grn.add_params('hotspot', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('scoexp', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('grnboost', {'prune_auc_threshold': 0.05, 'rank_threshold': 3000, 'auc_threshold': 0.05})

    # run analysis
    grn.main(database_fn,
             motif_anno_fn,
             tfs_fn,
             num_workers=cpu_count(),
             cache=False,
             save_tmp=True,
             c_threshold=0.2,
             layers=None,
             latent_obsm_key='spatial',
             model='danb',  #bernoulli
             n_neighbors=30,
             weighted_graph=False,
             cluster_label='celltype',
             method=method,
             prefix=prefix,
             noweights=False,
             rho_mask_dropouts=False)

    # PLOTing
    # auc_mtx = pd.read_csv(f'{out_dir}/{method}_auc.csv',index_col=0)
    # # remove all zero columns (which have no variation at all)
    # auc_mtx = auc_mtx.loc[:, (auc_mtx != 0).any(axis=0)]
    #
    # prn.auc_heatmap(data,
    #         auc_mtx,
    #         cluster_label='celltype',
    #         rss_fn=f'{out_dir}/{method}_regulon_specificity_scores.txt',
    #         topn=10,
    #         subset=False,
    #         save=True,
    #         fn=f'{out_dir}/{method}_clusters_heatmap_top20.png',
    #         legend_fn=f"{out_dir}/{method}_rss_celltype_legend_top20.png")
    #
    # regs = grn.regulons
    # for reg in list(regs.keys()):
    #     print(f'plotting {reg}')
    #     prn.plot_2d(grn.data, 'spatial', grn.auc_mtx, reg_name=reg, fn=f'{reg.strip("(+)")}.png')

