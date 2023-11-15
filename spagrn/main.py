#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: main.py
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
    parser.add_argument("--method", type=str, default='grnboost', choices=['grnboost', 'spg', 'scc'], help='method to calculate TF-gene similarity')
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
    grn.add_params('spg', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('scc', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('grnboost', {'prune_auc_threshold': 0.05, 'rank_threshold': 3000, 'auc_threshold': 0.05})

    # niche data
    niche_human = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_human.csv')
    niche_mouse = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_mouse.csv')
    niches = pd.concat([niche_mouse, niche_human])

    # run analysis
    grn.infer(database_fn,
              motif_anno_fn,
              tfs_fn,
              niche_df=niches,
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

