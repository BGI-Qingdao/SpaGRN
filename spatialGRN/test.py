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
import json
import argparse
import pandas as pd
from multiprocessing import cpu_count
from regulatory_network import InferenceRegulatoryNetwork as irn
from plot import PlotRegulatoryNetwork as prn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spaGRN tester')
    parser.add_argument("--data", '-i', type=str, help='experiment data file, in h5ad/loom format')
    parser.add_argument("--tf", '-t', type=str, help='TF list file')
    parser.add_argument("--database", '-d', type=str, help='ranked motifs database file, in feather format')
    parser.add_argument("--motif_anno", '-m', type=str, help='motifs annotation file, in tbl format')
    parser.add_argument("--method", type=str, default='grnboost', choices=['grnboost', 'hotspot', 'scoexp'], help='method to calculate TF-gene similarity')
    parser.add_argument("--output", '-o', type=str, help='output directory')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # supporting files
    #tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/tfs/mm_mgi_tfs.txt'
    # tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/tfs/test_mm_mgi_tfs.txt'
    #database_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/database/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    #motif_anno_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/motifs/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'
    # clustering output for stereopy data]
    #meta_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/meta_mousebrain.csv'
    # h5ad data
    #fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/MouseBrainCellbin.h5ad'
    # data = InferenceRegulatoryNetwork.load_anndata_by_cluster(fn, 'psuedo_class', ['HBGLU9', 'TEINH12', 'HBSER4'])
    # alternative: small h5ad for testing
    # fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/test_grn_MouseBrainCellbin.h5ad'
    # stereopy data
    # fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/SS200000135TL_D1.cellbin.gef'
    # fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/SS200000135TL_D1.raw.gef'

    # # 2023-03-18
    # tfs = InferenceRegulatoryNetwork.load_tfs(tfs_fn)
    # data_genes = data.var_names
    # common_tfs = set(tfs).intersection(set(data_genes))
    # data = data[:, list(common_tfs)]
    # data = read_gef(fn, bin_size=200)

    fn = args.data
    tfs_fn = args.tf
    database_fn = args.database
    motif_anno_fn = args.motif_anno
    out_dir = args.output
    method = args.method
    prefix = os.path.join(out_dir, method)

    # load data
    data = irn.read_file(fn)
    data = irn.preprocess(data)

    # create grn
    grn = irn(data)
    grn_plot = prn(data)

    # run analysis
    grn.main(database_fn,
             motif_anno_fn,
             tfs_fn,
             num_workers=cpu_count(),
             cache=False,
             save=True,
             method=method,
             prefix=prefix)

    grn_plot.auc_heatmap(grn.auc_mtx, fn=f'{method}_auc_heatmap.png')
    # grn.main(database_fn, motif_anno_fn, tfs_fn, num_workers=cpu_count(), cache=False, save=True, method='hotspot', prefix='hotspot')
    # grn_plot.auc_heatmap(grn.auc_mtx, fn='hotspot_auc_heatmap.png')
    # grn.main(database_fn, motif_anno_fn, tfs_fn, num_workers=cpu_count(), cache=False, save=True, method='scoexp', prefix='scoexp')
    # grn_plot.auc_heatmap(grn.auc_mtx, fn='scoexp_auc_heatmap.png')


    #grn_plot.dotplot_anndata(data, grn.gene_names, cluster_label='psuedo_class')
    #grn_plot.plot_2d_reg_h5ad(data, 'spatial', grn.auc_mtx, 'Zfp354c')
