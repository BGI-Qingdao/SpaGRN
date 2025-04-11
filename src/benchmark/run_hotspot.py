#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 27 Sep 2023 09:43
# @Author: Yao LI
# @File: spagrn/hotspot.py
import sys

import hotspot
import scanpy as sc
import pandas as pd
import seaborn as sns
import pickle
import json
from typing import Union


def get_cluster_label(g: sns.matrix.ClusterGrid, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract dendrogram x and y labels
    :param g:
    :param df:
    :return:
    """
    import scipy
    den = scipy.cluster.hierarchy.dendrogram(g.dendrogram_col.linkage,
                                             labels=df.index,
                                             color_threshold=0.60)

    def get_cluster_classes(den: dict, label='ivl'):
        from collections import defaultdict
        cluster_idxs = defaultdict(list)
        for c, pi in zip(den['color_list'], den['icoord']):
            for leg in pi[1:3]:
                i = (leg - 5.0) / 10.0
                if abs(i - int(i)) < 1e-5:
                    cluster_idxs[c].append(int(i))

        cluster_classes = {}
        for c, l in cluster_idxs.items():
            i_l = [den[label][i] for i in l]
            cluster_classes[c] = i_l

        return cluster_classes

    clusters = get_cluster_classes(den)

    cluster = []
    for i in df.index:
        included = False
        for j in clusters.keys():
            if i in clusters[j]:
                cluster.append(j)
                included = True
        if not included:
            cluster.append(None)

    df["cluster"] = cluster
    return df


def get_module_cols(adata, module_scores: pd.DataFrame) -> list:
    """

    :param adata:
    :param module_scores:
    :return:
    """
    module_cols = []
    for c in module_scores.columns:
        key = f"Module {c}"
        adata.obs[key] = module_scores[c]
        module_cols.append(key)
    return module_cols


def has_tf(tfs: list, l: list, module_id: Union[int, str]) -> Union[None, str]:
    """

    :param tfs:
    :param l:
    :param module_id:
    :return:
    """
    com_tf = set(tfs).intersection(set(l))
    if len(com_tf) == 1:
        return list(com_tf)[0]
    elif len(com_tf) == 0:
        return None
    elif len(com_tf) > 1:
        print(com_tf)
        print(f'module {module_id}')


def create_regulons(hs: hotspot.Hotspot, tf_names: list) -> dict:
    """

    :param hs:
    :param tf_names:
    :return:
    """
    mm = list(set(hs.modules))
    mm.remove(-1)
    regs = {}
    for i in mm:
        tg = list(hs.modules[hs.modules == i].index)
        supposed_tf = has_tf(tf_names, tg, i)
        if supposed_tf:
            regs[supposed_tf] = tg
    return regs


def lc2adj(local_correlations: pd.DataFrame, common_tf_list: list) -> pd.DataFrame:
    """

    :param local_correlations:
    :param common_tf_list:
    :return:
    """
    local_correlations['TF'] = local_correlations.columns
    local_correlations = local_correlations.melt(id_vars=['TF'])
    local_correlations.columns = ['TF', 'target', 'importance']
    local_correlations = local_correlations[local_correlations.TF.isin(common_tf_list)]
    return local_correlations


def main():
    adata = sc.read_h5ad(sys.argv[1])
    hs = hotspot.Hotspot(
        adata,
        layer_key=None,
        model='danb',
        latent_obsm_key="spatial",
    )
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    hs_results = hs.compute_autocorrelations(jobs=20)
    hs_genes = hs_results.loc[hs_results.FDR < 0.05].index  # Select genes

    local_correlations = hs.compute_local_correlations(hs_genes, jobs=20)  # jobs for parallelization
    local_correlations.to_csv('local_correlations.csv')

    hs.create_modules(
        min_gene_threshold=50, core_only=True, fdr_threshold=0.05
    )
    # calculate module scores
    module_scores = hs.calculate_module_scores()
    get_module_cols(adata, module_scores)

    # create regulons
    tf_names = ['Adf1', 'Aef1', 'grh', 'kn', 'tll', 'disco-r', 'Med', 'Dfd', 'br', 'so']
    regulons = create_regulons(hs, tf_names)
    with open('HOTSPOT_regulons.json', 'w') as f:
        json.dump(regulons, f, indent=4)

    # save results into adata
    adj = lc2adj(local_correlations, tf_names)
    adata.uns['adj'] = adj
    adata.uns['regulon_dict'] = regulons
    adata.write_h5ad('hotspot.h5ad')

    with open("hs.plk", 'wb') as f:
        pickle.dump(hs, f)

    # 画图
    hs.plot_local_correlations()


if __name__ == '__main__':
    main()
    # hs = pickle.load(open('hs.plk','rb'))
