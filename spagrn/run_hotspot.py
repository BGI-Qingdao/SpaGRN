#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 27 Sep 2023 09:43
# @Author: Yao LI
# @File: spagrn/hotspot.py
import sys
import hotspot
import scanpy as sc
import pickle


def get_cluster_label(fig):
    pass


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

    modules = hs.create_modules(
        min_gene_threshold=30, core_only=True, fdr_threshold=0.05
    )
    print(modules)


    # 画图
    fig, cm = hs.plot_local_correlations()
    # Extract module labels in clustermap
    modules_labels = cm.ax_heatmap.yaxis.get_majorticklabels()
    genes_labels = cm.ax_heatmap.xaxis.get_majorticklabels()
    print(modules_labels)
    print(genes_labels)
    with open('modules_labels.txt', 'w') as f:
        f.writelines('\n'.join(modules_labels))
    with open('genes_labels.txt', 'w') as f:
        f.writelines('\n'.join(genes_labels))
    adata.obsm['modules'] = modules_labels
    adata.obsm['genes'] = genes_labels

    # ?---------------------------------------
    module_scores = hs.calculate_module_scores()
    print(module_scores)

    module_cols = []
    for c in module_scores.columns:
        key = f"Module {c}"
        adata.obs[key] = module_scores[c]
        module_cols.append(key)
    print(module_cols)
    adata.write_h5ad('hetreo_hotspot.h5ad')

    with open("hs.plk", 'wb') as f:
        pickle.dump(hs, f)
    # hs = pickle.load(open('hs.plk','rb'))


if __name__ == '__main__':
    main()
