#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 13:39
# @Author: Yao LI
# @File: SpaGRN/autocor.py
import os
import sys
import time

import scipy
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad

from pynndescent import NNDescent
from scipy.stats import chi2, norm
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, issparse

from esda.getisord import G
from esda.moran import Moran
from esda.geary import Geary

import multiprocessing
from tqdm import tqdm


# -----------------------------------------------------#
# spatial weights
# -----------------------------------------------------#
def save_array(array, fn='array.json'):
    import json
    from json import JSONEncoder
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    encodedNumpyData = json.dumps(array, cls=NumpyEncoder)  # use dump() to write array into file
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(encodedNumpyData, f)


def save_list(l, fn='list.txt'):
    with open(fn, 'w') as f:
        f.write('\n'.join(l))


def read_list(fn):
    with open(fn, 'r') as f:
        l = f.read().splitlines()
    return l


def compute_weights(distances, neighborhood_factor=3):
    from math import ceil
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)
    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * distances ** 2 / sigma ** 2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm
    return weights


def neighbors_and_weights(data,
                          latent_obsm_key="spatial",
                          n_neighbors=30,
                          neighborhood_factor=3):
    """
    :param data:
    :param latent_obsm_key:
    :param n_neighbors:
    :param neighborhood_factor:
    :param approx_neighbors:
    :return:
    """
    from sklearn.neighbors import NearestNeighbors
    coords = data.obsm[latent_obsm_key]
    # get Nearest n Neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(coords)
    dist, ind = nbrs.kneighbors()  # neighbor name and index
    # get spatial weights between two points
    weights = compute_weights(
        dist, neighborhood_factor=neighborhood_factor)
    ind_df = pd.DataFrame(ind, index=data.obs_names)
    neighbors = ind_df
    weights = pd.DataFrame(weights, index=neighbors.index,
                           columns=neighbors.columns)
    return ind, neighbors, weights


def get_neighbor_weight_matrix(df):
    """
    3 columns weight matrix, first column is cell, second column is neighbor cells, third column is weight values.
    Create {cell: [neighbors]} index dictionary
    and {cell: [weights]} value dictionary
    :param df:
    :return: neighbor index dict and neighbor weight dict for pysal.lib to create a weights.W object.
    """
    unique_cells = sorted(df['Cell_x'].unique())
    cell_to_index = {cell: idx for idx, cell in enumerate(unique_cells)}

    df['Cell_x'] = df['Cell_x'].map(cell_to_index)
    df['Cell_y'] = df['Cell_y'].map(cell_to_index)

    nei_dict = (df.groupby('Cell_x')['Cell_y'].apply(list).to_dict())

    weights_grouped = df.groupby('Cell_x')['Weight'].apply(list).to_dict()
    w_dict = {cell: weights_grouped[cell] for cell in nei_dict}
    return nei_dict, w_dict


def get_w(ind, weights_n):
    """Create a Weight object for esda program"""
    nind = pd.DataFrame(data=ind)
    nei = nind.transpose().to_dict('list')
    w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    from pysal.lib import weights
    w = weights.W(nei, weights=w_dict)
    return w


def flat_weights(cell_names, ind, weights, n_neighbors=30):
    """
    Turn neighbor index into
    :param cell_names:
    :param ind:
    :param weights:
    :param n_neighbors:
    :return:
    """
    cell1 = np.repeat(cell_names, n_neighbors)
    cell2_indices = ind.flatten()  # starts from 0
    cell2 = cell_names[cell2_indices]
    print(len(cell1))
    print(len(cell2))
    weight = weights.to_numpy().flatten()
    df = pd.DataFrame({
        "Cell_x": cell1,
        "Cell_y": cell2,
        "Weight": weight
    })
    return df


def square_weights(flat_weights_matrix):
    full_weights_matrix = pd.pivot_table(flat_weights_matrix,
                                         index='Cell_x',
                                         columns='Cell_y',
                                         values='Weight',
                                         fill_value=0)
    return full_weights_matrix


def fdr(ps, axis=0):
    """
    Apply the Benjamini-Hochberg procedure (FDR) of an array of p-values
    :param ps:
    :param axis:
    :return:
    """
    ps = np.asarray(ps)
    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")
    if axis is None:
        axis = 0
        ps = ps.ravel()
    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")
    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]
    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps
    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m + 1)
    ps *= m / i
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)
    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)
    return np.clip(ps, 0, 1)


def cal_s0(w):
    """
    s0=\sum_i \sum_j w_{i,j}
    """
    s0 = np.sum(w)
    return s0


def cal_s1(w):
    """
    s1 = 1/2 * sum_i sum_j (w_ij + w_ji)^2
    """
    # broadcast, create w_ij + w_ji 矩阵
    w_sum = w + w.T
    # 计算 s1
    s1 = 0.5 * np.sum(w_sum ** 2)
    return s1


def cal_s2(w):
    """
    s2 = \sum_j (\sum_i w_{i,j} + \sum_i w_{j,i})^2
    """
    # 计算行和列的和
    row_sums = np.sum(w, axis=1)
    col_sums = np.sum(w, axis=0)
    # 计算 \sum_i w_{i,j} + \sum_i w_{j,i} 对每个 j
    total_sums = row_sums + col_sums
    s2 = np.sum(total_sums ** 2)
    return s2


def format_gene_array(gene_array):
    if scipy.sparse.issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)


def cal_k(gene_expression_matrix:np.ndarray, gene_x_id, n):
    gene_x_exp_mean = gene_expression_matrix[:, gene_x_id].mean()
    gene_x_exp = format_gene_array(gene_expression_matrix[:, gene_x_id])
    denominator = np.square(np.sum(np.square(gene_x_exp - gene_x_exp_mean)))
    numerator = n * np.sum(np.power(gene_x_exp - gene_x_exp_mean, 4))
    K = numerator / denominator
    return K


# -----------------------------------------------------#
# SOMDE
# -----------------------------------------------------#
def somde_p_values(adata, k=20, layer_key='raw_counts', latent_obsm_key="spatial"):
    if layer_key:
        exp = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        exp = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    df = pd.DataFrame(data=exp.T,
                      columns=adata.obs_names,
                      index=adata.var_names)
    cell_coordinates = adata.obsm[latent_obsm_key]
    corinfo = pd.DataFrame({
        "x": cell_coordinates[:, 0],
        "y": cell_coordinates[:, 1],
    })
    corinfo.index = adata.obs_names
    corinfo["total_count"] = exp.sum(1)
    X = corinfo[['x', 'y']].values.astype(np.float32)
    from somde import SomNode
    som = SomNode(X, k)
    som.mtx(df)
    som.norm()
    result, SVnum = som.run()
    p_values = result.pval
    adjusted_p_values = fdr(p_values)
    som.view()
    return adjusted_p_values


def view(som, raw=True, c=False, line=False):
    import matplotlib.pyplot as plt
    rr = som.som.codebook
    sizenum = np.ones([rr.shape[0], rr.shape[1]]) * 30
    rr = np.reshape(rr, [-1, 2])
    if raw:
        plt.scatter(som.X[:, 0], som.X[:, 1], s=3, label='original')
    for i in range(som.X.shape[0]):
        v, u = som.som.bmus[i]
        sizenum[u, v] += 2
        if line:
            plt.plot([som.X[i, 0], som.som.codebook[u, v, 0]], [som.X[i, 1], som.som.codebook[u, v, 1]])
    sizenum = np.reshape(sizenum, [-1, ])
    if c:
        plt.scatter(rr[:, 0], rr[:, 1], s=sizenum, label=str(som.somn) + 'X' + str(som.somn) + ' SOM nodes',
                    c=sizenum, cmap='hot')
        plt.colorbar()
    else:
        plt.scatter(rr[:, 0], rr[:, 1], s=sizenum, label=str(som.somn) + 'X' + str(som.somn) + ' SOM nodes', c='r')
    plt.savefig('somde.png')
    plt.close()


# -----------------------------------------------------#
# Main
# -----------------------------------------------------#
# def spatial_autocorrelation(adata,
#                             layer_key="raw_counts",
#                             latent_obsm_key="spatial",
#                             n_neighbors=10,
#                             somde_k=20,
#                             n_processes=None,
#                             prefix='',
#                             output_dir='.'):
#     """
#     Calculate spatial autocorrelation values using Moran's I, Geary'C, Getis's G and SOMDE algorithms
#     :param adata:
#     :param layer_key:
#     :param latent_obsm_key:
#     :param n_neighbors:
#     :param somde_k:
#     :param n_processes:
#     :param prefix:
#     :param output_dir:
#     :return:
#     """
#     print('Computing spatial weights matrix...')
#     sc.pp.filter_cells(adata, min_genes=100)
#     ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
#     Weights = get_w(ind, weights_n)
#     print("Computing Moran's I...")
#     # morans_ps = morans_i_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
#     # # np.savetxt(f'{output_dir}/{prefix}_morans_ps.txt', morans_ps)
#     # fdr_morans_ps = fdr(morans_ps)
#     # # np.savetxt(f'{output_dir}/{prefix}_fdr_morans_ps.txt', fdr_morans_ps)
#     # print("Computing Geary's C...")
#     # gearys_cs = gearys_c_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
#     # # np.savetxt(f'{output_dir}/{prefix}_gearys_cs.txt', gearys_cs)
#     # fdr_gearys_cs = fdr(gearys_cs)
#     # # np.savetxt(f'{output_dir}/{prefix}_fdr_gearys_cs.txt', fdr_gearys_cs)
#     # print("Computing Getis G...")
#     getis_gs = getis_g_p_values(adata, Weights, n_processes=n_processes, layer_key=layer_key)
#     np.savetxt(f'{output_dir}/{prefix}_getis_gs.txt', getis_gs)
#     # fdr_getis_gs = fdr(getis_gs)
#     # np.savetxt(f'{output_dir}/{prefix}_fdr_getis_gs.txt', fdr_getis_gs)
#     morans_ps = np.loadtxt(f'{output_dir}/{prefix}_morans_ps.txt')
#     fdr_morans_ps = np.loadtxt(f'{output_dir}/{prefix}_fdr_morans_ps.txt')
#     gearys_cs = np.loadtxt(f'{output_dir}/{prefix}_gearys_cs.txt')
#     fdr_gearys_cs = np.loadtxt(f'{output_dir}/{prefix}_fdr_gearys_cs.txt')
#     # getis_gs = np.loadtxt(f'{output_dir}/{prefix}_getis_gs.txt')
#     # fdr_getis_gs = np.loadtxt(f'{output_dir}/{prefix}_fdr_getis_gs.txt')
#     print('Computing SOMDE...')
#     adjusted_p_values = somde_p_values(adata, k=somde_k, layer_key=layer_key, latent_obsm_key=latent_obsm_key)
#     # np.savetxt(f'{output_dir}/{prefix}_fdr_SOMDE.txt', adjusted_p_values)
#     more_stats = pd.DataFrame({
#         'C': gearys_cs,
#         'FDR_C': fdr_gearys_cs,
#         'I': morans_ps,
#         'FDR_I': fdr_morans_ps,
#         'G': getis_gs,
#         'FDR_G': fdr_getis_gs,
#         'FDR_SOMDE': adjusted_p_values
#     }, index=adata.var_names)
#     more_stats.to_csv(f'{output_dir}/{prefix}_more_stats.csv', sep='\t')
#     return more_stats
#
#
# def combind_fdrs(pvalue_df, method='fisher') -> np.array:
#     """method options are {}"""
#     from scipy.stats import combine_pvalues
#     combined = np.apply_along_axis(combine_pvalues, 1, pvalue_df, method=method)[:, 1]
#     return combined  # shape (n_gene, )
#
#
# def preprocess(adata: ad.AnnData, min_genes=0, min_cells=3, min_counts=1, max_gene_num=4000):
#     adata.var_names_make_unique()  # compute the number of genes per cell (computes ‘n_genes' column)
#     # # find mito genes
#     sc.pp.ﬁlter_cells(adata, min_genes=0)
#     # add the total counts per cell as observations-annotation to adata
#     adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))
#     # ﬁltering with basic thresholds for genes and cells
#     sc.pp.ﬁlter_cells(adata, min_genes=min_genes)
#     sc.pp.ﬁlter_genes(adata, min_cells=min_cells)
#     sc.pp.ﬁlter_genes(adata, min_counts=min_counts)
#     adata = adata[adata.obs['n_genes'] < max_gene_num, :]
#     return adata
#
#
# def hot(data, layer_key="raw_counts", latent_obsm_key="spatial"):
#     import hotspot
#     hs = hotspot.Hotspot(data,
#                          layer_key=layer_key,
#                          model='bernoulli',
#                          latent_obsm_key=latent_obsm_key)
#     hs.create_knn_graph(weighted_graph=False, n_neighbors=10)
#     hs_results = hs.compute_autocorrelations()
#     return hs, hs_results
#
#
# def select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=True):
#     """
#     Select genes...
#     :param more_stats:
#     :param hs_results:
#     :param fdr_threshold:
#     :param combine: To select genes, combine p-values then choose genes have
#     :return:
#     """
#     # if combine:
#     more_stats['FDR'] = hs_results.FDR
#     cfdrs = combind_fdrs(more_stats[['FDR_C', 'FDR_I', 'FDR_G', 'FDR_SOMDE', 'FDR']])
#     more_stats['combined'] = cfdrs
#     cgenes = more_stats.loc[more_stats['combined'] < fdr_threshold].index
#     print(f"Combinded FDRs gives: {len(cgenes)} genes")  # 6961
#     # return genes
#     # else:
#     moran_genes = more_stats.loc[more_stats.FDR_I < fdr_threshold].index
#     geary_genes = more_stats.loc[more_stats.FDR_C < fdr_threshold].index
#     getis_genes = more_stats.loc[more_stats.FDR_G < fdr_threshold].index
#     somde_genes = more_stats.loc[more_stats.FDR_SOMDE < fdr_threshold].index
#     hs_genes = hs_results.loc[(hs_results.FDR < fdr_threshold)].index
#     inter_genes = set.intersection(set(moran_genes), set(geary_genes), set(getis_genes), set(somde_genes))
#     print(f"Moran's I find {len(moran_genes)} genes")  # 3860
#     print(f"Geary's C find {len(geary_genes)} genes")  # 7063
#     print(f'getis find {len(getis_genes)} genes')  # 4285
#     print(f'SOMDE find {len(somde_genes)} genes')  # 2593
#     print(f'intersection gene num: {len(inter_genes)}')  # 3416
#     # check somde results
#     t = set(somde_genes).intersection(set(hs_genes))
#     print(f'SOMDE genes {len(t)} in hs_genes')
#     # select
#     genes = inter_genes.intersection(set(hs_genes))
#     print(f'4 methods: inter_genes intersection with FDR genes: {len(genes)}')
#     global_genes = set.intersection(set(moran_genes), set(geary_genes), set(getis_genes)).intersection(set(hs_genes))
#     print(f'Global inter_genes intersection with FDR genes: {len(genes)}')  # 2728
#     return genes
#
#
# def main(prefix,
#          fn,
#          output_dir,
#          n_process=4,
#          min_genes=0,
#          min_cells=3,
#          min_counts=1,
#          layer_key=None,
#          latent_obsm_key="spatial",
#          n_neighbors=10,
#          somde_k=20):
#     """
#     Main function to calculate spatial autocorrelation values for genes
#     :param prefix: project name
#     :param fn: h5ad file name
#     :param output_dir: output directory
#     :param n_process: number of processes when computing in parallel
#     :param min_genes: filter cells
#     :param min_cells: filter genes
#     :param min_counts: filter genes
#     :param layer_key: layers containing gene expression matrix
#     :param latent_obsm_key: key stored cell/spot spatial coordinates
#     :param n_neighbors: number of neighbors when building KNN
#     :param somde_k: number of neighbors when using SOMDE model
#     :return:
#     """
#     adata = sc.read_h5ad(fn)
#     adata = preprocess(adata, min_genes=min_genes, min_cells=min_cells, min_counts=min_counts)
#     # ---- HOTSPOT autocorrelation
#     hs, hs_results = hot(adata, layer_key=layer_key, latent_obsm_key=latent_obsm_key)
#     # Select genes
#     more_stats = spatial_autocorrelation(adata,
#                                          layer_key=layer_key,
#                                          latent_obsm_key=latent_obsm_key,
#                                          n_neighbors=n_neighbors,
#                                          somde_k=somde_k,
#                                          n_processes=n_process,
#                                          prefix=prefix,
#                                          output_dir=output_dir)
#     select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=False)
#
#
# if __name__ == '__main__':
#     project_id = sys.argv[1]
#     # 1. E14-16h
#     # more_stats = spatial_autocorrelation(sub_adata,
#     #                                      layer_key="raw_counts",
#     #                                      latent_obsm_key="spatial",
#     #                                      n_neighbors=10,
#     #                                      somde_k=20,
#     #                                      n_processes=None,
#     #                                      prefix='',
#     #                                      output_dir='.')
#     # select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=False)
#     if project_id == 'E14':
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E14-16h_pca.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h'
#         prefix = 'E14-16h'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=3, layer_key="raw_counts",latent_obsm_key="spatial",n_neighbors=10,somde_k=20)
#     elif project_id == 'E16':
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E16-18h_pca.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E16-18h'
#         prefix = 'E16-18h'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=3, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 'L1':
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/L1_pca.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/L1'
#         prefix = 'L1'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=3, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 'L2':
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/L2_pca.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/L2'
#         prefix = 'L2'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=3, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 'L3':
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/L3_pca.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/L3'
#         prefix = 'L3'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=3, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#
#     project_id = int(project_id)
#     # 1. dryad.8t8s248, MERFISH, 2024-07-16
#     # mouse brain, preoptic region
#     if project_id == 1:
#         fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/1.merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/1.merfish'
#         prefix = 'merfish'
#         print(f'Running for {prefix} project...')
#         main(prefix, fn, output_dir, n_process=20, layer_key=None,latent_obsm_key="spatial",n_neighbors=10,somde_k=20)
#
#     # 2. zenodo.7551712, 10X Visium, human colorectal cancer (CRC)
#     # 7 individuals; two samples per patient
#     # GRN: using known TF-target interactions from DoRothEA
#     # GRCh38
#     elif project_id == 2:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_BelgianCohort/sp.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/2.zenodo.7551712_BelgianCohort'
#         prefix = 'zenodo.7551712_BelgianCohort'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None, latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 3:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_KoreanCohort/sp.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/3.zenodo.7551712_KoreanCohort'
#         prefix = 'zenodo.7551712_KoreanCohort'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 4:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_LiverMetastasis/sp.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/4.zenodo.7551712_LiverMetastasis'
#         prefix = 'zenodo.7551712_LiverMetastasis'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#
#     # 3. human ovarian tumour
#     # 10X Visium (GSE211956) & CosMx (zenodo.8287970)
#     # High-grade serous ovarian tumours from 10 patients diagnosed with stage III-IV cancers
#     elif project_id == 5:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_LiverMetastasis/sp.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/5.zenodo.7551712_LiverMetastasis'
#         prefix = 'zenodo.7551712_LiverMetastasis'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#     elif project_id == 6:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_LiverMetastasis/sp.h5ad'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/5.zenodo.7551712_LiverMetastasis'
#         prefix = 'zenodo.7551712_LiverMetastasis'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#
#     # 4. mouse, CosMx, brain with PFF induced PD
#     # 1 female and 2 male mic
#     # zenodo.10729766
#     elif project_id == 6:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.10729766/seurat_object_3mon.rds'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/6.zenodo.10729766_mouse_brain'
#         prefix = 'zenodo.10729766_mouse_brain'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=1, min_cells=3, min_counts=1, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
#
#     # 5. zenodo.8063124
#     # human, Clear cell renal cell carcinoma (ccRCC)
#     # 12 tumor sections and 2 NAT controls
#     elif project_id == 7:
#         fn = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.10729766/seurat_object_3mon.rds'
#         output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/7.zenodo.8063124_human'
#         prefix = 'zenodo.10729766_mouse_brain'
#         print(f'Running for [{prefix}] project...')
#         main(prefix, fn, output_dir, n_process=3, min_genes=1, min_cells=3, min_counts=1, layer_key=None,
#              latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
