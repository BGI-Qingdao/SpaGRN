#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 13:39
# @Author: Yao LI
# @File: SpaGRN/autocor.py
import os
import sys
import time

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
from pysal.lib import weights

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
    if issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)


def cal_k(adata, gene_x_id, n):
    gene_x_exp_mean = adata.X[:, gene_x_id].mean()
    gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
    denominator = np.square(np.sum(np.square(gene_x_exp - gene_x_exp_mean)))
    numerator = n * np.sum(np.power(gene_x_exp - gene_x_exp_mean, 4))
    K = numerator / denominator
    return K


# -----------------------------------------------------#
# Getis Ord General G
# -----------------------------------------------------#
def _getis_g(x, w):
    """
    Calculate getis ord general g for one gene
    :param x:
    :param w:
    :return:
    """
    x = np.asarray(x)
    w = np.asarray(w)
    numerator = np.sum(np.sum(w * np.outer(x, x)))
    denominator = np.sum(np.outer(x, x))
    G = numerator / denominator
    return G


def _getis_g_p_value_one_gene(G, w, x):
    n = w.shape[0]
    s0 = cal_s0(w)
    s02 = s0 * s0
    s1 = cal_s1(w)
    b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
    b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
    b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
    b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
    b4 = s1 - s2 + s02
    EG = s0 / (n * (n - 1))
    numerator = b0 * (np.square(np.sum(x ** 2))) + b1 * np.sum(np.power(x, 4)) + b2 * np.square(np.sum(x)) * np.sum(
        x ** 2) + b3 * np.sum(x) * np.sum(np.power(x, 3)) + b4 * np.power(np.sum(x), 4)
    denominator = np.square((np.square(np.sum(x)) - np.sum(x ** 2))) * n * (n - 1) * (n - 2) * (n - 3)
    VG = numerator / denominator - np.square(EG)
    Z = (G - EG) / np.sqrt(VG)
    p_value = 1 - norm.cdf(Z)
    # print(f'G: {G}\nVG: {VG}\nZ: {Z}\np_value: {p_value}')
    return p_value


def getis_g_p_values_one_gene(adata, gene_x_id, ind, weights_n, layer_key='raw_counts'):
    nind = pd.DataFrame(data=ind)
    nei = nind.transpose().to_dict('list')
    w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    w = weights.W(nei, weights=w_dict)
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X
    g = G(gene_expression_matrix[:, gene_x_id], w)
    return g.p_norm


# parallel computing
def _compute_g_for_gene(args):
    adata, gene_x_id, ind, weights_n, layer_key = args
    Gp = getis_g_p_values_one_gene(adata, gene_x_id, ind, weights_n, layer_key)
    print(f'gene{gene_x_id}: p_value: {Gp}')
    return Gp


def _getis_g_parallel(adata, ind, weights_n, n_genes, n_processes=None, layer_key=None):
    pool_args = [(adata, gene_x_id, ind, weights_n, layer_key) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        Gp_values = pool.map(_compute_g_for_gene, pool_args)
    return np.array(Gp_values)


def getis_g_p_values(adata: ad.AnnData,
                     ind,
                     weights_n,
                     n_processes=None,
                     layer_key=None):
    """
    Calculate getis ord general g for all genes and return getis_g_p_values values as a numpy.array
    :param adata: data containing gene expression matrix and cell-feature spatial coordinates array
    :param layer_key: layer key storing target gene expression matrix. if not provided, use raw counts adata.X as input
    :param n_processes: number of jobs when computing parallelly
    :return: (numpy.array) dimension: (n_genes, )
    """
    n_genes = len(adata.var_names)
    if n_processes:
        p_values = _getis_g_parallel(adata, ind, weights_n, n_genes, n_processes=n_processes, layer_key=layer_key)
    else:
        p_values = []
        for gene_x_id, gene_name in enumerate(adata.var_names):
            p_value = getis_g_p_value_one_gene(adata, gene_x_id, ind, weights_n)
            p_values.append(p_value)
        p_values = np.array(p_values)
    return p_values


# -----------------------------------------------------#
# M's I
# -----------------------------------------------------#
def _morans_i(adata, weights, layer_key='raw_counts'):
    """
    Calculate Moran’s I Global Autocorrelation Statistic for all genes in data
    :param adata: AnnData
    :param weights: neighbors connectivities array to use
    :param layer_key: Key for adata.layers to choose vals
    :return:
    """
    if 'connectivities' not in adata.obsp.keys():
        adata.obsp['connectivities'] = weights
    morans_i_array = sc.metrics.morans_i(adata, layer=layer_key)
    # shape: (n_genes, )
    return morans_i_array


def _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array):
    """
    Calculate p-value for one variable (gene)
    :param adata:
    :param gene_x_id:
    :param weights:
    :param morans_i_array:
    :return:
    """
    I = morans_i_array[gene_x_id]  # moran's I stats for the gene
    n = len(adata.obs_names)  # number of cells
    EI = -1 / (n - 1)  # Moran’s I expected value
    K = cal_k(adata, gene_x_id, n)
    S0 = cal_s0(weights)
    S1 = cal_s1(weights)
    S2 = cal_s2(weights)
    # Variance
    part1 = (n * (S1 * (n ** 2 - 3 * n + 3) - n * S2 + 3 * np.square(S0))) / (
            (n - 1) * (n - 2) * (n - 3) * np.square(S0))
    part2 = (K * (S1 * (n ** 2 - n) - 2 * n * S2 + 6 * np.square(S0))) / ((n - 1) * (n - 2) * (n - 3) * np.square(S0))
    VI = part1 - part2 - np.square(EI)
    stdI = np.sqrt(VI)
    # Z score
    Z = (I - EI) / stdI
    # Perform one-tail test one z score
    p_value = 1 - norm.cdf(Z)  # right tail
    return p_value


def morans_i_p_value_one_gene(x, w):
    i = Moran(x, w)
    return i.p_norm


# parallel computing
def _compute_i_for_gene(args):
    x, w = args
    Ip = morans_i_p_value_one_gene(x, w)
    return Ip


def _morans_i_parallel(n_genes, gene_expression_matrix, w, n_processes=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        p_values = pool.map(_compute_i_for_gene, pool_args)
    return np.array(p_values)


def morans_i_p_values(adata, ind, weights_n, layer_key='raw_counts', n_process=None):
    """
    Calculate Moran’s I Global Autocorrelation Statistic and its adjusted p-value
    :param adata: Anndata
    :param weights:
    :param layer_key:
    :param n_process:
    :return:
    """
    n_genes = len(adata.var_names)
    nind = pd.DataFrame(data=ind)
    nei = nind.transpose().to_dict('list')
    w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    w = weights.W(nei, weights=w_dict)
    if layer_key:
        gene_expression_matrix = adata.layers['raw_counts']
    else:
        gene_expression_matrix = adata.X
    p_values = _morans_i_parallel(n_genes, gene_expression_matrix, w, n_processes=n_process)
    return p_values


# -----------------------------------------------------#
# G's C
# -----------------------------------------------------#
def _gearys_c(adata, weights, layer_key='raw_counts'):
    if 'connectivities' not in adata.obsp.keys():
        adata.obsp['connectivities'] = weights
    gearys_c_array = sc.metrics.gearys_c(adata, layer=layer_key)
    # shape: (n_genes, )
    return gearys_c_array


def _gearys_c_p_value_one_gene(adata, gene_x_id, weights, gearys_c_array):
    C = gearys_c_array[gene_x_id]

    n = len(adata.obs_names)
    EC = 1
    K = cal_k(adata, gene_x_id, n)
    S0 = cal_s0(weights)
    S1 = cal_s1(weights)
    S2 = cal_s2(weights)
    part1 = (n - 1) * S1 * (n ** 2 - 3 * n + 3 - K * (n - 1)) / (np.square(S0) * n * (n - 2) * (n - 3))
    part2 = (n ** 2 - 3 - K * np.square(n - 1)) / (n * (n - 2) * (n - 3))
    part3 = (n - 1) * S2 * (n ** 2 + 3 * n - 6 - K * (n ** 2 - n + 2)) / (4 * n * (n - 2) * (n - 3) * np.square(S0))
    VC = part1 + part2 - part3
    # variance = (2 * (n ** 2) * S1 - n * S2 + 3 * (S0 ** 2)) / (S0 ** 2 * (n - 1) * (n - 2) * (n - 3))
    VC_norm = (1 / (2 * (n + 1) * S0 ** 2)) * ((2 * S1 + S2) * (n - 1) - 4 * S0 ** 2)
    Z = (C - EC) / np.sqrt(VC_norm)
    p_value = 1 - norm.cdf(Z)
    print(f'C: {C}\nVC: {VC}\nVC_norm: {VC_norm}\nZ: {Z}\np_value: {p_value}')
    return p_value


def gearys_c_p_value_one_gene(x, w):
    c = Geary(x, w)
    return c.p_norm


# parallel computing
def _compute_c_for_gene(args):
    x, w = args
    Cp = gearys_c_p_value_one_gene(x, w)
    return Cp


def _gearys_c_parallel(n_genes, gene_expression_matrix, w, n_processes=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        p_values = pool.map(_compute_c_for_gene, pool_args)
    return np.array(p_values)


def gearys_c_p_values(adata, ind, weights_n, layer_key='raw_counts', n_process=None):
    n_genes = len(adata.var_names)
    nind = pd.DataFrame(data=ind)
    nei = nind.transpose().to_dict('list')
    w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    w = weights.W(nei, weights=w_dict)
    if layer_key:
        gene_expression_matrix = adata.layers['raw_counts']
    else:
        gene_expression_matrix = adata.X
    p_values = _gearys_c_parallel(n_genes, gene_expression_matrix, w, n_processes=n_process)
    return p_values


# -----------------------------------------------------#
# SOMDE
# -----------------------------------------------------#
def somde_p_values(adata, k=20, layer_key='raw_counts', latent_obsm_key="spatial"):
    if layer_key:
        exp = adata.layers[layer_key]
    else:
        exp = adata.X
    df = pd.DataFrame(data=exp.T,
                      columns=adata.obs_names,
                      index=adata.var_names)
    cell_coordinates = adata.obsm[latent_obsm_key]
    corinfo = pd.DataFrame({
        "x": cell_coordinates[:, 0],
        "y": cell_coordinates[:, 1],
        "z": cell_coordinates[:, 2]
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
def spatial_autocorrelation(adata,
                            layer_key="raw_counts",
                            latent_obsm_key="spatial",
                            n_neighbors=10,
                            somde_k=20,
                            n_processes=None,
                            prefix='',
                            output_dir='.'):
    print('Computing spatial weights matrix...')
    ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
    print("Computing Moran's I...")
    morans_ps = morans_i_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
    fdr_morans_ps = fdr(morans_ps)
    print("Computing Geary's C...")
    gearys_cs = gearys_c_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
    fdr_gearys_cs = fdr(gearys_cs)
    print("Computing Getis G...")
    getis_gs = getis_g_p_values(adata, ind, weights_n, n_processes=n_processes, layer_key=layer_key)
    fdr_getis_gs = fdr(getis_gs)
    print('Computing SOMDE...')
    adjusted_p_values = somde_p_values(adata, k=somde_k, layer_key=layer_key, latent_obsm_key=latent_obsm_key)

    more_stats = pd.DataFrame({
        'C': gearys_cs,
        'FDR_C': fdr_gearys_cs,
        'I': morans_ps,
        'FDR_I': fdr_morans_ps,
        'G': getis_gs,
        'FDR_G': fdr_getis_gs,
        'FDR_SOMDE': adjusted_p_values
    }, index=adata.var_names)

    more_stats.to_csv(f'{output_dir}/{prefix}_more_stats.csv', sep='\t')

    return more_stats


def combind_fdrs(pvalue_df, method='fisher') -> np.array:
    """method options are {}"""
    from scipy.stats import combine_pvalues
    combined = np.apply_along_axis(combine_pvalues, 1, pvalue_df, method=method)[:, 1]
    return combined  # shape (n_gene, )


def preprocess(adata: ad.AnnData, min_genes=0, min_cells=3, min_counts=1, max_gene_num=4000):
    """
    Perform cleaning and quality control on the imported data before constructing gene regulatory network
    :param min_genes:
    :param min_cells:
    :param min_counts:
    :param max_gene_num:
    :return: a anndata.AnnData
    """
    adata.var_names_make_unique()  # compute the number of genes per cell (computes ‘n_genes' column)
    # # find mito genes
    sc.pp.ﬁlter_cells(adata, min_genes=0)
    # add the total counts per cell as observations-annotation to adata
    adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))
    # ﬁltering with basic thresholds for genes and cells
    sc.pp.ﬁlter_cells(adata, min_genes=min_genes)
    sc.pp.ﬁlter_genes(adata, min_cells=min_cells)
    sc.pp.ﬁlter_genes(adata, min_counts=min_counts)
    adata = adata[adata.obs['n_genes'] < max_gene_num, :]
    return adata


def hot(data, layer_key="raw_counts", latent_obsm_key="spatial"):
    import hotspot
    hs = hotspot.Hotspot(data,
                         layer_key=layer_key,
                         model='bernoulli',
                         latent_obsm_key=latent_obsm_key)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=10)
    hs_results = hs.compute_autocorrelations()
    return hs, hs_results


def select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=True):
    """

    :param more_stats:
    :param hs_results:
    :param fdr_threshold:
    :param combine: To select genes, combine p-values then choose genes have
    :return:
    """
    # if combine:
    more_stats['FDR'] = hs_results.FDR
    cfdrs = combind_fdrs(more_stats[['FDR_C', 'FDR_I', 'FDR_G', 'FDR_SOMDE', 'FDR']])
    more_stats['combined'] = cfdrs
    cgenes = more_stats.loc[more_stats['combined'] < fdr_threshold].index
    print(f"Combinded FDRs gives: {len(cgenes)} genes")  # 6961
    # return genes
    # else:
    moran_genes = more_stats.loc[more_stats.FDR_I < fdr_threshold].index
    geary_genes = more_stats.loc[more_stats.FDR_C < fdr_threshold].index
    getis_genes = more_stats.loc[more_stats.FDR_G < fdr_threshold].index
    somde_genes = more_stats.loc[more_stats.FDR_SOMDE < fdr_threshold].index
    hs_genes = hs_results.loc[(hs_results.FDR < fdr_threshold)].index
    inter_genes = set.intersection(set(moran_genes), set(geary_genes), set(getis_genes), set(somde_genes))
    print(f"Moran's I find {len(moran_genes)} genes")  # 3860
    print(f"Geary's C find {len(geary_genes)} genes")  # 7063
    print(f'getis find {len(getis_genes)} genes')  # 4285
    print(f'SOMDE find {len(somde_genes)} genes')  # 2593
    print(f'intersection gene num: {len(inter_genes)}')  # 3416
    # select
    genes = inter_genes.intersection(set(hs_genes))
    print(f'inter_genes intersection with FDR genes: {len(genes)}')  # 2728
    return genes


def main(prefix,
         fn,
         output_dir,
         n_process=4,
         min_genes=0,
         min_cells=3,
         min_counts=1,
         layer_key=None,
         latent_obsm_key="spatial",
         n_neighbors=10,
         somde_k=20):
    adata = sc.read_h5ad(fn)
    adata = preprocess(adata, min_genes=min_genes, min_cells=min_cells, min_counts=min_counts)
    # ---- HOTSPOT autocorrelation
    # print('HOTSPOT autocorrelation computing...')
    hs, hs_results = hot(adata, layer_key=layer_key, latent_obsm_key=latent_obsm_key)
    # Select genes
    # hs_genes = hs_results.index
    # print(f'hs_genes: {len(hs_genes)}')  # hs_genes: 12097
    # print('Selecting gene which FDR is lower than 0.05')
    # select_genes = hs_results.loc[hs_results.FDR < 0.05].index
    # print(f'select_genes: {len(select_genes)}')  # select_genes: 3181 / 3188
    # save_list(list(select_genes), f'{output_dir}/hotspot_select_genes.txt')
    # sub_adata = adata[:, select_genes]
    more_stats = spatial_autocorrelation(adata,
                                         layer_key=layer_key,
                                         latent_obsm_key=latent_obsm_key,
                                         n_neighbors=n_neighbors,
                                         somde_k=somde_k,
                                         n_processes=n_process,
                                         prefix=prefix,
                                         output_dir=output_dir)
    select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=False)


if __name__ == '__main__':
    # print(len(sys.argv))
    # print('Loading experimental data...')
    # prefix = sys.argv[1]
    # fn = sys.argv[2]
    # output_dir = sys.argv[3]
    # n_process = int(sys.argv[4])
    # if len(sys.argv) > 5:
    #     select_genes_fn = sys.argv[5]
    # else:
    #     select_genes_fn = None
    # adata = sc.read_h5ad(fn)
    # adata = preprocess(adata)
    #
    # # ---- HOTSPOT autocorrelation
    # print('HOTSPOT autocorrelation computing...')
    # hs, hs_results = hot(adata, layer_key=None)
    # # Select genes
    # if select_genes_fn:
    #     select_genes = read_list(select_genes_fn)
    # else:
    #     hs_genes = hs_results.index
    #     print(f'hs_genes: {len(hs_genes)}')  # hs_genes: 12097
    #     print('Selecting gene which FDR is lower than 0.05')
    #     select_genes = hs_results.loc[hs_results.FDR < 0.05].index
    #     print(f'select_genes: {len(select_genes)}')  # select_genes: 3181 / 3188
    #     save_list(list(select_genes), f'{output_dir}/hotspot_select_genes.txt')
    # sub_adata = adata[:, select_genes]

    # ---- LOCAL AUTOCORRELATION
    # 1. E14-16h
    # more_stats = spatial_autocorrelation(sub_adata,
    #                                      layer_key="raw_counts",
    #                                      latent_obsm_key="spatial",
    #                                      n_neighbors=10,
    #                                      somde_k=20,
    #                                      n_processes=None,
    #                                      prefix='',
    #                                      output_dir='.')
    # select_genes(more_stats, hs_results, fdr_threshold=0.05, combine=False)

    # 2. dryad.8t8s248, MERFISH, 2024-07-16
    fn2 = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/new_data/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.h5ad'
    output_dir2 = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/1.merfish'
    main('merfish', fn2, output_dir2, n_process=3, layer_key=None,latent_obsm_key="spatial",n_neighbors=10,somde_k=20)

    # 3. 10X Visium human
    # fn3 = '/zfsqd1/ST_OCEAN/USRS/hankai/database/SpaGRN/zenodo.7551712/DeconvolutionResults_ST_CRC_BelgianCohort/sp.h5ad'
    # output_dir3 = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/2.zenodo.7551712_BelgianCohort'
    # main('zenodo.7551712_BelgianCohort', fn3, output_dir3, n_process=3, min_genes=10, min_cells=50, min_counts=10, layer_key=None, latent_obsm_key="spatial", n_neighbors=10, somde_k=20)
