#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 13:39
# @Author: Yao LI
# @File: SpaGRN/autocor.py
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad

from pynndescent import NNDescent
from scipy.stats import chi2, norm
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, issparse

import multiprocessing


# -----------------------------------------------------#
# spatial weights
# -----------------------------------------------------#
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


# def cal_s1(w, n):
#     """
#     s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2
#     """
#     s1 = 0.5 * np.sum([
#         (w[i, j] + w[j, i]) ** 2
#         for i in range(n)
#         for j in range(n)
#     ])
#     return s1


def cal_s1(w, n):
    """
    s1 = 1/2 * sum_i sum_j (w_ij + w_ji)^2
    """
    # n = w.shape[0]  # 获取矩阵的维度 n
    # 使用广播机制构造 w_ij + w_ji 矩阵
    w_sum = w + w.T
    # 计算 s1
    s1 = 0.5 * np.sum(w_sum ** 2)
    return s1


def cal_s2(w, n):
    """
    s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2
    """
    s2 = np.sum([
        (np.sum(w[i, :]) + np.sum(w[:, i])) ** 2
        for i in range(n)
    ])
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


def cal_bs(w, n):
    '''for getis_g'''
    s0 = cal_s0(w)
    s02 = s0 * s0
    s1 = cal_s1(w, n)
    s2 = cal_s2(w, n)
    n2 = n * n
    b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
    b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
    b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
    b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
    b4 = s1 - s2 + s02
    return b0, b1, b2, b3, b4


# -----------------------------------------------------#
# Getis Ord General G
# -----------------------------------------------------#
def getis_g(x, w):
    """
    Calculate getis ord general g for one gene
    :param x:
    :param w:
    :return:
    """
    x = np.asarray(x)
    w = np.asarray(w)
    # Check for NaNs in gene expression values
    if np.any(np.isnan(x)):
        raise ValueError("Gene expression values contain NaNs.")
    # Check for zeros in gene expression values
    if np.all(x == 0):
        raise ValueError("All gene expression values are zero.")
    numerator = np.sum(np.sum(w * np.outer(x, x)))
    denominator = np.sum(x ** 2)
    # Check for zero denominator
    if denominator == 0:
        raise ValueError("Denominator is zero, which indicates all gene expression values are zero.")
    G = numerator / denominator
    return G


def getis_g_p_value_one_gene(G, w, x):
    n = w.shape[0]
    s0 = cal_s0(w)
    s02 = s0 * s0
    s1 = cal_s1(w, n)

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
    return p_value


# parallel computing
def _compute_g_for_gene(args):
    G, w, x = args
    Gp = getis_g_p_value_one_gene(G, w, x)
    return Gp


def _getis_g_parallel(gene_expression_matrix, n_processes=None):
    n_genes = gene_expression_matrix.shape[1]
    pool_args = [(G, w, gene_expression_matrix[:, gene_x_id]) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        Gp_values = pool.map(_compute_g_for_gene, pool_args)
    return np.array(Gp_values)


def getis_g_p_values(adata: ad.AnnData,
                     weights,
                     layer_key: str = 'raw_counts',
                     n_processes=None):
    """
    Calculate getis ord general g for all genes and return getis_g_p_values values as a numpy.array
    :param adata: data containing gene expression matrix and cell-feature spatial coordinates array
    :param layer_key: layer key storing target gene expression matrix. if not provided, use raw counts adata.X as input
    :param n_processes: number of jobs when computing parallelly
    :return: (numpy.array) dimension: (n_genes, )
    """
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X
    if n_processes:
        pass
    else:
        p_values = []
        for gene_x_id, gene_name in enumerate(adata.var_names):
            x = gene_expression_matrix[:, gene_x_id]
            G = getis_g(x, weights)
            p_value = getis_g_p_value_one_gene(G, w, x)
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
    print(morans_i_array.shape)
    print(morans_i_array.max(), morans_i_array.min())
    return morans_i_array


# def cal_I(adata, gene_x_id, weights):
#     gene_x_exp_mean = adata.X[:, gene_x_id].mean()
#     gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
#     s0 = cal_s0(weights)
#     n = weights.shape[0]
#     I = n / s0 * np.sum(weights * (gene_x_exp - gene_x_exp_mean) * (gene_x_exp - gene_x_exp_mean)) / np.sum(
#         np.square(gene_x_exp - gene_x_exp_mean))
#     return I


def _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array):
    """
    Calculate p-value for one variable (gene)
    :param adata:
    :param gene_x_id:
    :param weights:
    :param morans_i_array:
    :return:
    """
    # gene_x_id = adata.var.index.get_loc(gene_name)
    I = morans_i_array[gene_x_id]  # moran's I stats for the gene
    # I = cal_I(adata, gene_x_id, weights)

    n = len(adata.obs_names)  # number of cells
    EI = -1 / (n - 1)  # Moran’s I expected value
    K = cal_k(adata, gene_x_id, n)
    S0 = cal_s0(weights)
    S1 = cal_s1(weights, n)
    S2 = cal_s2(weights, n)
    # Variance
    part1 = (n * (S1 * (n ** 2 - 3 * n + 3) - n * S2 + 3 * np.square(S0))) / (
                (n - 1) * (n - 2) * (n - 3) * np.square(S0))
    part2 = (K * (S1 * (n ** 2 - n) - 2 * n * S2 + 6 * np.square(S0))) / ((n - 1) * (n - 2) * (n - 3) * np.square(S0))
    VI = part1 - part2 - np.square(EI)

    n2 = n * n
    S02 = S0 * S0
    v_num = n2 * S1 - n * S2 + 3 * S02
    v_den = (n - 1) * (n + 1) * S02
    VI_norm = v_num / v_den - (1.0 / (n - 1)) ** 2

    seI_norm = VI_norm ** (1 / 2.0)
    z_norm = (I - EI) / seI_norm
    p_norm = stats.norm.sf(z_norm)

    stdI = np.sqrt(VI)
    # Z score
    Z = (I - EI) / stdI
    # Perform one-tail test one z score
    p_value = 1 - norm.cdf(Z)  # right tail
    # fdr_p_value = fdr(p_value)

    if gene_x_id < 10:
        print(f'gene_x_id: {gene_x_id}')
        print(morans_i_array[gene_x_id])
        print(f'I: {I}')
        print(f'VI_norm: {VI_norm}')
        print(f'VI: {VI}')
        print(f'p_norm: {p_norm}')
        print(f'EI: {EI}, VI: {VI}, p_value: {p_value}')
    # print(f'one fdr: {fdr_p_value}')
    # return fdr_p_value
    # return p_value
    return p_norm


# parallel computing
def _compute_i_for_gene(args):
    adata, gene_x_id, weights, morans_i_array = args
    Ip = _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array)
    return Ip


def _morans_i_parallel(adata, weights, morans_i_array, n_processes=None):
    n_genes = len(adata.var_names)
    pool_args = [(adata, gene_x_id, weights, morans_i_array) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        p_values = pool.map(_compute_i_for_gene, pool_args)
    return np.array(p_values)


def morans_i_p_values(adata, weights, layer_key='raw_counts', n_process=None):
    """
    Calculate Moran’s I Global Autocorrelation Statistic and its adjusted p-value
    :param adata: Anndata
    :param weights:
    :param layer_key:
    :param n_process:
    :return:
    """
    morans_i_array = _morans_i(adata, weights, layer_key=layer_key)
    print(f'morans_i_array: {morans_i_array}')
    print(morans_i_array.max(), morans_i_array.min())
    if n_process:
        p_values = _morans_i_parallel(adata, weights, morans_i_array, n_processes=n_process)
    else:
        p_values = []
        for gene_x_id, gene_name in enumerate(adata.var_names):
            p = _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array)
            p_values.append(p)
        p_values = np.array(p_values)
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
    # gene_x_id = adata.var.index.get_loc(gene_name)
    C = gearys_c_array[gene_x_id]

    n = len(adata.obs_names)
    EC = 1
    K = cal_k(adata, gene_x_id, n)
    S0 = cal_s0(weights)
    S1 = cal_s1(weights, n)
    S2 = cal_s2(weights, n)
    part1 = (n - 1) * S1 * (n ** 2 - 3 * n + 3 - K * (n - 1)) / (np.square(S0) * n * (n - 2) * (n - 3))
    part2 = (n ** 2 - 3 - K * np.square(n - 1)) / (n * (n - 2) * (n - 3))
    part3 = (n - 1) * S2 * (n ** 2 + 3 * n - 6 - K * (n ** 2 - n + 2)) / (4 * n * (n - 2) * (n - 3) * np.square(S0))
    VC = part1 + part2 - part3

    Z = (C - EC) / np.sqrt(VC)
    p_value = 1 - norm.cdf(Z)
    fdr_p_value = fdr(p_value)
    return fdr_p_value


# parallel computing
def _compute_c_for_gene(args):
    adata, gene_x_id, weights, gearys_c_array = args
    Cp = _gearys_c_p_value_one_gene(adata, gene_x_id, weights, gearys_c_array)
    return Cp


def _gearys_c_parallel(adata, weights, gearys_c_array, n_processes=None):
    n_genes = len(adata.var_names)
    pool_args = [(adata, gene_x_id, weights, gearys_c_array) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        p_values = pool.map(_compute_c_for_gene, pool_args)
    return np.array(p_values)


def gearys_c_p_values(adata, weights, layer_key='raw_counts', n_process=None):
    gearys_c_array = _gearys_c(adata, weights, layer_key=layer_key)
    if n_process:
        p_values = _gearys_c_parallel(adata, weights, gearys_c_array, n_processes=n_process)
    else:
        p_values = []
        for gene_x_id, gene_name in enumerate(adata.var_names):
            p = _gearys_c_p_value_one_gene(adata, gene_x_id, weights, gearys_c_array)
            p_values.append(p)
        p_values = np.array(p_values)
    return p_values


# -----------------------------------------------------#
# SOMDE
# -----------------------------------------------------#
def somde_p_values(adata, k=20, latent_obsm_key="spatial"):
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
    ndf, ninfo = som.mtx(df)
    nres = som.norm()
    result, SVnum = som.run()
    p_values = result.pval
    adjusted_p_values = fdr(p_values)
    selected_genes = list(result[adjusted_p_values < 0.05].g)
    return p_values, selected_genes


# -----------------------------------------------------#
# Main
# -----------------------------------------------------#
def spatial_autocorrelation(adata,
                            layer_key="raw_counts",
                            latent_obsm_key="spatial",
                            n_neighbors=10,
                            n_processes=None):
    ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
    cell_names = adata.obs_names
    fw = flat_weights(cell_names, ind, weights_n, n_neighbors=n_neighbors)
    sw = square_weights(fw)
    weights = csr_matrix(sw)

    morans_ps = morans_i_p_values(adata, weights, layer_key=layer_key, n_process=n_processes)
    print(f'morans_ps: {morans_ps}')
    print(morans_ps.max(), morans_ps.min())
    print(type(morans_ps))
    print(f'p<0.05 gene nums: {morans_ps[morans_ps < 0.05].shape[0]}')
    fdr_morans_ps = fdr(morans_ps)
    print(f'fdr_morans_ps: {fdr_morans_ps}')
    print(fdr_morans_ps.max(), fdr_morans_ps.min())
    print(type(fdr_morans_ps))
    print(f'fdr<0.05 gene nums: {fdr_morans_ps[fdr_morans_ps < 0.05].shape[0]}')

    # gearys_cs = gearys_c_p_values(adata, weights, n_process=n_processes)

    # getis_gs = getis_g_p_values(adata, weights, layer_key=layer_key, n_processes=n_processes)
    #
    # more_stats = pd.DataFrame({
    #     'C': gearys_cs,
    #     'I': morans_is,
    #     'G': getis_gs
    # })
    # return more_stats


def combind_fdrs(more_stats, Hx, method='fisher'):
    from scipy.stats import combine_pvalues
    more_stats['Hx'] = Hx
    # p_values_array = more_stats.to_numpy()
    # combined = combine_pvalues(more_stats.to_numpy(), method=method)
    combined = np.apply_along_axis(combine_pvalues, 1, df, method=method)[:, 1]
    return combined


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


def hot(data):
    import hotspot
    hs = hotspot.Hotspot(data,
                         layer_key="raw_counts",
                         model='bernoulli',
                         latent_obsm_key="spatial")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=10)
    hs_results = hs.compute_autocorrelations()
    return hs_results


if __name__ == '__main__':
    adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h_pca.h5ad')
    # adata = adata[:100, :500]
    adata = preprocess(adata)
    hs_results = hot(adata)
    hs_genes = hs_results.index
    print(f'hs_genes: {len(hs_genes)}')  # hs_genes: 12097
    select_genes = hs_results.loc[hs_results.FDR < 0.05].index
    print(f'select_genes: {len(select_genes)}')  # select_genes: 3181
    sub_adata = adata[:, select_genes]

    local_correlations = hs.compute_local_correlations(select_genes[:500], jobs=12)  # jobs for parallelization
    print(local_correlations)


    spatial_autocorrelation(sub_adata, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10,n_processes=None)

    # from esda.getisord import G
    # from pysal.lib import weights
    # k = 10
    # ind, neighbors, weights_mtx = neighbors_and_weights(adata, n_neighbors=k)
    # gene_expression = adata.layers['raw_counts']
    # g = G(gene_expression[0, :], w)
    # print(g)
