#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 13:39
# @Author: Yao LI
# @File: SpaGRN/autocor.py
import os
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
from memory_profiler import profile, memory_usage


def memory_time_decorator(func):
    import time
    from memory_profiler import memory_usage
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        # Measure memory usage
        mem_usage = memory_usage((func, args, kwargs), max_usage=True)
        # Record end time
        end_time = time.time()
        # Calculate total time taken
        total_time = end_time - start_time
        print(f"Maximum memory usage: {mem_usage[0]:.2f} MiB")
        print(f"Total time taken: {total_time:.4f} seconds")
        return func(*args, **kwargs)
    return wrapper


class AutocorStats:
    def __init__(self, w=None):
        self.w = w
        self.n = w.shape[0]
        self.s0 = None
        self.s1 = None
        self.s2 = None
        # self.Stat = None
        self.E = None
        self.V = None

        self.b0 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None

        # calculation
        self.cal_s0()
        self.cal_s1()
        self.cal_s2()
        self.cal_bs()

    def cal_s0(self):
        """
        s0=\sum_i \sum_j w_{i,j}
        """
        self.s0 = np.sum(self.w)
        return self.s0

    def cal_s1(self):
        """
        s1 = 1/2 * sum_i sum_j (w_ij + w_ji)^2
        """
        # broadcast, create w_ij + w_ji 矩阵
        w_sum = self.w + self.w.T
        # 计算 s1
        self.s1 = 0.5 * np.sum(w_sum ** 2)
        return self.s1

    def cal_s2(self):
        """
        s2 = \sum_j (\sum_i w_{i,j} + \sum_i w_{j,i})^2
        """
        # 计算行和列的和
        row_sums = np.sum(self.w, axis=1)
        col_sums = np.sum(self.w, axis=0)
        # 计算 \sum_i w_{i,j} + \sum_i w_{j,i} 对每个 j
        total_sums = row_sums + col_sums
        self.s2 = np.sum(total_sums ** 2)
        return self.s2

    def cal_bs(self):
        self.b0 = (self.n ** 2 - 3 * self.n + 3) * self.s1 - self.n * self.s2 + 3 * self.s0 ** 2
        self.b1 = (-1.0) * ((self.n ** 2 - self.n) * self.s1 - 2 * self.n * self.s2 + 6 * self.s0 ** 2)
        self.b2 = (-1.0) * (2 * self.n * self.s1 - (self.n + 3) * self.s2 + 6 * self.s0 ** 2)
        self.b3 = 4 * (self.n - 1) * self.s1 - 2 * (self.n + 1) * self.s2 + 8 * self.s0 ** 2
        self.b4 = self.s1 - self.s2 + self.s0 ** 2

    def cal_EG(self):
        self.E = self.s0 / (self.n * (self.n - 1))
        return self.E

    # def cal_VG(self):
    #     numerator = self.b0 * (np.square(np.sum(x ** 2))) + b1 * np.sum(np.power(x, 4)) + b2 * np.square(np.sum(x)) * np.sum(x ** 2) + b3 * np.sum(x) * np.sum(np.power(x, 3)) + b4 * np.power(np.sum(x), 4)
    #     denominator = np.square((np.square(np.sum(x)) - np.sum(x ** 2))) * n * (n - 1) * (n - 2) * (n - 3)
    #     self.V = numerator / denominator - np.square(EG)

    def cal_EI(self):
        self.E = -1 / (self.n - 1)


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
    # x = np.asarray(x)
    # w = np.asarray(w)
    numerator = np.sum(np.sum(w * np.outer(x, x)))
    # denominator = np.sum(x ** 2)
    denominator = np.sum(np.outer(x, x))
    G = numerator / denominator
    return G


# def lag(w, y):
#     return w.sparse * y


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
    print(f'G: {G}\nVG: {VG}\nZ: {Z}\np_value: {p_value}')
    return p_value


def getis_g_p_values_one_gene(adata, gene_x_id, ind, weights_n):
    nind = pd.DataFrame(data=ind)
    nei = nind.transpose().to_dict('list')
    w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    w = weights.W(nei, weights=w_dict)
    gene_expression_matrix = adata.layers['raw_counts']
    g = G(gene_expression_matrix[:, gene_x_id], w)
    return g.p_norm


# parallel computing
# def _compute_g_for_gene(args):
#     w, x = args
#     G = getis_g(x, w)
#     Gp = getis_g_p_value_one_gene(G, w, x)
#     return Gp
def _compute_g_for_gene(args):
    adata, gene_x_id, ind, weights_n = args
    Gp = getis_g_p_values_one_gene(adata, gene_x_id, ind, weights_n)
    print(f'gene{gene_x_id}: p_value: {Gp}')
    return Gp


def _getis_g_parallel(adata, ind, weights_n, n_genes, n_processes=None):
    pool_args = [(adata, gene_x_id, ind, weights_n) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        Gp_values = pool.map(_compute_g_for_gene, pool_args)
    return np.array(Gp_values)


def getis_g_p_values(adata: ad.AnnData,
                     ind,
                     weights_n,
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
    n_genes = len(adata.var_names)
    if n_processes:
        p_values = _getis_g_parallel(adata, ind, weights_n, n_genes, n_processes=n_processes)
    else:
        p_values = []
        for gene_x_id, gene_name in enumerate(adata.var_names):
            # x = gene_expression_matrix[:, gene_x_id]
            # G = getis_g(x, weights)
            # p_value = getis_g_p_value_one_gene(G, w, x)
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
    # nind = pd.DataFrame(data=ind)
    # nei = nind.transpose().to_dict('list')
    # w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    # w = weights.W(nei, weights=w_dict)
    # gene_expression_matrix = adata.layers['raw_counts']
    i = Moran(x, w)
    return i.p_norm


# parallel computing
# def _compute_i_for_gene(args):
#     adata, gene_x_id, weights, morans_i_array = args
#     Ip = _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array)
#     return Ip
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
    # morans_i_array = _morans_i(adata, weights, layer_key=layer_key)
    # stats = AutocorStats(weights)
    # print(f'morans_i_array: {morans_i_array}')
    # print(morans_i_array.max(), morans_i_array.min())
    # if n_process:
    #     p_values = _morans_i_parallel(adata, weights, morans_i_array, n_processes=n_process)
    # else:
    #     p_values = []
    #     for gene_x_id, gene_name in enumerate(adata.var_names):
    #         p = _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array)
    #         p_values.append(p)
    #     p_values = np.array(p_values)
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
    # nind = pd.DataFrame(data=ind)
    # nei = nind.transpose().to_dict('list')
    # w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    # w = weights.W(nei, weights=w_dict)
    # gene_expression_matrix = adata.layers['raw_counts']
    c = Geary(x, w)
    return c.p_norm


# parallel computing
# def _compute_c_for_gene(args):
#     adata, gene_x_id, weights, gearys_c_array = args
#     Cp = _gearys_c_p_value_one_gene(adata, gene_x_id, weights, gearys_c_array)
#     return Cp
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
    # gearys_c_array = _gearys_c(adata, weights, layer_key=layer_key)
    # if n_process:
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
    # else:
    #     p_values = []
    #     for gene_x_id, gene_name in enumerate(adata.var_names):
    #         print(gene_x_id)
    #         p = _gearys_c_p_value_one_gene(adata, gene_x_id, weights, gearys_c_array)
    #         p_values.append(p)
    #     p_values = np.array(p_values)
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
    print(f'ndf: {ndf}')
    print(f'ninfo: {ninfo}')
    nres = som.norm()
    print(f'nres: {nres}')
    result, SVnum = som.run()
    p_values = result.pval
    adjusted_p_values = fdr(p_values)
    selected_genes = list(result[adjusted_p_values < 0.05].g)
    return p_values, selected_genes


# -----------------------------------------------------#
# Main
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


def spatial_autocorrelation(adata,
                            layer_key="raw_counts",
                            latent_obsm_key="spatial",
                            n_neighbors=10,
                            n_processes=None,
                            output='.'):
    ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
    # cell_names = adata.obs_names
    # fw = flat_weights(cell_names, ind, weights_n, n_neighbors=n_neighbors)
    # sw = square_weights(fw)
    # weights = csr_matrix(sw)

    morans_ps = morans_i_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
    # print(f'morans_ps: {morans_ps}')
    # print(morans_ps.max(), morans_ps.min())
    # print(type(morans_ps))
    # print(f'p<0.05 gene nums: {morans_ps[morans_ps < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'morans_ps.txt'), morans_ps)
    fdr_morans_ps = fdr(morans_ps)
    # print(f'fdr_morans_ps: {fdr_morans_ps}')
    # print(fdr_morans_ps.max(), fdr_morans_ps.min())
    # print(type(fdr_morans_ps))
    # print(f'fdr<0.05 gene nums: {fdr_morans_ps[fdr_morans_ps < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'fdr_morans_ps.txt'), fdr_morans_ps)

    gearys_cs = gearys_c_p_values(adata, ind, weights_n, layer_key=layer_key, n_process=n_processes)
    # print(f'gearys_cs: {gearys_cs}')
    # print(gearys_cs.max(), gearys_cs.min())
    # print(type(gearys_cs))
    # print(f'p<0.05 gene nums: {gearys_cs[gearys_cs < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'gearys_cs.txt'), gearys_cs)
    fdr_gearys_cs = fdr(gearys_cs)
    # print(f'fdr_gearys_cs: {fdr_gearys_cs}')
    # print(fdr_gearys_cs.max(), fdr_gearys_cs.min())
    # print(type(fdr_gearys_cs))
    # print(f'fdr<0.05 gene nums: {fdr_gearys_cs[fdr_gearys_cs < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'fdr_gearys_cs.txt'), fdr_gearys_cs)

    getis_gs = getis_g_p_values(adata, ind, weights_n, layer_key=layer_key, n_processes=n_processes)
    # print(f'getis_gs: {getis_gs}')
    # print(getis_gs.max(), getis_gs.min())
    # print(type(getis_gs))
    # print(f'p<0.05 gene nums: {getis_gs[getis_gs < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'getis_gs.txt'), getis_gs)
    fdr_getis_gs = fdr(getis_gs)
    # print(f'fdr_getis_gs: {fdr_getis_gs}')
    # print(fdr_getis_gs.max(), fdr_getis_gs.min())
    # print(type(fdr_getis_gs))
    # print(f'fdr<0.05 gene nums: {fdr_getis_gs[fdr_getis_gs < 0.05].shape[0]}')
    np.savetxt(os.path.join(output, 'fdr_getis_gs.txt'), fdr_getis_gs)

    # morans_ps = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/morans_ps.txt')
    # fdr_morans_ps = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/fdr_morans_ps.txt')
    # gearys_cs = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/gearys_cs.txt')
    # fdr_gearys_cs = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/fdr_gearys_cs.txt')
    # getis_gs = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/getis_gs.txt')
    # fdr_getis_gs = np.loadtxt('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/fdr_getis_gs.txt')

    more_stats = pd.DataFrame({
        'C': gearys_cs,
        'FDR_C': fdr_gearys_cs,
        'I': morans_ps,
        'FDR_I': fdr_morans_ps,
        'G': getis_gs,
        'FDR_G': fdr_getis_gs
    }, index=adata.var_names)
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


def hot(data):
    import hotspot
    hs = hotspot.Hotspot(data,
                         layer_key="raw_counts",
                         model='bernoulli',
                         latent_obsm_key="spatial")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=10)
    hs_results = hs.compute_autocorrelations()
    return hs, hs_results


# -----------------------------------------------------#
#               Bivariate Moran's I                    #
# -----------------------------------------------------#
def task_generator(adata, weights, selected_genes, n_genes):
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            yield (i, j, adata, weights, selected_genes)


def global_bivariate_moran_R_two_genes(adata, weights, gene_x, gene_y):
    """
    Compute bivariate Moran's R value of two given genes
        :param adata: experimental data
    :param weights: spatial weights between two cells/spots
    :param gene_x: gene name
    :param gene_y: gene name
    :return: float, bivariate Moran's R value
    """
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    # 3 get average
    gene_x_exp_mean = adata.X[:, gene_x_id].mean()
    gene_y_exp_mean = adata.X[:, gene_y_id].mean()
    # 4 calculate denominator
    gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
    gene_y_exp = format_gene_array(adata.X[:, gene_y_id])
    denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                  np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    # 5 calulate numerator
    gene_x_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_x_id])
    gene_y_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_y_id])
    wij = weights['Weight'].to_numpy()
    numerator = np.sum(wij * (gene_x_in_cell_x - gene_x_exp_mean) * (gene_y_in_cell_y - gene_y_exp_mean))
    return numerator / denominator


def compute_pair_m(args):
    """Apply function on two genes"""
    i, j, adata, weights, selected_genes = args
    gene1 = selected_genes[i]
    gene2 = selected_genes[j]
    value = global_bivariate_moran_R_two_genes(adata, weights, gene1, gene2)
    return i, j, value


def run_in_parallel_m(adata, weights: pd.DataFrame, selected_genes, n_genes, num_workers):
    """

    :param adata:
    :param weights:
    :param selected_genes:
    :param n_genes:
    :param num_workers:
    :return:
    """
    print('Starting function run_in_parallel...')
    result_matrix = np.zeros((n_genes, n_genes))
    pool = multiprocessing.Pool(processes=num_workers)
    start_time = time.time()
    tasks = task_generator(adata, weights, selected_genes, n_genes)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"task_generator: time taken: {total_time:.4f} seconds")
    for result in pool.imap_unordered(compute_pair_m, tasks):
        i, j, value = result
        result_matrix[i, j] = value
        result_matrix[j, i] = value
    pool.close()
    pool.join()
    return result_matrix


def global_bivariate_moran_R(adata, weights: pd.DataFrame, selected_genes, num_workers=4):
    """

    :param adata:
    :param weights:
    :param selected_genes:
    :param num_workers:
    :return:
    """
    n_genes = len(selected_genes)
    start_time = time.time()
    result_matrix = run_in_parallel_m(adata, weights, selected_genes, n_genes, num_workers)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_m: Total time taken: {total_time:.4f} seconds")
    df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
    return df
# def global_bivariate_moran_R(adata, weights, selected_genes):
#     n_genes = len(selected_genes)
#     result_matrix = np.zeros((n_genes, n_genes))
#     for i in range(n_genes):
#         for j in range(i+1, n_genes):
#             result_matrix[i, j] = global_bivariate_moran_R_two_genes(adata, weights, selected_genes[i], selected_genes[j])
#             result_matrix[j, i] = result_matrix[i, j]
#     df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
#     return df


# -----------------------------------------------------#
#               Bivariate Greay's C                    #
# -----------------------------------------------------#
def global_bivariate_gearys_C_two_genes(adata, weights: pd.DataFrame, gene_x: str, gene_y: str):
    """
    Compute bivariate Geary's C value of two given genes
    :param adata: experimental data
    :param weights: spatial weights between two cells/spots
    :param gene_x: gene name
    :param gene_y: gene name
    :return: float, bivariate Geary's C value
    """
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    # 3 get average
    gene_x_exp_mean = adata.X[:, gene_x_id].mean()
    gene_y_exp_mean = adata.X[:, gene_y_id].mean()
    # 4 calculate denominator
    gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
    gene_y_exp = format_gene_array(adata.X[:, gene_y_id])
    denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                  np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    # 5 calulate numerator
    gene_x_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_x_id])
    gene_x_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_x_id])
    gene_y_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_y_id])
    gene_y_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_y_id])
    wij = weights['Weight'].to_numpy()
    numerator = np.sum(wij * (gene_x_in_cell_x - gene_y_in_cell_x) * (gene_y_in_cell_y - gene_x_in_cell_y))
    return numerator / denominator


def compute_pair_c(args):
    """Apply function on two genes"""
    i, j, adata, weights, selected_genes = args
    # Log process information to confirm parallel execution
    process_id = os.getpid()
    # print(f"Computing pair ({i}, {j}) in process ID {process_id}")
    gene1 = selected_genes[i]
    gene2 = selected_genes[j]
    value = global_bivariate_gearys_C_two_genes(adata, weights, gene1, gene2)
    return i, j, value


def run_in_parallel(adata, weights: pd.DataFrame, selected_genes, n_genes, num_workers):
    """

    :param adata:
    :param weights:
    :param selected_genes:
    :param n_genes:
    :param num_workers:
    :return:
    """
    print('Starting function run_in_parallel...')
    result_matrix = np.zeros((n_genes, n_genes))
    pool = multiprocessing.Pool(processes=num_workers)
    start_time = time.time()
    tasks = [(i, j, adata, weights, selected_genes) for i in range(n_genes) for j in range(i + 1, n_genes)]
    end_time = time.time()
    total_time = end_time - start_time
    print(f"list comprehension: time taken: {total_time:.4f} seconds")
    for result in pool.imap_unordered(compute_pair_c, tasks):
        i, j, value = result
        result_matrix[i, j] = value
        result_matrix[j, i] = value
    pool.close()
    pool.join()
    return result_matrix


def run_in_parallel_c(adata, weights: pd.DataFrame, selected_genes, n_genes, num_workers):
    """

    :param adata:
    :param weights:
    :param selected_genes:
    :param n_genes:
    :param num_workers:
    :return:
    """
    print('Starting function run_in_parallel...')
    result_matrix = np.zeros((n_genes, n_genes))
    pool = multiprocessing.Pool(processes=num_workers)
    start_time = time.time()
    tasks = task_generator(adata, weights, selected_genes, n_genes)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"task_generator: time taken: {total_time:.4f} seconds")
    for result in pool.imap_unordered(compute_pair_c, tasks):
        i, j, value = result
        result_matrix[i, j] = value
        result_matrix[j, i] = value
    pool.close()
    pool.join()
    return result_matrix


def global_bivariate_gearys_C(adata, weights: pd.DataFrame, selected_genes, num_workers=4):
    """

    :param adata:
    :param weights:
    :param selected_genes:
    :param num_workers:
    :return:
    """
    n_genes = len(selected_genes)
    start_time = time.time()
    result_matrix = run_in_parallel_c(adata, weights, selected_genes, n_genes, num_workers)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_c: Total time taken: {total_time:.4f} seconds")
    df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
    return df


if __name__ == '__main__':
    print('Loading experimental data...')
    adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h_pca.h5ad')
    # fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E16-18h_pca.h5ad'
    # adata = sc.read_h5ad(fn)
    adata = preprocess(adata)

    # ----TEST SOMDE
    # p_values, selected_genes = somde_p_values(adata, k=20, latent_obsm_key="spatial")
    # print(p_values)
    # print(f'gene num: {len(selected_genes)}')

    # ---- HOTSPOT autocorrelation
    print('HOTSPOT autocorrelation computing...')
    hs, hs_results = hot(adata)
    hs_genes = hs_results.index
    print(f'hs_genes: {len(hs_genes)}')  # hs_genes: 12097
    print('Selecting gene which FDR is lower than 0.05')
    select_genes = hs_results.loc[hs_results.FDR < 0.05].index
    print(f'select_genes: {len(select_genes)}')  # select_genes: 3181 / 3188
    save_list(list(select_genes), 'hotspot_select_genes.txt')
    # select_genes = read_list('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/hotspot_select_genes.txt')
    # sub_adata = adata[:, select_genes]

    # ---- HOTSPOT co-expression
    # local_correlations = hs.compute_local_correlations(select_genes, jobs=12)  # jobs for parallelization
    # print(local_correlations)

    # ---- TESTS
    print('Computing spatial weights matrix...')
    ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key="spatial", n_neighbors=10)
    cell_names = adata.obs_names
    print('Shifting spatial weight matrix shape...')
    fw = flat_weights(cell_names, ind, weights_n, n_neighbors=10)
    # output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E16-18'
    # more_stats = spatial_autocorrelation(adata, layer_key="raw_counts", latent_obsm_key="spatial", n_neighbors=10,
    #                                      n_processes=20, output=output_dir)
    # moran_genes = more_stats.loc[more_stats.FDR_I < 0.05].index
    # print(f'moran find {len(moran_genes)} genes')  # 3860
    # geary_genes = more_stats.loc[more_stats.FDR_C < 0.05].index
    # print(f'geary find {len(geary_genes)} genes')  # 7063
    # getis_genes = more_stats.loc[more_stats.FDR_G < 0.05].index
    # print(f'getis find {len(getis_genes)} genes')  # 4285
    # inter_genes = set.intersection(set(moran_genes), set(geary_genes), set(getis_genes))
    # print(f'intersection gene num (of three stats): {len(inter_genes)}')  # 3416
    # print(f'inter_genes intersection with hotspot genes: {len(inter_genes.intersection(set(select_genes)))}')  # 2728
    #
    # combind_fdrs = combind_fdrs(more_stats[['FDR_C', 'FDR_I', 'FDR_G']])
    # print(combind_fdrs)
    # print(combind_fdrs[combind_fdrs < 0.05].shape[0])  # 6961
    # more_stats['combinded'] = combind_fdrs
    # more_stats.to_csv('more_stats.csv', sep='\t')

    # ----
    # from esda.getisord import G
    # from pysal.lib import weights
    # k = 10
    # ind, neighbors, weights_mtx = neighbors_and_weights(adata, n_neighbors=k)
    # nind = pd.DataFrame(data=ind)
    # nei = nind.transpose().to_dict('list')
    # w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    # w = weights.W(nei, weights=w_dict)
    # gene_expression_matrix = adata.layers['raw_counts']
    # g = G(gene_expression_matrix[:, 0], w)
    # print(g.G)

    # ---TEST BV
    num_w = 20
    print("Computing global bivariate geary'C value in parallel...")
    local_correlations_bv_gc = global_bivariate_gearys_C(adata, fw, select_genes, num_workers=num_w)
    local_correlations_bv_gc.to_csv('local_correlations_bv_gc.csv')

    print("Computing global bivariate Moran's I value in parallel...")
    local_correlations_bv_mr = global_bivariate_moran_R(adata, fw, select_genes, num_workers=num_w)
    local_correlations_bv_mr.to_csv('local_correlations_bv_mr.csv')

