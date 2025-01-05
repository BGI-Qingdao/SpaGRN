#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Dec 2024 15:01
# @Author: Yao LI
# @File: SpaGRN/m_autocor.py
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
# M's I
# -----------------------------------------------------#
# def _morans_i(adata, weights, layer_key='raw_counts'):
#     if 'connectivities' not in adata.obsp.keys():
#         adata.obsp['connectivities'] = weights
#     morans_i_array = sc.metrics.morans_i(adata, layer=layer_key)
#     # shape: (n_genes, )
#     return morans_i_array
#
#
# def _morans_i_p_value_one_gene(adata, gene_x_id, weights, morans_i_array):
#     I = morans_i_array[gene_x_id]  # moran's I stats for the gene
#     n = len(adata.obs_names)  # number of cells
#     EI = -1 / (n - 1)  # Moran’s I expected value
#     K = cal_k(adata, gene_x_id, n)
#     S0 = cal_s0(weights)
#     S1 = cal_s1(weights)
#     S2 = cal_s2(weights)
#     # Variance
#     part1 = (n * (S1 * (n ** 2 - 3 * n + 3) - n * S2 + 3 * np.square(S0))) / (
#             (n - 1) * (n - 2) * (n - 3) * np.square(S0))
#     part2 = (K * (S1 * (n ** 2 - n) - 2 * n * S2 + 6 * np.square(S0))) / ((n - 1) * (n - 2) * (n - 3) * np.square(S0))
#     VI = part1 - part2 - np.square(EI)
#     stdI = np.sqrt(VI)
#     # Z score
#     Z = (I - EI) / stdI
#     return Z
#     # Perform one-tail test one z score
#     # p_value = 1 - norm.cdf(Z)  # right tail
#     # return p_value
#
#
# def morans_i_p_value_one_gene(x, w):
#     i = Moran(x, w)
#     return i.p_norm


# parallel computing
def _compute_i_for_gene(args):
    x, w = args
    i = Moran(x, w)
    return i.p_norm


def _compute_i_zscore_for_gene(args):
    x, w = args
    i = Moran(x, w)
    return i.z_norm


def _morans_i_parallel(n_genes, gene_expression_matrix, w, n_processes=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        p_values = pool.map(_compute_i_for_gene, pool_args)
    return np.array(p_values)


def _morans_i_zscore_parallel(n_genes, gene_expression_matrix, w, n_processes=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        z_scores = pool.map(_compute_i_zscore_for_gene, pool_args)
    return np.array(z_scores)


def morans_i_p_values(adata, Weights, layer_key='raw_counts', n_process=None):
    """
    Calculate Moran’s I Global Autocorrelation Statistic and its adjusted p-value
    :param adata: Anndata
    :param Weights:
    :param layer_key:
    :param n_process:
    :return:
    """
    n_genes = len(adata.var_names)
    # nind = pd.DataFrame(data=ind)
    # nei = nind.transpose().to_dict('list')
    # w_dict = weights_n.reset_index(drop=True).transpose().to_dict('list')
    # w = weights.W(nei, weights=w_dict)
    # w = get_w(ind, weights_n)
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    p_values = _morans_i_parallel(n_genes, gene_expression_matrix, Weights, n_processes=n_process)
    return p_values


def morans_i_zscore(adata, Weights, layer_key='raw_counts', n_process=None):
    """
    Calculate Moran’s I Global Autocorrelation Statistic and its adjusted p-value
    :param adata: Anndata
    :param Weights:
    :param layer_key:
    :param n_process:
    :return:
    """
    n_genes = len(adata.var_names)
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    z_scores = _morans_i_zscore_parallel(n_genes, gene_expression_matrix, Weights, n_processes=n_process)
    return z_scores
