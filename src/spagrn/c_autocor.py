#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Dec 2024 15:02
# @Author: Yao LI
# @File: SpaGRN/c_autocor.py
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
# G's C
# -----------------------------------------------------#
# def _gearys_c(adata, weights, layer_key='raw_counts'):
#     if 'connectivities' not in adata.obsp.keys():
#         adata.obsp['connectivities'] = weights
#     gearys_c_array = sc.metrics.gearys_c(adata, layer=layer_key)
#     # shape: (n_genes, )
#     return gearys_c_array
#
#
# def _gearys_c_p_value_one_gene(gene_expression_matrix, n_genes, gene_x_id, weights, gearys_c_array):
#     C = gearys_c_array[gene_x_id]
#     EC = 1
#     K = cal_k(gene_expression_matrix, gene_x_id, n_genes)
#     S0 = cal_s0(weights)
#     S1 = cal_s1(weights)
#     S2 = cal_s2(weights)
#     part1 = (n_genes - 1) * S1 * (n_genes ** 2 - 3 * n_genes + 3 - K * (n_genes - 1)) / (np.square(S0) * n_genes * (n_genes - 2) * (n_genes - 3))
#     part2 = (n_genes ** 2 - 3 - K * np.square(n_genes - 1)) / (n_genes * (n_genes - 2) * (n_genes - 3))
#     part3 = (n_genes - 1) * S2 * (n_genes ** 2 + 3 * n_genes - 6 - K * (n_genes ** 2 - n_genes + 2)) / (4 * n_genes * (n_genes - 2) * (n_genes - 3) * np.square(S0))
#     VC = part1 + part2 - part3
#     # variance = (2 * (n ** 2) * S1 - n * S2 + 3 * (S0 ** 2)) / (S0 ** 2 * (n - 1) * (n - 2) * (n - 3))
#     VC_norm = (1 / (2 * (n_genes + 1) * S0 ** 2)) * ((2 * S1 + S2) * (n_genes - 1) - 4 * S0 ** 2)
#     Z = (C - EC) / np.sqrt(VC_norm)
#     p_value = 1 - norm.cdf(Z)
#     print(f'C: {C}\nVC: {VC}\nVC_norm: {VC_norm}\nZ: {Z}\np_value: {p_value}')
#     return p_value


def gearys_c_p_value_one_gene(x, w):
    c = Geary(x, w)
    return c.p_norm


def gearys_c_z_score_one_gene(x, w):
    c = Geary(x, w)
    return c.z_norm


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


def _compute_c_z_score_for_gene(args):
    x, w = args
    Cz = gearys_c_z_score_one_gene(x, w)
    return Cz


def _gearys_c_z_score_parallel(n_genes, gene_expression_matrix, w, n_processes=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        z_scores = pool.map(_compute_c_z_score_for_gene, pool_args)
    return np.array(z_scores)


def gearys_c(adata, Weights, layer_key='raw_counts', n_process=None, mode='pvalue'):
    """
    Main function to calculate Geary's C and its p-values
    :param adata:
    :param Weights:
    :param layer_key:
    :param n_process:
    :param mode
    :return:
    """
    n_genes = len(adata.var_names)
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    if mode =='pvalue':
        p_values = _gearys_c_parallel(n_genes, gene_expression_matrix, Weights, n_processes=n_process)
        return p_values
    elif mode == 'zscore':
        z_scores = _gearys_c_z_score_parallel(n_genes, gene_expression_matrix, Weights, n_processes=n_process)
        return z_scores

