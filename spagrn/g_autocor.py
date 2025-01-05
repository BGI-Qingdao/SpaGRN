#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Dec 2024 15:00
# @Author: Yao LI
# @File: SpaGRN/g_autocor.py
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
# Getis Ord General G
# -----------------------------------------------------#
# def _getis_g(x, w):
#     x = np.asarray(x)
#     w = np.asarray(w)
#     numerator = np.sum(np.sum(w * np.outer(x, x)))
#     denominator = np.sum(np.outer(x, x))
#     G = numerator / denominator
#     return G
#
#
# def _getis_g_p_value_one_gene(G, w, x):
#     n = w.shape[0]
#     s0 = cal_s0(w)
#     s02 = s0 * s0
#     s1 = cal_s1(w)
#     b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
#     b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
#     b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
#     b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
#     b4 = s1 - s2 + s02
#     EG = s0 / (n * (n - 1))
#     numerator = b0 * (np.square(np.sum(x ** 2))) + b1 * np.sum(np.power(x, 4)) + b2 * np.square(np.sum(x)) * np.sum(
#         x ** 2) + b3 * np.sum(x) * np.sum(np.power(x, 3)) + b4 * np.power(np.sum(x), 4)
#     denominator = np.square((np.square(np.sum(x)) - np.sum(x ** 2))) * n * (n - 1) * (n - 2) * (n - 3)
#     VG = numerator / denominator - np.square(EG)
#     Z = (G - EG) / np.sqrt(VG)
#     p_value = 1 - norm.cdf(Z)
#     # print(f'G: {G}\nVG: {VG}\nZ: {Z}\np_value: {p_value}')
#     return p_value
#
#
# def getis_g_p_values_one_gene(gene_expression_matrix, gene_x_id, w):
#     g = G(gene_expression_matrix[:, gene_x_id], w)
#     return g.p_norm


# parallel computing
# p-values
def _compute_g_for_gene(args):
    gene_expression_matrix, gene_x_id, w = args
    g = G(gene_expression_matrix[:, gene_x_id], w)
    # print(f'gene{gene_x_id}: p_value: {g.p_norm}')
    return g.p_norm


def _getis_g_parallel(gene_expression_matrix, w, n_genes, n_processes=None):
    pool_args = [(gene_expression_matrix, gene_x_id, w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        Gp_values = pool.map(_compute_g_for_gene, pool_args)
    return np.array(Gp_values)


# z-scores
def _compute_g_zscore_for_gene(args):
    gene_expression_matrix, gene_x_id, w = args
    g = G(gene_expression_matrix[:, gene_x_id], w)
    # print(f'gene{gene_x_id}: z_score: {g.z_norm}')
    return g.z_norm


def _getis_g_zscore_parallel(gene_expression_matrix, w, n_genes, n_processes=None):
    pool_args = [(gene_expression_matrix, gene_x_id, w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        Gp_values = pool.map(_compute_g_zscore_for_gene, pool_args)
    return np.array(Gp_values)


def getis_g(adata,
            Weights,
            n_processes=None,
            layer_key=None,
            mode='pvalue'):
    """
    Calculate getis ord general g for all genes and return getis_g_p_values values as a numpy.array
    :param adata: data containing gene expression matrix and cell-feature spatial coordinates array
    :param Weights:
    :param n_processes: number of jobs when computing parallelly
    :param layer_key:
    :param mode: to calculate p-values or z-scores for the statistics
    :return: (numpy.array) dimension: (n_genes, )
    """
    assert mode in ['pvalue', 'zscore']
    n_genes = len(adata.var_names)
    # prepare gene expression matrix format
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key].toarray() if scipy.sparse.issparse(
            adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

    # start calculation
    if mode == 'pvalue':
        p_values = _getis_g_parallel(gene_expression_matrix, Weights, n_genes, n_processes=n_processes)
        return p_values
    elif mode == 'zscore':
        z_scores = _getis_g_zscore_parallel(gene_expression_matrix, Weights, n_genes, n_processes=n_processes)
        return z_scores
