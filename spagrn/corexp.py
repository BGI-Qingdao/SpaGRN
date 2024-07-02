#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 23:01
# @Author: Yao LI
# @File: SpaGRN/corexp.py
import numpy as np
import pandas as pd
import anndata as ad
import multiprocess
from scipy.sparse import issparse


# --------------------- CO-EXPRESSION --------------------------
# in case sparse X in h5ad
def format_gene_array(gene_array):
    if scipy.sparse.issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)


# -----------------------------------------------------#
#               Bivariate Moran's I                    #
# -----------------------------------------------------#
def global_bivariate_moran_R_two_genes(adata, weights, gene_x, gene_y):
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs
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


def global_bivariate_moran_R(adata, weights):
    gbc_list = []
    for gene_x, gene_y in zip():
        gbc = global_bivariate_moran_R_two_genes(adata, weights, gene_x, gene_y)
        gbc_list.append(gbc)


# -----------------------------------------------------#
#               Bivariate Greay's C                    #
# -----------------------------------------------------#
def global_bivariate_gearys_C_two_genes(adata, weights, gene_x, gene_y):
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs
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


def global_bivariate_gearys_C(adata, weights):
    gbc_list = []
    for gene_x, gene_y in zip():
        gbc = global_bivariate_gearys_C_two_genes(adata, weights, gene_x, gene_y)
        gbc_list.append(gbc)
