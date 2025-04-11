#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 23:01
# @Author: Yao LI
# @File: SpaGRN/corexp.py
import os
import sys
import time
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import scipy
from scipy.sparse import csr_matrix, issparse
import multiprocessing
from tqdm import tqdm


# --------------------- CO-EXPRESSION --------------------------
# in case sparse X in h5ad
def format_gene_array(gene_array):
    if scipy.sparse.issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)

class D:
    def __init__(self):
        pass
# Bivariate Moran's R
def bv_moran_r(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx):
    """Compute bivariate Moran's R value of two given genes"""
    # 3 get average
    gene_x_exp_mean = mtx[:, gene_x_id].mean()
    gene_y_exp_mean = mtx[:, gene_y_id].mean()
    # 4 calculate denominator
    gene_x_exp = format_gene_array(mtx[:, gene_x_id])
    gene_y_exp = format_gene_array(mtx[:, gene_y_id])
    denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                  np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    # 5 calulate numerator
    gene_x_in_cell_i = format_gene_array(mtx[cell_x_id, gene_x_id])
    gene_y_in_cell_j = format_gene_array(mtx[cell_y_id, gene_y_id])
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_exp_mean) * (gene_y_in_cell_j - gene_y_exp_mean))
    return numerator / denominator


def compute_pairs_m(args):
    """Apply function on two genes"""
    gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx = args
    value = bv_moran_r(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx)
    tf = gene_names[gene_x_id]
    gene = gene_names[gene_y_id]
    return tf, gene, value


def global_bivariate_moran_R(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list, layer_key='raw_counts', num_workers: int = 4) -> pd.DataFrame:
    """
    :param adata:
    :param weights:
    :param tfs_in_data:
    :param select_genes:
    :param layer_key:
    :param num_workers:
    :return:
    """
    gene_names = adata.var.index
    # 1 gene name to matrix id
    tf_ids = adata.var.index.get_indexer(tfs_in_data)
    target_ids = adata.var.index.get_indexer(select_genes)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    if layer_key:
        mtx = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[
            layer_key]
    else:
        mtx = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    wij = weights['Weight'].to_numpy()
    start_time = time.time()
    pool_args = [(gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx) for gene_x_id in tf_ids for gene_y_id in target_ids]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_pairs_m, pool_args)
    results_df = pd.DataFrame(results, columns=['TF', 'target', 'importance'])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_m: Total time taken: {total_time:.4f} seconds")
    return results_df


# Bivariate Greay's C
def bv_geary_c(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx):
    """Compute bivariate Geary's C value of two given genes"""
    gene_x_exp_mean = mtx[:, gene_x_id].mean()
    gene_y_exp_mean = mtx[:, gene_y_id].mean()
    # 4 calculate denominator
    gene_x_exp = format_gene_array(mtx[:, gene_x_id])
    gene_y_exp = format_gene_array(mtx[:, gene_y_id])
    denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                  np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    # 5 calulate numerator
    gene_x_in_cell_i = format_gene_array(mtx[cell_x_id, gene_x_id])
    gene_x_in_cell_j = format_gene_array(mtx[cell_y_id, gene_x_id])
    gene_y_in_cell_i = format_gene_array(mtx[cell_x_id, gene_y_id])
    gene_y_in_cell_j = format_gene_array(mtx[cell_y_id, gene_y_id])
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_in_cell_j) * (gene_y_in_cell_i - gene_y_in_cell_j))
    return numerator / denominator


def compute_pairs_c(args):
    """Apply function on two genes"""
    gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx = args
    value = bv_geary_c(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx)
    tf = gene_names[gene_x_id]
    gene = gene_names[gene_y_id]
    return tf, gene, value


def global_bivariate_gearys_C(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list, layer_key='raw_counts', num_workers: int = 4) -> pd.DataFrame:
    """
    :param adata:
    :param weights:
    :param tfs_in_data:
    :param select_genes:
    :param layer_key:
    :param num_workers:
    :return:
    """
    gene_names = adata.var.index
    # 1 gene name to matrix id
    tf_ids = adata.var.index.get_indexer(tfs_in_data)
    target_ids = adata.var.index.get_indexer(select_genes)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    # 3 get average
    if layer_key:
        mtx = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        mtx = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    wij = weights['Weight'].to_numpy()
    start_time = time.time()
    pool_args = [(gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx) for gene_x_id in tf_ids for gene_y_id in target_ids]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_pairs_c, pool_args)
    results_df = pd.DataFrame(results, columns=['TF', 'target', 'importance'])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_m: Total time taken: {total_time:.4f} seconds")
    return results_df


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
                          n_neighbors=10,
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


def get_p_M(I, adata, weights, gene_x, gene_y, permutation_num=99):
    def __calc(w, zx, zy, den):
        wzy = slag(w, zy)
        num = (zx * wzy).sum()
        return num / den
    gene_matrix = adata.X
    x = gene_matrix[:, gene_x]
    x = np.asarray(x).flatten()
    y = gene_matrix[:, gene_y]
    y = np.asarray(y).flatten()
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    den = x.shape[0] - 1.0
    sim = [__calc(weights, zx, np.random.permutation(zy), den) for i in range(permutation_num)]
    sim = np.array(sim)
    above = sim >= I
    larger = above.sum()
    if (permutation_num - larger) < larger:
        larger = permutation_num - larger
    p_sim = (larger + 1.0) / (permutation_num + 1.0)
    return p_sim


def get_p_C(C, adata, weights, gene_x, gene_y, permutation_num=99):
    sim = [bv_geary_c(adata, weights, gene_x, np.random.permutation(gene_y)) for i in range(permutation_num)]
    sim = np.array(sim)
    above = sim >= C
    larger = above.sum()
    if (permutation_num - larger) < larger:
        larger = permutation_num - larger
    p_sim = (larger + 1.0) / (permutation_num + 1.0)
    return p_sim


def main(data_fn, tfs_fn, genes_fn, layer_key='raw_counts', latent_obsm_key="spatial", n_neighbors=10, fw_fn=None, output_dir='.', num_workers=6):
    print('Loading experimental data...')
    adata = sc.read_h5ad(data_fn)
    adata = preprocess(adata)
    tfs = read_list(tfs_fn)
    select_genes = read_list(genes_fn)
    adata = adata[:, select_genes]
    tfs_in_data = list(set(tfs).intersection(set(adata.var_names)))
    print(f'{len(tfs_in_data)} TFs in data')
    select_genes_not_tfs = list(set(select_genes) - set(tfs_in_data))
    print(f'{len(select_genes_not_tfs)} genes to use.')
    # ---- Weights
    if fw_fn:
        fw = pd.read_csv(fw_fn)
    else:
        print('Computing spatial weights matrix...')
        ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
        cell_names = adata.obs_names
        print('Shifting spatial weight matrix shape...')
        fw = flat_weights(cell_names, ind, weights_n, n_neighbors=n_neighbors)
        fw.to_csv(f'{output_dir}/flat_weights.csv', index=False)
    # --- TEST BV
    print("Computing global bivariate geary'C value in parallel...")
    local_correlations_bv_gc = global_bivariate_gearys_C(adata,
                                                         fw,
                                                         tfs_in_data,
                                                         select_genes,
                                                         num_workers=num_workers,
                                                         layer_key=layer_key)
    local_correlations_bv_gc.to_csv(f'{output_dir}/local_correlations_bv_gc.csv', index=None)
    print("Computing global bivariate Moran's R value in parallel...")
    local_correlations_bv_mr = global_bivariate_moran_R(adata,
                                                        fw,
                                                        tfs_in_data,
                                                        select_genes,
                                                        num_workers=num_workers,
                                                        layer_key=layer_key)
    local_correlations_bv_mr.to_csv(f'{output_dir}/local_correlations_bv_mr.csv', index=None)


if __name__ == '__main__':
    project_id = sys.argv[1]
    if project_id == 'E14-16h':
        data_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E14-16h_pca.h5ad'
        tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/tfs/allTFs_dmel.txt'
        genes_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/hotspot_og/hotspot_select_genes.txt'
        fw_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/E14-16h_flat_weights.csv'
        output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h'
        main(data_fn, tfs_fn, genes_fn, fw_fn=fw_fn, output_dir=output_dir, num_workers=6)
    elif project_id == 'mouse_brain':
        data_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/Mouse_brain_cell_bin.h5ad'
        tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/tfs/allTFs_mm.txt'
        genes_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/mouse_brain/global_genes.txt'
        fw_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/mouse_brain/flat_weights.csv'
        output_dir = f'/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/{project_id}'
        main(data_fn, tfs_fn, genes_fn, layer_key='counts', latent_obsm_key="spatial", n_neighbors=10, fw_fn=fw_fn,
             output_dir=output_dir, num_workers=20)
