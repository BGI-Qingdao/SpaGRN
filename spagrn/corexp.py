#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 23:01
# @Author: Yao LI
# @File: SpaGRN/corexp.py
# import numpy as np
# import pandas as pd
# import anndata as ad
# from scipy.sparse import issparse
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


# def memory_time_decorator(func):
#     from memory_profiler import memory_usage
#     def wrapper(*args, **kwargs):
#         # Record start time
#         start_time = time.time()
#         # Measure memory usage
#         mem_usage = memory_usage((func, args, kwargs), max_usage=True)
#         # Record end time
#         end_time = time.time()
#         # Calculate total time taken
#         total_time = end_time - start_time
#         print(f"Maximum memory usage: {mem_usage[0]:.2f} MiB")
#         print(f"Total time taken: {total_time:.4f} seconds")
#         return func(*args, **kwargs)
#     return wrapper
#
#
# # -----------------------------------------------------#
# #               Bivariate Moran's I                    #
# # -----------------------------------------------------#
# def global_bivariate_moran_R_two_genes(adata, weights, gene_x, gene_y):
#     # 1 gene name to matrix id
#     gene_x_id = adata.var.index.get_loc(gene_x)
#     gene_y_id = adata.var.index.get_loc(gene_y)
#     # 2 cell name to matrix id
#     tmp_obs = adata.obs
#     tmp_obs['id'] = np.arange(len(adata.obs))
#     cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
#     cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
#     # 3 get average
#     gene_x_exp_mean = adata.X[:, gene_x_id].mean()
#     gene_y_exp_mean = adata.X[:, gene_y_id].mean()
#     # 4 calculate denominator
#     gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
#     gene_y_exp = format_gene_array(adata.X[:, gene_y_id])
#     denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
#                   np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
#     # 5 calulate numerator
#     gene_x_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_x_id])
#     gene_y_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_y_id])
#     wij = weights['Weight'].to_numpy()
#     numerator = np.sum(wij * (gene_x_in_cell_x - gene_x_exp_mean) * (gene_y_in_cell_y - gene_y_exp_mean))
#     return numerator / denominator
#
#
# @memory_time_decorator
# def global_bivariate_moran_R(adata, weights, selected_genes):
#     n_genes = len(selected_genes)
#     result_matrix = np.zeros((n_genes, n_genes))
#     for i in range(n_genes):
#         for j in range(i+1, n_genes):
#             result_matrix[i, j] = global_bivariate_moran_R_two_genes(adata, weights, selected_genes[i], selected_genes[j])
#             result_matrix[j, i] = result_matrix[i, j]
#     df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
#     return df
#
#
# # -----------------------------------------------------#
# #               Bivariate Greay's C                    #
# # -----------------------------------------------------#
# def global_bivariate_gearys_C_two_genes(adata, weights, gene_x, gene_y):
#     # 1 gene name to matrix id
#     gene_x_id = adata.var.index.get_loc(gene_x)
#     gene_y_id = adata.var.index.get_loc(gene_y)
#     # 2 cell name to matrix id
#     tmp_obs = adata.obs
#     tmp_obs['id'] = np.arange(len(adata.obs))
#     cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
#     cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
#     # 3 get average
#     gene_x_exp_mean = adata.X[:, gene_x_id].mean()
#     gene_y_exp_mean = adata.X[:, gene_y_id].mean()
#     # 4 calculate denominator
#     gene_x_exp = format_gene_array(adata.X[:, gene_x_id])
#     gene_y_exp = format_gene_array(adata.X[:, gene_y_id])
#     denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
#                   np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
#     # 5 calulate numerator
#     gene_x_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_x_id])
#     gene_x_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_x_id])
#     gene_y_in_cell_x = format_gene_array(adata.X[cell_x_id, gene_y_id])
#     gene_y_in_cell_y = format_gene_array(adata.X[cell_y_id, gene_y_id])
#     wij = weights['Weight'].to_numpy()
#     numerator = np.sum(wij * (gene_x_in_cell_x - gene_y_in_cell_x) * (gene_y_in_cell_y - gene_x_in_cell_y))
#     return numerator / denominator
#
#
# # @memory_time_decorator
# # def global_bivariate_gearys_C(adata, weights, selected_genes):
# #     n_genes = len(selected_genes)
# #     result_matrix = np.zeros((n_genes, n_genes))
# #     for i in range(n_genes):
# #         for j in range(i+1, n_genes):
# #             result_matrix[i, j] = global_bivariate_gearys_C_two_genes(adata, weights, selected_genes[i], selected_genes[j])
# #             result_matrix[j, i] = result_matrix[i, j]
# #     df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
# #     return df
#
#
# def compute_pair(i, j, adata, weights, selected_genes):
#     gene_x = selected_genes[i]
#     gene_y = selected_genes[j]
#     result = global_bivariate_gearys_C_two_genes(adata, weights, gene_x, gene_y)
#     return i, j, result
#
#
# def global_bivariate_gearys_C(adata, weights, selected_genes):
#     from concurrent.futures import ProcessPoolExecutor
#     n_genes = len(selected_genes)
#     result_matrix = np.zeros((n_genes, n_genes))
#     # Create a list to store the tasks
#     tasks = []
#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor() as executor:
#         # Create tasks for each pair (i, j) where i < j
#         for i in range(n_genes):
#             for j in range(i + 1, n_genes):
#                 tasks.append(executor.submit(compute_pair, i, j, adata, weights, selected_genes))
#         # Collect the results as they complete
#         for task in tasks:
#             i, j, result = task.result()
#             result_matrix[i, j] = result
#             result_matrix[j, i] = result
#     df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
#     return df


# -----------------------------------------------------#
#               Bivariate Moran's I                    #
# -----------------------------------------------------#
def task_generator(adata, weights: pd.DataFrame, tfs, selected_genes: list):
    for i in range(len(tfs)):
        for j in range(len(selected_genes)):
            yield (i, j, adata, weights, tfs, selected_genes)


def global_bivariate_moran_R_two_genes(adata, weights: pd.DataFrame, gene_x: str, gene_y: str):
    """Compute bivariate Moran's R value of two given genes"""
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    # 3 get average
    mtx = adata.layers['raw_counts']
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
    wij = weights['Weight'].to_numpy()
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_exp_mean) * (gene_y_in_cell_j - gene_y_exp_mean))
    return numerator / denominator


def compute_pair_m(args):
    """Apply function on two genes"""
    # i, j, adata, weights, tfs, selected_genes = args
    adata, weights, tf, gene = args
    print(f'TF: {tf}, gene: {gene}')
    # tf = tfs[i]
    # gene = selected_genes[j]
    value = global_bivariate_moran_R_two_genes(adata, weights, tf, gene)
    return tf, gene, value


def run_in_parallel_m(adata, weights: pd.DataFrame, tfs:list, selected_genes: list, num_workers: int) -> np.array:
    print('Starting function run_in_parallel...')
    # result_matrix = np.zeros((n_genes, n_genes))
    # results = {}
    # tf_list = []
    # gene_list = []
    # bv_mr_list = []
    # pool = multiprocessing.Pool(processes=num_workers)
    # start_time = time.time()
    # tasks = task_generator(adata, weights, tfs, selected_genes)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"task_generator: time taken: {total_time:.4f} seconds")
    # gene_pairs = [(tf, gene) for tf in tfs for gene in selected_genes]
    # pool_args = [(i, j, adata, weights, tfs, selected_genes) for pair in gene_pairs]
    pool_args = [(adata, weights, tf, gene) for tf in tfs for gene in selected_genes]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_pair_m, pool_args)
    # for process, result in enumerate(pool.imap_unordered(compute_pair_m, tasks)):
    #     results[process+1] = result
        # tf, gene, value = result
        # tf_list.append(tf)
        # gene_list.append(gene)
        # bv_mr_list.append(value)
        # result_matrix[i, j] = value
        # result_matrix[j, i] = value
    # pool.close()
    # pool.join()
    # result_df = pd.DataFrame({'TF': tf_list, 'target': gene_list, 'BV_R': bv_mr_list})
    results_df = pd.DataFrame(results, columns=['TF', 'target', 'importance'])
    return results_df


def global_bivariate_moran_R(adata, weights: pd.DataFrame, tfs, selected_genes: list, num_workers: int = 4) -> pd.DataFrame:
    """
    Compute bivariate Moran's R
    :param adata: data containing gene expression matrix
    :param weights: spatial weights matrix for cells/spots
    :param selected_genes: interest genes to compute bivariate Moran's R for
    :param num_workers: number of parallel jobs
    :return: gene x gene, value is bivariate Moran's R
    """
    start_time = time.time()
    df = run_in_parallel_m(adata, weights, tfs, selected_genes, num_workers)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_m: Total time taken: {total_time:.4f} seconds")
    # df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
    return df


# -----------------------------------------------------#
#               Bivariate Greay's C                    #
# -----------------------------------------------------#
def global_bivariate_gearys_C_two_genes(adata, weights: pd.DataFrame, gene_x: str, gene_y: str):
    """Compute bivariate Geary's C value of two given genes"""
    # 1 gene name to matrix id
    gene_x_id = adata.var.index.get_loc(gene_x)
    gene_y_id = adata.var.index.get_loc(gene_y)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    # 3 get average
    mtx = adata.layers['raw_counts']
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
    wij = weights['Weight'].to_numpy()
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_in_cell_j) * (gene_y_in_cell_i - gene_y_in_cell_j))
    return numerator / denominator


def compute_pair_c(args):
    """Apply function on two genes"""
    i, j, adata, weights, selected_genes = args
    process_id = os.getpid()
    print(f"Computing pair ({i}, {j}) in process ID {process_id}")
    gene1 = selected_genes[i]
    gene2 = selected_genes[j]
    value = global_bivariate_gearys_C_two_genes(adata, weights, gene1, gene2)
    return i, j, value


def run_in_parallel_c(adata, weights: pd.DataFrame, selected_genes, n_genes, num_workers) -> np.array:
    """Compute bivariate Geary's C between two genes in a list of genes, in parallel"""
    print('Starting function run_in_parallel...')
    result_matrix = np.zeros((n_genes, n_genes))
    pool = multiprocessing.Pool(processes=num_workers)
    start_time = time.time()
    # tasks = task_generator(adata, weights, selected_genes, n_genes)
    tasks = [(i, j, adata, weights, selected_genes) for i in range(n_genes) for j in range(i + 1, n_genes)]
    print(tasks[10])
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


def global_bivariate_gearys_C(adata, weights: pd.DataFrame, selected_genes: list, num_workers: int = 4) -> pd.DataFrame:
    """
    Compute bivariate Geary's C
    :param adata: data containing gene expression matrix
    :param weights: spatial weights matrix for cells/spots
    :param selected_genes: interest genes to compute bivariate Geary's C for
    :param num_workers: number of parallel jobs
    :return: gene x gene, value is bivariate Geary's C
    """
    n_genes = len(selected_genes)
    start_time = time.time()
    result_matrix = run_in_parallel_c(adata, weights, selected_genes, n_genes, num_workers)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_c: Total time taken: {total_time:.4f} seconds")
    df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
    return df


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


def M(adata, weights, selected_genes, cal_p=False):
    n_genes = len(selected_genes)
    result_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            result_matrix[i, j] = global_bivariate_moran_R_two_genes(adata, weights, selected_genes[i], selected_genes[j])
            result_matrix[j, i] = result_matrix[i, j]
    df = pd.DataFrame(data=result_matrix, index=selected_genes, columns=selected_genes)
    return df


def M2(adata, weights, gene_pairs: pd.DataFrame, cal_p=False):
    bv_rs = []
    ps = []
    for index, row in gene_pairs.iterrows():
    # for pair in gene_pairs:
        gene_x = row['X']
        gene_y = row['Y']
        # gene_x=pair[0]
        # gene_y = pair[1]
        bv_r = global_bivariate_moran_R_two_genes(adata, weights, gene_x, gene_y)
        bv_rs.append(bv_r)
        if cal_p:
            pR = get_p_R(C, adata, weights, gene_x, gene_y, permutation_num=99)
            ps.append(pR)
    gene_pairs['BV_R'] = bv_rs
    if cal_p:
        gene_pairs['BV_P'] = ps
    return gene_pairs


def C(adata, weights, target_genes, cal_p=False):
    target_n_genes = len(target_genes)
    all_genes = adata.var_names
    result_matrix = np.zeros((target_n_genes, len(all_genes)))
    pCs = []
    for i in range(target_n_genes):
        for j in range(len(all_genes)):
            print(f"Computing pair ({target_genes[i]}, {all_genes[j]})")
            if target_genes[i] == all_genes[j]:
                result_matrix[i, j] = 0
            else:
                C = global_bivariate_gearys_C_two_genes(adata, weights, target_genes[i], all_genes[j])
                result_matrix[i, j] = C
                if cal_p:
                    pC = get_p_C(C, adata, weights, gene_x, gene_y, permutation_num=99)
                    pCs.append(pC)
                # result_matrix[j, i] = result_matrix[i, j]
            print(result_matrix.sum())
    df = pd.DataFrame(data=result_matrix, index=target_genes, columns=all_genes)
    return df


def C2(adata, weights, gene_pairs: pd.DataFrame, cal_p=False):
    bv_cs = []
    ps = []
    for index, row in gene_pairs.iterrows():
        gene_x = row['X']
        gene_y = row['Y']
        bv_c = global_bivariate_gearys_C_two_genes(adata, weights, gene_x, gene_y)
        bv_cs.append(bv_c)
        if cal_p:
            pC = get_p_C(C, adata, weights, gene_x, gene_y, permutation_num=99)
            ps.append(pC)
    gene_pairs['BV_C'] = bv_cs
    if cal_p:
        gene_pairs['BV_P'] = ps
    return gene_pairs


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
    sim = [global_bivariate_gearys_C_two_genes(adata, weights, gene_x, np.random.permutation(gene_y)) for i in range(permutation_num)]
    sim = np.array(sim)
    above = sim >= C
    larger = above.sum()
    if (permutation_num - larger) < larger:
        larger = permutation_num - larger
    p_sim = (larger + 1.0) / (permutation_num + 1.0)
    return p_sim


def get_gene_pairs(genes_list):
    gene_pairs = [(genes_list[i], genes_list[j]) for i in range(len(genes_list)) for j in range(i + 1, len(genes_list))]
    return gene_pairs


if __name__ == '__main__':
    print('Loading experimental data...')
    adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E14-16h_pca.h5ad')
    adata = preprocess(adata)

    tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/tfs/allTFs_dmel.txt'
    tfs = read_list(tfs_fn)

    # fn = sys.argv[1]
    # file_index = sys.argv[2]
    # output_dir = sys.argv[3]
    # gene_pairs = pd.read_csv(fn, header=None)
    # gene_pairs.columns = ['X', 'Y']
    select_genes = read_list('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/hotspot_og/hotspot_select_genes.txt')
    adata = adata[:, select_genes]

    # ---- Weights
    # print('Computing spatial weights matrix...')
    # ind, neighbors, weights_n = neighbors_and_weights(sub_adata, latent_obsm_key="spatial", n_neighbors=10)
    # cell_names = sub_adata.obs_names
    # print('Shifting spatial weight matrix shape...')
    # fw = flat_weights(cell_names, ind, weights_n, n_neighbors=10)
    # fw.to_csv('E14-16h_flat_weights.csv', index=False)
    fw = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/E14-16h_flat_weights.csv')

    # --- TEST BV
    # print("Computing global bivariate geary'C value in parallel...")
    # local_correlations_bv_gc = global_bivariate_gearys_C(sub_adata, fw, select_genes, num_workers=num_files)
    # local_correlations_bv_gc = C(adata, fw, target_genes)
    # local_correlations_bv_gc = C2(adata, fw, gene_pairs) # 50genes, Total time taken: 127.5423 seconds
    # local_correlations_bv_gc.to_csv(f'{output_dir}/local_correlations_bv_gc_{file_index}.csv', index=None)

    print("Computing global bivariate Moran's R value in parallel...")
    tfs_in_data = list(set(tfs).intersection(set(adata.var_names)))
    print(f'{len(tfs_in_data)} TFs in data')
    select_genes_not_tfs = list(set(select_genes)-set(tfs_in_data))
    print(f'{len(select_genes_not_tfs)} genes to use.')
    local_correlations_bv_mr = global_bivariate_moran_R(adata, fw, tfs_in_data, select_genes, num_workers=12)
    local_correlations_bv_mr.to_csv('local_correlations_bv_mr.csv', index=None)
    # gene_pairs = get_gene_pairs(list(adata.var_names))
    # print(f'{len(gene_pairs)} gene pairs')
    # start_time = time.time()
    # local_correlations_bv_mr = M2(adata, fw, gene_pairs)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total time taken: {total_time:.4f} seconds")  # 50genes, Total time taken: 120.3324 seconds
    # local_correlations_bv_mr.to_csv(f'{output_dir}/local_correlations_bv_mr_{file_index}.csv', index=None)
