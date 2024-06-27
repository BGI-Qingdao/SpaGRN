#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 19 Jun 2024 09:05
# @Author: Yao LI
# @File: SpaGRN/gis.py
import scanpy as sc
import numpy as np
import pandas as pd

import hotspot
from math import ceil
from pynndescent import NNDescent
from scipy.stats import chi2, norm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import multiprocessing


def compute_weights(distances, neighborhood_factor=3):
    """
    Computes weights on the nearest neighbors based on a
    gaussian kernel and their distances
    Kernel width is set to the num_neighbors / neighborhood_factor's distance
    distances:  cells x neighbors ndarray
    neighborhood_factor: float
    returns weights:  cells x neighbors ndarray
    """
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)
    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * distances ** 2 / sigma ** 2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm
    return weights


def neighbors_and_weights(data, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True):
    """
    Computes nearest neighbors and associated weights for data
    Uses euclidean distance between rows of `data`

    Parameters
    ==========
    data: pandas.Dataframe num_cells x num_features

    Returns
    =======
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """

    coords = data.obsm['spatial']

    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm="ball_tree").fit(coords)
    dist, ind = nbrs.kneighbors()
    print(type(dist))
    print(type(ind))

    weights = compute_weights(
        dist, neighborhood_factor=neighborhood_factor)

    ind_df = pd.DataFrame(ind, index=data.obs_names)
    neighbors = ind_df
    print(neighbors)
    weights = pd.DataFrame(weights, index=neighbors.index,
                           columns=neighbors.columns)

    return ind, neighbors, weights


def flat_weights(cell_names, ind, weights, n_neighbors=30):
    cell1 = np.repeat(cell_names, n_neighbors)
    cell2_indices = ind.flatten()  # starts from 0
    cell2 = cell_names[cell2_indices]
    weight = weights.flatten()

    df = pd.DataFrame({
        "Cell_x": cell1,
        "Cell_y": cell2,
        "Weight": weight
    })
    return df


# Spatial Weights
def compute_spatial_weights(coords):
    """
    Use euclidean distance as spatial weights
    :param coords:
    :return:
    """
    distances = squareform(pdist(coords))  # euclidean distance
    # weights = np.zeros_like(distances)
    # Inverse distance weights
    np.fill_diagonal(distances, 1)  # Avoid division by zero for self-distances
    weights = 1 / distances
    np.fill_diagonal(weights, 0)  # when i=j, w=0. Set diagonal to 0 to ignore self-weights
    return weights


def compute_sparse_spatial_weights(coords, k=5):
    """
    Compute the sparse spatial weights matrix using nearest neighbors approach for 3D coordinates.
    Parameters:
    coords (array-like): 2D array of 3D spatial coordinates.
    k (int): Number of nearest neighbors to consider.
    Returns:
    csr_matrix: Sparse spatial weights matrix.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # weights = np.zeros_like(distances)
    # weights[:, 1:] = 1 / distances[:, 1:]  # Inverse distance weights excluding self-distance

    radius_ii = ceil(distances.shape[1] / 3)
    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * distances ** 2 / sigma ** 2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm

    row_indices = np.repeat(np.arange(coords.shape[0]), k)
    col_indices = indices[:, 1:].flatten()
    values = weights[:, 1:].flatten()
    spatial_weights = csr_matrix((values, (row_indices, col_indices)), shape=(coords.shape[0], coords.shape[0]))
    spatial_weights = spatial_weights + spatial_weights.T  # Make symmetric
    return spatial_weights


# Spatial Auto-correlation Statics
def getis_ord_general_g(x, w):
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


def compute_g_for_gene(args):
    gene_idx, gene_expression_matrix, spatial_weights = args
    gene_expression = gene_expression_matrix[:, gene_idx]
    G = getis_ord_general_g(gene_expression, spatial_weights)
    return G


def getis_g_parallel(gene_expression_matrix, spatial_weights, n_processes=None):
    n_genes = gene_expression_matrix.shape[1]
    pool_args = [(gene_idx, gene_expression_matrix, spatial_weights) for gene_idx in range(n_genes)]
    with multiprocessing.Pool(processes=n_processes) as pool:
        G_values = pool.map(compute_g_for_gene, pool_args)
    return np.array(G_values)


def getis_g(adata, weight='knn', layer_key='raw_counts', latent_obsm_key="spatial", n_neighbors=10):
    # Step1: get cell coordinates and gene expression matrix
    cell_coordinates = adata.obsm[latent_obsm_key]
    if layer_key:
        gene_expression_matrix = adata.layers[layer_key]
    else:
        gene_expression_matrix = adata.X
    # Step2: Compute the spatial weights matrix
    weight = compute_sparse_spatial_weights(cell_coordinates)
    print(weight.shape)
    neighbors, weights = neighbors_and_weights(cell_coordinates, n_neighbors=n_neighbors)
    print(neighbors)
    print(weights.shape)
    # Step3: Calculate Getis-Ord General G for each gene
    G_values = []
    for gene_idx in range(gene_expression_matrix.shape[1]):
        gene_expression = gene_expression_matrix[:, gene_idx]
        G = getis_ord_general_g(gene_expression, weights)
        G_values.append(G)
        # print(f"Getis-Ord General G statistic for gene {gene_idx + 1}: {G}")
    G_values = np.array(G_values)
    return G_values


def pseudo_data(num_cells=10, num_genes=5):
    # Generate a non-zero gene expression matrix (cells x genes)
    # Values are randomly chosen from 1 to 100
    np.random.seed(0)  # For reproducibility
    gene_expression_matrix = np.random.randint(1, 101, size=(num_cells, num_genes))
    # Generate a non-zero 3D spatial coordinates matrix for cells
    # Values are randomly chosen from 1 to 50
    cell_coordinates = np.random.randint(1, 51, size=(num_cells, 3))
    return gene_expression_matrix, cell_coordinates


# Compute p-values for spatial auto-correlation statics
def tests(adata, Hx: np.array, layer_key='raw_counts', latent_obsm_key="spatial", n_neighbors=10) -> np.array:
    """
    :param adata:
    :param Hx:
    :param layer_key:
    :param latent_obsm_key:
    :return:
    """
    # sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    ind, neighbors, weights = neighbors_and_weights(cell_coordinates, n_neighbors=n_neighbors)
    adata.uns["neighbors"]["connectivities"] = weights
    pc_c_m = sc.metrics.morans_i(adata, layer=layer_key)  # shape: (n_gene, )
    pc_c_c = sc.metrics.gearys_c(adata, layer=layer_key)
    pc_c_g = getis_g(adata, layer_key=layer_key, latent_obsm_key=latent_obsm_key)

    p_values = np.array([pc_c_m, pc_c_c, pc_c_g, Hx])
    return p_values


# def calculate_p_values(stat_values, n_cells):
#     expected_value = -1 / (n_cells - 1)
#     print(expected_value)
#     variance = (n_cells ** 2 - 3 * n_cells + 3) / ((n_cells - 1) * (n_cells - 2) * (n_cells + 1)) - expected_value ** 2
#     print(variance)
#     z_scores = (stat_values - expected_value) / np.sqrt(variance)
#     p_values = 2 * norm.cdf(-np.abs(z_scores))  # 双尾检验
#     return p_values


def calculate_morans_i_p_value(moran_i, n_cells, weights):  # TODO: why use spatial weights here?
    # Calculate the expected value of Moran's I
    E_I = -1 / (n_cells - 1)
    print(E_I)
    # Calculate the variance of Moran's I
    S0 = np.sum(weights)
    S1 = 0.5 * np.sum((weights + weights.T) ** 2)
    S2 = np.sum((np.sum(weights, axis=0) + np.sum(weights, axis=1)) ** 2)
    EI_squared = E_I ** 2
    Var_I = (n_cells ** 2 * S1 - n_cells * S2 + 3 * S0 ** 2) / ((n_cells - 1) * (n_cells + 1) * S0 ** 2) - EI_squared
    print(Var_I)
    # Calculate the standard deviation of Moran's I
    std_I = np.sqrt(Var_I)
    # Calculate the Z-score
    z_score = (moran_i - E_I) / std_I
    # Calculate the p-value based on the Z-score
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    return p_value


def gearys_c_variance(x, w):
    n = len(x)
    W = np.sum(w)
    mean_x = np.mean(x)
    # Calculate the terms needed for variance
    S0 = W
    S1 = np.sum([(w[i, j] * (x[i] - x[j]) ** 2) ** 2 for i in range(n) for j in range(n)])
    S2 = np.sum([(np.sum([w[i, j] * (x[i] - x[j]) ** 2 for j in range(n)])) ** 2 for i in range(n)])
    # Expected value of C
    E_C = 1
    # Variance of C
    var_C = ((n - 1) ** 2 * S1 - (n - 1) * S2 + 3 * S0 ** 2) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)
    return var_C


def fdr(pvalues, alpha=0.05):
    """
    Calculate the p-value cut-off to control for
    the false discovery rate (FDR) for multiple testing.
    If by controlling for FDR, all of n null hypotheses
    are rejected, the conservative Bonferroni bound (alpha/n)
    is returned instead.
    Parameters
    ----------
    pvalues     : array
                  (n, ), p values for n multiple tests.
    alpha       : float, optional
                  Significance level. Default is 0.05.
    Returns
    -------
                : float
                  Adjusted criterion for rejecting the null hypothesis.
                  If by controlling for FDR, all of n null hypotheses
                  are rejected, the conservative Bonferroni bound (alpha/n)
                  is returned.
    Notes
    -----
    For technical details see :cite:`Benjamini:2001` and
    :cite:`Castro:2006tz`.
    """

    n = len(pvalues)
    p_sort = np.sort(pvalues)[::-1]
    index = np.arange(n, 0, -1)
    p_fdr = index * alpha / n
    search = p_sort < p_fdr
    sig_all = np.where(search)[0]
    if len(sig_all) == 0:
        return alpha / n
    else:
        return p_fdr[sig_all[0]]


# Combine p-values
def fishers_method(p_values: np.array) -> np.array:
    """
    Combines p-values using Fisher's Combined Probability Test.

    Fisher's method combines multiple p-values into a single p-value
    by summing the negative natural logarithms of the individual p-values
    and using a chi-squared distribution.
    :param p_values: A 2D NumPy array where each row represents a set of p-values
        to be combined. Shape should be (m, n), where m is the number
        of tests and n is the number of p-values per test.
    :return: A 1D array of combined p-values for each test, with length m.
    """
    chi2_stat = -2 * np.sum(np.log(p_values), axis=0)
    p_combined = chi2.sf(chi2_stat, 2 * p_values.shape[0])
    return p_combined


def stouffers_method(p_values: np.array) -> np.array:
    """
    Combines p-values using Stouffer's Z-score Method.

    Stouffer's method combines multiple p-values into a single p-value
    by converting each p-value to a z-score, summing the z-scores,
    and then converting the sum back to a p-value using the standard
    normal distribution.
    :param p_values: A 2D NumPy array where each row represents a set of p-values
        to be combined. Shape should be (m, n), where m is the number
        of tests and n is the number of p-values per test.
    :return:
    """
    z_scores = norm.ppf(1 - p_values)
    combined_z = np.sum(z_scores, axis=0) / np.sqrt(p_values.shape[0])
    p_combined = norm.sf(combined_z)
    return p_combined


def combine(p_values: np.array, method: str) -> np.array:
    """
    Combine p-values
    """
    if method not in ['fisher', 'stouffer']:
        print()
    global combined_p_values
    if method == 'fisher':
        combined_p_values = fishers_method(p_values)
        print("Fisher's combined p-values:", combined_p_values)
    elif method == 'stouffer':
        combined_p_values = stouffers_method(p_values)
        print("Stouffer's combined p-values:", combined_p_values)
    return combined_p_values


# Main Function
def get_combined_p_values(adata,
                          Hx,
                          layer_key='raw_counts',
                          latent_obsm_key="spatial",
                          n_neighbors=10,
                          method='fisher'):
    p_values = tests(adata, Hx, layer_key=layer_key, latent_obsm_key=latent_obsm_key, n_neighbors=n_neighbors)
    combined_p_values = combine(p_values, method=method)
    return combined_p_values


if __name__ == '__main__':
    layer_key = 'raw_counts'
    model = 'bernoulli'
    latent_obsm_key = "spatial"
    umi_counts_obs_key = None
    weighted_graph = False
    n_neighbors = 10

    adata = sc.read_h5ad('E14-16h_pca.h5ad')
    print(type(adata.obs_names))
    print(adata.obs_names)
    cell_coordinates = adata.obsm['spatial']
    gene_expression_matrix = adata.layers['raw_counts']

    # weight = compute_sparse_spatial_weights(cell_coordinates)
    # print(f'weight shape: {weight.shape}')
    ind, neighbors, weights = neighbors_and_weights(adata, n_neighbors=n_neighbors)
    print(f'neighbors: \n{neighbors}')
    print(f'weights shape: {weights.shape}')
    fw = flat_weights(adata.obs_names, ind, weights, n_neighbors=n_neighbors)
    print(fw)

    # Test OG hotspot
    # hs = hotspot.Hotspot(adata,
    #                      layer_key=layer_key,
    #                      model=model,
    #                      latent_obsm_key=latent_obsm_key,
    #                      umi_counts_obs_key=umi_counts_obs_key)
    # hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)
    # hs_results = hs.compute_autocorrelations()

    # /dellfsqd2/ST_OCEAN/USER/liyao1/tools/anaconda3/envs/test/lib/python3.8/site-packages/hotspot/
