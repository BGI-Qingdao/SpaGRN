#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

# python core modules
import os
import csv
import sys
import logging
from typing import Union

# third party modules
import json
import glob
import anndata
import hotspot
import scipy.sparse
import pandas as pd
import numpy as np
import scanpy as sc
from copy import deepcopy
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.rss import regulon_specificity_scores
from pyscenic.aucell import aucell

# modules in self project
from .spa_logger import logger


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]


class ScoexpMatrix:
    """
    Algorithms to calulate Scoexp matrix 
    based on CellTrek (10.1038/s41587-022-01233-1) 
    see CellTrek from https://github.com/navinlabcode/CellTrek
    """

    @staticmethod
    def rbfk(dis_mat, sigm, zero_diag=True):
        """
        Radial basis function kernel
        
        :param dis_mat: Distance matrix
        :param sigm: Width of rbfk
        :param zero_diag:
        :return rbf matrix
        """
        rbfk_out = np.exp(-1 * np.square(dis_mat) / (2 * sigm ** 2))
        if zero_diag:
            rbfk_out[np.diag_indices_from(rbfk_out)] = 0
        return rbfk_out

    @staticmethod
    def wcor(X, W, method='pearson', na_zero=True):
        """
        Weighted cross correlation
        
        :param X: Expression matrix, n X p
        :param W: Weight matrix, n X n
        :param method: Correlation method, pearson or spearman
        :param na_zero: Na to zero
        :return correlation matrix
        """
        from scipy.stats import rankdata
        from sklearn.preprocessing import scale
        if method == 'spearman':
            X = np.apply_along_axis(rankdata, 0, X)  # rank each columns
        X = scale(X, axis=0)  # scale for each columns
        W_cov_temp = np.matmul(np.matmul(X.T, W), X)
        W_diag_mat = np.sqrt(np.matmul(np.diag(W_cov_temp), np.diag(W_cov_temp).T))
        cor_mat = W_cov_temp / W_diag_mat
        if na_zero:
            np.nan_to_num(cor_mat, False)
        return cor_mat

    @staticmethod
    def scoexp(irn_data,
               gene_list: list = [],
               tf_list: list = [],
               sigm=15,
               zero_cutoff=5,
               cor_method='spearman',
               save: bool = True,
               fn: str = 'adj.csv',
               ):
        """
        Main logic for scoexp calculation
 
        :param irn_data: object of InferenceRegulatoryNetwork
        :param sigm: sigma for RBF kernel, default 15.
        :param gene_list: filter gene by exp cell > zero_cutoff% of all cells if len(gene_list)<2, otherwise use this gene set.
        :param tf_list: tf gene list. Use gene_list if tf_list is empty.
        :param zero_cutoff: filter gene by exp cell > zero_cutoff% if if len(gene_list)<2
        :param cor_method: 'spearman' or 'pearson'
        :return: dataframe of tf-gene-importances
        """
        from scipy.spatial import distance_matrix
        cell_gene_matrix = irn_data.matrix
        if not isinstance(cell_gene_matrix, np.ndarray):
            cell_gene_matrix = cell_gene_matrix.toarray()
        # check gene_list
        if len(gene_list) < 2:
            logger.info('gene filtering...')
            feature_nz = np.apply_along_axis(lambda x: np.mean(x != 0) * 100, 0, cell_gene_matrix)
            features = irn_data.gene_names[feature_nz > zero_cutoff]
            logger.info(f'{len(features)} features after filtering...')
        else:
            features = np.intersect1d(np.array(gene_list), irn_data.gene_names)
            if len(features) < 2:
                logger.error('No enough genes in gene_list detected, exit...')
                sys.exit(12)
        # check tf_list
        if len(tf_list) < 1:
            tf_list = features
        else:
            tf_list = np.intersect1d(np.array(tf_list), features)

        gene_select = np.isin(irn_data.gene_names, features, assume_unique=True)
        celltrek_inp = cell_gene_matrix[:, gene_select]
        dist_mat = distance_matrix(irn_data.position,
                                   irn_data.position)
        kern_mat = ScoexpMatrix.rbfk(dist_mat, sigm=sigm, zero_diag=False)
        logger.info('Calculating spatial-weighted cross-correlation...')
        wcor_mat = ScoexpMatrix.wcor(X=celltrek_inp, W=kern_mat, method=cor_method)
        logger.info('Calculating spatial-weighted cross-correlation done.')
        df = pd.DataFrame(data=wcor_mat, index=features, columns=features)
        # extract tf-gene-importances
        df = df[tf_list].copy().T
        df['TF'] = tf_list
        ret = df.melt(id_vars=['TF'])
        ret.columns = ['TF', 'target', 'importance0']
        maxV = ret['importance0'].max()
        ret['importance'] = ret['importance0'] / maxV
        ret['importance'] = ret['importance'] * 1000
        ret.drop(columns=['importance0'], inplace=True)
        ret['valid'] = ret.apply(lambda row: row['TF'] != row['target'], axis=1)
        ret = ret[ret['valid']].copy()
        ret.drop(columns=['valid'], inplace=True)
        # ret.to_csv('adj.csv',header=True,index=False)
        if save:
            ret.to_csv(fn, index=False)
        return ret


class InferRegulatoryNetwork:
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    def __init__(self, data, pos_label='spatial'):
        """
        Constructor of this Object.
        :param data:
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        # input
        self._data = data
        self._matrix = None  # pd.DataFrame
        self._gene_names = None
        self._cell_names = None
        self._position = None  # np.array
        self._tfs = None  # list

        self.load_data_info(pos_label)

        # network calculated attributes
        self._regulon_list = None  # list
        self._auc_mtx = None  # pd.DataFrame
        self._adjacencies = None  # pd.DataFrame
        self._regulon_dict = None  # dictionary
        self._rss = None  # pd.DataFrame

        # other settings
        self._params = {
             'hotspot': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': False,
            },
            'grnboost': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': False,
            },
            'scoexp': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': True,
            }}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: anndata.AnnData, pos_label='spatial'):
        """
        re-assign data for this object.
        :param data:
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        self._data = data
        self.load_data_info(pos_label)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def gene_names(self):
        return self._gene_names

    @gene_names.setter
    def gene_names(self, value):
        self._gene_names = value

    @property
    def cell_names(self):
        return self._cell_names

    @cell_names.setter
    def cell_names(self, value):
        self._cell_names = value

    @property
    def adjacencies(self):
        return self._adjacencies

    @adjacencies.setter
    def adjacencies(self, value):
        self._adjacencies = value

    @property
    def regulon_list(self):
        return self._regulon_list

    @regulon_list.setter
    def regulon_list(self, value):
        self._regulon_list = value

    @property
    def regulon_dict(self):
        return self._regulon_dict

    @regulon_dict.setter
    def regulon_dict(self, value):
        self._regulon_dict = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def rss(self):
        return self._rss

    @rss.setter
    def rss(self, value):
        self._rss = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        """only use this function when setting params as a whole.
        use add_params to solely update/add some of the params and keep the rest unchanged"""
        self._params = value

    def add_params(self, method: str, dic: dict):
        """
        :param method:
        :param dic:

        Example:
            grn = InferenceRegulatoryNetwork(data)
            grn.add_params('hotspot', {'num_worker':12, 'auc_threshold': 0.001})
        """
        og_params = deepcopy(self._params)
        try:
            for key, value in dic:
                self._params[method][key] = value
        except KeyError:
            logger.warning('KeyError, params did not change')  # TODO
            self._params = og_params

    def load_data_info(self, pos_label):
        """
        Load useful data to properties.
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        # if isinstance(self._data, StereoExpData):
        #     self._matrix = self._data.exp_matrix
        #     self._gene_names = self._data.gene_names
        #     self._cell_names = self._data.cell_names
        #     self._position = self._data.position
        # elif isinstance(self._data, anndata.AnnData):
        self._matrix = self._data.X
        self._gene_names = self._data.var_names
        self._cell_names = self._data.obs_names
        self._position = self._data.obsm[pos_label]

    @staticmethod
    def is_valid_exp_matrix(mtx: pd.DataFrame):
        """
        check if the exp matrix is valid for the grn pipeline
        :param mtx:
        :return:
        """
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    # Data loading methods
    @staticmethod
    def read_file(fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            * gef
            * gem
            * loom
            * h5ad
        Recommended formats:
            * h5ad
            * gef
        :param fn:
        :param bin_type:
        :return:

        Example:
            grn.read_file('test.gef', bin_type='bins')
            or
            grn.read_file('test.h5ad')
        """
        logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        logger.info(f'file extension is {extension}')
        if extension == '.csv':
            logger.error('read_file method does not support csv files')
            raise TypeError('this method does not support csv files, '
                            'please read this file using functions outside of the InferenceRegulatoryNetwork class, '
                            'e.g. pandas.read_csv')
        elif extension == '.loom':
            data = sc.read_loom(fn)
            return data
        elif extension == '.h5ad':
            data = sc.read_h5ad(fn)
            return data
        elif extension == '.gef':
            from stereo.io.reader import read_gef
            data = read_gef(file_path=fn, bin_type=bin_type)
            return data

    @staticmethod
    def load_anndata_by_cluster(fn: str,
                                cluster_label: str,
                                target_clusters: list) -> anndata.AnnData:
        """
        When loading anndata, only load in wanted clusters
        One must perform Clustering beforehand
        :param fn: data file name
        :param cluster_label: where the clustering results are stored
        :param target_clusters: a list of interested cluster names
        :return:

        Example:
            sub_data = load_anndata_by_cluster(data, 'psuedo_class', ['HBGLU9'])
        """
        data = InferRegulatoryNetwork.read_file(fn)
        if isinstance(data, anndata.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]
        else:
            raise TypeError('data must be anndata.Anndata object')

    @staticmethod
    # def load_stdata_by_cluster(data: StereoExpData,
    #                            meta: pd.DataFrame,
    #                            cluster_label: str,
    #                            target_clusters: list) -> scipy.sparse.csc_matrix:
    #     """
    #
    #     :param data:
    #     :param meta:
    #     :param cluster_label:
    #     :param target_clusters:
    #     :return:
    #     """
    #
    #     return data.exp_matrix[meta[cluster_label].isin(target_clusters)]

    @staticmethod
    def read_motif_file(fname):
        """

        :param fname:
        :return:
        """
        df = pd.read_csv(fname, sep=',', index_col=[0, 1], header=[0, 1], skipinitialspace=True)
        df[('Enrichment', 'Context')] = df[('Enrichment', 'Context')].apply(lambda s: eval(s))
        df[('Enrichment', 'TargetGenes')] = df[('Enrichment', 'TargetGenes')].apply(lambda s: eval(s))
        return df

    @staticmethod
    def load_tfs(fn: str) -> list:
        """

        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    @staticmethod
    def preprocess(adata: anndata.AnnData, min_genes=0, min_cells=3, min_counts=1, max_gene_num=4000):
        """
        Perform cleaning and quality control on the imported data before constructing gene regulatory network
        :param adata:
        :param min_genes:
        :param min_cells:
        :param min_counts:
        :param max_gene_num:
        :return: a anndata.AnnData
        """
        adata.var_names_make_unique()  # compute the number of genes per cell (computes ‘n_genes' column)
        # # find mito genes
        # sc.pp.ﬁlter_cells(adata, min_genes=0)
        # add the total counts per cell as observations-annotation to adata
        adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))

        logger.info('Start filtering data...')
        # ﬁltering with basic thresholds for genes and cells
        sc.pp.ﬁlter_cells(adata, min_genes=min_genes)
        sc.pp.ﬁlter_genes(adata, min_cells=min_cells)
        sc.pp.ﬁlter_genes(adata, min_counts=min_counts)
        adata = adata[adata.obs['n_genes'] < max_gene_num, :]
        return adata

    # ------------------------------------------------------#
    #           step1: CALCULATE TF-GENE PAIRS              #
    # ------------------------------------------------------#
    @staticmethod
    def _set_client(num_workers: int) -> Client:
        """

        :param num_workers:
        :return:
        """
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        custom_client = Client(local_cluster)
        return custom_client

    def grn_inference(self,
                      matrix,
                      tf_names,
                      genes: list,
                      num_workers: int,
                      verbose: bool = True,
                      cache: bool = True,
                      save: bool = True,
                      fn: str = 'adj.csv',
                      **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules via grnboost2 method
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param tf_names: list of target TFs or all
        :param genes: list of interested genes
        :param num_workers: number of thread
        :param verbose: if print out running details
        :param cache:
        :param save: if save adjacencies result into a file
        :param fn: adjacencies file name
        :return:

        Example:

        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            adjacencies = pd.read_csv(fn)
            self.adjacencies = adjacencies
            return adjacencies
        else:
            logger.info('cached file not found, running grnboost2 now')

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = InferRegulatoryNetwork._set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                **kwargs)
        if save:
            adjacencies.to_csv(fn, index=False)
        self.adjacencies = adjacencies
        return adjacencies

    def uniq_genes(self, adjacencies):
        """
        Detect unique genes
        :param adjacencies:
        :return:
        """
        df = self._data.to_df()
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(df.columns)
        logger.info(f'find {len(unique_adj_genes) / len(set(df.columns))} unique genes')
        return unique_adj_genes

    @staticmethod
    def hotspot_matrix(data: anndata.AnnData,
                       layer_key=None,
                       model='bernoulli',
                       latent_obsm_key="spatial",
                       umi_counts_obs_key=None,
                       weighted_graph=False,
                       n_neighbors=30,
                       fdr_threshold=0.05,
                       tf_list=None,
                       save=True,
                       jobs=None,
                       fn: str = 'adj.csv',
                       **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules via hotspot method
        :param data: Count matrix (shape is cells by genes)
        :param layer_key: Key in adata.layers with count data, uses adata.X if None.
        :param model: Specifies the null model to use for gene expression.
            Valid choices are:
                * 'danb': Depth-Adjusted Negative Binomial
                * 'bernoulli': Models probability of detection
                * 'normal': Depth-Adjusted Normal
                * 'none': Assumes data has been pre-standardized
        :param latent_obsm_key: Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        :param distances_obsp_key: Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        :param umi_counts_obs_key: Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used. 'total_counts'
        :param weighted_graph: Whether or not to create a weighted graph
        :param n_neighbors: Neighborhood size
        :param neighborhood_factor: Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        :param approx_neighbors: Use approximate nearest neighbors or exact scikit-learn neighbors. Only
            when hotspot initialized with `latent`.
        :param fdr_threshold: Correlation theshold at which to stop assigning genes to modules
        :param tf_list: predefined TF names
        :param save: if save results onto disk
        :param jobs: Number of parallel jobs to run
        :param fn: output file name
        :return: A dataframe, local correlation Z-scores between genes (shape is genes x genes)
        """
        if os.path.isfile('local_correlations.csv'):
            local_correlations = pd.read_csv('local_correlations.csv', index_col=0)
        else:
            hs = hotspot.Hotspot(data,
                                 layer_key=layer_key,
                                 model=model,
                                 latent_obsm_key=latent_obsm_key,
                                 umi_counts_obs_key=umi_counts_obs_key,
                                 **kwargs)
            hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)
            hs_results = hs.compute_autocorrelations()
            hs_genes = hs_results.loc[hs_results.FDR < fdr_threshold].index  # Select genes
            local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)  # jobs for parallelization
            local_correlations.to_csv('local_correlations.csv')
        logger.info('Network Inference DONE')
        logger.info(f'Hotspot: create {local_correlations.shape[0]} features')

        # subset by TFs
        if tf_list:
            common_tf_list = list(set(tf_list).intersection(set(local_correlations.columns)))
            logger.info(f'detected {len(common_tf_list)} predefined TF in data')
            assert len(common_tf_list) > 0, 'predefined TFs not found in data'
        else:
            common_tf_list = local_correlations.columns

        # reshape matrix
        local_correlations['TF'] = local_correlations.columns
        local_correlations = local_correlations.melt(id_vars=['TF'])
        local_correlations.columns = ['TF', 'target', 'importance']
        local_correlations = local_correlations[local_correlations.TF.isin(common_tf_list)]
        # remove if TF = target
        local_correlations = local_correlations[local_correlations.TF != local_correlations.target]

        if save:
            local_correlations.to_csv(fn, index=False)
        return local_correlations

    @staticmethod
    def load_database(database_dir: str) -> list:
        """
        Load ranked database
        :param database_dir:
        :return:
        """
        logger.info('Loading ranked databases...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=_name(fname)) for fname in db_fnames]
        return dbs

    # ------------------------------------------------------#
    #           step2:               #
    # ------------------------------------------------------#
    @staticmethod
    def get_modules(adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    **kwargs):
        """
        Create of co-expression modules

        :param adjacencies:
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param rho_mask_dropouts:
        :return:
        """
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        logger.info(f'generated {len(modules)} modules')
        return modules

    def prune_modules(self,
                      modules: list,
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int,
                      cache: bool = True,
                      save: bool = True,
                      fn: str = 'motifs.csv',
                      **kwargs):
        """
        First, calculate a list of enriched motifs and the corresponding target genes for all modules.
        Then, create regulon_list from this table of enriched motifs.
        :param modules: The sequence of modules.
        :param dbs: The sequence of databases.
        :param motif_anno_fn: The name of the file that contains the motif annotations to use.
        :param rank_threshold: The total number of ranked genes to take into account when creating a recovery curve.
        :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
            Area Under the recovery Curve.
        :param nes_threshold: The Normalized Enrichment Score (NES) threshold to select enriched features.
        :param motif_similarity_fdr: The maximum False Discovery Rate to find factor annotations for enriched motifs.
        :param orthologuous_identity_threshold: The minimum orthologuous identity to find factor annotations
            for enriched motifs.
        :param weighted_recovery: Use weights of a gene signature when calculating recovery curves?
        :param num_workers: If not using a cluster, the number of workers to use for the calculation.
            None of all available CPUs need to be used.
        :param module_chunksize: The size of the chunk to use when using the dask framework.
        :param cache:
        :param save:
        :param fn:
        :param kwargs:
        :return: A dataframe.
        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            df = self.read_motif_file(fn)
            regulon_list = df2regulons(df)
            # alternative:
            # regulon_list = load_signatures(fn)
            self.regulon_list = regulon_list
            return regulon_list
        else:
            logger.info('cached file not found, running prune modules now')

        if num_workers is None:
            num_workers = cpu_count()
        with ProgressBar():
            df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)

        regulon_list = df2regulons(df)
        self.regulon_list = regulon_list

        if save:
            df.to_csv(fn)
        return regulon_list

    @staticmethod
    def get_regulon_dict(regulon_list: list) -> dict:
        """
        Form dictionary of { TF : Target } pairs from 'pyscenic ctx' output.
        :param regulon_list:
        :return:
        """
        regulon_dict = {}
        for reg in regulon_list:
            targets = [target for target in reg.gene2weight]
            regulon_dict[reg.name] = targets
        return regulon_dict

    # ------------------------------------------------------#
    #           step3: CALCULATE TF-GENE PAIRS              #
    # ------------------------------------------------------#
    def auc_activity_level(self,
                           matrix,
                           regulons: list,
                           auc_threshold: float,
                           num_workers: int,
                           noweights: bool = False,
                           normalize: bool = False,
                           seed=None,
                           cache: bool = True,
                           save: bool = True,
                           fn='auc.csv',
                           **kwargs) -> pd.DataFrame:
        """
        Calculate enrichment of gene signatures for cells/spots.

        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons: list of ctxcore.genesig.Regulon objects
        :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
            Area Under the recovery Curve.
        :param num_workers: The number of cores to use.
        :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
        :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
        :param cache:
        :param save:
        :param fn:
        :return: A dataframe with the AUCs (n_cells x n_modules).
        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            auc_mtx = pd.read_csv(fn, index_col=0)  # important! cell must be index, not one of the column
            self.auc_mtx = auc_mtx
            return auc_mtx
        else:
            logger.info('cached file not found, calculating auc_activity_level now')

        if num_workers is None:
            num_workers = cpu_count()

        auc_mtx = aucell(matrix,
                         regulons,
                         auc_threshold=auc_threshold,
                         num_workers=num_workers,
                         noweights=noweights,
                         normalize=normalize,
                         seed=seed,
                         **kwargs)
        # check if there were regulons contain all zero auc values
        if not auc_mtx.loc[:, auc_mtx.ne(0).any()].empty:
            logger.warning('auc matrix contains all zero columns')
        auc_mtx = auc_mtx.loc[:, ~auc_mtx.ne(0).any()]
        
        self.auc_mtx = auc_mtx

        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    # ------------------------------------------------------#
    #                       HANDLE DATA                     #
    # ------------------------------------------------------#
    def cal_regulon_score(self, cluster_label='annotation', fn='regulon_specificity_scores.txt'):
        """
        Regulon specificity scores (RSS) across predicted cell types
        :param cluster_label:
        :return:
        """
        rss_cellType = regulon_specificity_scores(self.auc_mtx, self.data.obs[cluster_label])
        rss_cellType.to_csv(fn)
        self.rss = rss_cellType
        return rss_cellType

    @staticmethod
    def get_top_regulons(data: anndata.AnnData, cluster_label: str, rss_cellType: pd.DataFrame, topn: int) -> list:
        """
        get top n regulons for each cell type based on regulon specificity scores (rss)
        :param data:
        :param cluster_label:
        :param rss_cellType:
        :param topn:
        :return: a list
        """
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(data.obs[cluster_label])))
        topreg = {}
        for i, c in enumerate(cats):
            topreg[c] = list(rss_cellType.T[c].sort_values(ascending=False)[:topn].index)
        return topreg

    # ------------------------------------------------------#
    #                 Results saving methods                #
    # ------------------------------------------------------#
    def regulons_to_json(self, regulon_list: list, fn='regulons.json'):
        """
        Write regulon dictionary into json file
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = self.get_regulon_dict(regulon_list)
        with open(fn, 'w') as f:
            json.dump(regulon_dict, f, indent=4)

    def regulons_to_csv(self, regulon_list: list, fn: str = 'regulon_list.csv'):
        """
        Save regulon_list (df2regulons output) into a csv file.
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = self.get_regulon_dict(regulon_list)
        # Optional: join list of target genes
        for key in regulon_dict.keys(): regulon_dict[key] = ";".join(regulon_dict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(regulon_dict.items())

    def to_loom(self, matrix: pd.DataFrame, auc_matrix: pd.DataFrame, regulons: list, fn: str = 'output.loom'):
        """
        Save GRN results in one loom file
        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=matrix, auc_mtx=auc_matrix,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
                    out_fname=fn)

    def to_cytoscape(self,
                     regulons: list,
                     adjacencies: pd.DataFrame,
                     tf: str,
                     fn: str = 'cytoscape.txt'):
        """
        Save GRN result of one TF, into Cytoscape format for down stream analysis
        :param regulons: list of regulon objects, output of prune step
        :param adjacencies: adjacencies matrix
        :param tf: one target TF name
        :param fn: output file name
        :return:

        Example:
            grn.to_cytoscape(regulons, adjacencies, 'Gnb4', 'Gnb4_cytoscape.txt')
        """
        # get TF data
        if isinstance(regulons, list):
            regulon_dict = self.get_regulon_dict(regulons)
        else:
            regulon_dict = regulons
        sub_adj = adjacencies[adjacencies.TF == tf]
        targets = regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')

    # GRN pipeline main logic
    def main(self,
             databases: str,
             motif_anno_fn: str,
             tfs_fn,
             target_genes=None,
             num_workers=None,
             save=True,
             cache=True,
             method='grnboost',
             sigm=15,
             prefix: str = 'project',

             layers='raw_counts',
             model='bernoulli',
             latent_obsm_key='spatial',
             umi_counts_obs_key=None,
             cluster_label='annotation',

             noweights = None ,
             normalize: bool = False):
        """

        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :param target_genes:
        :param num_workers:
        :param save:
        :param cache:
        :param method: method from [grnboost/hotspot/scoexp]
        :param sigm: sigma for scoexp, default 15 (assumption for 15um)
        :param prefix:
        :param layers:
        :param model:
        :param latent_obsm_key:
        :param umi_counts_obs_key:
        :param cluster_label:
        :param noweights:
        :param normalize:
        :return:
        """
        assert method in ['grnboost', 'hotspot', 'scoexp'], "method options are grnboost/hotspot/scoexp"
        global adjacencies
        matrix = self._matrix
        df = self._data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self._gene_names
        
        if noweights is None:
            noweights = self.params[method]["noweights"]

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
            # tfs = self._gene_names
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)

        # 3. GRN inference
        if method == 'grnboost':
            adjacencies = self.grn_inference(matrix,
                                             genes=target_genes,
                                             tf_names=tfs,
                                             num_workers=num_workers,
                                             cache=cache,
                                             save=save,
                                             fn=f'{prefix}_adj.csv')
        elif method == 'scoexp':
            adjacencies = ScoexpMatrix.scoexp(self,
                                              target_genes,
                                              tfs,
                                              sigm=sigm,
                                              save=save,
                                              fn=f'{prefix}_adj.csv')
        elif method == 'hotspot':
            adjacencies = self.hotspot_matrix(self.data,
                                              tf_list=tfs,
                                              jobs=num_workers,
                                              layer_key=layers,
                                              model=model,
                                              latent_obsm_key=latent_obsm_key,
                                              umi_counts_obs_key=umi_counts_obs_key,
                                              fn=f'{prefix}_adj.csv')

        modules = self.get_modules(adjacencies, df)

        # 4. Regulons prediction aka cisTarget
        regulons = self.prune_modules(modules,
                                      dbs,
                                      motif_anno_fn,
                                      num_workers=num_workers,
                                      save=save,
                                      cache=cache,
                                      fn=f'{prefix}_motifs.csv',
                                      rank_threshold=self.params[method]["rank_threshold"],
                                      auc_threshold=self.params[method]["prune_auc_threshold"],
                                      nes_threshold=self.params[method]["nes_threshold"],
                                      motif_similarity_fdr=self.params[method]["motif_similarity_fdr"])

        self.regulon_dict = self.get_regulon_dict(regulons)

        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df,
                                             regulons,
                                             auc_threshold=self.params[method]["auc_threshold"],
                                             num_workers=num_workers,
                                             save=save,
                                             cache=cache,
                                             noweights=noweights,
                                             normalize=normalize,
                                             fn=f'{prefix}_auc.csv')
        logger.info('auc calculation DONE')

        self.cal_regulon_score(cluster_label=cluster_label, fn=f'{prefix}_regulon_specificity_scores.txt')

        # save results
        if save:
            logger.info('saving results...')
            self.regulons_to_json(regulons, fn=f'{prefix}_regulons.json')
            self.to_loom(df, auc_matrix, regulons, fn=f'{prefix}_output.loom')
            # self.to_cytoscape(regulons, adjacencies, 'Zfp354c')
            logger.info('results saving DONE')
