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
from typing import Union

# third party modules
import json
import glob
import anndata
import logging
import scipy.sparse
import pandas as pd
import numpy as np
import scanpy as sc
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell
from stereo.core.stereo_exp_data import StereoExpData
from stereo.io.reader import read_gef

# modules in self project

logger = logging.getLogger()


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
        
        :param dis_mat Distance matrix
        :param sigm Width of rbfk
        :param zero_diag
        :return rbf matrix
        """
        rbfk_out = np.exp(-1 * np.square(dis_mat) / (2*sigm**2) )
        if zero_diag:
            rbfk_out[np.diag_indices_from(rbfk_out)]=0
        return rbfk_out

    @staticmethod
    def wcor(X, W, method='pearson', na_zero=True) :
        """
        Weighted cross correlation
        
        :param X Expression matrix, n X p
        :param W Weight matrix, n X n
        :param method Correlation method, pearson or spearman
        :param na_zero Na to zero
        :return correlation matrix
        """
        from scipy.stats import rankdata
        from sklearn.preprocessing import scale
        if method == 'spearman':
            X = np.apply_along_axis(rankdata ,0, X) # rank each columns
        X = scale(X,axis=0) # scale for each columns
        W_cov_temp = np.matmul( np.matmul(X.T, W), X )
        W_diag_mat = np.sqrt( np.matmul(np.diag(W_cov_temp), np.diag(W_cov_temp).T ) )
        cor_mat = W_cov_temp / W_diag_mat
        if na_zero:
            np.nan_to_num(cor_mat,False)
        return cor_mat

    @staticmethod
    def scoexp( irn_data,
                gene_list =[],
                tf_list = [],
                sigm=15,
                zero_cutoff=5,
                cor_method='spearman',
                ):
        """
        Main logic for scoexp calculation
 
        :param irn_data: object of InferenceRegulatoryNetwork
        :param sigm: sigma for RBF kernel, default 15.
        :param gene_list: filter gene by exp cell > zero_cutoff% of all cells if len(gene_list)<2, otherwise use this gene set.
        :param tf_list, tf gene list. Use gene_list if tf_list is empty.
        :param zero_cutoff: filter gene by exp cell > zero_cutoff% if if len(gene_list)<2
        :param cor_method: 'spearman' or 'pearson'
        :return: dataframe of tf-gene-importances
        """
        from scipy.spatial import distance_matrix
        cell_gene_matrix = irn_data.matrix()
        if not isinstance(cell_gene_matrix,np.ndarray):
            cell_gene_matrix = cell_gene_matrix.toarray()
        # check gene_list
        if len(gene_list)<2:
            print('gene filtering...',flush=True)
            feature_nz = np.apply_along_axis(lambda x: np.mean(x!=0)*100,0, cell_gene_matrix)
            features = irn_data.gene_names()[feature_nz > zero_cutoff]
            print(f'{len(features)} features after filtering...',flush=True)
        else:
            features = np.intersect1d(np.array(gene_list),irn_data.gene_names())
            if len(features)<2:
                print('No enough genes in gene_list detected, exit...',flush=True)
                sys.exit(12)
        # check tf_list
        if len(tf_list) < 1:
            tf_list = features
        else:
            tf_list = np.intersect1d(np.array(tf_list),features)

        gene_select = np.isin(irn_data.gene_names(),features,assume_unique=True)
        celltrek_inp = cell_gene_matrix[:,gene_select]
        dist_mat = distance_matrix( irn_data.pos(),
                                    irn_data.pos() )
        kern_mat = rbfk(dist_mat, sigm=sigm, zero_diag=False)
        print('Calculating spatial-weighted cross-correlation...',flush=True)
        wcor_mat = wcor(X=celltrek_inp, W=kern_mat, method=cor_method)
        print('Calculating spatial-weighted cross-correlation done.',flush=True)
        df = pd.DataFrame(data=wcor_mat, index=features, columns=features)
        #extract tf-gene-importances
        df = df[tf_list].copy().T
        df['TF'] = tf_list
        ret = df.melt(id_vars=['TF'])
        ret.columns = ['TF','target','importance']
        return ret


class InferenceRegulatoryNetwork:
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    def __init__(self, data, pos_label = 'spatial'):
        """
        Constructor of this Object.
        :param data:
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        # input
        self._data = data
        self._matrix = None  # pd.DataFrame
        self._gene_names = []
        self._cell_names = []

        self.load_data_info(pos_label)

        self._tfs = []

        # network calculated attributes
        self._regulon_list = None  # list
        self._auc_mtx = None
        self._adjacencies = None  # pd.DataFrame
        self._regulon_dict = None

        # other settings
        # self._num_workers = num_workers
        # self._thld = auc_thld

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Union[StereoExpData, anndata.AnnData], pos_label = 'spatial'):
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
    def pos(self):
        return self._pos    

    # @property
    # def num_workers(self):
    #     return self._num_workers
    #
    # @num_workers.setter
    # def num_workers(self, value):
    #     self._num_workers = value
    #
    # @property
    # def thld(self):
    #     return self._thld
    #
    # @thld.setter
    # def thld(self, value):
    #     self._thld = value

    def load_data_info(self, pos_label):
        """
        Load useful data to properties.
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        if isinstance(self._data, StereoExpData):
            self._matrix = self._data.exp_matrix
            self._gene_names = self._data.gene_names
            self._cell_names = self._data.cell_names
            self._pos = self._data.position
        elif isinstance(self._data, anndata.AnnData):
            self._gene_names = self._data.var_names
            self._cell_names = self._data.obs_names
            self._pos = self._data.obsm[pos_label]

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
        data = InferenceRegulatoryNetwork.read_file(fn)
        if isinstance(data, anndata.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]
        else:
            raise TypeError('data must be anndata.Anndata object')

    @staticmethod
    def load_stdata_by_cluster(data: StereoExpData,
                               meta: pd.DataFrame,
                               cluster_label: str,
                               target_clusters: list) -> scipy.sparse.csc_matrix:
        """

        :param data:
        :param meta:
        :param cluster_label:
        :param target_clusters:
        :return:
        """
        return data.exp_matrix[meta[cluster_label].isin(target_clusters)]

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

    # Gene Regulatory Network inference methods
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
        Inference of co-expression modules
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
        custom_client = InferenceRegulatoryNetwork._set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                **kwargs)
        if save:
            adjacencies.to_csv(fn, index=False)  # adj.csv, don't have to save into a file
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

    def get_modules(self,
                    adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    **kwargs):
        """
        Inference of co-expression modules

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
        return modules

    def prune_modules(self,
                      modules: list,
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int,
                      is_prune: bool = True,
                      cache: bool = True,
                      save: bool = True,
                      fn: str = 'motifs.csv',
                      **kwargs):
        """
        First, calculate a list of enriched motifs and the corresponding target genes for all modules.
        Then, create regulon_list from this table of enriched motifs.
        :param modules:
        :param dbs:
        :param motif_anno_fn:
        :param num_workers:
        :param is_prune:
        :param cache:
        :param save:
        :param fn:
        :return:
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
        if is_prune:
            with ProgressBar():
                df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)
                df.to_csv(fn)  # motifs filename
            regulon_list = df2regulons(df)
            self.regulon_list = regulon_list

            if save:
                self.regulons_to_json(regulon_list)

            # alternative way of getting regulon_list, without creating df first
            # regulon_list = prune(dbs, modules, motif_anno_fn)
            return regulon_list
        else:
            logger.warning('if prune_modules is set to False')

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

    def auc_activity_level(self,
                           matrix,
                           regulons: list,
                           auc_threshold: float,
                           num_workers: int,
                           cache: bool = True,
                           save: bool = True,
                           fn='auc.csv',
                           **kwargs) -> pd.DataFrame:
        """

        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons: list of ctxcore.genesig.Regulon objects
        :param auc_threshold:
        :param num_workers:
        :param cache:
        :param save:
        :param fn:
        :return:
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

        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers, **kwargs)
        self.auc_mtx = auc_mtx

        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    # Results saving methods
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

    def to_loom(self, matrix: pd.DataFrame, auc_matrix: pd.DataFrame, regulons: list, loom_fn: str = 'output.loom'):
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
                    out_fname=loom_fn)

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
        regulon_dict = self.get_regulon_dict(regulons)
        sub_adj = adjacencies[adjacencies.TF == tf]
        targets = regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')

    # HOTSPOT related
    def get_input_spatial_matrix(self, data):
        pass

    # GRN pipeline main logic
    def main(self,
             databases: str,
             motif_anno_fn: str,
             tfs_fn,
             target_genes=None,
             num_workers=None,
             save=True,
             method='grnboost2',
             sigm=15,
        ):
        """
        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :param target_genes:
        :param num_workers:
        :param save:
        :param method: method from [grnboost2/hotspot/scoexp]
        :param sigm: sigma for scoexp, default 15 (assumption for 15um)
        :return:
        """
        matrix = self._matrix
        df = self._data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self._gene_names

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
            # tfs = self._gene_names
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)
        # 3. GRN inference
        if method == 'grnboost2':
            adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers)
        elif method == 'scoexp':
            adjacencies = ScoexpMatrix.scoexp(self,target_genes,tfs,sigm=sigm)
        modules = self.get_modules(adjacencies, df)
        # 4. Regulons prediction aka cisTarget
        regulons = self.prune_modules(modules, dbs, motif_anno_fn, num_workers=24)
        self.regulon_dict = self.get_regulon_dict(regulons)
        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df, regulons, auc_threshold=0.5, num_workers=num_workers)

        # save results
        if save:
            self.regulons_to_csv(regulons)
            self.regulons_to_json(regulons)
            self.to_loom(df, auc_matrix, regulons)
            self.to_cytoscape(regulons, adjacencies, 'Zfp354c')

