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
import warnings
from typing import Union

# third party modules
import json
import glob
import anndata
import scipy.sparse
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.cli.utils import load_signatures
from pyscenic.rss import regulon_specificity_scores
from pyscenic.aucell import aucell
from stereo.core.stereo_exp_data import StereoExpData
from stereo.io.reader import read_gef

# modules in self project


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]


class InferenceRegulatoryNetwork:
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    def __init__(self, data):
        # input
        self._data = data
        self._matrix = None  # pd.DataFrame
        self._gene_names = []
        self._cell_names = []

        self.load_data_info()

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
    def data(self, data: Union[StereoExpData, anndata.AnnData]):
        self._data = data
        self.load_data_info()

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

    def load_data_info(self):
        """"""
        if isinstance(self._data, StereoExpData):
            self._matrix = self._data.exp_matrix
            self._gene_names = self._data.gene_names
            self._cell_names = self._data.cell_names
        elif isinstance(self._data, anndata.AnnData):
            self._gene_names = self._data.var_names
            self._cell_names = self._data.obs_names

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
             save=True):
        """
        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :param target_genes:
        :param num_workers:
        :param save:
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
        adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers)
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


