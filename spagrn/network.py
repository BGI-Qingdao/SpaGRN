# !/usr/bin/env python -*- coding: utf-8 -*-
# @Date: Created on 31 Oct 2023 15:00
# @Author: Yao LI
# @File: spagrn/network.py
# @Description: A Gene Regulatory Network object. A typical network essentially contains regulators
# (e.g. TFs), target genes. and cell types, regulons score between cell types and activity level among cells. and
# regulator-target regulatory effect.

import os

import json
import pickle
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as an
from typing import Sequence

from ctxcore.genesig import Regulon
from pyscenic.rss import regulon_specificity_scores


def remove_all_zero(auc_mtx):
    # check if there were regulons contain all zero auc values
    auc_mtx = auc_mtx.loc[:, ~auc_mtx.ne(0).any()]
    # remove all zero columns (which have no variation at all)
    auc_mtx = auc_mtx.loc[:, (auc_mtx != 0).any(axis=0)]
    return auc_mtx


class Network:
    def __init__(self):
        """
        Constructor of the (Gene Regulatory) Network Object.
        """
        # input
        self._data = None  # anndata.Anndata
        self._matrix = None  # pd.DataFrame
        self._gene_names = None  # list of strings
        self._cell_names = None  # list of strings
        self._position = None  # np.array
        self._tfs = None  # list

        # Network calculated attributes
        self._regulons = None  # list of ctxcore.genesig.Regulon instances
        self._modules = None  # list of ctxcore.genesig.Regulon instances
        self._auc_mtx = None  # pd.DataFrame
        self._adjacencies = None  # pd.DataFrame
        self._regulon_dict = None  # dictionary
        self._rss = None  # pd.DataFrame

        # Receptors
        self._filtered = None  # dictionary
        self._receptors = None  # set
        self.receptor_dict = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

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
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

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
    def modules(self):
        return self._modules

    @modules.setter
    def modules(self, value):
        self._modules = value

    @property
    def filtered(self):
        return self._filtered

    @filtered.setter
    def filtered(self, value):
        self._filtered = value

    @property
    def receptors(self):
        return self._receptors

    @receptors.setter
    def receptors(self, value):
        self._receptors = value

    # ------------------------------------------------------#
    #                Data loading methods                   #
    # ------------------------------------------------------#
    def load_data_info(self, pos_label='spatial'):
        """
        (for raw data)
        Load useful data to properties.
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata.
        :return:
        """
        if self.data:
            self.matrix = self.data.X
            self.gene_names = self.data.var_names
            self.cell_names = self.data.obs_names
            self.position = self.data.obsm[pos_label]

    def load_results(self, modules_fn=None, regulons_fn=None):
        """
        (for derived data)
        Load results generate by SpaGRN. Mainly contains
        :param modules_fn:
        :param regulons_fn:
        :return:
        """
        try:
            self.regulon_dict = self.data.uns['regulon_dict']
            self.adjacencies = self.data.uns['adj']
            self.auc_mtx = self.data.obsm['auc_mtx']
            self.rss = self.data.uns['rss']
        except KeyError as e:
            print(f"WARNING: {e.args[0]} does not exist")
        if modules_fn:
            self.modules = pickle.load(open(modules_fn, 'rb'))
        if regulons_fn:
            self.regulons = pickle.load(open(regulons_fn, 'rb'))

    @staticmethod
    def read_file(fn):
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
        :return:

        Example:
            grn.read_file('test.gef', bin_type='bins')
            or
            grn.read_file('test.h5ad')
        """
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            raise TypeError('this method does not support csv files, '
                            'please read this file using functions outside of the InferenceRegulatoryNetwork class, '
                            'e.g. pandas.read_csv')
        elif extension == '.loom':
            data = sc.read_loom(fn)
            return data
        elif extension == '.h5ad':
            data = sc.read_h5ad(fn)
            return data

    def load_anndata_by_cluster(self, fn: str,
                                cluster_label: str,
                                target_clusters: list) -> an.AnnData:
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
        data = self.read_file(fn)
        if isinstance(data, an.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]
        else:
            raise TypeError('data must be anndata.Anndata object')

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

    @staticmethod
    def preprocess(adata: an.AnnData, min_genes=0, min_cells=3, min_counts=1, max_gene_num=4000):
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

    def uniq_genes(self, adjacencies):
        """
        Detect unique genes
        :param adjacencies:
        :return:
        """
        df = self._data.to_df()
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(df.columns)
        return unique_adj_genes

    @staticmethod
    def get_regulon_dict(regulon_list: Sequence[Regulon]) -> dict:
        """
        Form dictionary of { TF : Target } pairs from Regulons.
        :param regulon_list:
        :return:
        """
        assert regulon_list is not None, "regulons is not available, calculate regulons or load regulons results first"
        regulon_dict = {}
        for reg in regulon_list:
            targets = [target for target in reg.gene2weight]
            regulon_dict[reg.name] = targets
        return regulon_dict

    # Save to files
    def regulons_to_json(self, fn='regulons.json'):
        """
        Write regulon dictionary into json file
        :param fn:
        :return:
        """
        if not self.regulon_dict:
            self.regulon_dict = self.get_regulon_dict(self.regulons)
            self.data.uns['regulon_dict'] = self.regulon_dict
        with open(fn, 'w') as f:
            json.dump(self.regulon_dict, f, sort_keys=True, indent=4)

    # Regulons and Cell Types
    def cal_regulon_score(self, cluster_label='annotation', save_tmp=False, fn='regulon_specificity_scores.txt'):
        """
        Regulon specificity scores (RSS) across predicted cell types
        :param fn:
        :param save_tmp:
        :param cluster_label:
        :return:
        """
        rss_cellType = regulon_specificity_scores(self.auc_mtx, self.data.obs[cluster_label])
        if save_tmp:
            rss_cellType.to_csv(fn)
        self.rss = rss_cellType
        self.data.uns['rss'] = rss_cellType  # for each cell type
        return rss_cellType

    def get_top_regulons(self, cluster_label: str, topn: int) -> dict:
        """
        get top n regulons for each cell type based on regulon specificity scores (rss)
        :param cluster_label:
        :param topn:
        :return: a list
        """
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(self.data.obs[cluster_label])))
        topreg = {}
        for i, c in enumerate(cats):
            topreg[c] = list(self.rss.T[c].sort_values(ascending=False)[:topn].index)
        return topreg
