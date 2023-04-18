#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
"""

# python core modules

# third party modules
from typing import Union

import anndata
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from pyscenic.cli.utils import load_signatures
from pyscenic.export import add_scenic_metadata
from pyscenic.rss import regulon_specificity_scores
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["ytick.labelright"] = True
mpl.rcParams["ytick.labelleft"] = False

# modules in self project

logger = logging.getLogger()


class PlotRegulatoryNetwork:
    """
    Plot Gene Regulatory Networks related plots
    """

    def __init__(self, data, cluster_label='annotation'):
        self._data = data
        self._regulon_list = None
        self._auc_mtx = None
        self._regulon_dict = None

        self._celltype_colors = [
            '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
            '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
            '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
            '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
            '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
            '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
        ]
        self._cluster_label = cluster_label

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

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
    def celltype_colors(self):
        return self._celltype_colors

    @celltype_colors.setter
    def celltype_colors(self, value):
        self._celltype_colors = value

    @property
    def cluster_label(self):
        return self._cluster_label

    @cluster_label.setter
    def cluster_label(self, value):
        self._cluster_label = value

    def add_color(self, value):
        if isinstance(value, list):
            self._celltype_colors.extend(value)
        elif isinstance(value, str):
            self._celltype_colors.append(value)
        else:
            logger.error('new color should be either a string or a list of strings')

    # dotplot method for anndata
    @staticmethod
    def dotplot_anndata(data: anndata.AnnData,
                        gene_names: list,
                        cluster_label: str,
                        save: bool = True,
                        **kwargs):
        """
        create a dotplot for Anndata object.
        a dotplot contains percent (of cells that) expressed (the genes) and average expression (of genes).

        :param data: gene data
        :param gene_names: interested gene names
        :param cluster_label: label of clustering output
        :param save: if save plot into a file
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return: plt axe object
        """
        if isinstance(data, anndata.AnnData):
            return sc.pl.dotplot(data, var_names=gene_names, groupby=cluster_label, save=save, **kwargs)

    @staticmethod
    def plot_2d_reg(data: anndata.AnnData,
                    pos_label,
                    auc_mtx,
                    reg_name: str,
                    fn: str,
                    **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :param fn:
        :return:
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        if fn is None:
            fn = f'{reg_name.strip("(+)")}.pdf'

        cell_coor = data.obsm[pos_label]
        auc_zscore = PlotRegulatoryNetwork.cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore.values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder],
                         cell_coor[:, 1][zorder],
                         c=sub_zscore.iloc[zorder],
                         marker='.',
                         edgecolors='none',
                         cmap='plasma',
                         lw=0,
                         **kwargs)
        plt.axis("equal")
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(fn, format='pdf')
        plt.close()

    @staticmethod
    def plot_3d_reg(data: anndata.AnnData,
                    pos_label,
                    auc_mtx,
                    reg_name: str,
                    fn: str,
                    view_vertical=222,
                    view_horizontal=-80,
                    **kwargs):
        """
        Plot genes of one regulon on a 3D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :param fn:
        :param view_vertical: vertical angle to view to the 3D object
        :param view_horizontal: horizontal angle to view the 3D object
        :return:

        Example:
            plot_3d_reg(data, 'spatial', auc_mtx, 'Zfp354c', view_vertical=30, view_horizontal=-30)
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        if fn is None:
            fn = f'{reg_name.strip("(+)")}.pdf'

        # prepare plotting data
        cell_coor = data.obsm[pos_label]
        auc_zscore = PlotRegulatoryNetwork.cal_zscore(auc_mtx)
        sub_zscore = auc_zscore[reg_name]

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        sc = ax.scatter(cell_coor[:, 0],
                        cell_coor[:, 1],
                        cell_coor[:, 2],
                        c=sub_zscore,
                        marker='.',
                        edgecolors='none',
                        cmap='plasma',
                        lw=0, **kwargs)
        # set view angle
        ax.view_init(view_vertical, view_horizontal)
        # scale axis
        xlen = cell_coor[:, 0].max() - cell_coor[:, 0].min()
        ylen = cell_coor[:, 1].max() - cell_coor[:, 1].min()
        zlen = cell_coor[:, 2].max() - cell_coor[:, 2].min()
        yscale = ylen / xlen
        zscale = zlen / xlen
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, yscale, zscale, 1]))

        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(fn, format='pdf')
        plt.close()

    @staticmethod
    def rss_heatmap(data: anndata.AnnData,
                    auc_mtx: pd.DataFrame,
                    cluster_label: str,
                    rss_fn: str = 'regulon_specificity_scores.txt',
                    topn=5,
                    save=True,
                    subset=True,
                    subset_size=5000, 
                    fn='clusters_heatmap_top5.pdf',
                    legend_fn="rss_celltype_legend_top5.png"):
        """
        Plot heatmap for Regulon specificity scores (RSS) value
        :param data: 
        :param auc_mtx: 
        :param cluster_label:
        :param rss_fn:
        :param topn:
        :param save:
        :param subset:
        :parma subset_size:
        :param fn:
        :return:
        """
        if subset and len(data.obs) > subset_size:
            fraction = subset_size / len(data.obs)
            #do stratified sampling
            draw_obs = data.obs.groupby(cluster_label, group_keys=False).apply(lambda x: x.sample(frac=fraction))
            # load the regulon_list from a file using the load_signatures function
            cell_order = draw_obs[cluster_label].sort_values()
        else:
            # load the regulon_list from a file using the load_signatures function
            cell_order = data.obs[cluster_label].sort_values()
        celltypes = sorted(list(set(data.obs[cluster_label])))

        # Regulon specificity scores (RSS) across predicted cell types
        rss_cellType = pd.read_csv(rss_fn, index_col=0)
        # Select the top 5 regulon_list from each cell type
        topreg = PlotRegulatoryNetwork.get_top_regulons(data, cluster_label, rss_cellType, topn=topn)


        obs_list = ['CNS', 'amnioserosa', 'carcass', 'epidermis', 'epidermis/CNS', 'fat body', 'fat body/trachea', 'foregut', 'foregut/garland cells', 'hemolymph', 'hindgut', 'hindgut/malpighian tubule', 'midgut', 'midgut/malpighian tubules', 'muscle', 'salivary gland', 'testis', 'trachea']

        colors = [
            '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
            '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
            '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
            '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
            '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
            '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
        ]
        colorsd = dict((i, c) for i, c in zip(obs_list, colors))
        colormap = [colorsd[x] for x in cell_order]        

        # plot legend
        #plot_legend(colormap, obs_list, legend_fn)

        # plot z-score
        auc_zscore = PlotRegulatoryNetwork.cal_zscore(auc_mtx)
        plot_data = auc_zscore[topreg].loc[cell_order.index]
        sns.set(font_scale=1.2)
        g = sns.clustermap(plot_data,
                           annot=False,
                           square=False,
                           linecolor='gray',
                           yticklabels=True, xticklabels=True,
                           vmin=-3, vmax=3,
                           cmap="YlGnBu",
                           row_colors=colormap,
                           row_cluster=False, col_cluster=True)
        g.cax.set_visible(True)
        g.ax_heatmap.set_yticks([])
        g.ax_heatmap.set_ylabel('')
        g.ax_heatmap.set_xlabel('')
        if save:
            plt.savefig(fn, format='pdf')
        return g

    @staticmethod
    def map_celltype_colors(data, celltype_colors: list, celltypes: list, cluster_label: str):
        """

        :param data:
        :param celltype_colors:
        :param celltypes:
        :param cluster_label:
        :return:
        """
        assert len(celltype_colors) >= len(celltypes)
        colorsd = dict((i, c) for i, c in zip(celltypes, celltype_colors))
        colormap = [colorsd[x] for x in data.obs[cluster_label]]
        return colorsd, colormap

    @staticmethod
    def get_top_regulons(data: anndata.AnnData, cluster_label: str, rss_celltype: pd.DataFrame, topn: int) -> list:
        """
        get top n regulons for each cell type based on regulon specificity scores (rss)
        :param data:
        :param cluster_label:
        :param rss_celltype:
        :param topn:
        :return: a list
        """
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(data.obs[cluster_label])))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_celltype.T[c].sort_values(ascending=False)[:topn].index)
            )
        topreg = list(set(topreg))
        return topreg

    @staticmethod
    def cal_zscore(auc_mtx: pd.DataFrame, save=False) -> pd.DataFrame:
        """
        calculate z-score for each gene among cells
        :param auc_mtx:
        :param save:
        :return:
        """
        func = lambda x: (x - x.mean()) / x.std(ddof=0)
        auc_zscore = auc_mtx.transform(func, axis=0)
        if save:
            auc_zscore.to_csv('auc_zscore.csv', index=False)
        return auc_zscore


def is_regulon_name(reg):
    """
    Decide if a string is a regulon_list name
    :param reg: the name of the regulon
    :return:
    """
    if '(+)' in reg or '(-)' in reg:
        return True


# Generate a heatmap
def palplot(pal, names, colors=None, size=1):
    """

    :param pal:
    :param names:
    :param colors:
    :param size:
    :return:
    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n), cmap=mpl.colors.ListedColormap(list(pal)), interpolation="nearest",
              aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    colors = n * ['k'] if colors is None else colors
    for idx, (name, color) in enumerate(zip(names, colors)):
        ax.text(0.0 + idx, 0.0, name, color=color, horizontalalignment='center', verticalalignment='center')
    return f


def plot_legend(colormap, obs_list, legend_fn):
    # plot legend
    sns.set()
    sns.set(font_scale=0.8)
    palplot(colormap, obs_list, size=1)
    plt.savefig(legend_fn, bbox_inches="tight")
    plt.close()

