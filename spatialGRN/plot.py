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
import anndata
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyscenic.cli.utils import load_signatures
from pyscenic.export import add_scenic_metadata
from pyscenic.rss import regulon_specificity_scores
from stereo.core.stereo_exp_data import StereoExpData

# modules in self project

logger = logging.getLogger()


class PlotRegulatoryNetwork:
    """
    Plot Gene Regulatory Networks related plots
    """

    def __init__(self, data):
        self._data = data
        self._regulon_list = None
        self._auc_mtx = None
        self._regulon_dict = None

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
        elif isinstance(data, StereoExpData):
            logger.warning('for StereoExpData object, please use function: dotplot_stereo')

    # dotplot method for StereoExpData
    @staticmethod
    def _cal_percent_df(exp_matrix: pd.DataFrame,
                        cluster_meta: pd.DataFrame,
                        regulon_genes: str,
                        celltype: list,
                        groupby: str,
                        cutoff: float = 0):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype:
        :param cutoff:
        :return:
        """
        # which cells are in cluster X
        cells = cluster_meta[cluster_meta[groupby] == celltype]['cell']
        ncells = set(exp_matrix.index).intersection(set(cells))
        # get expression data for cells
        ct_exp = exp_matrix.loc[ncells]
        # input genes in regulon Y
        # get expression data for regulon Y genes in cluster X cells
        g_ct_exp = ct_exp[regulon_genes]
        # count the number of genes which expressed in cluster X cells
        regulon_cell_num = g_ct_exp[g_ct_exp > cutoff].count().count()
        total_cell_num = g_ct_exp.shape[0] * g_ct_exp.shape[1]
        if total_cell_num == 0:
            return 0
        else:
            return regulon_cell_num / total_cell_num

    @staticmethod
    def _cal_exp_df(exp_matrix, cluster_meta, regulon_genes, celltype: str, groupby: str):
        """
        Calculate average expression level for regulon Y genes in cluster X cells
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype
        :return: numpy.float32
        """
        # get expression data for regulon Y genes in cluster X cells
        cells = cluster_meta[cluster_meta[groupby] == celltype]['cell']
        ncells = set(exp_matrix.index).intersection(set(cells))
        ct_exp = exp_matrix.loc[ncells]
        g_ct_exp = ct_exp[regulon_genes]
        if g_ct_exp.empty:
            return 0
        else:
            return np.mean(g_ct_exp)

    @staticmethod
    def dotplot_stereo(data: StereoExpData,
                       meta: pd.DataFrame,
                       regulon_dict,
                       regulon_names: list,
                       celltypes: list,
                       groupby: str,
                       palette: str = 'RdYlBu_r',
                       **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (blue is high).

        :param data:
        :param meta:
        :param regulon_dict:
        :param regulon_names:
        :param celltypes:
        :param groupby:
        :param palette:
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return:
        """
        expr_matrix = data.to_df()
        dot_data = {'cell type': [], 'regulons': [], 'percentage': [], 'avg exp': []}

        for reg in regulon_names:
            target_genes = regulon_dict[f'{reg}(+)']
            for ct in celltypes:
                reg_ct_percent = PlotRegulatoryNetwork._cal_percent_df(exp_matrix=expr_matrix,
                                                                       cluster_meta=meta,
                                                                       regulon_genes=target_genes,
                                                                       celltype=ct, groupby=groupby)
                reg_ct_avg_exp = PlotRegulatoryNetwork._cal_exp_df(exp_matrix=expr_matrix,
                                                                   cluster_meta=meta,
                                                                   regulon_genes=target_genes,
                                                                   celltype=ct, groupby=groupby)
                dot_data['regulons'].append(reg)
                dot_data['cell type'].append(ct)
                dot_data['percentage'].append(reg_ct_percent)
                dot_data['avg exp'].append(reg_ct_avg_exp)

        dot_df = pd.DataFrame(dot_data)
        dot_df.to_csv('dot_df.csv', index=False)
        g = sns.scatterplot(data=dot_df, size='percentage', hue='avg exp', x='regulons', y='cell type', sizes=(20, 200),
                            marker='o', palette=palette, legend='full', **kwargs)
        plt.legend(frameon=False, loc=(1.04, 0))
        plt.tick_params(axis='both', length=0, labelsize=6)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('dot.png')
        return g

    @staticmethod
    def plot_2d_reg_stereo(data: StereoExpData, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.position
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore.values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore.iloc[zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def plot_2d_reg_h5ad(data: anndata.AnnData, pos_label, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:

        Example:
            plot_2d_reg_h5ad(data, 'spatial', auc_mtx, 'Zfp354c')
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.obsm[pos_label]
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore.values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore.iloc[zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def rss_heatmap(data: anndata.AnnData,
                    regulons_fn,
                    auc_mtx: pd.DataFrame,
                    cluster_label: str,
                    topn=5,
                    save=True,
                    fn='clusters_heatmap_top5.png'):
        """
        Plot heatmap for Regulon specificity scores (RSS) value
        :param data: 
        :param auc_mtx: 
        :param regulons_fn:
        :param cluster_label:
        :param topn:
        :param save:
        :param fn:
        :return:
        """
        # load the regulon_list from a file using the load_signatures function
        regulons = load_signatures(regulons_fn)  # regulons_df -> list of regulon_list
        data = add_scenic_metadata(data, auc_mtx, regulons)

        # Regulon specificity scores (RSS) across predicted cell types
        rss_cellType = regulon_specificity_scores(auc_mtx, data.obs[cluster_label])
        rss_cellType.to_csv('regulon_specificity_scores.txt')
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(data.obs[cluster_label])))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_cellType.T[c].sort_values(ascending=False)[:topn].index)
            )
        topreg = list(set(topreg))

        colors = [
            '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
            '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
            '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
            '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
            '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
            '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
        ]
        colorsd = dict((f'c{i}', c) for i, c in enumerate(colors))
        colormap = [colorsd[x] for x in data.obs[cluster_label]]

        # plot legend
        sns.set()
        sns.set(font_scale=0.8)
        fig = palplot(colors, cats, size=1)
        plt.savefig("hood_inte.seurat_clusters-heatmap-legend-top5.pdf", dpi=600, bbox_inches="tight")

        # plot z-score
        auc_zscore = cal_zscore(auc_mtx)
        sns.set(font_scale=1.2)
        g = sns.clustermap(auc_zscore[topreg], annot=False, square=False, linecolor='gray', yticklabels=True,
                           xticklabels=True, vmin=-2, vmax=6, cmap="YlGnBu", figsize=(21, 16), row_colors=colormap)
        g.cax.set_visible(True)
        g.ax_heatmap.set_ylabel('')
        g.ax_heatmap.set_xlabel('')
        if save:
            plt.savefig(fn)
        return g


def cal_zscore(auc_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    calculate z-score for each gene among cells
    :param auc_mtx:
    :return:
    """
    func = lambda x: (x - x.mean()) / x.std(ddof=0)
    auc_zscore = auc_mtx.transform(func, axis=0)
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
