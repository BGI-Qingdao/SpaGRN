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
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from pyscenic.rss import regulon_specificity_scores
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["ytick.labelright"] = True
mpl.rcParams["ytick.labelleft"] = False

# modules in self project


################################################
#                                              #
#        Plot Gene Regulatory Networks         #
#                                              #
################################################
COLORS = [
    '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
    '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
    '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
    '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
    '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
    '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
]


# dotplot method for anndata
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

    cell_coor = data.obsm[pos_label]
    auc_zscore = cal_zscore(auc_mtx)
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


def plot_3d_reg(data: anndata.AnnData,
                pos_label,
                auc_mtx,
                reg_name: str,
                fn: str,
                view_vertical=222,
                view_horizontal=-80,
                show_bg=False,
                xscale=1,
                yscale=1,
                zscale=1,
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
    :param show_bg: if show background
    :param xscale:
    :param yscale:
    :param zscale:
    :return:

    Example:
        plot_3d_reg(data, 'spatial', auc_mtx, 'Zfp354c', view_vertical=30, view_horizontal=-30)
    """
    if '(+)' not in reg_name:
        reg_name = reg_name + '(+)'

    # prepare plotting data
    cell_coor = data.obsm[pos_label]
    auc_zscore = cal_zscore(auc_mtx)
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
    _xscale = xscale
    _yscale = ylen / xlen * yscale
    _zscale = zlen / xlen * zscale
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([_xscale, _yscale, _zscale, 1]))

    if not show_bg:
        plt.box(False)
        plt.axis('off')
    plt.colorbar(sc, shrink=0.35)
    plt.savefig(fn)
    plt.close()


def plot_3d_tf(data: anndata.AnnData,
                pos_label,
                auc_mtx,
                reg_name: str,
                fn: str,
                view_vertical=222,
                view_horizontal=-80,
                show_bg=False,
                xscale=1,
                yscale=1,
                zscale=1,
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
    :param show_bg: if show background
    :param xscale:
    :param yscale:
    :param zscale:
    :return:

    Example:
        plot_3d_reg(data, 'spatial', auc_mtx, 'Zfp354c', view_vertical=30, view_horizontal=-30)
    """
    if '(+)' not in reg_name:
        reg_name = reg_name + '(+)'

    # prepare plotting data
    cell_coor = data.obsm[pos_label]
    auc_zscore = cal_zscore(auc_mtx)
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
    _xscale = xscale
    _yscale = ylen / xlen * yscale
    _zscale = zlen / xlen * zscale
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([_xscale, _yscale, _zscale, 1]))

    if not show_bg:
        plt.box(False)
        plt.axis('off')
    plt.colorbar(sc, shrink=0.35)
    plt.savefig(fn)
    plt.close()


def rss_heatmap(data: anndata.AnnData,
                auc_mtx: pd.DataFrame,
                cluster_label: str,
                rss_fn: str,
                topn=5,
                save=True,
                subset=True,
                subset_size=5000,
                fn='clusters_heatmap_top5.png',
                legend_fn="rss_celltype_legend.png",
                cluster_list=None):
    """
    Plot heatmap for Regulon specificity scores (RSS) value
    :param data:
    :param auc_mtx:
    :param cluster_label:
    :param rss_fn:
    :param topn:
    :param save:
    :param subset:
    :param subset_size:
    :param fn:
    :param legend_fn:
    :param cluster_list: list of cluster names one prefer to use
    :return:

    Example:
        # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
        rss_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
                    rss_fn='regulon_specificity_scores.txt',
                    cluster_list=['CNS', 'amnioserosa', 'carcass'])
    """
    if subset and len(data.obs) > subset_size:
        fraction = subset_size / len(data.obs)
        # do stratified sampling
        draw_obs = data.obs.groupby(cluster_label, group_keys=False).apply(lambda x: x.sample(frac=fraction))
        # load the regulon_list from a file using the load_signatures function
        cell_order = draw_obs[cluster_label].sort_values()
    else:
        # load the regulon_list from a file using the load_signatures function
        cell_order = data.obs[cluster_label].sort_values()
    celltypes = sorted(list(set(data.obs[cluster_label])))

    # Regulon specificity scores (RSS) across predicted cell types
    if rss_fn is None:
        rss_cellType = regulon_specificity_scores(auc_mtx, data.obs[cluster_label])
    else:
        rss_cellType = pd.read_csv(rss_fn, index_col=0)
    # Select the top 5 regulon_list from each cell type
    topreg = get_top_regulons(data, cluster_label, rss_cellType, topn=topn)

    if cluster_list is None:
        cluster_list = celltypes.copy()
    colorsd = dict((i, c) for i, c in zip(cluster_list, COLORS))
    colormap = [colorsd[x] for x in cell_order]

    # plot legend
    plot_legend(colorsd, fn=legend_fn)

    # plot z-score
    auc_zscore = cal_zscore(auc_mtx)
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
        plt.savefig(fn)
    return g


def rss_heatmap_uneven(data: anndata.AnnData,
                auc_mtx: pd.DataFrame,
                cluster_label: str,
                rss_fn: str,
                topn=5,
                target_celltype: str= 'ventricular-specific CM',
                save=True,
                subset=True,
                subset_size=5000,
                fn='clusters_heatmap_top5.png',
                legend_fn="rss_celltype_legend.png",
                cluster_list=None):
    """
    Plot heatmap for Regulon specificity scores (RSS) value
    :param data:
    :param auc_mtx:
    :param cluster_label:
    :param rss_fn:
    :param topn:
    :param save:
    :param subset:
    :param subset_size:
    :param fn:
    :param legend_fn:
    :param cluster_list: list of cluster names one prefer to use
    :return:

    Example:
        # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
        rss_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
                    rss_fn='regulon_specificity_scores.txt',
                    cluster_list=['CNS', 'amnioserosa', 'carcass'])
    """
    if subset and len(data.obs) > subset_size:
        fraction = subset_size / len(data.obs)
        # do stratified sampling
        draw_obs = data.obs.groupby(cluster_label, group_keys=False).apply(lambda x: x.sample(frac=fraction))
        # load the regulon_list from a file using the load_signatures function
        cell_order = draw_obs[cluster_label].sort_values()
    else:
        # load the regulon_list from a file using the load_signatures function
        cell_order = data.obs[cluster_label].sort_values()
    celltypes = sorted(list(set(data.obs[cluster_label])))

    # Regulon specificity scores (RSS) across predicted cell types
    if rss_fn is None:
        rss_cellType = regulon_specificity_scores(auc_mtx, data.obs[cluster_label])
    else:
        rss_cellType = pd.read_csv(rss_fn, index_col=0)
    # Select the top 5 regulon_list from each cell type
    topreg = get_top_regulons_uneven(data, cluster_label, rss_cellType, topn=topn, target_celltype=target_celltype)

    if cluster_list is None:
        cluster_list = celltypes.copy()
    colorsd = dict((i, c) for i, c in zip(cluster_list, COLORS))
    colormap = [colorsd[x] for x in cell_order]

    # plot legend
    plot_legend(colorsd, fn=legend_fn)

    # plot z-score
    auc_zscore = cal_zscore(auc_mtx)
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
        plt.savefig(fn)
    return g


def highlight_key(color_dir: dict,
                  new_value: str = '#8a8787',
                  key_to_highlight=None
                  ) -> dict:
    """
    Highlight one or more interested keys/celltypes when plotting,
    the rest of keys/celltypes will be set to gray by default.
    :param color_dir
    :param new_value
    :param key_to_highlight
    :return dict
    """
    # assert key_to_highlight in color_dir.keys()
    if key_to_highlight is None:
        key_to_highlight = ['Cardiac muscle lineages']
    for k, v in color_dir.items():
        if k not in key_to_highlight:
            color_dir[k] = new_value
    return color_dir


def plot_legend(color_dir, marker='o', linestyle='', numpoints=1, ncol=3, loc='center', figsize=(10, 5),
                fn='legend.png', **kwargs):
    """
    Make separate legend file for heatmap
    :param color_dir:
    :param linestyle: legend icon style
    :param marker: legend icon style
    :param numpoints:
    :param ncol: number of columns, legend layout
    :param loc: location of the legend
    :param figsize: (width, height)
    :param fn:
    :param kwargs:
    :return:

    Example:
        color_dir = {'celltype1': #000000}
        plot_legend(color_dir)
    """
    fig = plt.figure(figsize=figsize)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker=marker, linestyle=linestyle, **kwargs)
               for color in color_dir.values()]
    plt.legend(markers, color_dir.keys(), numpoints=numpoints, ncol=ncol, loc=loc)
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def get_top_regulons(data: anndata.AnnData,
                     cluster_label: str,
                     rss_celltype: pd.DataFrame,
                     topn: int) -> list:
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


def get_top_regulons_uneven(data: anndata.AnnData,
                            cluster_label: str,
                            rss_celltype: pd.DataFrame,
                            topn: int = 10,
                            target_celltype: str = 'ventricular-specific CM',
                            target_topn: int = 20) -> list:
    """
    get target_topn regulons for interested cell type and topn regulons for the rest cell types
    based on regulon specificity scores (rss)
    :param data:
    :param cluster_label:
    :param rss_celltype:
    :param topn:
    :param target_celltype:
    :param target_topn:
    :return: a list
    """
    cats = sorted(list(set(data.obs[cluster_label])))
    topreg = []
    for i, c in enumerate(cats):
        if c == target_celltype:
            topreg.extend(list(rss_celltype.T[c].sort_values(ascending=False)[:target_topn].index))
        else:
            topreg.extend(list(rss_celltype.T[c].sort_values(ascending=False)[:topn].index))
    return topreg


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


if __name__ == '__main__':
    # total celltypes for Drosophilidae data
    cluster_list = ['CNS', 'amnioserosa', 'carcass', 'epidermis', 'epidermis/CNS', 'fat body', 'fat body/trachea',
                'foregut', 'foregut/garland cells', 'hemolymph', 'hindgut', 'hindgut/malpighian tubule', 'midgut',
                'midgut/malpighian tubules', 'muscle', 'salivary gland', 'testis', 'trachea']
