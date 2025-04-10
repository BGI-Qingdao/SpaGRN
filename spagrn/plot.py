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

# third party modules
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
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
CM = 1 / 2.54


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
                auc_mtx,
                reg_name: str,
                fn: str,
                pos_label='spatial',
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
                auc_mtx,
                reg_name: str,
                fn: str,
                view_vertical=222,
                view_horizontal=-80,
                show_bg=False,
                xscale=1,
                yscale=1,
                zscale=1,
                pos_label='spatial',
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
    reg_name = f'reg_name(+)' if '(+)' not in reg_name else reg_name

    # prepare plotting data
    cell_coor = data.obsm[pos_label]
    if isinstance(cell_coor, pd.DataFrame):
        cell_coor = cell_coor.to_numpy()
    sub_zscore = auc_mtx[reg_name]

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
               auc_mtx,
               reg_name: str,
               fn: str,
               view_vertical=222,
               view_horizontal=-80,
               show_bg=False,
               xscale=1,
               yscale=1,
               zscale=1,
               pos_label='spatial',
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
    if isinstance(cell_coor, pd.DataFrame):
        cell_coor = cell_coor.to_numpy()
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


def plot_3d_web(data, auc_mtx, reg_name, prefix='', zscale=1, xscale=1, yscale=1):
    """

    :param data:
    :param auc_mtx:
    :param reg_name:
    :param prefix:
    :param zscale:
    :param xscale:
    :param yscale:
    :return:
    """
    coor = data.obsm['spatial_regis']

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sub_zscore = auc_mtx[reg_name]
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{'type': 'scene'}]],
                        subplot_titles=('regulon'),
                        )
    cell = go.Scatter3d(
        x=coor[:, 0],
        y=coor[:, 1],
        z=coor[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=sub_zscore,
            opacity=1,
            colorbar=dict(thickness=20)
        ),
        legendgroup='cell',
        showlegend=True,
    )
    fig.add_trace(cell, row=1, col=1)
    axis = dict(
        showbackground=False,
        showline=True,
        zeroline=False,
        showgrid=True,
        showticklabels=False,
        title='',
    )
    fig.update_layout(
        margin=dict(l=10, r=10, b=0, t=0),
        showlegend=True,
        scene=dict(
            aspectmode='data',
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
        ),
        paper_bgcolor='rgba(0, 0, 0, 1)',
        plot_bgcolor='rgba(0, 0, 0, 1)',
    )
    # manually force the z-axis to appear twice as big as the other two
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=xscale, y=yscale, z=zscale))
    fig.write_html(f'{prefix}_regulon.html')


def auc_heatmap(data: anndata.AnnData,
                auc_mtx: pd.DataFrame,
                cluster_label: str,
                rss_fn=None,
                topn=5,
                save=True,
                subset=True,
                subset_size=5000,
                fn='clusters_heatmap_top5.pdf',
                legend_fn="rss_celltype_legend.pdf",
                cluster_list=None,
                row_cluster=False,
                col_cluster=True,
                cmap="YlGnBu",
                vmin=-3, vmax=3,
                yticklabels=True, xticklabels=True,
                **kwargs):
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
    :param row_cluster:
    :param col_cluster:
    :param cmap:
    :param vmin:
    :param vmax:
    :param yticklabels:
    :param xticklabels:
    :param kwargs:
    :return:

    Example:
        # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
        auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
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
    if rss_fn is None:  # input rss file has highest priority
        if 'rss' in data.uns:
            rss_cellType = data.uns['rss']
        else:
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
    try:
        plot_data = auc_zscore[topreg].loc[cell_order.index]
    except KeyError:
        com_topreg = list(set(topreg).intersection(set(auc_zscore.columns)))
        plot_data = auc_zscore[com_topreg].loc[cell_order.index]
    sns.set(font_scale=1.2)
    g = sns.clustermap(plot_data,
                       annot=False,
                       square=False,
                       linecolor='gray',
                       yticklabels=yticklabels, xticklabels=xticklabels,
                       vmin=vmin, vmax=vmax,
                       cmap=cmap,
                       row_colors=colormap,
                       row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       **kwargs)
    g.cax.set_visible(False)  # set colorbar
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    if save:
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def isr_heatmap(data: anndata.AnnData,
                cluster_label: str,
                isr_mtx=None,
                rss_fn=None,
                topn=5,
                save=True,
                subset=True,
                subset_size=5000,
                fn='clusters_heatmap_top5.pdf',
                legend_fn="rss_celltype_legend.pdf",
                cluster_list=None,
                row_cluster=False,
                col_cluster=True,
                cmap="YlGnBu",
                vmin=-3, vmax=3,
                yticklabels=True, xticklabels=True,
                **kwargs):
    """
    Plot heatmap for Regulon specificity scores (RSS) value
    :param data:
    :param cluster_label:
    :param isr_mtx:
    :param rss_fn:
    :param topn:
    :param save:
    :param subset:
    :param subset_size:
    :param fn:
    :param legend_fn:
    :param cluster_list:
    :param row_cluster:
    :param col_cluster:
    :param cmap:
    :param vmin:
    :param vmax:
    :param yticklabels:
    :param xticklabels:
    :param kwargs:
    :return:

    Example:
        # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
        auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
                    rss_fn='regulon_specificity_scores.txt',
                    cluster_list=['CNS', 'amnioserosa', 'carcass'])
    """
    if isr_mtx is None:
        isr_mtx = data.obsm['isr']
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
        rss_cellType = regulon_specificity_scores(isr_mtx, data.obs[cluster_label])
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
    isr_zscore = cal_zscore(isr_mtx)
    try:
        plot_data = isr_zscore[topreg].loc[cell_order.index]
    except KeyError:
        com_topreg = list(set(topreg).intersection(set(isr_zscore.columns)))
        plot_data = isr_zscore[com_topreg].loc[cell_order.index]
    sns.set(font_scale=1.2)
    g = sns.clustermap(plot_data,
                       annot=False,
                       square=False,
                       linecolor='gray',
                       yticklabels=yticklabels, xticklabels=xticklabels,
                       vmin=vmin, vmax=vmax,
                       cmap=cmap,
                       row_colors=colormap,
                       row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       **kwargs)
    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    if save:
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def auc_heatmap_uneven(data: anndata.AnnData,
                       auc_mtx: pd.DataFrame,
                       cluster_label: str,
                       rss_fn: str,
                       topn=5,
                       target_celltype: str = 'ventricular-specific CM',
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
        auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
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
                       cmap='magma',  # "YlGnBu",
                       row_colors=colormap,
                       row_cluster=False, col_cluster=True,
                       figsize=(3 * CM, 5.5 * CM))
    g.cax.set_visible(True)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    if save:
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def auc_heatmap_reorder(data: anndata.AnnData,
                        auc_mtx: pd.DataFrame,
                        cluster_label: str,
                        rss_fn: str,
                        order_fn: Optional[str] = None,
                        topn=10,
                        save=True,
                        subset=True,
                        subset_size=5000,
                        fn='clusters_heatmap_top5.png',
                        legend_fn="rss_celltype_legend.png",
                        cluster_list=None,
                        **kwargs):
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
        auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
                    rss_fn='regulon_specificity_scores.txt',
                    cluster_list=['CNS', 'amnioserosa', 'carcass'])
    """
    # 1. Custom regulons
    # Select top regulons
    if order_fn is None:
        # Regulon specificity scores (RSS) across predicted cell types
        if rss_fn is None:
            rss_cellType = regulon_specificity_scores(auc_mtx, data.obs[cluster_label])
        else:
            rss_cellType = pd.read_csv(rss_fn, index_col=0)
        topreg = get_top_regulons(data, cluster_label, rss_cellType, topn=topn)
    else:
        with open(order_fn, 'r') as f:
            topreg = f.read().splitlines()

    # # 2. If Subset
    # if subset and len(data.obs) > subset_size:
    #     fraction = subset_size / len(data.obs)
    #     # do stratified sampling
    #     draw_obs = data.obs.groupby(cluster_label, group_keys=False).apply(lambda x: x.sample(frac=fraction))
    #     # load the regulon_list from a file using the load_signatures function
    #     cell_order = draw_obs[cluster_label].sort_values()
    # else:
    #     # load the regulon_list from a file using the load_signatures function
    #     cell_order = data.obs[cluster_label].sort_values()

    # #3. Custom cell type/cluster order
    # SET CELL ORDERS
    celltypes = sorted(list(set(data.obs[cluster_label])))
    if cluster_list is None:
        # cluster_list = celltypes
        cell_order = data.obs[cluster_label].sort_values()
    else:  # when provide cluster list
        # sort cell types in a custom order
        data.obs[cluster_label] = pd.Categorical(data.obs[cluster_label], categories=cluster_list, ordered=True)
        data.obs = data.obs.sort_values(cluster_label)
        cell_order = data.obs[cluster_label]

    # 4. Make plotting data
    # map color to cell type
    colorsd = dict((i, c) for i, c in zip(celltypes, COLORS))
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
                       vmin=-1.5, vmax=2.5,
                       cmap="YlGnBu",
                       row_colors=colormap,
                       row_cluster=False, col_cluster=True,
                       **kwargs)
    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    if save:
        plt.tight_layout()
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def generate_plot_data(data, auc_mtx, cluster_label, mode='mean', subset=False, subset_size=None, rss_fn=None, order_fn=None,
                       topn=None, cluster_list=None):
    if cluster_list:
        data = data[data.obs[cluster_label].isin(cluster_list)].copy()
        auc_mtx = auc_mtx.loc[list(data.obs_names)]

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

    # load regulon order list
    if order_fn:
        with open(order_fn, 'r') as f:
            topreg = f.read().splitlines()
    elif topn is None:
        topreg = list(auc_mtx.columns)
    else:
        topreg = get_top_regulons(data, cluster_label, rss_cellType, topn=topn)

    if cluster_list is None:
        cluster_list = celltypes.copy()
    colorsd = dict(zip(cluster_list, COLORS))
    colormap = [colorsd[x] for x in cluster_list]

    # plot z-score
    auc_zscore = cal_zscore(auc_mtx)
    try:
        plot_data = auc_zscore[topreg].loc[cell_order.index]
    except KeyError:
        com_topreg = list(set(topreg).intersection(set(auc_zscore.columns)))
        plot_data = auc_zscore[com_topreg].loc[cell_order.index]
    # calculate mean values for each celltype
    plot_data['celltype'] = cell_order
    if mode == 'mean':
        plot_data = plot_data.groupby(['celltype']).mean()
    return plot_data, colorsd, colormap


def auc_heatmap_reorder2(data: anndata.AnnData,
                         auc_mtx: pd.DataFrame,
                         cluster_label: str,
                         rss_fn: str,
                         order_fn: Optional[str] = None,
                         topn: Optional[int] = 10,
                         save=True,
                         subset=True,
                         subset_size=5000,
                         fn='clusters_heatmap_top5.png',
                         legend_fn="rss_celltype_legend.png",
                         cluster_list=None,
                         figsize=(3 * 3 * CM, 5.5 * 3 * CM),
                         annot=False,
                         square=False,
                         linecolor='gray',
                         yticklabels=True,
                         xticklabels=5,
                         vmin=-1,
                         vmax=1,
                         cmap="magma",
                         row_cluster=False,
                         col_cluster=False,
                         **kwargs):
    """
    Plot heatmap for Regulon specificity scores (RSS) value
    :param data:
    :param auc_mtx:
    :param cluster_label:
    :param rss_fn:
    :param order_fn:
    :param topn:
    :param save:
    :param subset:
    :param subset_size:
    :param fn:
    :param legend_fn:
    :param cluster_list: list of cluster names one prefer to use
    :param figsize:
    :param annot:
    :param square:
    :param linecolor:
    :param yticklabels:
    :param xticklabels:
    :param vmin:
    :param vmax:
    :param cmap:
    :param row_cluster:
    :param col_cluster:
    :return:

    Example:
    # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
    auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
                rss_fn='regulon_specificity_scores.txt',
                cluster_list=['CNS', 'amnioserosa', 'carcass'],
                xticklabels='auto')
    """
    plot_data, colorsd, colormap = generate_plot_data(data, auc_mtx, cluster_label, subset=subset,
                                                      subset_size=subset_size, rss_fn=rss_fn, order_fn=order_fn,
                                                      topn=topn, cluster_list=cluster_list)
    # generate legend file
    plot_legend(colorsd, fn=legend_fn)

    # sns.set(font_scale=1.2)
    g = sns.clustermap(plot_data,
                       annot=annot,
                       square=square,
                       linecolor=linecolor,
                       yticklabels=yticklabels, xticklabels=xticklabels,
                       vmin=vmin, vmax=vmax,
                       cmap=cmap,  # YlGnBu, RdYlBu
                       row_colors=colormap,
                       row_cluster=row_cluster, col_cluster=col_cluster,
                       figsize=figsize,
                       **kwargs)
    g.cax.set_visible(True)
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    if save:
        plt.tight_layout()
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def auc_heatmap_reorder3(data: anndata.AnnData,
                         auc_mtx: pd.DataFrame,
                         cluster_label: str,
                         rss_fn: str,
                         order_fn: str,
                         target_celltype: str = 'ventricular-specific CM',
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
    :param save:
    :param subset:
    :param subset_size:
    :param fn:
    :param legend_fn:
    :param cluster_list: list of cluster names one prefer to use
    :return:

    Example:
        # only plot ['CNS', 'amnioserosa', 'carcass'] clusters and their corresponding top regulons
        auc_heatmap(adata, auc_mtx, cluster_label='celltypes', subset=False,
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

    # load regulon order list
    with open(order_fn, 'r') as f:
        topreg = f.read().splitlines()

    if cluster_list is None:
        cluster_list = celltypes.copy()
    colorsd = dict((i, c) for i, c in zip(cluster_list, COLORS))
    colormap = [colorsd[x] for x in cell_order]

    # plot z-score
    auc_zscore = cal_zscore(auc_mtx)
    plot_data = auc_zscore[topreg].loc[cell_order.index]
    # calculate mean values for each celltype
    plot_data['celltype'] = cell_order

    plot_data = plot_data.groupby(["celltype"], sort=False).apply(
        lambda x: x.sort_values(list(plot_data.columns), ascending=False))  # .reset_index(drop=True)

    colormap = [colorsd[x] for x in plot_data.celltype]

    sns.set(font_scale=1.2)
    g = sns.clustermap(plot_data.drop(['celltype'], axis=1),
                       annot=False,
                       square=False,
                       linecolor='gray',
                       yticklabels=True, xticklabels=True,
                       vmin=-3, vmax=3,
                       cmap="YlGnBu",
                       row_colors=colormap,
                       row_cluster=False, col_cluster=False)  # YlGnBu, RdYlBu
    g.cax.set_visible(True)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    if save:
        plt.tight_layout()
        file_format = os.path.splitext(fn)[1].replace('.','')
        plt.savefig(fn, format=file_format)
    return g


def func(df):
    total_cell_order = []
    ct = df['celltype']
    celltypes = sorted(set(ct))
    for celltype in celltypes:
        ct_df = df[df.celltype == celltype]
        g = sns.clustermap(ct_df.drop(['celltype'], axis=1),
                           annot=False,
                           square=False,
                           linecolor='gray',
                           yticklabels=True, xticklabels=True,
                           vmin=-1.5, vmax=2.5,
                           cmap="YlGnBu",
                           row_cluster=False, col_cluster=False)  # YlGnBu, RdYlBu
        labels = get_labels(g)
        total_cell_order += labels
        return total_cell_order


def get_labels(clustermap):
    labels = clustermap.ax_heatmap.yaxis.get_majorticklabels()
    cluster_labels = [label.get_text() for label in labels]
    return cluster_labels


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
                fn='legend.pdf', **kwargs):
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
    plt.figure(figsize=figsize)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker=marker, linestyle=linestyle, **kwargs)
               for color in color_dir.values()]
    plt.legend(markers, color_dir.keys(), numpoints=numpoints, ncol=ncol, loc=loc, frameon=False)
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    file_format = os.path.splitext(fn)[1].replace('.','')
    plt.savefig(fn, format=file_format)
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
    # all cell type labels should be strings
    if not all(isinstance(x, str) for x in cats):
        cats = [str(x) for x in cats]
    rss_celltype.index = rss_celltype.index.astype(str)

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
    topreg = list(set(topreg))
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


def go_bar(fn):
    name = fn.split('.')[0]
    # df = pd.read_excel(fn)
    df = pd.read_csv(fn)
    CM = 1 / 2.54
    y_pos = range(len(df.Description))
    plt.figure(figsize=(20 * CM, 12 * CM))
    ax = plt.barh(y_pos, df['LogP'])
    plt.yticks(y_pos, labels=df.Description)
    plt.tick_params(axis='y', length=0)
    # ax.yaxis.tick_right()
    # plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{name}.pdf', format='pdf')
    plt.close()


def plot_3D_legend(adata, cluster_label='leiden', prefix=''):
    """
    :param adata:
    :param cluster_label:
    :param is_deconv:
    :param prefix:
    :return:

    Example:
        plot_3D_legend(adata, cluster_label='leiden', prefix=n)
    """
    import plotly.express as px

    obs = adata.obs.copy()
    obs['x'] = adata.obsm['spatial'][:, 0]
    obs['y'] = adata.obsm['spatial'][:, 1]
    obs['z'] = adata.obsm['spatial'][:, 2]
    xs = adata.obsm['spatial'][:, 0]
    ys = adata.obsm['spatial'][:, 1]
    zs = adata.obsm['spatial'][:, 2]
    fig = px.scatter_3d(obs,
                        x=xs,
                        y=ys,
                        z=zs,
                        color=cluster_label,
                        opacity=0.7,
                        )
    fig.update_traces(marker_size=3)
    fig.layout.update(
        scene=dict(aspectmode='data')
    )
    fig.write_html(f'{prefix}_{cluster_label}.html')


def spatial_plot_2d(adata, color='annotation', prefix='2d_plot'):
    """

    :param adata:
    :param color:
    :param prefix:
    :return:

    Example:
        spatial_plot_2d(adata, color='leiden', prefix=prefix)
    """
    import matplotlib
    palette = list(matplotlib.colors.CSS4_COLORS.values())

    x, y = zip(*adata.obsm['spatial'])
    annotations = adata.obs[color].unique()
    colors = palette[:len(annotations)]
    color_dict = dict((anno, color) for anno, color in zip(annotations, colors))
    c = [color_dict[anno] for anno in adata.obs[color]]
    plt.scatter(x, y, s=1, c=c, lw=0, edgecolors='none')
    plt.gca().set_aspect('equal')
    plt.show()
    plt.savefig(f'{prefix}_{color}_2d.png')
    plt.close()


def plot_2d(data: anndata.AnnData,
            auc_mtx,
            reg_name: str,
            fn: str,
            pos_label='spatial',
            marker='.',
            edgecolors='none',
            cmap='plasma',
            lw=0,
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

    if isinstance(data.obsm[pos_label], np.ndarray):
        cell_coor = data.obsm[pos_label]
    else:
        cell_coor = data.obsm[pos_label].to_numpy()
    auc_zscore = cal_zscore(auc_mtx)
    # prepare plotting data
    sub_zscore = auc_zscore[reg_name]
    # sort data points by zscore (low to high), because first dot will be covered by latter dots
    zorder = np.argsort(sub_zscore.values)
    # plot cell/bin dot, x y coor
    sc = plt.scatter(cell_coor[:, 0][zorder],
                     cell_coor[:, 1][zorder],
                     c=sub_zscore.iloc[zorder],
                     marker=marker,
                     edgecolors=edgecolors,
                     cmap=cmap,
                     lw=lw,
                     **kwargs)
    plt.axis("equal")
    plt.title(reg_name)
    plt.box(False)
    plt.axis('off')
    plt.colorbar(sc, shrink=0.35)
    file_format = os.path.splitext(fn)[1].replace('.', '')
    plt.savefig(fn, format=file_format)
    plt.close()


def plot_celltype(adata, color='annotation', fn='cell_type.png', custom_labels=None, spatial_label='spatial', s=1, marker='.'):
    """

    :param adata:
    :param color:
    :param prefix:
    :param custom_labels: labels associate with cell types. e.g. it's celltype1 in annotation, but you want to label it as 'ct1'
    :param spatial_label:
    :return:

    Example:
        tfs = ['Adf1', 'Aef1', 'grh', 'kn', 'tll']
        plot_celltype(adata, color='celltype', prefix='ground_truth', custom_labels=tfs)
    """
    annotations = adata.obs[color].unique()
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(annotations)))
    color_dict = dict((anno, color) for anno, color in zip(annotations, colors))

    if custom_labels:
        labels = custom_labels
    else:
        labels = list(annotations)

    for i, label in enumerate(labels):
        # data = adata[adata.obs[color] == (i + 1)]
        data = adata[adata.obs[color] == label]
        c = [color_dict[anno] for anno in data.obs[color]]
        # plt.scatter(data.obsm[spatial_label]['x'], data.obsm[spatial_label]['y'], s=s, c=c, marker=marker, label=label)
        plt.scatter(data.obsm[spatial_label][:, 0], data.obsm[spatial_label][:, 1], s=s, c=c, marker=marker, label=label)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if fn is None:
        fn ='cell_type.png'
    file_format = os.path.splitext(fn)[1].replace('.', '')
    plt.savefig(fn, format=file_format)
    plt.close()


def plot_leiden(adata):
    colors = adata.uns['leiden_colors']
    celltypes = list(adata.obs['leiden'].unique())
    color_dict = dict(zip(celltypes, colors))
    for i, label in enumerate(celltypes):
        data = adata[adata.obs['leiden'] == label]
        c = [color_dict[anno] for anno in data.obs['leiden']]
        plt.scatter(data.obsm['spatial']['x'], data.obsm['spatial']['y'], s=1, c=c, marker='.', label=label)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.box(False)
    plt.axis('off')
    plt.savefig('mouse_brain_E15_leiden_spatial_0.9.png')
    plt.show()
    plt.close()


def plot_cancer(adata):
    colors = COLORS
    celltypes = list(adata.obs['bayes_clusters'].unique())
    color_dict = dict(zip(celltypes, colors))
    for i, label in enumerate(celltypes):
        data = adata[adata.obs['bayes_clusters'] == label]
        c = [color_dict[anno] for anno in data.obs['bayes_clusters']]
        plt.scatter(data.obsm['spatial']['x'], data.obsm['spatial']['y'], s=1, c=c, marker='.', label=label)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.box(False)
    plt.axis('off')
    plt.savefig('cancer_celltype.pdf', format='pdf')
    plt.close()


def plot_ct(adata, colors, ctlabel='leiden'):
    celltypes = list(adata.obs[ctlabel].unique())
    color_dict = dict(zip(celltypes, colors))
    for i, label in enumerate(celltypes):
        data = adata[adata.obs[ctlabel] == label]
        c = [color_dict[anno] for anno in data.obs[ctlabel]]
        plt.scatter(data.obsm['spatial']['x'], data.obsm['spatial']['y'], s=50, c=c, marker='.', label=label)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_gene(data: anndata.AnnData,
              gene_name: str,
              fn: str,
              pos_label='spatial',
              show_bg=False,
              marker='.',
              edgecolors='none',
              cmap='plasma',
              lw=0,
              **kwargs):
    """
    Plot a gene on a 3D map
    :param lw:
    :param cmap:
    :param edgecolors:
    :param marker:
    :param show_bg:
    :param pos_label:
    :param data:
    :param gene_name:
    :param fn:
    :return:

    Example:
        plot_gene(data, 'spatial', 'Zfp354c', 'Zfp354c.png')
    """
    # prepare plotting data
    cell_coor = data.obsm[pos_label]
    exp_mtx = data.to_df()

    sc = plt.scatter(cell_coor['x'],
                     cell_coor['y'],
                     c=exp_mtx[gene_name],
                     marker=marker,
                     edgecolors=edgecolors,
                     cmap=cmap,
                     lw=lw, **kwargs)

    if not show_bg:
        plt.box(False)
        plt.axis('off')
    plt.axis("equal")
    plt.title(gene_name)
    plt.colorbar(sc, shrink=0.35)
    file_format = os.path.splitext(fn)[1].replace('.', '')
    plt.savefig(fn, format=file_format)
    plt.close()


def plot_ligand_receptor(data,
                         ligand: str,
                         receptor: str,
                         fn: str,
                         pos_label='spatial',
                         show_bg=False):
    """

    :param data:
    :param ligand:
    :param receptor:
    :param fn:
    :param pos_label:
    :param show_bg:
    :return:
    """
    # prepare plotting data
    cell_coor = data.obsm[pos_label]
    exp_mtx = data.to_df()

    # coor when exp is more than 0
    x = cell_coor['x']
    y = cell_coor['y']
    data1 = exp_mtx[ligand]
    data2 = exp_mtx[receptor]
    z1 = cal_zscore(exp_mtx[ligand])
    z2 = cal_zscore(exp_mtx[receptor])

    plt.scatter(x,
                y,
                c=data1,
                marker='.',
                edgecolors='none',
                cmap='Blues',
                lw=0, label='ligand', alpha=0.5)
    plt.scatter(x,
                y,
                c=data2,
                marker='.',
                edgecolors='none',
                cmap='Oranges',
                lw=0, label='receptor', alpha=0.5)

    # Highlight overlapping points with a different color
    for i in range(len(x)):
        if z1[i] > 1 and z2[i] > 1:
            plt.scatter(x[i], y[i], color='red', marker='.', s=100)

    if not show_bg:
        plt.box(False)
        plt.axis('off')
    plt.axis("equal")
    plt.title(f'{ligand}_{receptor}')
    # plt.colorbar(sc, shrink=0.35)
    file_format = os.path.splitext(fn)[1].replace('.', '')
    plt.savefig(fn, format=file_format)
    plt.close()


class PlotDataParameters:
    def __init__(self, cluster_list=None, cluster_label=None, subset=None, subset_size=None, rss_fn=None, order_fn=None, topn=None, mode=None):
        self.cluster_list = cluster_list
        self.cluster_label = cluster_label
        self.subset = subset
        self.rss_fn = rss_fn
        self.order_fn = order_fn
        self.topn = topn
        self.subset_size = subset_size
        self.mode = mode


def generate_plot_data(data, auc_mtx, parameters):
    cluster_list = parameters.cluster_list
    subset = parameters.subset
    rss_fn = parameters.rss_fn
    order_fn = parameters.order_fn
    topn = parameters.topreg
    subset_size = parameters.subset_size
    mode = parameters.mode
    cluster_label = parameters.cluster_label

    # Generate plot data based on the provided parameters
    if cluster_list:
        data = data[data.obs[cluster_label].isin(cluster_list)].copy()
        auc_mtx = auc_mtx.loc[list(data.obs_names)]

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

    # load regulon order list
    if order_fn:
        with open(order_fn, 'r') as f:
            topreg = f.read().splitlines()
    elif topn is None:
        topreg = list(auc_mtx.columns)
    else:
        topreg = get_top_regulons(data, cluster_label, rss_cellType, topn=topn)

    if cluster_list is None:
        cluster_list = celltypes.copy()
    colorsd = dict(zip(cluster_list, COLORS))
    colormap = [colorsd[x] for x in cluster_list]

    # plot z-score
    auc_zscore = cal_zscore(auc_mtx)
    try:
        plot_data = auc_zscore[topreg].loc[cell_order.index]
    except KeyError:
        com_topreg = list(set(topreg).intersection(set(auc_zscore.columns)))
        plot_data = auc_zscore[com_topreg].loc[cell_order.index]
    # calculate mean values for each celltype
    plot_data['celltype'] = cell_order
    if mode == 'mean':
        plot_data = plot_data.groupby(['celltype']).mean()
    return plot_data, colorsd, colormap


def t():
    exp = np.array()
    array = np.array()
    for i,row in coordinates:
        x,y = row
        array[x,y] = exp[i,:]


def plot_isr(mtx,
             cell_coor,
             reg_name: str,
             receptor_name,
             fn: str,
             show_bg=False,
             marker='.',
             edgecolors='none',
             cmap='plasma',
             lw=0,
             s=1,
             **kwargs):
    sc = plt.scatter(cell_coor['x'],
                     cell_coor['y'],
                     c=mtx,
                     marker=marker,
                     edgecolors=edgecolors,
                     cmap=cmap,
                     s=s,
                     lw=lw, **kwargs)
    if not show_bg:
        plt.box(False)
        plt.axis('off')
    plt.axis("equal")
    plt.title(f'{reg_name}(+) {receptor_name}')
    plt.colorbar(sc, shrink=0.35)
    file_format = os.path.splitext(fn)[1].replace('.', '')
    plt.savefig(fn, format=file_format)
    plt.close()
