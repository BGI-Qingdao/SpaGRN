#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: spagrn_parser.py
@time: 2023/Nov/01
@description: test file for inference gene regulatory networks module
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

import os
import sys
import argparse
from multiprocessing import cpu_count
from spagrn.regulatory_network import InferNetwork as irn
import spagrn.plot as prn


def inference_command(args):
    # avoid TypeError: stat: path should be string, bytes, os.PathLike or integer, not _io.TextIOWrapper
    # load data
    data = irn.read_file(args.expression_mtx_fname)
    data = irn.preprocess(data)
    # create grn
    grn = irn(data, project_name=args.project_name)
    # run_all analysis
    grn.infer(args.database,
              args.motif,
              args.tfs_fname,
              niche_df=args.reference,
              num_workers=args.num_workers,
              cache=args.cache,
              output_dir=args.output,
              save_tmp=args.save_tmp,
              layers=args.layer_key,
              latent_obsm_key=args.spatial,  # IMPORTANT!
              umi_counts_obs_key=args.umi_counts_obs_key,
              model=args.model,
              n_neighbors=args.n_neighbors,
              methods=args.methods,
              local=args.local,
              somde_k=args.somde_k,
              operation=args.operation,
              combine=args.combine,
              mode=args.mode,
              weighted_graph=args.weighted_graph,
              cluster_label=args.cluster_label,
              noweights=args.noweights,
              normalize=args.normalize,
              rho_mask_dropouts=args.rho_mask_dropouts)


def add_computation_parameters(parser):
    group = parser.add_argument_group("computation arguments")
    group.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="",
    )
    group.add_argument(
        "--save_tmp",
        default=True,
        help="",
    )
    group.add_argument(
        "--cache",
        default=False,
        help="",
    )
    return parser


def add_spatial_autocor_parameters(parser):
    group = parser.add_argument_group("spatial autocorrelation module generation arguments")
    group.add_argument(
        "--methods",
        default=None,
        help="The algorithm for computing spatial autocorrelation, input a list. choose from ['FDR_C', 'FDR_I', 'FDR_G', 'FDR']",
    )
    group.add_argument(
        "--local",
        default=False,
        help="If to use local spatial autocorrelation algorithm SOMDE.",
    )
    group.add_argument(
        "--somde_k",
        default=20,
        type=int,
        help="kernel value k when using local spatial autocorrelation algorithm SOMDE.",
    )
    group.add_argument(
        "--mode",
        choices=["moran", "geary"],
        default="moran",
        help="The algorithm for computing spatial genes co-expression, bi-variate version (default: moran).",
    )
    group.add_argument(
        "--operation",
        choices=["intersection", "union"],
        default="intersection",
        help="When provide several methods to compute spatial autocorrelation, ",
    )
    group.add_argument(
        "--combine",
        default=False,
        help="combine",
    )
    group.add_argument(
        "--layer_key",
        type=str,
        default=None,
        help="",
    )
    group.add_argument(
        "--model",
        type=str,
        default='danb',
        choices=["bernoulli", "danb", "normal", "none"],
        help="",
    )
    # group.add_argument(
    #     "--latent_obsm_key",
    #     type=str,
    #     default='spatial',
    #     help="",
    # )
    group.add_argument(
        "--umi_counts_obs_key",
        type=str,
        default=None,
        help="",
    )
    group.add_argument(
        "--n_neighbors",
        type=int,
        default=30,
        help="",
    )
    group.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="",
    )
    group.add_argument(
        "--distances_obsp_key",
        type=str,
        help="",
    )
    group.add_argument(
        "--neighborhood_factor",
        type=str,
        help="",
    )
    group.add_argument(
        "--approx_neighbors",
        type=str,
        help="",
    )
    group.add_argument(
        "--weighted_graph",
        default=False,
        help="",
    )
    # group.add_argument(
    #     "--sigma",
    #     type=int,
    #     default=15,
    #     help="",
    # )
    # group.add_argument(
    #     "--zero_cutoff",
    #     type=int,
    #     default=5,
    #     help="",
    # )
    # group.add_argument(
    #     "--cor_method",
    #     type=str,
    #     default='spearman',
    #     help="",
    # )
    return parser


def add_ctx_parameters(parser):
    group = parser.add_argument_group("Regulons Generation arguments")
    group.add_argument(
        "--rho_mask_dropouts",
        action="store_const",
        const=True,
        default=False,
        help="",
    )
    group.add_argument(
        "--rank_threshold",
        type=int,
        default=1500,
        help="",
    )
    group.add_argument(
        "--prune_auc_threshold",
        type=float,
        default=0.5,
        help="",
    )
    group.add_argument(
        "--nes_threshold",
        type=float,
        default=0.5,
        help="",
    )
    group.add_argument(
        "--motif_similarity_fdr",
        type=float,
        default=0.05,
        help="",
    )
    group.add_argument(
        "--orthologuous_identity_threshold",
        help="",
    )
    group.add_argument(
        "--weighted_recovery",
        help="",
    )
    group.add_argument(
        "--module_chunksize",
        help="",
    )


def add_aucell_parameters(parser):
    group = parser.add_argument_group("Regulon Activity Calculation arguments")
    group.add_argument(
        "--auc_threshold",
        type=float,
        default=0.5,
        help="",
    )
    group.add_argument(
        "--noweights",
        default=False,
        help="",
    )
    group.add_argument(
        "--normalize",
        default=False,
        help="",
    )
    group.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="seed for generating random numbers",
    )


def add_receptor_parameters(parser):
    group = parser.add_argument_group("Receptors Detection arguments")
    group.add_argument(
        "--reference",
        # type=argparse.FileType("r"),
        # type=str,
        help="pandas.DataFrame containing ligand-receptor information. MUST be pandas.DataFrame, not the name of the file",
        default=None,
        required=False,
    )
    # group.add_argument(
    #     "--receptor_key",
    #     type=str,
    #     default='to',
    #     help="",
    #     required=False,
    # )


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog='spagrn',
        # prog=os.path.splitext(os.path.basename(__file__))[0],
        description="Spatial Gene Regulatory Network inference",
        fromfile_prefix_chars="@",
        add_help=True,
        epilog="Arguments can be read from file using a @args.txt construct. "
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # --------------------------------------------
    # create the parser for the "infer" command
    # --------------------------------------------
    parser_infer = subparsers.add_parser(
        "infer", help="Derive regulons from SRT expression matrix and spatial coordinates."
    )
    parser_infer.add_argument(
        "expression_mtx_fname",
        # type=argparse.FileType("r"),
        type=str,
        help="The name of the file that contains the expression matrix for the single cell experiment."
             " H5AD file formats are supported.",
    )
    parser_infer.add_argument(
        "tfs_fname",
        # type=argparse.FileType("r"),
        type=str,
        help="TF list file.",
    )
    parser_infer.add_argument(
        "-db",
        "--database",
        # type=argparse.FileType("r"),
        type=str,
        required=True,
        help="ranked motifs database file, in feather format.",
    )
    parser_infer.add_argument(
        "--motif",
        # type=argparse.FileType("r"),
        type=str,
        required=True,
        help="motif annotation file, in tbl format.",
    )
    parser_infer.add_argument(
        "-p",
        "--project_name",
        default="",
        help="Project name/Prefix of output files.",
    )
    parser_infer.add_argument(
        "-c",
        "--cluster_label",
        default="annotation",
        help="label storing cell type/cluster annotation.",
    )
    parser_infer.add_argument(
        "-o",
        "--output",
        # type=argparse.FileType("w"),
        type=str,
        # default=str(sys.stdout),
        help="Output file/stream, i.e. the adjacencies table with correlations (csv, tsv).",
    )
    parser_infer.add_argument(
        "--spatial",
        type=str,
        default='spatial',
        help="",
    )
    add_computation_parameters(parser_infer)
    add_spatial_autocor_parameters(parser_infer)
    add_ctx_parameters(parser_infer)
    add_aucell_parameters(parser_infer)
    add_receptor_parameters(parser_infer)
    parser_infer.set_defaults(func=inference_command)

    # ----------------------------------------------
    # create the parser for the "receptor" command
    # ----------------------------------------------
    # parser_receptor = subparsers.add_parser(
    #     "receptor",
    #     help="[Optional] Receptor detection.",
    # )
    # parser_receptor.add_argument(
    #     "spagrn_output_fname",
    #     type=str,
    #     help="",
    # )
    # add_receptor_parameters(parser_receptor)
    # parser_receptor.set_defaults(func=receptor_command)

    # ------------------------------------------
    # create the parser for the "plot" command
    # ------------------------------------------
    # 'plot' subcommand
    plot_parser = subparsers.add_parser('plot', help='Plot subcommand')
    plot_parser.add_argument(
        '-d',
        '--data',
        type=str,
        # required=True,
        help='Path to the data file'
    )
    plot_parser.add_argument(
        '-n',
        '--name',
        # required=True,
        type=str,
        help='name for gene/regulon to plot.'
    )
    plot_parser.add_argument(
        '--color',
        default='celltype',
        type=str,
        help=''
    )
    plot_parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=['gene', 'regulon', 'celltype'],
        default='regulon',
        help="choose to plot gene, regulon or celltype. (default: regulon) ",
    )
    plot_parser.add_argument(
        "--dimension",
        type=str,
        choices=['2d', '3d', '2D', '3D'],
        default='2d',
        help="choose data dimension to plot, choices are {2d, 3d, 2D, 3D}. (default: 2d) ",
    )
    plot_parser.add_argument(
        '-p',
        '--pos_label',
        default='spatial',
        type=str,
        help=''
    )
    plot_parser.add_argument(
        '--custom_labels',
        default=None,
        help=''
    )
    plot_parser.add_argument(
        '-o',
        '--output',
        # required=True,
        type=str,
        help=''
    )
    add_plot_parameters(plot_parser)
    add_3d_parameters(plot_parser)
    plot_parser.set_defaults(func=plot_command)

    # for more plotting functions/alternatives
    # make subcommands
    plot_subparsers = plot_parser.add_subparsers(title='plot subcommands', dest='plot_command',
                                                 help='plot subcommand help')

    # 'heatmap' subcommand under 'plot'
    heatmap_parser = plot_subparsers.add_parser('heatmap', help='Generate a heatmap')
    heatmap_parser.add_argument(
        '-d',
        '--data',
        type=str,
        required=True,
        help='Path to the data file'
    )
    heatmap_parser.add_argument(
        '--cluster_label',
        type=str,
        default='annotation',
        help='label for cell type/cluster annotation'
    )
    heatmap_parser.add_argument(
        '--rss_fn',
        type=str,
        default=None,
        help='Regulon specificity scores (RSS) file, a text file.'
    )
    heatmap_parser.add_argument(
        '--topn',
        type=int,
        default=10,
        help='select top N regulons for each cell type'
    )
    heatmap_parser.add_argument(
        '--subset',
        default=False,
        help='If subset data by cells'
    )
    heatmap_parser.add_argument(
        '--subset_size',
        type=int,
        default=5000,
        help='If subset, number of cells to keep.'
    )
    heatmap_parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='clusters_heatmap_top.png',
        required=True,
        help='Output image file name.'
    )
    heatmap_parser.add_argument(
        '--legend_fn',
        type=str,
        default='rss_celltype_legend.png',
        help='Output legend image file name.'
    )
    heatmap_parser.add_argument(
        '--cluster_list',
        default=None,
        help='list of clusters to map celltype-color, generate legend.'
    )
    heatmap_parser.add_argument(
        '--row_cluster',
        default=False,
    )
    heatmap_parser.add_argument(
        '--col_cluster',
        default=False,
    )
    add_plot_parameters(heatmap_parser)
    heatmap_parser.set_defaults(func=heatmap_command)

    # 'heatmap' subcommand under 'plot'
    web_parser = plot_subparsers.add_parser('web', help='Generate plot html file via Plotly')
    web_parser.add_argument(
        '-d',
        '--data',
        type=str,
        required=True,
        help='Path to the data file'
    )

    return parser


def add_plot_parameters(parser):
    group = parser.add_argument_group("General Plots arguments")
    group.add_argument(
        '--figsize',
        default=(3, 5.5),
        help='heatmap figure size, a tuple. (default: (3, 5.5))'
    )
    group.add_argument(
        '--marker',
        type=str,
        default='.',
        help=''
    )
    group.add_argument(
        '--show_bg',
        default=False,
        help=''
    )
    group.add_argument(
        '--cmap',
        type=str,
        default="YlGnBu",
        help='Colormap for the heatmap'
    )
    group.add_argument(
        '--annot',
        default=False,
        help=''
    )
    group.add_argument(
        '--square',
        default=False,
        help=''
    )
    group.add_argument(
        '--linecolor',
        default="gray",
        help=''
    )
    group.add_argument(
        '--yticklabels',
        default=False,
        help=''
    )
    group.add_argument(
        '--xticklabels',
        default=True,
        help=''
    )
    group.add_argument(
        '--vmin',
        default=-3,
        help='minimum value'
    )
    group.add_argument(
        '--vmax',
        default=3,
        help='max value'
    )
    group.add_argument(
        '--s',
        default=1,
        type=float,
        help='scatter s'
    )
    group.add_argument(
        '--edgecolors',
        type=str,
        default='none',
        help=''
    )
    group.add_argument(
        '--lw',
        default=0,
        help='line weight'
    )


def add_3d_parameters(parser):
    group = parser.add_argument_group("3D Plots arguments")
    group.add_argument(
        '-vv',
        '--view_vertical',
        type=int,
        default=222,
        help='vertical angle to view to the 3D object. range from 0 to 360.'
    )
    group.add_argument(
        '-vh',
        '--view_horizontal',
        type=int,
        default=-80,
        help='horizontal angle to view the 3D object. range from 0 to 360.'
    )
    group.add_argument(
        '--xscale',
        type=float,
        default=1,
        help='scale of x axis. (default: 1)'
    )
    group.add_argument(
        '--yscale',
        type=float,
        default=1,
        help=''
    )
    group.add_argument(
        '--zscale',
        type=float,
        default=1,
        help=''
    )


def plot_command(args):
    import scanpy as sc
    adata = sc.read_h5ad(args.data)
    auc_mtx = adata.obsm['auc_mtx']
    if args.mode == 'celltype':
        prn.plot_celltype(adata, color=args.color, spatial_label=args.pos_label, custom_labels=args.custom_labels,
                          fn=args.output)
                          # marker=args.marker, s=args.s)  # why the program keep crushing after added this line
    elif args.mode == 'gene':
        prn.plot_gene(adata, gene_name=args.name, fn=args.output, pos_label=args.pos_label, show_bg=args.show_bg,
                      cmap=args.cmap)
                # marker=args.marker, edgecolors=args.edgecolors, lw=args.lw)
    elif args.mode == 'regulon':
        if args.dimension in ['2d', '2D']:
            prn.plot_2d(adata, auc_mtx, reg_name=args.name, fn=args.output, pos_label=args.pos_label, cmap=args.cmap)
                        # marker=args.marker, edgecolors=args.edgecolors, lw=args.lw)
        elif args.dimension in ['3d', '3D']:
            prn.plot_3d_reg(adata, auc_mtx, reg_name=args.name, fn=args.output, view_vertical=args.view_vertical,
                            view_horizontal=args.view_horizontal, show_bg=args.show_bg, pos_label=args.pos_label,
                            xscale=args.xscale, yscale=args.yscale, zscale=args.zscale)


def heatmap_command(args):
    import scanpy as sc
    adata = sc.read_h5ad(args.data)
    auc_mtx = adata.obsm['auc_mtx']
    prn.auc_heatmap(adata, auc_mtx, args.cluster_label, args.rss_fn, topn=args.topn, cmap=args.cmap,
                    subset=args.subset, subset_size=args.subset_size, fn=args.output, legend_fn=args.legend_fn,
                    cluster_list=args.cluster_list, row_cluster=args.row_cluster, col_cluster=args.col_cluster)


def main(argv=None):
    # Parse arguments.
    parser = create_argument_parser()
    args = parser.parse_args(args=argv)
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
