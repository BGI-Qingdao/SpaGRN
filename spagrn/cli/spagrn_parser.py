#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: spagrn.py
@time: 2023/Nov/01
@description: test file for inference gene regulatory networks module
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

import os
import sys
sys.path.append('/Users/Oreo/PycharmProjects/SpaGRN/')
import argparse
import pandas as pd
from pathlib import Path, PurePath
from multiprocessing import cpu_count
from spagrn.regulatory_network import InferNetwork as irn
# import spagrn.plot as prn
import scanpy as sc


def scc_command(args):
    """
        Infer co-expression modules.
        """
    print("Loading expression anndata.")
    try:
        ex_mtx = irn.read_file(
            args.expression_mtx_fname.name)
    except ValueError as e:
        print(e)
        sys.exit(1)

    tf_names = irn.load_tfs(args.tfs_fname.name)

    if args.sparse:
        n_total_genes = len(ex_mtx[1])
        n_matching_genes = len(ex_mtx[1].isin(tf_names))
    else:
        n_total_genes = len(ex_mtx.columns)
        n_matching_genes = len(ex_mtx.columns.isin(tf_names))
    if n_total_genes == 0:
        print(
            "The expression matrix supplied does not contain any genes. "
            "Make sure the extension of the file matches the format (tab separation for TSV and "
            "comma sepatration for CSV)."
        )
        sys.exit(1)
    if float(n_matching_genes) / n_total_genes < 0.80:
        print(
            "Expression data is available for less than 80% of the supplied transcription factors."
        )

    print("Inferring regulatory networks.")
    client, shutdown_callback = _prepare_client(
        args.client_or_address, num_workers=args.num_workers
    )
    method = grnboost2 if args.method == "grnboost2" else genie3
    try:
        if args.sparse:
            network = method(
                expression_data=ex_mtx[0],
                gene_names=ex_mtx[1],
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=args.seed,
            )
        else:
            network = method(
                expression_data=ex_mtx,
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=args.seed,
            )
    finally:
        shutdown_callback(False)

    print("Writing results to file.")
    extension = PurePath(args.output.name).suffixes
    network.to_csv(args.output.name, index=False, sep=suffixes_to_separator(extension))


def spg_command(args):
    """
    Infer co-expression modules.
    """
    print("Loading expression anndata.")
    try:
        ex_mtx = irn.read_file(
            args.expression_mtx_fname.name)
    except ValueError as e:
        print(e)
        sys.exit(1)

    tf_names = irn.load_tfs(args.tfs_fname.name)

    if args.sparse:
        n_total_genes = len(ex_mtx[1])
        n_matching_genes = len(ex_mtx[1].isin(tf_names))
    else:
        n_total_genes = len(ex_mtx.columns)
        n_matching_genes = len(ex_mtx.columns.isin(tf_names))
    if n_total_genes == 0:
        print(
            "The expression matrix supplied does not contain any genes. "
            "Make sure the extension of the file matches the format (tab separation for TSV and "
            "comma sepatration for CSV)."
        )
        sys.exit(1)
    if float(n_matching_genes) / n_total_genes < 0.80:
        print(
            "Expression data is available for less than 80% of the supplied transcription factors."
        )

    print("Inferring regulatory networks.")
    client, shutdown_callback = _prepare_client(
        args.client_or_address, num_workers=args.num_workers
    )
    method = spg if args.method == "spg" else scc
    try:
        if args.sparse:
            network = method(
                expression_data=ex_mtx[0],
                gene_names=ex_mtx[1],
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=args.seed,
            )
        else:
            network = method(
                expression_data=ex_mtx,
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=args.seed,
            )
    finally:
        shutdown_callback(False)

    print("Writing results to file.")
    extension = PurePath(args.output.name).suffixes
    network.to_csv(args.output.name, index=False, sep=suffixes_to_separator(extension))


def run_all(args):
    fn = args.data
    tfs_fn = args.tf
    database_fn = args.database
    motif_anno_fn = args.motif_anno
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    method = args.method
    prefix = os.path.join(out_dir, method)

    # load data
    data = irn.read_file(fn)
    data = irn.preprocess(data)
    sc.tl.pca(data)

    # create grn
    grn = irn(data)

    # set parameters
    grn.add_params('hotspot', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('scc', {'prune_auc_threshold': 0.05, 'rank_threshold': 9000, 'auc_threshold': 0.05})
    grn.add_params('grnboost', {'prune_auc_threshold': 0.05, 'rank_threshold': 3000, 'auc_threshold': 0.05})

    # niche data
    niche_human = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_human.csv')
    niche_mouse = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/resource/lr_network_mouse.csv')
    niches = pd.concat([niche_mouse, niche_human])

    # run_all analysis
    grn.infer(database_fn,
              motif_anno_fn,
              tfs_fn,
              niche_df=niches,
              num_workers=cpu_count(),
              cache=False,
              save_tmp=True,
              c_threshold=0.2,
              layers=None,
              latent_obsm_key='spatial',
              model='danb',  # bernoulli
              n_neighbors=30,
              weighted_graph=False,
              cluster_label='celltype',
              method=method,
              prefix=prefix,
              noweights=False,
              rho_mask_dropouts=False)


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


def add_coexp_parameters(parser):
    group = parser.add_argument_group("co-expressed module generation arguments")
    group.add_argument(
        "--c_threshold",
        type=float,
        default=-1,
        help="",
    )
    group.add_argument(
        "--layer_key",
        type=argparse.FileType("r"),
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
    group.add_argument(
        "--latent_obsm_key",
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
        "--min_genes",
        type=int,
        default=20,
        help="",
    )
    group.add_argument(
        "--expression_mtx_fname",
        type=argparse.FileType("r"),
        help="",
    )
    group.add_argument(
        "--mask_dropouts",
        action="store_const",
        const=True,
        default=False,
        help="",
    )
    return parser


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.splitext(os.path.basename(__file__))[0],
        description="Spatial Gene Regulatory Network inference",
        fromfile_prefix_chars="@",
        add_help=True,
        epilog="Arguments can be read from file using a @args.txt construct. "
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # --------------------------------------------
    # create the parser for the "spg" command
    # --------------------------------------------
    parser_spg = subparsers.add_parser(
        "spg", help="Derive regulons from expression matrix by spatial-proximity-graph (SPG) model."
    )
    parser_spg.add_argument(
        "expression_mtx_fname",
        type=argparse.FileType("r"),
        help="",
    )
    parser_spg.add_argument(
        "tfs_fname",
        type=argparse.FileType("r"),
        help="",
    )
    parser_spg.add_argument(
        "-d",
        "--database",
        type=argparse.FileType("r"),
        help="ranked motifs database file, in feather format",
    )
    parser_spg.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file/stream, i.e. a table of TF-target genes (CSV).",
    )
    parser_spg.add_argument(
        "-m",
        "--method",
        choices=["genie3", "grnboost2"],
        default="grnboost2",
        help="The algorithm for gene regulatory network reconstruction (default: grnboost2).",
    )
    parser_spg.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="Seed value for regressor random state initialization. Applies to both GENIE3 and GRNBoost2. The default is to use a random seed.",
    )
    add_computation_parameters(parser_spg)
    parser_spg.set_defaults(func=spg_command)

    # -----------------------------------------
    # create the parser for the "scc" command
    # -----------------------------------------
    parser_scc = subparsers.add_parser(
        "scc",
        help='[Optional] Add Pearson correlations based on TF-gene expression to the network adjacencies output from the GRN step, and output these to a new adjacencies file. This will normally be done during the "ctx" step.',
    )
    parser_scc.add_argument(
        "adjacencies",
        type=argparse.FileType("r"),
        help="The name of the file that contains the GRN adjacencies (output from the GRN step).",
    )
    parser_scc.add_argument(
        "expression_mtx_fname",
        type=argparse.FileType("r"),
        help="The name of the file that contains the expression matrix for the single cell experiment."
        " Two file formats are supported: csv (rows=cells x columns=genes) or loom (rows=genes x columns=cells).",
    )
    parser_scc.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file/stream, i.e. the adjacencies table with correlations (csv, tsv).",
    )
    add_coexp_parameters(parser_scc)
    parser_scc.set_defaults(func=scc_command)

    # -----------------------------------------
    # create the parser for the "plot" command
    # -----------------------------------------
    parser_plot = subparsers.add_parser(
        "plot",
        help="Find enriched motifs for a gene signature and optionally prune targets from this signature based on cis-regulatory cues.",
    )
    parser_plot.add_argument(
        "module_fname",
        type=argparse.FileType("r"),
        help="The name of the file that contains the signature or the co-expression modules. "
        "The following formats are supported: CSV or TSV (adjacencies), YAML, GMT and DAT (modules)",
    )
    parser_plot.add_argument(
        "database_fname",
        type=argparse.FileType("r"),
        nargs="+",
        help="The name(s) of the regulatory feature databases. "
        "Two file formats are supported: feather or db (legacy).",
    )
    parser_plot.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file/stream, i.e. a table of enriched motifs and target genes (csv, tsv)"
        " or collection of regulons (yaml, gmt, dat, json).",
    )
    parser_plot.add_argument(
        "-n",
        "--no_pruning",
        action="store_const",
        const="yes",
        help="Do not perform pruning, i.e. find enriched motifs.",
    )
    parser_plot.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="The size of the module chunks assigned to a node in the dask graph (default: 100).",
    )
    parser_plot.add_argument(
        "--mode",
        choices=["custom_multiprocessing", "dask_multiprocessing", "dask_cluster"],
        default="custom_multiprocessing",
        help="The mode to be used for computing (default: custom_multiprocessing).",
    )
    parser_plot.add_argument(
        "-a",
        "--all_modules",
        action="store_const",
        const="yes",
        default="no",
        help="Included positive and negative regulons in the analysis (default: no, i.e. only positive).",
    )
    parser_plot.add_argument(
        "-t",
        "--transpose",
        action="store_const",
        const="yes",
        help="Transpose the expression matrix (rows=genes x columns=cells).",
    )

    # --------------------------------------------
    # create the parser for the "util" command
    # -------------------------------------------
    parser_util = subparsers.add_parser(
        "util", help="Quantify activity of gene signatures across single cells."
    )

    # Mandatory arguments
    parser_util.add_argument(
        "expression_mtx_fname",
        type=argparse.FileType("r"),
        help="The name of the file that contains the expression matrix for the single cell experiment."
        " Two file formats are supported: csv (rows=cells x columns=genes) or loom (rows=genes x columns=cells).",
    )
    parser_util.add_argument(
        "signatures_fname",
        type=argparse.FileType("r"),
        help="The name of the file that contains the gene signatures."
        " Three file formats are supported: gmt, yaml or dat (pickle).",
    )
    # Optional arguments
    parser_util.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file/stream, a matrix of AUC values."
        " Two file formats are supported: csv or loom."
        " If loom file is specified the loom file while contain the original expression matrix and the"
        " calculated AUC values as extra column attributes.",
    )
    parser_util.add_argument(
        "-t",
        "--transpose",
        action="store_const",
        const="yes",
        help="Transpose the expression matrix if supplied as csv (rows=genes x columns=cells).",
    )
    parser_util.add_argument(
        "-w",
        "--weights",
        action="store_const",
        const="yes",
        help="Use weights associated with genes in recovery analysis."
        " Is only relevant when gene signatures are supplied as json format.",
    )
    parser_util.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="The number of workers to use (default: {}).".format(cpu_count()),
    )
    parser_util.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="Seed for the expression matrix ranking step. The default is to use a random seed.",
    )

    return parser


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
