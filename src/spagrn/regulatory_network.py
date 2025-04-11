#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: infer gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

# python core modules
import os

# third party modules
import warnings
import json
import glob
import anndata
import hotspot
import pickle
import scipy
import pandas as pd
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Sequence, Type, Optional, Optional, List

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from ctxcore.genesig import Regulon, GeneSignature
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell, derive_auc_threshold
from pyscenic.prune import prune2df, df2regulons

# modules in self project

from .autocor import *
from .corexp import *
from .c_autocor import gearys_c
from .m_autocor import morans_i_p_values, morans_i_zscore
from .g_autocor import getis_g
from .network import Network


def intersection_ci(iterableA, iterableB, key=lambda x: x) -> list:
    """
    Return the intersection of two iterables with respect to `key` function.
    (ci: case insensitive)
    :param iterableA: list no.1
    :param iterableB: list no.2
    :param key:
    :return:
    """

    def unify(iterable):
        d = {}
        for item in iterable:
            d.setdefault(key(item), []).append(item)
        return d

    A, B = unify(iterableA), unify(iterableB)
    matched = []
    for k in A:
        if k in B:
            matched.append(B[k][0])
    return matched


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]


def _set_client(num_workers: int) -> Client:
    """
    set number of processes when perform parallel computing
    :param num_workers:
    :return:
    """
    local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    custom_client = Client(local_cluster)
    return custom_client


def save_list(l, fn='list.txt'):
    """save a list into a text file"""
    with open(fn, 'w') as f:
        f.write('\n'.join(l))


class InferNetwork(Network):
    """
    Algorithms to infer Gene Regulatory Networks (GRNs)
    """

    def __init__(self, adata=None, project_name: str = 'project', ):
        """
        Constructor of this Object.
        :param data: sequencing data in AnnData format
        :return:
        """
        super().__init__()
        self.data = adata
        self.project_name = project_name

        self.more_stats = None
        self.weights = None
        self.ind = None
        self.weights_n = None

        # other settings
        self._params = {
            'rank_threshold': 1500,
            'prune_auc_threshold': 0.05,
            'nes_threshold': 3.0,
            'motif_similarity_fdr': 0.05,
            'auc_threshold': 0.05,
            'noweights': False,
        }
        self.tmp_dir = None

    # GRN pipeline infer logic
    def infer(self,
              databases: str,
              motif_anno_fn: str,
              tfs_fn,
              gene_list: Optional[List] = None,
              cluster_label='annotation',
              niche_df=None,
              receptor_key='to',
              num_workers=None,
              save_tmp=False,
              cache=False,
              output_dir=None,

              layers='raw_counts',
              model='bernoulli',
              latent_obsm_key='spatial',
              umi_counts_obs_key=None,
              n_neighbors=30,
              weighted_graph=False,
              rho_mask_dropouts=False,
              local=False,
              methods=None,
              operation='intersection',
              combine=False,
              mode='moran',
              somde_k=20,
              noweights=None,
              normalize: bool = False):
        print('----------------------------------------')
        # Set project name
        print(f'Project name is {self.project_name}')
        # Set general output directory
        if output_dir is None:  # when output dir is not set
            output_dir = os.path.dirname(os.path.abspath(__file__))  # set output dir to current working dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f'Saving output files into {output_dir}')
        # Set project tmp directory to save temporary files
        if save_tmp:
            self.tmp_dir = os.path.join(output_dir, 'tmp_files')
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            print(f'Saving temporary files to {self.tmp_dir}')
        print('----------------------------------------')

        global adjacencies
        exp_mat = self._data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if noweights is None:
            noweights = self.params["noweights"]

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)

        # 3. GRN Inference
        adjacencies = self.spg(self.data,
                               gene_list=gene_list,
                               tf_list=tfs,
                               jobs=num_workers,
                               layer_key=layers,
                               model=model,
                               latent_obsm_key=latent_obsm_key,
                               umi_counts_obs_key=umi_counts_obs_key,
                               n_neighbors=n_neighbors,
                               weighted_graph=weighted_graph,
                               cache=cache,
                               save_tmp=save_tmp,
                               fn=os.path.join(self.tmp_dir, f'{mode}_adj.csv'),
                               local=local,
                               methods=methods,
                               operation=operation,
                               combine=combine,
                               mode=mode,
                               somde_k=somde_k)

        # 4. Compute Modules
        # ctxcore.genesig.Regulon
        modules = self.get_modules(adjacencies,
                                   exp_mat,
                                   cache=cache,
                                   save_tmp=save_tmp,
                                   rho_mask_dropouts=rho_mask_dropouts)
        # before_cistarget(tfs, modules, project_name)

        # 5. Regulons Prediction aka cisTarget
        # ctxcore.genesig.Regulon
        regulons = self.prune_modules(modules,
                                      dbs,
                                      motif_anno_fn,
                                      num_workers=num_workers,
                                      save_tmp=save_tmp,
                                      cache=cache,
                                      fn=os.path.join(self.tmp_dir, 'motifs.csv'),
                                      rank_threshold=self.params["rank_threshold"],
                                      auc_threshold=self.params["prune_auc_threshold"],
                                      nes_threshold=self.params["nes_threshold"],
                                      motif_similarity_fdr=self.params["motif_similarity_fdr"])

        # 6.0. Cellular Enrichment (aka AUCell)
        self.cal_auc(exp_mat,
                     regulons,
                     auc_threshold=self.params["auc_threshold"],
                     num_workers=num_workers,
                     save_tmp=save_tmp,
                     cache=cache,
                     noweights=noweights,
                     normalize=normalize,
                     fn=os.path.join(self.tmp_dir, 'auc_mtx.csv'))

        # 6.1. Receptor AUCs
        if niche_df is not None:
            self.get_filtered_receptors(niche_df, receptor_key=receptor_key)
            receptor_auc_mtx = self.receptor_auc()
            self.isr(receptor_auc_mtx)

        # 7. Calculate Regulon Specificity Scores
        self.cal_regulon_score(cluster_label=cluster_label, save_tmp=save_tmp,
                               fn=f'{self.tmp_dir}/regulon_specificity_scores.txt')

        # 8. Save results to h5ad file
        # dtype=object
        self.data.write_h5ad(os.path.join(output_dir, f'{self.project_name}_spagrn.h5ad'))
        return self.data

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        """only use this function when setting params as a whole.
        use add_params to solely update/add some of the params and keep the rest unchanged"""
        self._params = value

    def add_params(self, dic: dict):
        """
        :param dic: keys are parameter name, values are parameter values

        Example:
            grn = InferNetwork(data)
            grn.add_params({'num_worker':12, 'auc_threshold': 0.001})
        """
        og_params = deepcopy(self._params)
        try:
            for key, value in dic.items():
                self._params[key] = value
        except KeyError:
            self._params = og_params

    # ------------------------------------------------------#
    #             step0: LOAD AUXILIARY DATA                #
    # ------------------------------------------------------#
    @staticmethod
    def read_motif_file(fname):
        """
        Read motifs.csv file generate by
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
        Get a list of interested TFs from a text file
        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    @staticmethod
    def load_database(database_dir: str) -> list:
        """
        Load motif ranking database
        :param database_dir:
        :return:
        """
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=_name(fname)) for fname in db_fnames]
        return dbs

    # ------------------------------------------------------#
    #           step1: CALCULATE TF-GENE PAIRS              #
    # ------------------------------------------------------#
    def rf_infer(self,
                 matrix,
                 tf_names,
                 genes: list,
                 num_workers: int,
                 verbose: bool = True,
                 cache: bool = True,
                 save_tmp: bool = True,
                 fn: str = 'adj.csv',
                 **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules via random forest (RF) module
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param tf_names: list of target TFs or a`ll
        :param genes: list of interested genes
        :param num_workers: number of thread
        :param verbose: if print out running details
        :param cache:
        :param save_tmp: if save adjacencies result into a file
        :param fn: adjacencies file name
        :return:
        """
        if cache and os.path.isfile(fn):
            adjacencies = pd.read_csv(fn)
            self.adjacencies = adjacencies
            self.data.uns['adj'] = adjacencies
            return adjacencies

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = _set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                **kwargs)
        if save_tmp:
            adjacencies.to_csv(os.path.join(self.tmp_dir, fn), index=False)
        self.adjacencies = adjacencies
        self.data.uns['adj'] = adjacencies
        return adjacencies

    def spatial_autocorrelation(self,
                                adata,
                                layer_key="raw_counts",
                                latent_obsm_key="spatial",
                                n_neighbors=10,
                                somde_k=20,
                                n_processes=None,
                                local=False,
                                cache=False):
        """
        Calculate spatial autocorrelation values using Moran's I, Geary'C, Getis's G and SOMDE algorithms
        :param adata:
        :param layer_key:
        :param latent_obsm_key:
        :param n_neighbors:
        :param somde_k:
        :param n_processes:
        :param local:
        :param cache:
        :return:
        """
        print('Computing spatial weights matrix...')
        self.ind, neighbors, self.weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key,
                                                                    n_neighbors=n_neighbors)
        Weights = get_w(self.ind, self.weights_n)
        self.weights = Weights

        more_stats = pd.DataFrame(index=adata.var_names)
        if local:
            if cache and os.path.isfile(f'{self.tmp_dir}/local_more_stats.csv.csv'):
                print(f'Found cached file {self.tmp_dir}/local_more_stats.csv')
                more_stats = pd.read_csv(f'{self.tmp_dir}/local_more_stats.csv', index_col=0, sep='\t')
                self.more_stats = more_stats
                return more_stats
            print('Computing SOMDE...')
            adjusted_p_values = somde_p_values(adata, k=somde_k, layer_key=layer_key, latent_obsm_key=latent_obsm_key)
            more_stats['FDR_SOMDE'] = adjusted_p_values
            more_stats.to_csv(f'{self.tmp_dir}/local_more_stats.csv', sep='\t')
        else:
            if cache and os.path.isfile(f'{self.tmp_dir}/more_stats.csv'):
                print(f'Found cached file {self.tmp_dir}/more_stats.csv')
                more_stats = pd.read_csv(f'{self.tmp_dir}/more_stats.csv', index_col=0, sep='\t')
                self.more_stats = more_stats
                return more_stats
            print("Computing Moran's I...")
            morans_ps = morans_i_p_values(adata, Weights, layer_key=layer_key, n_process=n_processes)
            fdr_morans_ps = fdr(morans_ps)
            print("Computing Geary's C...")
            gearys_cs = gearys_c(adata, Weights, layer_key=layer_key, n_process=n_processes, mode='pvalue')
            fdr_gearys_cs = fdr(gearys_cs)
            print("Computing Getis G...")
            getis_gs = getis_g(adata, Weights, n_processes=n_processes, layer_key=layer_key, mode='pvalue')
            fdr_getis_gs = fdr(getis_gs)
            # save results
            more_stats = pd.DataFrame({
                'C': gearys_cs,
                'FDR_C': fdr_gearys_cs,
                'I': morans_ps,
                'FDR_I': fdr_morans_ps,
                'G': getis_gs,
                'FDR_G': fdr_getis_gs
            }, index=adata.var_names)

        self.more_stats = more_stats
        return more_stats

    @staticmethod
    def spatial_autocorrelation_zscore(adata,
                                       layer_key="raw_counts",
                                       latent_obsm_key="spatial",
                                       n_neighbors=10,
                                       n_processes=None):
        """
        Calculate spatial autocorrelation values using Moran's I, Geary'C, Getis's G and SOMDE algorithms
        :param adata:
        :param layer_key:
        :param latent_obsm_key:
        :param n_neighbors:
        :param n_processes:
        :return:
        """
        print('Computing spatial weights matrix...')
        ind, neighbors, weights_n = neighbors_and_weights(adata, latent_obsm_key=latent_obsm_key,
                                                          n_neighbors=n_neighbors)
        Weights = get_w(ind, weights_n)

        print("Computing Moran's I...")
        morans_ps = morans_i_zscore(adata, Weights, layer_key=layer_key, n_process=n_processes)
        print("Computing Geary's C...")
        gearys_cs = gearys_c(adata, Weights, layer_key=layer_key, n_process=n_processes, mode='zscore')
        print("Computing Getis G...")
        getis_gs = getis_g(adata, Weights, n_processes=n_processes, layer_key=layer_key, mode='zscore')
        # save results
        more_stats = pd.DataFrame({
            'C_zscore': gearys_cs,
            'I_zscore': morans_ps,
            'G_zscore': getis_gs,
        }, index=adata.var_names)
        return more_stats

    def select_genes(self, methods=None, fdr_threshold=0.05, local=True, combine=True, operation='intersection'):
        """
        Select genes based FDR values...
        :param methods:
        :param fdr_threshold:
        :param local:
        :param combine:
        :param operation:
        :return:
        """
        if methods is None:
            methods = ['FDR_C', 'FDR_I', 'FDR_G', 'FDR']
        # 1. LOCAL
        if local:
            somde_genes = self.more_stats.loc[self.more_stats.FDR_SOMDE < fdr_threshold].index
            print(f'SOMDE find {len(somde_genes)} genes')
            return somde_genes
        # 2. GLOBAL
        # 2.1 combine p-values
        elif combine:
            cfdrs = combind_fdrs(self.more_stats[['FDR_C', 'FDR_I', 'FDR_G', 'FDR']])  # combine 4 types of p-values
            self.more_stats['combined'] = cfdrs
            genes = self.more_stats.loc[self.more_stats['combined'] < fdr_threshold].index
            print(f"Combinded FDRs gives: {len(cgenes)} genes")
            return genes
        # 2.2 individual p-values
        elif methods:
            indices_list = [set(self.more_stats[self.more_stats[m] < fdr_threshold].index) for m in methods]
            if operation == 'intersection':
                global_inter_genes = set.intersection(*indices_list)
                print(f'global spatial gene num (intersection): {len(global_inter_genes)}')
                return global_inter_genes
            elif operation == 'union':
                global_union_genes = set().union(*indices_list)
                print(f'global spatial gene num (union): {len(global_union_genes)}')
                return global_union_genes

    @staticmethod
    def check_stats(more_stats):
        """Compute gene numbers for each"""
        moran_genes = more_stats.loc[more_stats.FDR_I < fdr_threshold].index
        geary_genes = more_stats.loc[more_stats.FDR_C < fdr_threshold].index
        getis_genes = more_stats.loc[more_stats.FDR_G < fdr_threshold].index
        hs_genes = more_stats.loc[(more_stats.FDR < fdr_threshold)].index
        print(f"Moran's I find {len(moran_genes)} genes")
        print(f"Geary's C find {len(geary_genes)} genes")
        print(f'Getis find {len(getis_genes)} genes')
        print(f"HOTSPOT find {len(hs_genes)} genes")
        if 'FDR_SOMDE' in more_stats.columns:
            somde_genes = more_stats.loc[more_stats.FDR_SOMDE < fdr_threshold].index
            print(f'SOMDE find {len(somde_genes)} genes')

    def spg(self,
            data: anndata.AnnData,
            gene_list: Optional[List] = None,
            layer_key=None,
            model='bernoulli',
            latent_obsm_key="spatial",
            umi_counts_obs_key=None,
            weighted_graph=False,
            n_neighbors=10,
            fdr_threshold=0.05,
            tf_list=None,
            save_tmp=False,
            jobs=None,
            cache=False,
            local=False,
            methods=None,
            operation='intersection',
            combine=True,
            somde_k=20,
            fn: str = 'adj.csv',
            mode='moran',
            **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules by spatial-proximity-graph (SPG) model.
        :param data: Count matrix (shape is cells by genes)
        :param gene_list: A list of interested genes to calculate co-expression values with TF genes.
                Could be HVGs or all genes in the count matrix. When not provided, will compute spatial autocorrelation
                values between all genes in the count matrix with TF genes and select interested genes with significant
                spatial variability.
        :param layer_key: Key in adata.layers with count data, uses adata.X if None.
        :param model: Specifies the null model to use for gene expression.
            Valid choices are:
                * 'danb': Depth-Adjusted Negative Binomial
                * 'bernoulli': Models probability of detection
                * 'normal': Depth-Adjusted Normal
                * 'none': Assumes data has been pre-standardized
        :param latent_obsm_key: Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        :param umi_counts_obs_key: Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used. 'total_counts'
        :param weighted_graph: Whether or not to create a weighted graph
        :param n_neighbors: Neighborhood size
        :param fdr_threshold: Correlation threshold at which to stop assigning genes to modules
        :param tf_list: predefined TF names
        :param save_tmp: if save results onto disk
        :param jobs: Number of parallel jobs to run_all
        :param cache:
        :param local:
        :param combine:
        :param mode:
        :param somde_k:
        :param operation:
        :param methods:
        :param fn: output file name
        :return: A dataframe, local correlation Z-scores between genes (shape is genes x genes)
        """
        global local_correlations
        if cache and os.path.isfile(fn):
            print(f'Found cached file {fn}')
            local_correlations = pd.read_csv(fn)
            self.data.uns['adj'] = local_correlations
            return local_correlations
        else:
            # 2024-12-20: select genes or provide a list of genes
            if gene_list:
                hs_genes = gene_list
                print('Computing spatial weights matrix...')
                self.ind, neighbors, self.weights_n = neighbors_and_weights(data, latent_obsm_key=latent_obsm_key,
                                                                            n_neighbors=n_neighbors)
                Weights = get_w(self.ind, self.weights_n)
                self.weights = Weights
            else:
                hs = hotspot.Hotspot(data,
                                     layer_key=layer_key,
                                     model=model,
                                     latent_obsm_key=latent_obsm_key,
                                     umi_counts_obs_key=umi_counts_obs_key,
                                     **kwargs)
                hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)
                hs_results = hs.compute_autocorrelations()

                # 1: Select genes
                self.spatial_autocorrelation(data,
                                             layer_key=layer_key,
                                             latent_obsm_key=latent_obsm_key,
                                             n_neighbors=n_neighbors,
                                             somde_k=somde_k,
                                             n_processes=jobs,
                                             local=local,
                                             cache=cache)
                self.more_stats['FDR'] = hs_results.FDR
                if save_tmp:
                    self.more_stats.to_csv(f'{self.tmp_dir}/more_stats.csv', sep='\t')

                hs_genes = self.select_genes(methods=methods,
                                             fdr_threshold=fdr_threshold,
                                             local=local,
                                             combine=combine,
                                             operation=operation)
                hs_genes = list(hs_genes)
                assert len(hs_genes) > 0
                if save_tmp:
                    save_list(hs_genes, fn=f'{self.tmp_dir}/selected_genes.txt')

            # 2. Define gene-gene relationships with pair-wise local correlations
            print(f'Current mode is {mode}')
            if mode == 'zscore':
                # subset by TFs
                local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)  # jobs for parallelization
                if tf_list:
                    common_tf_list = list(set(tf_list).intersection(set(local_correlations.columns)))
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

            elif mode == 'moran':
                tfs_in_data = list(set(tf_list).intersection(set(data.var_names)))
                select_genes_not_tfs = list(set(hs_genes) - set(tfs_in_data))
                fw = flat_weights(data.obs_names, self.ind, self.weights_n, n_neighbors=n_neighbors)
                local_correlations = global_bivariate_moran_R(data,
                                                              fw,
                                                              tfs_in_data,
                                                              select_genes_not_tfs,
                                                              num_workers=jobs,
                                                              layer_key=layer_key)

            elif mode == 'geary':
                tfs_in_data = list(set(tf_list).intersection(set(data.var_names)))
                select_genes_not_tfs = list(set(hs_genes) - set(tfs_in_data))
                fw = flat_weights(data.obs_names, self.ind, self.weights_n, n_neighbors=n_neighbors)
                local_correlations = global_bivariate_gearys_C(data,
                                                               fw,
                                                               tfs_in_data,
                                                               select_genes_not_tfs,
                                                               num_workers=jobs,
                                                               layer_key=layer_key)

        local_correlations['importance'] = local_correlations['importance'].astype(np.float64)
        self.data.uns['adj'] = local_correlations
        if save_tmp:
            local_correlations.to_csv(os.path.join(self.tmp_dir, f'{mode}_adj.csv'), index=False)
        return local_correlations

    # ------------------------------------------------------#
    #            step2:  FILTER TFS AND TARGETS             #
    # ------------------------------------------------------#
    def get_modules(self,
                    adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    save_tmp=False,
                    cache=False,
                    **kwargs) -> Sequence[Regulon]:
        """
        Create of co-expression modules
        :param adjacencies:
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param rho_mask_dropouts:
        :param save_tmp:
        :param cache:
        :return:
        """
        if cache and os.path.isfile(f'{self.tmp_dir}/modules.pkl'):
            print(f'Find cached file {self.tmp_dir}/modules.pkl')
            modules = pickle.load(open(f'{self.tmp_dir}/modules.pkl', 'rb'))
            self.modules = modules
            return modules
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        self.modules = modules
        if save_tmp:
            with open(f'{self.tmp_dir}/modules.pkl', "wb") as f:
                pickle.dump(modules, f)
        return modules

    # ------------------------------------------------------#
    #            step3:  FILTER TFS AND TARGETS             #
    # ------------------------------------------------------#
    def prune_modules(self,
                      modules: Sequence[Regulon],
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int,
                      cache: bool = False,
                      save_tmp: bool = False,
                      fn: str = 'motifs.csv',
                      **kwargs) -> Sequence[Regulon]:
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
        :param save_tmp:
        :param fn:
        :param kwargs:
        :return: A dataframe.
        """
        if cache and os.path.isfile(fn):
            print(f'Find cached file {fn}')
            df = self.read_motif_file(fn)
            regulon_list = df2regulons(df)
            self.regulons = regulon_list
            self.regulon_dict = self.get_regulon_dict(regulon_list)
            self.data.uns['regulon_dict'] = self.regulon_dict
            return regulon_list

        if num_workers is None:
            num_workers = cpu_count()
        # infer function
        # #1.
        with ProgressBar():
            df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)  # rank_threshold
            df.to_csv(fn)
        # this function actually did two things. 1. get df, 2. turn df into list of Regulons
        # #2.
        regulon_list = df2regulons(df)
        self.regulons = regulon_list

        # #3. handle results
        # convert Regulon list to dictionaries for easy access and readability
        self.regulon_dict = self.get_regulon_dict(regulon_list)
        self.data.uns['regulon_dict'] = self.regulon_dict

        # save to data
        if save_tmp:
            with open(f'{self.tmp_dir}/regulons.json', 'w') as f:
                json.dump(self.regulon_dict, f, sort_keys=True, indent=4)
        return regulon_list

    # ------------------------------------------------------#
    #           step4: CALCULATE TF-GENE PAIRS              #
    # ------------------------------------------------------#
    def cal_auc(self,
                matrix,
                regulons: Sequence[Type[GeneSignature]],
                auc_threshold: float,
                num_workers: int,
                noweights: bool = False,
                normalize: bool = False,
                seed=None,
                cache: bool = True,
                save_tmp: bool = True,
                fn='auc.csv') -> pd.DataFrame:
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
        :param seed: seed for generating random numbers
        :param cache:
        :param save_tmp:
        :param fn:
        :return: A dataframe with the AUCs (n_cells x n_modules).
        """
        if cache and os.path.isfile(fn):
            print(f'Find cached file {fn}')
            auc_mtx = pd.read_csv(fn, index_col=0)  # important! cell must be index, not one of the column
            self.auc_mtx = auc_mtx
            self.data.obsm['auc_mtx'] = self.auc_mtx
            return auc_mtx

        if num_workers is None:
            num_workers = cpu_count()

        auc_mtx = aucell(matrix,
                         regulons,
                         auc_threshold=auc_threshold,
                         num_workers=num_workers,
                         noweights=noweights,
                         normalize=normalize,
                         seed=seed)

        self.auc_mtx = auc_mtx
        self.data.obsm['auc_mtx'] = self.auc_mtx
        if save_tmp:
            auc_mtx.to_csv(os.path.join(self.tmp_dir, 'auc_mtx.csv'))
        return auc_mtx

    def receptor_auc(self, auc_threshold=None, p_range=0.01, num_workers=20) -> Optional[pd.DataFrame]:
        """
        Calculate AUC value for modules that detected receptor genes within
        :param auc_threshold:
        :param p_range:
        :param num_workers:
        :return:
        """
        if self.receptor_dict is None:
            print('receptor dict not found. run_all get_receptors first.')
            return
        # 1. create new modules
        receptor_modules = list(
            map(
                lambda x: GeneSignature(
                    name=x,
                    gene2weight=self.receptor_dict[x],
                ),
                self.receptor_dict,
            )
        )
        ex_matrix = self.data.to_df()
        if auc_threshold is None:
            percentiles = derive_auc_threshold(ex_matrix)
            a_value = percentiles[p_range]
        else:
            a_value = auc_threshold
        receptor_auc_mtx = aucell(ex_matrix, receptor_modules, auc_threshold=a_value, num_workers=num_workers)
        return receptor_auc_mtx

    def isr(self, receptor_auc_mtx) -> pd.DataFrame:
        """
        Calculate ISR matrix for all the regulons
        :param receptor_auc_mtx: auc matrix for modules containing receptor genes
        :return:
        """
        auc_mtx = self.data.obsm['auc_mtx']
        # change receptor auc matrix column names so it can aligned with the auc matrix column names
        col_names = receptor_auc_mtx.columns.copy()
        col_names = [f'{i}(+)' for i in col_names]
        receptor_auc_mtx.columns = col_names
        # ! only uses modules that occurs in the auc matrix aka they have been identified as regulons
        later_regulon_names = list(set(auc_mtx.columns).intersection(set(col_names)))
        receptor_auc_mtx = receptor_auc_mtx[later_regulon_names]
        # combine two dfs
        df = pd.concat([auc_mtx, receptor_auc_mtx], axis=1)
        # sum values
        isr_df = df.groupby(level=0, axis=1).sum()
        self.data.obsm['isr'] = isr_df
        return isr_df

    # ------------------------------------------------------ #
    #              step2-3: Receptors Detection              #
    # ------------------------------------------------------ #
    def get_filtered_genes(self):
        """
        Detect genes filtered by cisTarget
        :return:
        """
        module_tf = []
        for i in self.modules:
            module_tf.append(i.transcription_factor)

        final_tf = [i.strip('(+)') for i in list(self.regulon_dict.keys())]
        com = set(final_tf).intersection(set(module_tf))

        before_tf = {}
        for tf in com:
            before_tf[tf] = []
            for i in self.modules:
                if tf == i.transcription_factor:
                    before_tf[tf] += list(i.genes)

        filtered = {}
        for tf in com:
            final_targets = self.regulon_dict[f'{tf}(+)']
            before_targets = set(before_tf[tf])
            filtered_targets = before_targets - set(final_targets)
            if tf in filtered_targets:
                filtered_targets.remove(tf)
            filtered[tf] = list(filtered_targets)
        self.filtered = filtered
        self.data.uns['filtered_genes'] = filtered
        return filtered

    def get_filtered_receptors(self, niche_df: pd.DataFrame, receptor_key='to'):
        """

        :type niche_df: pd.DataFrame
        :param receptor_key: column name of receptor
        :param save_tmp:
        :param niche_df:
        :return:
        """
        if niche_df is None:
            warnings.warn("Ligand-Receptor reference database is missing, skipping get_filtered_receptors method")
            return

        receptor_tf = {}
        total_receptor = set()

        self.get_filtered_genes()
        for tf, targets in self.filtered.items():
            rtf = set(intersection_ci(set(niche_df[receptor_key]), set(targets), key=str.lower))
            if len(rtf) > 0:
                receptor_tf[tf] = list(rtf)
                total_receptor = total_receptor | rtf
        self.receptors = total_receptor
        self.receptor_dict = receptor_tf
        self.data.uns['receptor_dict'] = receptor_tf

    # ------------------------------------------------------------------------------------------------
    # def compute_regulons(self, adjacencies, exp_mat, dbs, motif_anno_fn, rho_mask_dropouts, num_workers, save_tmp,
    #                      cache,
    #                      noweights, normalize):
    #     modules = self.get_modules(adjacencies, exp_mat, rho_mask_dropouts=rho_mask_dropouts, prefix=self.project_name)
    #
    #     regulons = self.prune_modules(modules,
    #                                   dbs,
    #                                   motif_anno_fn,
    #                                   num_workers=num_workers,
    #                                   save_tmp=save_tmp,
    #                                   cache=cache,
    #                                   fn=f'{self.project_name}_motifs.csv',
    #                                   prefix=self.project_name,
    #                                   rank_threshold=self.params["rank_threshold"],
    #                                   auc_threshold=self.params["prune_auc_threshold"],
    #                                   nes_threshold=self.params["nes_threshold"],
    #                                   motif_similarity_fdr=self.params["motif_similarity_fdr"])
    #
    #     self.cal_auc(exp_mat,
    #                  regulons,
    #                  auc_threshold=self.params["auc_threshold"],
    #                  num_workers=num_workers,
    #                  save_tmp=save_tmp,
    #                  cache=cache,
    #                  noweights=noweights,
    #                  normalize=normalize,
    #                  fn=f'{self.project_name}_auc.csv')
