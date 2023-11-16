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
import pandas as pd
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Sequence, Type, Optional

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from ctxcore.genesig import Regulon, GeneSignature
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell, derive_auc_threshold
from pyscenic.prune import prune2df, df2regulons

# modules in self project
from .scoexp import ScoexpMatrix
from .network import Network


def before_cistarget(tfs: list, modules: Sequence[Regulon], prefix: str):
    """
    Detect genes that were generated in the get_modules step
    :param tfs:
    :param modules:
    :param prefix:
    :return:
    """
    d = {}
    for tf in tfs:
        d[tf] = {}
        tf_mods = [x for x in modules if x.transcription_factor == tf]
        for i, mod in enumerate(tf_mods):
            d[tf][f'module {str(i)}'] = list(mod.genes)
        with open(f'{prefix}_before_cistarget.json', 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def get_module_targets(modules):
    """
    同上 (before_cistarget)
    :param modules:
    :return:
    """
    d = {}
    for module in modules:
        tf = module.transcription_factor
        tf_mods = [x for x in modules if x.transcription_factor == tf]
        targets = []
        for i, mod in enumerate(tf_mods):
            targets += list(mod.genes)
        d[tf] = list(set(targets))
    return d


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


class InferNetwork(Network):
    """
    Algorithms to infer Gene Regulatory Networks (GRNs)
    """

    def __init__(self, adata=None, pos_label='spatial'):
        """
        Constructor of this Object.
        :param data:
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata. 
        :return:
        """
        super().__init__()
        self.data = adata
        self.load_data_info(pos_label)

        # other settings
        self._params = {
            'spg': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': False,
            },
            'boost': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': False,
            },
            'scc': {
                'rank_threshold': 1500,
                'prune_auc_threshold': 0.07,
                'nes_threshold': 3.0,
                'motif_similarity_fdr': 0.05,
                'auc_threshold': 0.5,
                'noweights': True,
            }}

    # GRN pipeline infer logic
    def infer(self,
              databases: str,
              motif_anno_fn: str,
              tfs_fn,
              cluster_label='annotation',  # TODO: shouldn't set default value
              niche_df=None,
              receptor_key='to',
              target_genes=None,
              num_workers=None,
              save_tmp=True,
              cache=True,
              prefix: str = 'project',

              method='spg',
              sigm=15,
              c_threshold=0.8,
              layers='raw_counts',
              model='bernoulli',
              latent_obsm_key='spatial',
              umi_counts_obs_key=None,
              n_neighbors=30,
              weighted_graph=False,
              rho_mask_dropouts=False,
              noweights=None,
              normalize: bool = False):
        """

        :param receptor_key:
        :param niche_df:
        :param c_threshold:
        :param n_neighbors:
        :param weighted_graph:
        :param rho_mask_dropouts:
        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :param target_genes:
        :param num_workers:
        :param save_tmp:
        :param cache:
        :param prefix:
        :param method: method from [boost/spg/scc]
        :param sigm: sigma for scc, default 15 (assumption for 15um)
        :param layers:
        :param model:
        :param latent_obsm_key:
        :param umi_counts_obs_key:
        :param cluster_label:
        :param noweights:
        :param normalize:
        :return:
        """
        assert method in ['boost', 'spg', 'scc'], "method options are boost/spg/scc"
        self.data.uns['method'] = method
        global adjacencies
        matrix = self._matrix
        df = self._data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self._gene_names

        if noweights is None:
            noweights = self.params[method]["noweights"]

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
            # tfs = self._gene_names
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)

        # 3. GRN Inference
        if method == 'boost':
            adjacencies = self.rf_infer(matrix,
                                        genes=target_genes,
                                        tf_names=tfs,
                                        num_workers=num_workers,
                                        cache=cache,
                                        save_tmp=save_tmp,
                                        fn=f'{prefix}_adj.csv')
        elif method == 'scc':
            adjacencies = ScoexpMatrix.scc(self,
                                           target_genes,
                                           tfs,
                                           sigm=sigm,
                                           save_tmp=save_tmp,
                                           fn=f'{prefix}_adj.csv')
        elif method == 'spg':
            adjacencies = self.spg(self.data,
                                   c_threshold=c_threshold,
                                   tf_list=tfs,
                                   jobs=num_workers,
                                   layer_key=layers,
                                   model=model,
                                   latent_obsm_key=latent_obsm_key,
                                   umi_counts_obs_key=umi_counts_obs_key,
                                   n_neighbors=n_neighbors,
                                   weighted_graph=weighted_graph,
                                   cache=cache,
                                   fn=f'{prefix}_adj.csv')

        # 4. Compute Modules
        # ctxcore.genesig.Regulon
        modules = self.get_modules(adjacencies, df, rho_mask_dropouts=rho_mask_dropouts, prefix=prefix)
        before_cistarget(tfs, modules, prefix)

        # 5. Regulons Prediction aka cisTarget
        # ctxcore.genesig.Regulon
        regulons = self.prune_modules(modules,
                                      dbs,
                                      motif_anno_fn,
                                      num_workers=num_workers,
                                      save_tmp=save_tmp,
                                      cache=cache,
                                      fn=f'{prefix}_motifs.csv',
                                      prefix=prefix,
                                      rank_threshold=self.params[method]["rank_threshold"],
                                      auc_threshold=self.params[method]["prune_auc_threshold"],
                                      nes_threshold=self.params[method]["nes_threshold"],
                                      motif_similarity_fdr=self.params[method]["motif_similarity_fdr"])

        # 6.0. Cellular Enrichment (aka AUCell)
        self.cal_auc(df,
                     regulons,
                     auc_threshold=self.params[method]["auc_threshold"],
                     num_workers=num_workers,
                     save_tmp=save_tmp, cache=cache,
                     noweights=noweights,
                     normalize=normalize,
                     fn=f'{prefix}_auc.csv')

        # 6.1. Receptor AUCs
        if niche_df is not None:
            self.get_receptors(niche_df, receptor_key=receptor_key, save_tmp=save_tmp,
                               fn=f'{prefix}_filtered_targets_receptor.json')
            self.receptor_auc()

        # 7. Calculate Regulon Specificity Scores
        self.cal_regulon_score(cluster_label=cluster_label, save_tmp=save_tmp,
                               fn=f'{prefix}_regulon_specificity_scores.txt')

        # 8. Save results to h5ad file
        # TODO: check if data has adj, regulon_dict, auc_mtx etc. before saving to disk
        # dtype=object
        self.data.write_h5ad(f'{prefix}_spagrn.h5ad')

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        """only use this function when setting params as a whole.
        use add_params to solely update/add some of the params and keep the rest unchanged"""
        self._params = value

    def add_params(self, method: str, dic: dict):
        """
        :param method:
        :param dic:

        Example:
            grn = InferNetwork(data)
            grn.add_params('spg', {'num_worker':12, 'auc_threshold': 0.001})
        """
        og_params = deepcopy(self._params)
        try:
            for key, value in dic.items():
                self._params[method][key] = value
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
        :param tf_names: list of target TFs or all
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
            adjacencies.to_csv(fn, index=False)
        self.adjacencies = adjacencies
        self.data.uns['adj'] = adjacencies
        return adjacencies

    def spg(self,
            data: anndata.AnnData,
            c_threshold: float,
            layer_key=None,
            model='bernoulli',
            latent_obsm_key="spatial",
            umi_counts_obs_key=None,
            weighted_graph=False,
            n_neighbors=30,
            fdr_threshold=0.05,
            tf_list=None,
            save_tmp=True,
            jobs=None,
            cache=False,
            fn: str = 'adj.csv',
            **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules by spatial-proximity-graph (SPG) model.
        :param data: Count matrix (shape is cells by genes)
        :param c_threshold:
        :param layer_key: Key in adata.layers with count data, uses adata.X if None.
        :param model: Specifies the null model to use for gene expression.
            Valid choices are:
                * 'danb': Depth-Adjusted Negative Binomial
                * 'bernoulli': Models probability of detection
                * 'normal': Depth-Adjusted Normal
                * 'none': Assumes data has been pre-standardized
        :param latent_obsm_key: Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        :param distances_obsp_key: Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        :param umi_counts_obs_key: Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used. 'total_counts'
        :param weighted_graph: Whether or not to create a weighted graph
        :param n_neighbors: Neighborhood size
        :param neighborhood_factor: Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        :param approx_neighbors: Use approximate nearest neighbors or exact scikit-learn neighbors. Only
            when hotspot initialized with `latent`.
        :param fdr_threshold: Correlation threshold at which to stop assigning genes to modules
        :param tf_list: predefined TF names
        :param save_tmp: if save results onto disk
        :param jobs: Number of parallel jobs to run_all
        :param cache:
        :param fn: output file name
        :return: A dataframe, local correlation Z-scores between genes (shape is genes x genes)
        """
        if cache and os.path.isfile(fn):
            local_correlations = pd.read_csv(fn)
            self.data.uns['adj'] = local_correlations
            return local_correlations
        else:
            hs = hotspot.Hotspot(data,
                                 layer_key=layer_key,
                                 model=model,
                                 latent_obsm_key=latent_obsm_key,
                                 umi_counts_obs_key=umi_counts_obs_key,
                                 **kwargs)
            hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)
            hs_results = hs.compute_autocorrelations()
            # Select genes
            hs_genes = hs_results.loc[(hs_results.FDR < fdr_threshold) & (hs_results.C > c_threshold)].index
            local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)  # jobs for parallelization

        # subset by TFs
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

        self.data.uns['adj'] = local_correlations
        if save_tmp:
            local_correlations.to_csv(fn, index=False)
        return local_correlations

    # ------------------------------------------------------#
    #            step2:  FILTER TFS AND TARGETS             #
    # ------------------------------------------------------#
    def get_modules(self,
                    adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    prefix: str = 'exp',
                    **kwargs) -> Sequence[Regulon]:
        """
        Create of co-expression modules
        :param adjacencies:
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param rho_mask_dropouts:
        :param prefix:
        :return:
        """
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        self.modules = modules
        # self.data.uns['modules'] = modules
        with open(f'{prefix}_modules.pkl', "wb") as f:
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
                      cache: bool = True,
                      save_tmp: bool = True,
                      fn: str = 'motifs.csv',
                      prefix: str = 'exp',
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
        :param prefix:
        :param kwargs:
        :return: A dataframe.
        """
        if cache and os.path.isfile(fn):
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

        # this function actually did two things. 1. get df, 2. turn df into list of Regulons
        # #2.
        regulon_list = df2regulons(df)
        self.regulons = regulon_list

        # #3. handle results
        # convert Regulon list to dictionaries for easy access and readability
        self.regulon_dict = self.get_regulon_dict(regulon_list)
        self.data.uns['regulon_dict'] = self.regulon_dict

        # save to data
        with open(f'{prefix}_regulons.pkl', "wb") as f:
            pickle.dump(regulon_list, f)
        # self.data.uns['regulons'] = self.regulon_list  TODO: is saving to a pickle file the only way?
        if save_tmp:
            df.to_csv(fn)
            self.regulons_to_json(fn=f'{prefix}_regulons.json')
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
            auc_mtx.to_csv(fn)
        return auc_mtx

    def receptor_auc(self, auc_threshold=None, p_range=0.01, num_workers=20) -> Optional[pd.DataFrame]:
        """

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
        self.data.obsm['rep_auc_mtx'] = receptor_auc_mtx
        return receptor_auc_mtx

    # ------------------------------------------------------ #
    #              step2-3: Receptors Detection              #
    # ------------------------------------------------------ #
    # def get_filtered_genes(self):
    #     """
    #     Detect genes filtered by cisTarget
    #     :return:
    #     """
    #     # if self.regulon_dict is None:
    #     #     self.regulon_dict = self.get_regulon_dict(self.regulons)
    #     module_tf = []
    #     for i in self.modules:
    #         module_tf.append(i.transcription_factor)
    #
    #     final_tf = [i.strip('(+)') for i in list(self.regulon_dict.keys())]
    #     com = set(final_tf).intersection(set(module_tf))
    #
    #     before_tf = {}
    #     for tf in com:
    #         before_tf[tf] = []
    #         for i in self.modules:
    #             if tf == i.transcription_factor:
    #                 before_tf[tf] += list(i.genes)
    #
    #     filtered = {}
    #     for tf in com:
    #         final_targets = self.regulon_dict[f'{tf}(+)']
    #         before_targets = set(before_tf[tf])
    #         filtered_targets = before_targets - set(final_targets)
    #         if tf in filtered_targets:
    #             filtered_targets.remove(tf)
    #         filtered[tf] = list(filtered_targets)
    #         filtered[tf] = list(filtered_targets)
    #     self.filtered = filtered
    #     self.data.uns['filtered_genes'] = filtered
    #     return filtered

    # def get_filtered_receptors(self, niche_df: pd.DataFrame, receptor_key='to', save_tmp=False,
    #                            fn='filtered_targets_receptor.json'):
    #     """
    #
    #     :type niche_df: pd.DataFrame
    #     :param receptor_key: column name of receptor
    #     :param save_tmp:
    #     :param niche_df:
    #     :param fn:
    #     :return:
    #     """
    #     if niche_df is None:
    #         warnings.warn("Ligand-Receptor reference database is missing, skipping get_filtered_receptors method")
    #         return
    #
    #     receptor_tf = {}
    #     total_receptor = set()
    #
    #     self.get_filtered_genes()
    #     for tf, targets in self.filtered.items():
    #         rtf = set(intersection_ci(set(niche_df[receptor_key]), set(targets), key=str.lower))
    #         if len(rtf) > 0:
    #             receptor_tf[tf] = list(rtf)
    #             total_receptor = total_receptor | rtf
    #     self.receptors = total_receptor
    #     self.receptor_dict = receptor_tf
    #     self.data.uns['receptors'] = list(total_receptor)  # warning: anndata cannot save class set to disk
    #     self.data.uns['receptor_dict'] = receptor_tf
    #
    #     if save_tmp:
    #         with open(fn, 'w') as fp:
    #             json.dump(receptor_tf, fp, sort_keys=True, indent=4)

    def get_receptors(self, niche_df: pd.DataFrame, receptor_key='to', save_tmp=False, fn='coexpressed_receptor.json'):
        """

        :param niche_df:
        :param receptor_key:
        :param save_tmp:
        :param fn:
        :return:
        """
        if niche_df is None:
            warnings.warn("Ligand-Receptor reference database is missing, skipping get_filtered_receptors method")
            return

        receptor_tf = {}
        total_receptor = set()

        module_targets = get_module_targets(self.modules)
        for tf, targets in module_targets.items():
            rtf = set(intersection_ci(set(niche_df[receptor_key]), set(targets), key=str.lower))
            if len(rtf) > 0:
                receptor_tf[tf] = list(rtf)
                receptor_tf[tf] = list(rtf)
                total_receptor = total_receptor | rtf
        self.receptors = total_receptor
        self.receptor_dict = receptor_tf
        self.data.uns['receptors_all'] = list(total_receptor)  # warning: anndata cannot save class set to disk
        self.data.uns['receptor_dict_all'] = receptor_tf

        if save_tmp:
            with open(fn, 'w') as fp:
                json.dump(receptor_tf, fp, sort_keys=True, indent=4)
