#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 23 Oct 2023 16:09
# @Author: Lidong GUO
# @File: spagrn/scc.py
import sys
# third party modules
import pandas as pd
import numpy as np


class ScoexpMatrix:
    """
    Algorithms to calulate Scoexp matrix
    based on CellTrek (10.1038/s41587-022-01233-1)
    see CellTrek from https://github.com/navinlabcode/CellTrek
    """

    @staticmethod
    def rbfk(dis_mat, sigm, zero_diag=True):
        """
        Radial basis function kernel

        :param dis_mat: Distance matrix
        :param sigm: Width of rbfk
        :param zero_diag:
        :return rbf matrix
        """
        rbfk_out = np.exp(-1 * np.square(dis_mat) / (2 * sigm ** 2))
        if zero_diag:
            rbfk_out[np.diag_indices_from(rbfk_out)] = 0
        return rbfk_out

    @staticmethod
    def wcor(X, W, method='pearson', na_zero=True):
        """
        Weighted cross correlation

        :param X: Expression matrix, n X p
        :param W: Weight matrix, n X n
        :param method: Correlation method, pearson or spearman
        :param na_zero: Na to zero
        :return correlation matrix
        """
        from scipy.stats import rankdata
        from sklearn.preprocessing import scale
        if method == 'spearman':
            X = np.apply_along_axis(rankdata, 0, X)  # rank each columns
        X = scale(X, axis=0)  # scale for each columns
        W_cov_temp = np.matmul(np.matmul(X.T, W), X)
        W_diag_mat = np.sqrt(np.matmul(np.diag(W_cov_temp), np.diag(W_cov_temp).T))
        cor_mat = W_cov_temp / W_diag_mat
        if na_zero:
            np.nan_to_num(cor_mat, False)
        return cor_mat

    @staticmethod
    def scc(irn_data,
            gene_list: list = [],
            tf_list: list = [],
            sigm=15,
            zero_cutoff=5,
            cor_method='spearman',
            save_tmp: bool = True,
            fn: str = 'adj.csv',
            ):
        """
        Main logic for infering gene regulatory network by spatially-aware cross-correlation (SCC) model
        :param irn_data: object of spagrn.network.Network
        :param sigm: sigma for RBF kernel, default 15.
        :param gene_list: filter gene by exp cell > zero_cutoff% of all cells if len(gene_list)<2, otherwise use this gene set.
        :param tf_list: tf gene list. Use gene_list if tf_list is empty.
        :param zero_cutoff: filter gene by exp cell > zero_cutoff% if if len(gene_list)<2
        :param cor_method: 'spearman' or 'pearson'
        :return: dataframe of tf-gene-importances
        """
        from scipy.spatial import distance_matrix
        cell_gene_matrix = irn_data.matrix
        if not isinstance(cell_gene_matrix, np.ndarray):
            cell_gene_matrix = cell_gene_matrix.toarray()
        # check gene_list
        if len(gene_list) < 2:
            # logger.info('gene filtering...')
            feature_nz = np.apply_along_axis(lambda x: np.mean(x != 0) * 100, 0, cell_gene_matrix)
            features = irn_data.gene_names[feature_nz > zero_cutoff]
            # logger.info(f'{len(features)} features after filtering...')
        else:
            features = np.intersect1d(np.array(gene_list), irn_data.gene_names)
            if len(features) < 2:
                # logger.error('No enough genes in gene_list detected, exit...')
                sys.exit(12)
        # check tf_list
        if len(tf_list) < 1:
            tf_list = features
        else:
            tf_list = np.intersect1d(np.array(tf_list), features)

        gene_select = np.isin(irn_data.gene_names, features, assume_unique=True)
        celltrek_inp = cell_gene_matrix[:, gene_select]
        dist_mat = distance_matrix(irn_data.position,
                                   irn_data.position)
        kern_mat = ScoexpMatrix.rbfk(dist_mat, sigm=sigm, zero_diag=False)
        # logger.info('Calculating spatial-weighted cross-correlation...')
        wcor_mat = ScoexpMatrix.wcor(X=celltrek_inp, W=kern_mat, method=cor_method)
        # logger.info('Calculating spatial-weighted cross-correlation done.')
        df = pd.DataFrame(data=wcor_mat, index=features, columns=features)
        # extract tf-gene-importances
        df = df[tf_list].copy().T
        df['TF'] = tf_list
        ret = df.melt(id_vars=['TF'])
        ret.columns = ['TF', 'target', 'importance0']
        maxV = ret['importance0'].max()
        ret['importance'] = ret['importance0'] / maxV
        ret['importance'] = ret['importance'] * 1000
        # plt.hist(ret['importance'])
        # plt.savefig('celltrek_importance.png')
        ret.drop(columns=['importance0'], inplace=True)
        ret['valid'] = ret.apply(lambda row: row['TF'] != row['target'], axis=1)
        ret = ret[ret['valid']].copy()
        ret.drop(columns=['valid'], inplace=True)
        # ret.to_csv('adj.csv',header=True,index=False)
        if save_tmp:
            ret.to_csv(fn, index=False)
        return ret
