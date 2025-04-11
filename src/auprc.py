#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Oct 2023 10:34
# @Author: Yao LI
# @File: spagrn/prc.py


import os
from typing import Union
import glob
import json
import anndata
import scanpy as sc
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
'''
python ../../../../spagrn_debug/au.py
'''


class AUPRC:
    def __init__(self, tfs=None, data=None, adj=None, name_df=None):
        self._adata = data  # only necessary when using spearman cc
        self._tfs = tfs

        self._ground_truth = None  # all genes
        self._prediction = None  # all genes
        self._baseline = None
        self._auprc_ratio = None
        self._auprc = None
        self._auroc = None

        self._adj = adj
        self._regulons = None

        # column names in Ground Truth file
        self._value_col = 'regulator.effect'
        self._tf_col = 'regulator.gene'
        self._target_col = 'regulated.gene'

        self._name_df = name_df  # ID-gene name chart
        self._true_df = None  # ground truth exclude noises
        self._pred_df = None  # prediction exclude genes that has been filtered

        self.prec = None
        self.recall = None
        self.thresholds = None

    @property
    def adata(self):
        return self._adata

    @adata.setter
    def adata(self, value):
        self._adata = value

    @property
    def tfs(self):
        return self._tfs

    @property
    def name_df(self):
        return self._name_df

    @name_df.setter
    def name_df(self, value):
        self._name_df = value

    @property
    def ground_truth(self):
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, value):
        self._ground_truth = value

    @property
    def true_df(self):
        return self._true_df

    @true_df.setter
    def true_df(self, value):
        self._true_df = value

    @property
    def pred_df(self):
        return self._pred_df

    @pred_df.setter
    def pred_df(self, value):
        self._pred_df = value

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value

    @property
    def auprc_ratio(self):
        return self._auprc_ratio

    @auprc_ratio.setter
    def auprc_ratio(self, value):
        self._auprc_ratio = value

    @property
    def auroc(self):
        return self._auroc

    @auroc.setter
    def auroc(self, value):
        self._auroc = value

    @property
    def auprc(self):
        return self._auprc

    @auprc.setter
    def auprc(self, value):
        self._auprc = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def value_col(self):
        return self._value_col

    @property
    def tf_col(self):
        return self._tf_col

    @property
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, value):
        self._target_col = value

    @property
    def adj(self):
        return self._adj

    @adj.setter
    def adj(self, value):
        self._adj = value

    @property
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

    def get_true_df(self, ground_truth_files):
        """

        :param ground_truth_files:
        :return:
        """
        fl = glob.glob(ground_truth_files)
        self.true_df = pd.concat([pd.read_csv(i) for i in fl]).astype(str)
        return self.true_df

    # def make_ground_truth(self, ground_truth_files, real_tfs=None, false_tfs=None):
    #     """
    #
    #     :param ground_truth_files:
    #     :param real_tfs:
    #     :param false_tfs:
    #     :return:
    #     """
    #     # names = pd.read_csv(naming_fn)
    #     fl = glob.glob(ground_truth_files)
    #     df_true = pd.concat([pd.read_csv(i) for i in fl]).astype(str)
    #
    #     # adata = sc.read_h5ad(adata_fn)
    #     # if self.adata is None:
    #     #     self.adata = adata
    #     all_genes = self.adata.var_names
    #     ground_truth = pd.DataFrame(product(self.tfs, all_genes), columns=['regulator.gene', 'regulated.gene']).astype(
    #         str)
    #     # ! make sure gene names are using the same nomenclature
    #     ground_truth['regulated.gene'] = ground_truth['regulated.gene'].replace(list(self.name_df['name']),
    #                                                                             list(self.name_df['id']))
    #     ground_truth['regulator.effect'] = [0] * ground_truth.shape[0]
    #     ground_truth = pd.concat([ground_truth, df_true])
    #     ground_truth = ground_truth.drop_duplicates(['regulator.gene', 'regulated.gene'], keep='last')
    #
    #     # if false TF exists
    #     if real_tfs and false_tfs:
    #         t_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(real_tfs)]
    #         f_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(false_tfs)]
    #         f_ground_truth['regulator.effect'] = [0.0] * f_ground_truth.shape[0]
    #         ground_truth = pd.concat([t_ground_truth, f_ground_truth])
    #
    #     ground_truth[['regulator.gene', 'regulated.gene']] = ground_truth[['regulator.gene', 'regulated.gene']].replace(
    #         list(self.name_df['id']), list(self.name_df['name']))
    #     ground_truth['regulator.effect'] = ground_truth['regulator.effect'].astype('float64')
    #     # convert y_true into a binary matrix
    #     ground_truth.loc[ground_truth['regulator.effect'] > 0, 'regulator.effect'] = 1
    #     # order of genes need to be consistent between ground_truth and prediction
    #     ground_truth = ground_truth.sort_values(['regulator.gene', 'regulated.gene'], ascending=[True, True])
    #     self.ground_truth = ground_truth
    #     return ground_truth

    def get_baseline(self):
        self.baseline = 1 - self.ground_truth[self.ground_truth[self.value_col] == 0].shape[0] / \
                        self.ground_truth.shape[0]
        print(f'Baseline is {self.baseline} (num of 1/num of total)')
        return self.baseline

    def get_pred_df(self, y_true_label=None, y_true_tf_col=None, y_true_target_col=None):
        """

        :param y_true_label:
        :return:
        """
        if self.adj is None:
            self.adj = self.adata.uns['adj']
        if y_true_label is None:
            y_true_label = self.value_col
        if y_true_tf_col is None:
            y_true_tf_col = self.tf_col
        if y_true_target_col is None:
            y_true_target_col = self.target_col

        # 2. input prediction value
        self.regulons = self.adata.uns['regulon_dict']
        mylist = [(key, x) for key, val in self.regulons.items() for x in val]
        df_pred = pd.DataFrame(mylist, columns=['Name', 'Values'])
        # TODO: if has (+)
        df_pred['Name'] = df_pred['Name'].str.strip('(+)')
        df_pred['prediction'] = [1] * df_pred.shape[0]

        # 1. get importance (pred_label) values
        df_pred = self.adj.merge(df_pred, left_on=['TF', 'target'], right_on=['Name', 'Values'], how='left')
        df_pred['prediction'] = df_pred['prediction'].fillna(0)

        # 3. introduce ground truth classification label
        df_pred = df_pred.merge(self.ground_truth, left_on=['TF', 'target'],
                                right_on=[y_true_tf_col, y_true_target_col],
                                how='left')

        df_pred = df_pred[['TF', 'target', 'importance', 'prediction', y_true_label]]
        df_pred.columns = [y_true_tf_col, y_true_target_col, 'importance', 'prediction', 'ground truth']
        # df_pred.to_csv('df_pred.csv', index=False)
        self.pred_df = df_pred
        return self.pred_df

    def get_pred_df_grnboost(self, y_true_label=None, y_true_tf_col=None, y_true_target_col=None):
        """

        :param y_true_label:
        :return:
        """
        if self.adj is None:
            self.adj = self.adata.uns['adj']
        if y_true_label is None:
            y_true_label = self.value_col
        if y_true_tf_col is None:
            y_true_tf_col = self.tf_col
        if y_true_target_col is None:
            y_true_target_col = self.target_col

        # 1. input prediction value
        df_pred = self.adj.copy()
        df_pred['prediction'] = [1] * df_pred.shape[0]

        # 3. introduce ground truth classification label
        df_pred = df_pred.merge(self.ground_truth, left_on=['TF', 'target'],
                                right_on=[y_true_tf_col, y_true_target_col],
                                how='left')

        df_pred = df_pred[['TF', 'target', 'importance', 'prediction', y_true_label]]
        df_pred.columns = [y_true_tf_col, y_true_target_col, 'importance', 'prediction', 'ground truth']
        # df_pred.to_csv('df_pred.csv', index=False)
        self.pred_df = df_pred
        return self.pred_df

    # alternative to get_pred_df
    # calculate spearman values
    def get_pred_df_spearman(self, data: Union[pd.DataFrame, anndata.AnnData] = None, y_true_label=None,
                             y_true_tf_col=None, y_true_target_col=None):
        """

        :param data:
        :param y_true_label:
        :param y_true_tf_col:
        :param y_true_target_col:
        :return:
        """
        if y_true_label is None:
            y_true_label = self.value_col
        if y_true_tf_col is None:
            y_true_tf_col = self.tf_col
        if y_true_target_col is None:
            y_true_target_col = self.target_col
        # 1. calculate spearman cc
        # adata = sc.read_h5ad(self.adata_fn)
        # if self.adata is None:
        #     self.adata = adata
        # df = adata.to_df()
        if data is None:
            raise ValueError('Must provide expression matrix when pred label is set to spearman')
        if isinstance(data, anndata.AnnData):
            data = data.to_df()
        if self.adj is None:
            self.adj = self.adata.uns['adj']
        s = []
        for i in self.adj.index:
            res = stats.spearmanr(data[self.adj.loc[i].TF], data[self.adj.loc[i].target])
            s.append(res.correlation)
        self.adj['spearman'] = s
        adj = self.adj.sort_values(['importance', 'spearman'], ascending=False)

        # 2. input prediction value
        regs = self.adata.uns['regulon_dict']
        mylist = [(key, x) for key, val in regs.items() for x in val]
        df_pred = pd.DataFrame(mylist, columns=['Name', 'Values'])
        df_pred['Name'] = df_pred['Name'].str.strip('(+)')
        df_pred['prediction'] = [1] * df_pred.shape[0]

        # 1. merge spearman df and prediction df
        df_pred = adj.merge(df_pred, left_on=['TF', 'target'], right_on=['Name', 'Values'], how='left')
        df_pred['prediction'].fillna(0)
        df_pred['prediction'] = df_pred['prediction'].fillna(0)

        # 3. introduce ground truth classification label
        df_pred = df_pred.merge(self.ground_truth, left_on=['TF', 'target'],
                                right_on=[y_true_tf_col, y_true_target_col],
                                how='left')
        df_pred = df_pred[['TF', 'target', 'importance', 'spearman', 'prediction', y_true_label]]
        # sort by spearman value
        tt1 = df_pred[df_pred.prediction > 0]
        tt0 = df_pred[df_pred.prediction == 0]
        tt1 = tt1.sort_values(['spearman'], ascending=False)
        tt0 = tt0.sort_values(['spearman'], ascending=False)
        # make sure 0 labels (negatives) spearman value is smaller than 1 labels
        tt0['spearman'] = tt0['spearman'] - 1
        df_prediction = pd.concat([tt1, tt0])
        df_prediction.columns = [y_true_tf_col, y_true_target_col, 'importance', 'spearman', 'prediction',
                                 'ground truth']
        # df_prediction.to_csv('df_pred.csv', index=False)
        self.pred_df = df_prediction
        # self.adata.uns['prediction'] = df_prediction
        return df_prediction

    def get_prediction_df(self, pred_label='spearman', y_true_tf_col=None, y_true_target_col=None):
        """
        get prediction for all genes (including genes had been filtered out by SpaGRN),
        so Ground Truth and Prediction have the same dimension (aka len(all_genes))
        value_col: column of value to pass in AUPRC calculation e.g. importance, spearman coefficient ...
        :param pred_label:
        :param y_true_tf_col:
        :param y_true_target_col:
        :return:
        """
        if y_true_tf_col is None:
            y_true_tf_col = self.tf_col
        if y_true_target_col is None:
            y_true_target_col = self.target_col

        pred_index = pd.merge(
            self.pred_df[[y_true_tf_col, y_true_target_col, pred_label, 'prediction', 'ground truth']],
            self.ground_truth[[y_true_tf_col, y_true_target_col]], on=[y_true_tf_col, y_true_target_col],
            how='outer')
        assert pred_index.shape[0] == self.ground_truth.shape[0]
        pred = pred_index.sort_values([y_true_tf_col, y_true_target_col], ascending=[True, True])
        if pred_label == 'spearman':
            pred = pred.fillna(int(pred[pred_label].min()) - 2)
        else:
            pred = pred.fillna(0)

        self.prediction = pred
        return self.prediction

    def get_auprc(self, y_true_label='regulator.effect', pred_label='spearman'):
        """

        :param y_true_label:
        :param pred_label:
        :return:
        """
        self.prec, self.recall, self.thresholds = precision_recall_curve(y_true=self.ground_truth[y_true_label],
                                                                         probas_pred=self.prediction[pred_label],
                                                                         pos_label=1)
        new_auc = auc(self.recall, self.prec)
        self.auprc = new_auc

    def get_ratio(self):
        """
        AUPRC ratio
        :return:
        """
        if self.baseline:  # walrus operator cannot be used on instance attributes?
            self.auprc_ratio = self.auprc / self.baseline
        else:
            self.get_baseline()
            self.auprc_ratio = self.auprc / self.baseline
        if self.adata is not None and isinstance(self.adata, anndata.AnnData):
            self.adata.uns['auprc_ratio'] = self.auprc_ratio
        print(f'AUPRC ratio is {self.auprc_ratio}.')

    def plot_prec_recall(self, fn='Precision-Recall.png'):
        if self.recall is None or self.prec is None:
            raise ValueError('Calculate prc first plotting. See method get_auprc')
        plt.fill_between(self.recall, self.prec)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Train Precision-Recall curve")
        plt.savefig(fn)
        plt.close()

    def get_auroc(self, y_true_label='regulator.effect', pred_label='spearman'):
        """

        :param y_true_label:
        :param pred_label:
        :return:
        """
        fpr, tpr, thresholds2 = roc_curve(y_true=self.ground_truth[y_true_label],
                                          y_score=self.prediction[pred_label],
                                          pos_label=1)
        auroc = auc(fpr, tpr)
        self.auroc = auroc
        return auroc
        # plt.fill_between(fpr, tpr)
        # plt.ylabel("true positive")
        # plt.xlabel("false positive")
        # plt.title("AUROC")
        # plt.savefig('aucroc.png')
        # plt.close()

    def prc(self,
            pred_label, ground_truth: pd.DataFrame,
            y_true_label='regulator.effect',
            fn='adata.h5ad',
            fig_fn='Precision-Recall.png'):
        """
        Main logic method. SpaGRN.AUPRC pipeline
        1. generate ground truth
        2.1. load in prediction output by SpaGRN
        2.2. fill in the blank
        3. Calculate AUPRC and plot result
        :param pred_label:
        :param ground_truth_files:
        :param y_true_label:
        :param fn:
        :return:
        """
        # 1.
        # self.make_ground_truth(ground_truth_files, real_tfs=['2', '232', '408', '805', '1006'],
        #                        false_tfs=['1140', '1141', '1142', '1143', '1144'])
        # self.get_baseline()
        self.ground_truth = ground_truth

        # 2.
        if pred_label == 'spearman':
            self.get_pred_df_spearman(data=self.adata)
        else:
            # self.adj = adj_df
            self.get_pred_df_grnboost()
            # self.get_pred_df()
        self.get_prediction_df(pred_label=pred_label)

        # 3.
        self.get_auprc(pred_label=pred_label, y_true_label=y_true_label)
        self.get_ratio()
        # self.plot_prec_recall(fn=fig_fn)

        # 4. save results
        # self.adata.write_h5ad(fn)

    def roc(self,
            pred_label,
            ground_truth: pd.DataFrame,
            adj_df=None,
            y_true_label='regulator.effect',
            fn='adata.h5ad',
            fig_fn='Precision-Recall.png'):
        """
        Main logic method. SpaGRN.AUPRC pipeline
        1. generate ground truth
        2.1. load in prediction output by SpaGRN
        2.2. fill in the blank
        3. Calculate AUPRC and plot result
        :param pred_label:
        :param ground_truth_files:
        :param y_true_label:
        :param fn:
        :return:
        """
        # 1.
        # self.make_ground_truth(ground_truth_files, real_tfs=['2', '232', '408', '805', '1006'],
        #                        false_tfs=['1140', '1141', '1142', '1143', '1144'])
        # self.get_baseline()
        self.ground_truth = ground_truth

        # 2.
        if pred_label == 'spearman':
            self.get_pred_df_spearman(data=self.adata)
        else:
            # self.adj = adj_df
            self.get_pred_df_grnboost()
            # self.get_pred_df()
        self.get_prediction_df(pred_label=pred_label)

        # 4.
        self.get_auroc(pred_label=pred_label, y_true_label=y_true_label)

        # 5. save results
        # self.adata.write_h5ad(fn)


def cal_auprc(tfs, name_df, ground_truth, pred_label='spearman', adata=None, adj_fn=None, fn='adata.h5ad'):
    """

    :param adata:
    :param tfs:
    :param name_df:
    :param adj_fn:
    :param reg_fn:
    :param pred_label:
    :param ground_truth_files:
    :return:
    """
    a = AUPRC(data=adata,
              adj=adj_fn,
              tfs=tfs,
              name_df=name_df)
    a.prc(pred_label=pred_label, ground_truth=ground_truth, fn=fn)
    return a.auprc_ratio


def cal_auroc(tfs, name_df, ground_truth, pred_label='spearman', adata=None, adj_fn=None, fn='adata.h5ad'):
    """

    :param adata:
    :param tfs:
    :param name_df:
    :param adj_fn:
    :param reg_fn:
    :param pred_label:
    :param ground_truth_files:
    :return:
    """
    a = AUPRC(data=adata,
              adj=adj_fn,
              tfs=tfs,
              name_df=name_df)
    a.roc(pred_label=pred_label, ground_truth=ground_truth, fn=fn)
    return a.auroc


def make_ground_truth(ground_truth_files, name_df: pd.DataFrame, all_genes: list, real_tfs=None, false_tfs=None):
    """

    :param ground_truth_files:
    :param name_df:
    :param all_genes:
    :param real_tfs:
    :param false_tfs:
    :return:
    """
    # names = pd.read_csv(naming_fn)
    fl = glob.glob(ground_truth_files)
    df_true = pd.concat([pd.read_csv(i) for i in fl]).astype(str)

    # adata = sc.read_h5ad(adata_fn)
    # if self.adata is None:
    #     self.adata = adata
    # all_genes = adata.var_names
    ground_truth = pd.DataFrame(product(tfs, all_genes), columns=['regulator.gene', 'regulated.gene']).astype(
        str)
    # ! make sure gene names are using the same nomenclature
    ground_truth['regulated.gene'] = ground_truth['regulated.gene'].replace(list(name_df['name']),
                                                                            list(name_df['id']))
    ground_truth['regulator.effect'] = [0] * ground_truth.shape[0]
    ground_truth = pd.concat([ground_truth, df_true])
    ground_truth = ground_truth.drop_duplicates(['regulator.gene', 'regulated.gene'], keep='last')

    # if false TF exists
    if real_tfs and false_tfs:
        t_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(real_tfs)]
        f_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(false_tfs)]
        f_ground_truth['regulator.effect'] = [0.0] * f_ground_truth.shape[0]
        ground_truth = pd.concat([t_ground_truth, f_ground_truth])

    ground_truth[['regulator.gene', 'regulated.gene']] = ground_truth[['regulator.gene', 'regulated.gene']].replace(
        list(name_df['id']), list(name_df['name']))
    ground_truth['regulator.effect'] = ground_truth['regulator.effect'].astype('float64')
    # convert y_true into a binary matrix
    ground_truth.loc[ground_truth['regulator.effect'] > 0, 'regulator.effect'] = 1
    # order of genes need to be consistent between ground_truth and prediction
    ground_truth = ground_truth.sort_values(['regulator.gene', 'regulated.gene'], ascending=[True, True])
    return ground_truth


def calculate_multi_samples(methods, data_fn_base, tfs, names, ver7_ground_truth):
    data_nums = list(range(2, 12))
    ratios = {}
    aurocs = {}
    for method in methods:
        ratios[method] = []
        aurocs[method] = []
        for num in data_nums:
            data_folder = os.path.join(data_fn_base, f'data{num}')
            if method == 'hotspot':
                data_fn = os.path.join(data_folder, f'{method}/{method}_spagrn.h5ad')
                adata = sc.read_h5ad(data_fn)
                pred_label = 'spearman'
                ratio = cal_auprc(tfs, names, adata=adata, pred_label=pred_label, ground_truth=ver7_ground_truth)
            elif method == 'HOTSPOT':
                data_fn = os.path.join(data_folder, f'{method}/hotspot.h5ad')
                adata = sc.read_h5ad(data_fn)
                pred_label = 'importance'
                ratio = cal_auprc(tfs, names, adata=adata, pred_label=pred_label, ground_truth=ver7_ground_truth)
            elif method == 'genie3':
                data_fn = os.path.join(data_folder, f'{method}/genie3.adj.csv')
                df = pd.read_csv(data_fn)
                ratio = cal_auprc(tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth, adj_fn=df)
                pred_label = 'importance'
            elif method == 'grnboost':
                data_fn = os.path.join(data_folder, f'{method}/grnboost_adj.csv')
                df = pd.read_csv(data_fn)
                ratio = cal_auprc(tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth, adj_fn=df)
                pred_label = 'importance'

            # adata = sc.read_h5ad(data_fn)
            ratios[method].append(ratio)
            auroc = cal_auroc(adata, tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth)
            aurocs[method].append(auroc)

    with open('ratios.json_sub', 'w') as f:
        json.dump(ratios, f, sort_keys=True, indent=4)
    with open('aurocs.json_sub', 'w') as f:
        json.dump(aurocs, f, sort_keys=True, indent=4)
    return ratios, aurocs


def ratios_boxplot(ratios, methods, x_labels=None, fn='auprc_ratio_boxplot.pdf'):
    if x_labels is None:
        x_labels = ['SpaGRN', 'GRNBoost2', 'GENIE3', 'HOTSPOT']
    df = pd.DataFrame.from_records(ratios).astype('float64')
    df = df[methods]
    ax = sns.boxplot(data=df)
    plt.title('AUPRC ratios')
    plt.ylabel('ratio')
    ax.set_xticklabels(x_labels)
    plt.tight_layout()
    plt.savefig(fn, format='pdf')
    plt.close()


def auroc_boxplot(aurocs, methods, x_labels=None, fn='auroc_ratio_boxplot.pdf'):
    if x_labels is None:
        x_labels = ['SpaGRN', 'GRNBoost2', 'GENIE3', 'HOTSPOT']
    df = pd.DataFrame.from_records(aurocs).astype('float64')
    df = df[methods]
    ax = sns.boxplot(data=df)
    plt.title('AUROC')
    plt.ylabel('auroc')
    ax.set_xticklabels(x_labels)
    plt.tight_layout()
    plt.savefig(fn, format='pdf')
    plt.close()


if __name__ == '__main__':
    # names = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/name_df.csv')
    '''
    print(names)
    id,name
    2,Adf1
    232,Aef1
    408,grh
    '''
    # tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
    # ver7_ground_truth = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/GRN_params_*.csv'
    # data_fn_base = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8'

    # for one sample
    # 2023-10-24: test h5ad adj & regulons: success
    # adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/data11/HOTSPOT.h5ad')
    # adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/data11/hetero_11.h5ad')
    # test run for GENIE3
    # adj_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/data11/genie3.adj.csv'
    # adj_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/grnboost_adj.csv'
    # reg_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/data11/grnboost/grnboost_regulons.json'
    # ratio = cal_auprc(adata, tfs, names, adj_fn=adj_fn, reg_fn=reg_fn, pred_label='importance', ground_truth_files=ver7_ground_truth)
    # 2023-10-24
    # ratio = cal_auprc(adata, tfs, names, pred_label='importance', ground_truth_files=ver7_ground_truth)
    # print(ratio)

    # if os.path.isfile('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ratios.json') \
    #         and os.path.isfile('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/aurocs.json'):
    #     rs = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ratios.json'))
    #     aus = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/aurocs.json'))
    # else:
    # 2023-10-26: final version
    # methods = ['hotspot', 'grnboost', 'genie3', 'HOTSPOT']
    # rs, aus = calculate_multi_samples(methods)
    # ratios_boxplot(rs, methods)
    # auroc_boxplot(aus, methods)

    # 2023-10-25, 27
    # 补run HOTSPOT
    # ratios = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ratios.json'))
    # aurocs = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/aurocs.json'))
    # data_nums = list(range(2, 12))
    # methods = ['HOTSPOT']
    # for method in methods:
    #     ratios[method] = []
    #     aurocs[method] = []
    #     for num in data_nums:
    #         data_folder = os.path.join(data_fn_base, f'data{num}')
    #         data_fn = os.path.join(data_folder, f'{method}/hotspot.h5ad')
    #         if method == 'hotspot':
    #             pred_label = 'spearman'
    #         else:
    #             pred_label = 'importance'
    #
    #         adata = sc.read_h5ad(data_fn)
    #         ratio = cal_auprc(adata, tfs, names, pred_label=pred_label, ground_truth_files=ver7_ground_truth)
    #         ratios[method].append(ratio)
    #         auroc = cal_auroc(adata, tfs, names, pred_label=pred_label, ground_truth_files=ver7_ground_truth)
    #         aurocs[method].append(auroc)
    # with open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ratios.json', 'w') as f:
    #     json.dump(ratios, f, sort_keys=True, indent=4)
    # with open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/aurocs.json', 'w') as f:
    #     json.dump(aurocs, f, sort_keys=True, indent=4)
    #
    # ratios_boxplot(ratios, methods=['hotspot', 'grnboost', 'genie3', 'HOTSPOT'])
    # auroc_boxplot(aurocs, methods=['hotspot', 'grnboost', 'genie3', 'HOTSPOT'])

    # 2023-11-13: run subdata results
    names = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/name_df.csv')
    # ver7_ground_truth = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/GRN_params_*.csv'
    adata = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/data11/hetero_11.h5ad')
    tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
    # a = AUPRC(data=adata,
    #           tfs=tfs,
    #           name_df=names)
    # gt = a.make_ground_truth(ver7_ground_truth, real_tfs=['2', '232', '408', '805', '1006'], false_tfs=['1140', '1141', '1142', '1143', '1144'])
    # print(gt)
    # gt.to_csv('ground_truth_ver7.csv', index=False)
    ver7_ground_truth_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ground_truth_ver7.csv'
    ver7_ground_truth = pd.read_csv(ver7_ground_truth_fn)

    data_fn_base200 = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/subdata_200/'
    data_fn_base100 = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/subdata_100/'

    # ratios = {}
    # aurocs = {}
    # data_nums = list(range(2, 12))
    # methods = ['hotspot', 'HOTSPOT']
    # for method in methods:
    #     ratios[method] = []
    #     aurocs[method] = []
    #     for num in data_nums:
    #         data_folder = os.path.join(data_fn_base100, f'data{num}')
    #         data_fn = os.path.join(data_folder, f'{method}.h5ad')
    #         if method == 'hotspot':
    #             pred_label = 'spearman'
    #         else:
    #             pred_label = 'importance'
    #         adata = sc.read_h5ad(data_fn)
    #         ratio = cal_auprc(adata, tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth)
    #         ratios[method].append(ratio)
    #         auroc = cal_auroc(adata, tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth)
    #         aurocs[method].append(auroc)
    # with open(f'{data_fn_base100}/ratios.json', 'w') as f:
    #     json.dump(ratios, f, sort_keys=True, indent=4)
    # with open(f'{data_fn_base100}/aurocs.json', 'w') as f:
    #     json.dump(aurocs, f, sort_keys=True, indent=4)

    # ratios = json.load(open(f'{data_fn_base200}/ratios.json'))
    # aurocs = json.load(open(f'{data_fn_base200}/aurocs.json'))
    # data_nums = list(range(2, 12))
    # methods = ['grnboost', 'genie3']
    # for method in methods:
    #     ratios[method] = []
    #     aurocs[method] = []
    #     for num in data_nums:
    #         data_folder = os.path.join(data_fn_base200, f'data{num}')
    #         data_fn = os.path.join(data_folder, f'{method}.csv')
    #         if method == 'hotspot':
    #             pred_label = 'spearman'
    #         else:
    #             pred_label = 'importance'
    #         df = pd.read_csv(data_fn)
    #         ratio = cal_auprc(tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth, adj_fn=df)
    #         ratios[method].append(ratio)
    #         auroc = cal_auroc(tfs, names, pred_label=pred_label, ground_truth=ver7_ground_truth, adj_fn=df)
    #         aurocs[method].append(auroc)
    # with open(f'{data_fn_base200}/ratios.json', 'w') as f:
    #     json.dump(ratios, f, sort_keys=True, indent=4)
    # with open(f'{data_fn_base200}/aurocs.json', 'w') as f:
    #     json.dump(aurocs, f, sort_keys=True, indent=4)
    def list_insert(my_list,n_blanks=2):
        insert_positions = range(n_blanks, len(my_list) + n_blanks * (len(my_list) // 10 + 1), 10)
        new_list = []
        for i, val in enumerate(my_list):
            new_list.append(val)
            if i + 1 in insert_positions:
                new_list.extend([None] * n_blanks)
        return new_list


    ratios = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/ratios.json'))
    aurocs = json.load(open('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver8/aurocs.json'))
    ratios200 = json.load(open(f'{data_fn_base200}/ratios.json'))
    aurocs200 = json.load(open(f'{data_fn_base200}/aurocs.json'))
    ratios100 = json.load(open(f'{data_fn_base100}/ratios.json'))
    aurocs100 = json.load(open(f'{data_fn_base100}/aurocs.json'))

    def plot_sub(ratios,ratios200,ratios100,fn='boxplot_subset.pdf'):
        x_labels = ['SpaGRN(1500)', 'SpaGRN(1000)', 'SpaGRN(500)',
                    'GRNBoost2(1500)', 'GRNBoost2(1000)', 'GRNBoost2(500)',
                    'GENIE3(1500)', 'GENIE3(1000)', 'GENIE3(500)',
                    'HOTSPOT(1500)', 'HOTSPOT(1000)', 'HOTSPOT(500)']
        df = pd.DataFrame.from_records(ratios).astype('float64')
        df = df[['hotspot', 'grnboost', 'genie3', 'HOTSPOT']]  # set order
        df200 = pd.DataFrame.from_records(ratios200).astype('float64')
        df200 = df200[['hotspot', 'grnboost', 'genie3', 'HOTSPOT']]  # set order
        df200.columns = ['hotspot200', 'grnboost200', 'genie3200', 'HOTSPOT200']
        df100 = pd.DataFrame.from_records(ratios100).astype('float64')
        df100 = df100[['hotspot', 'grnboost', 'genie3', 'HOTSPOT']]  # set order
        df100.columns = ['hotspot100', 'grnboost100', 'genie3100', 'HOTSPOT100']
        df = pd.concat([df, df200, df100])
        df = df[['hotspot', 'hotspot200', 'hotspot100',
                 'grnboost', 'grnboost200', 'grnboost100',
                 'genie3', 'genie3200', 'genie3100',
                 'HOTSPOT', 'HOTSPOT200', 'HOTSPOT100']]

        # Reshape the dataframe using melt()
        df_melted = df.melt(var_name='Group', value_name='Value')
        # cate = ['A'] * 90 + ['B'] * 90 + ['C'] * 90 + ['D'] * 90
        # df_melted['category'] = cate
        #
        colors = ['green']*3 + ['orange']*3+['blue']*3+['red']*3
        # # colors = ['green','orange','blue','red']
        sns.set_palette(colors)
        # # ax = sns.boxplot(data=df)
        ax = sns.boxplot(x='Group', y='Value', data=df_melted)

        plt.title('AUPRC ratios')
        plt.ylabel('ratio')
        ax.set_xticklabels(x_labels)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fn, format='pdf')
        plt.close()


    plot_sub(aurocs,aurocs200,aurocs100)
