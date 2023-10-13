#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Oct 2023 10:34
# @Author: Yao LI
# @File: spagrn/auprc.py


import sys
from typing import Union
import glob
import json
import anndata
import scanpy as sc
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class AUPRC:
    def __init__(self, tfs, name_df, adj_fn, reg_fn, data=None):
        self._adata = data  # only necessary when using spearman cc
        # self.ground_truth_files = ground_truth_files
        self.adj_fn = adj_fn
        self.reg_fn = reg_fn
        self._tfs = tfs

        self._ground_truth = None  # all genes
        self._prediction = None  # all genes
        self._baseline = None
        self._auprc_ratio = None

        self.adj = None
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

    @property
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

    def get_true_df(self, ground_truth_files):
        fl = glob.glob(ground_truth_files)
        self.true_df = pd.concat([pd.read_csv(i) for i in fl]).astype(str)
        return self.true_df

    def make_ground_truth(self, ground_truth_files, real_tfs=None, false_tfs=None):
        """

        :param all_genes:
        :param data_fn:
        :param ground_truth_files:
        :param naming_fn:
        :return:
        """
        # names = pd.read_csv(naming_fn)
        fl = glob.glob(ground_truth_files)
        df_true = pd.concat([pd.read_csv(i) for i in fl]).astype(str)

        # adata = sc.read_h5ad(adata_fn)
        # if self.adata is None:
        #     self.adata = adata
        all_genes = self.adata.var_names
        ground_truth = pd.DataFrame(product(self.tfs, all_genes), columns=['regulator.gene', 'regulated.gene']).astype(
            str)
        # ! make sure gene names are using the same nomenclature
        ground_truth['regulated.gene'] = ground_truth['regulated.gene'].replace(list(self.name_df['name']),
                                                                                list(self.name_df['id']))
        ground_truth['regulator.effect'] = [0] * ground_truth.shape[0]
        ground_truth = pd.concat([ground_truth, df_true])
        ground_truth = ground_truth.drop_duplicates(['regulator.gene', 'regulated.gene'], keep='last')

        # if false TF exists
        if real_tfs and false_tfs:
            # real_tfs = ['2', '232', '408', '805', '1006']
            # false_tfs = ['1140', '1141', '1142', '1143', '1144']
            t_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(real_tfs)]
            f_ground_truth = ground_truth[ground_truth['regulator.gene'].isin(false_tfs)]
            f_ground_truth['regulator.effect'] = [0.0] * f_ground_truth.shape[0]
            ground_truth = pd.concat([t_ground_truth, f_ground_truth])

        ground_truth[['regulator.gene', 'regulated.gene']] = ground_truth[['regulator.gene', 'regulated.gene']].replace(
            list(self.name_df['id']), list(self.name_df['name']))
        # ground_truth.to_csv('ground_truth_all_and_noise.csv', index=False)
        ground_truth['regulator.effect'] = ground_truth['regulator.effect'].astype('float64')
        # convert y_true into a binary matrix
        ground_truth.loc[ground_truth['regulator.effect'] > 0, 'regulator.effect'] = 1
        # order of genes need to be consistent between ground_truth and prediction
        ground_truth = ground_truth.sort_values(['regulator.gene', 'regulated.gene'], ascending=[True, True])
        self.ground_truth = ground_truth
        return ground_truth

    def get_pred_df(self, y_true_label=None):
        if self.adj is None:
            self.adj = pd.read_csv(self.adj_fn)
        if y_true_label is None:
            y_true_label = self.value_col

        self.regulons = json.load(open(self.reg_fn))
        mylist = [(key, x) for key, val in self.regulons.items() for x in val]
        df_pred = pd.DataFrame(mylist, columns=['Name', 'Values'])
        df_pred['Name'] = df_pred['Name'].str.strip('(+)')
        df_pred['prediction'] = [1] * df_pred.shape[0]

        df_pred = self.adj.merge(df_pred, left_on=['TF', 'target'], right_on=['Name', 'Values'], how='left')
        df_pred['prediction'].fillna(0)
        df_pred['prediction'] = df_pred['prediction'].fillna(0)

        df_pred = df_pred[['TF', 'target', 'importance', 'prediction', y_true_label]]
        df_pred.columns = ['TF', 'target', 'importance', 'prediction', 'ground truth']

        self.pred_df = df_pred
        return self.pred_df

    # alternative to get_pred_df
    # calculate spearman values
    def get_pred_df_spearman(self, data: Union[pd.DataFrame, anndata.AnnData] = None, y_true_label=None,
                             y_true_tf_col=None, y_true_target_col=None):
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
            self.adj = pd.read_csv(self.adj_fn)
        s = []
        for i in self.adj.index:
            res = stats.spearmanr(data[self.adj.loc[i].TF], data[self.adj.loc[i].target])
            s.append(res.correlation)
        self.adj['spearman'] = s
        adj = self.adj.sort_values(['importance', 'spearman'], ascending=False)

        # input prediction value
        regs = json.load(open(self.reg_fn))
        mylist = [(key, x) for key, val in regs.items() for x in val]
        df_pred = pd.DataFrame(mylist, columns=['Name', 'Values'])
        df_pred['Name'] = df_pred['Name'].str.strip('(+)')
        df_pred['prediction'] = [1] * df_pred.shape[0]

        # merge spearman df and prediction df
        df_pred = adj.merge(df_pred, left_on=['TF', 'target'], right_on=['Name', 'Values'], how='left')
        df_pred['prediction'].fillna(0)
        df_pred['prediction'] = df_pred['prediction'].fillna(0)

        # introduce ground truth classification label
        df_pred = df_pred.merge(self.ground_truth, left_on=['TF', 'target'],
                                right_on=[y_true_tf_col, y_true_target_col],
                                how='left')
        df_pred = df_pred[['TF', 'target', 'importance', 'spearman', 'prediction', y_true_label]]
        df_pred.columns = ['TF', 'target', 'importance', 'spearman', 'prediction', 'ground truth']
        # sort by spearman value
        tt1 = df_pred[df_pred.prediction > 0]
        tt0 = df_pred[df_pred.prediction == 0]
        tt1 = tt1.sort_values(['spearman'], ascending=False)
        tt0 = tt0.sort_values(['spearman'], ascending=False)
        # make sure 0 labels (negatives) spearman value is smaller than 1 labels
        tt0['spearman'] = tt0['spearman'] - 1
        df_prediction = pd.concat([tt1, tt0])
        df_prediction.columns = [y_true_tf_col, y_true_target_col, y_true_label, 'spearman', 'prediction',
                                 'ground truth']
        df_prediction.to_csv('df_pred.csv', index=False)
        # use this df as auprc input
        self.pred_df = df_prediction
        return df_prediction

    def get_baseline(self):
        self.baseline = 1 - self.ground_truth[self.ground_truth[self.value_col] == 0].shape[0] / \
                        self.ground_truth.shape[0]
        print(f'Baseline is {self.baseline} (num of 1/num of total)')
        return self.baseline

    def get_prediction_df(self, pred_label='spearman', y_true_label=None, y_true_tf_col=None, y_true_target_col=None):
        """
        get prediction for all genes (including genes had been filtered out by SpaGRN),
        so Ground Truth and Prediction have the same dimension (aka len(all_genes))
        value_col: column of value to pass in AUPRC calculation e.g. importance, spearman coefficient ...
        :return:
        """
        if y_true_label is None:
            y_true_label = self.value_col
        if y_true_tf_col is None:
            y_true_tf_col = self.tf_col
        if y_true_target_col is None:
            y_true_target_col = self.target_col
        pred_index = pd.merge(
            self.pred_df[[y_true_tf_col, y_true_target_col, y_true_label, pred_label, 'prediction', 'ground truth']],
            self.ground_truth[[y_true_tf_col, y_true_target_col]], on=[y_true_tf_col, y_true_target_col],
            how='outer')
        assert pred_index.shape[0] == self.ground_truth.shape[0]
        pred = pred_index.sort_values([y_true_tf_col, y_true_target_col], ascending=[True, True])
        pred = pred.fillna(int(pred[pred_label].min()) - 2)
        pred[y_true_label] = pred[y_true_label].astype('float64')
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

    def get_ratio(self):
        """
        AUPRC ratio
        :return:
        """
        new_auc = auc(self.recall, self.prec)
        if self.baseline:  # walrus operator cannot be used on instance attributes?
            self.auprc_ratio = new_auc / self.baseline
        else:
            self.get_baseline()
            self.auprc_ratio = new_auc / self.baseline
        print(f'AUPRC ratio is {self.auprc_ratio}.')
        with open('AUPRC_ratio.txt', 'w') as f:
            f.writelines(f'{self.auprc_ratio}')

    def plot_prec_recall(self, fn='Precision-Recall.png'):
        if self.recall is None or self.prec is None:
            raise ValueError('Calculate auprc first plotting. See method get_auprc')
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
        plt.fill_between(fpr, tpr)
        plt.ylabel("true positive")
        plt.xlabel("false positive")
        plt.title("AUROC")
        plt.savefig('aucroc.png')
        plt.close()

    def auprc(self,
              pred_label,
              ground_truth_files,
              y_true_label='regulator.effect',
              fn='Precision-Recall.png'):
        """
        Main logic method. SpaGRN.AUPRC pipeline
        1. generate ground truth
        2.1. load in prediction output by SpaGRN
        2.2. fill in the blank
        3. Calculate AUPRC and plot result
        :param pred_label:
        :param y_true_label:
        :param adata_fn:
        :param adj_fn:
        :param reg_fn:
        :param fn:
        :return:
        """
        # 1.
        self.make_ground_truth(ground_truth_files, real_tfs=['2', '232', '408', '805', '1006'],
                               false_tfs=['1140', '1141', '1142', '1143', '1144'])
        # self.get_baseline()

        # 2.
        if pred_label == 'spearman':
            self.get_pred_df_spearman(data=self.adata)
        else:
            self.get_pred_df()
        self.get_prediction_df(pred_label=pred_label)

        # 3.
        self.get_auprc(pred_label=pred_label, y_true_label=y_true_label)
        self.plot_prec_recall(fn=fn)
        self.get_ratio()


if __name__ == '__main__':
    import sys

    adata1 = sc.read_h5ad(sys.argv[1])
    names = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/name_df.csv')
    tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
    ver7_gt = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/GRN_params_*.csv'

    # SpaGRN output:
    a = AUPRC(data=adata1,
              tfs=tfs,
              name_df=names,
              adj_fn=sys.argv[3],
              reg_fn=sys.argv[2])
    a.auprc(pred_label='spearman', ground_truth_files=ver7_gt)

    # # GrnBoost2 output:
    # adata2 = sc.read_h5ad('')
    # b = AUPRC(data=adata2,
    #           tfs=tfs,
    #           name_df=names,
    #           ground_truth_files=ver7_gt,
    #           adj_fn='',
    #           reg_fn='')
    # b.auprc(pred_label='regulator.effect')
