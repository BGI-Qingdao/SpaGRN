#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 09 Oct 2023 09:08
# @Author: Yao LI
# @File: spagrn/auroc.py

import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
tf_names = ['Adf1', 'Aef1', 'grh', 'kn', 'tll', 'disco-r', 'Med', 'Dfd', 'br', 'so']
import json

regs = json.load(open('hotspot_danb/hotspot_regulons.json'))
# noise_genes = [i for i in names.id if 'gene' in i]

# df_true = pd.read_csv('ground_truth.csv')
fl = glob.glob('./GRN_params_*.csv')
df_true = pd.concat([pd.read_csv(i) for i in fl]).astype(str)
dir_true = df_true.to_dict()
fl_c = glob.glob('./counts_*.csv')
counts = pd.concat([pd.read_csv(i, index_col=0) for i in fl_c]).fillna(0).astype(int)
all_genes = list(counts.index)
df_pred = pd.read_csv('hotspot_danb/predicted_outcome_ver7.csv')
adj = pd.read_csv('hotspot_danb/hotspot_adj.csv')
# all_genes = list(df_pred['regulated.gene'])
# ture_genes = df_true['regulated.gene']
# noise_genes = list(set(all_genes)-set(ture_genes))

# noise_true = pd.DataFrame(product(tfs, noise_genes), columns=['regulator.gene', 'regulated.gene']).astype(str)
# noise_true['regulator.effect'] = [0] * noise_true.shape[0]
ground_truth = pd.DataFrame(product(tfs, all_genes), columns=['regulator.gene', 'regulated.gene']).astype(str)
ground_truth['regulator.effect'] = [0] * ground_truth.shape[0]
ground_truth = pd.concat([ground_truth, df_true])
ground_truth = ground_truth.drop_duplicates(['regulator.gene', 'regulated.gene'], keep='last')
names = pd.read_csv('name_df.csv')
ground_truth[['regulator.gene', 'regulated.gene']] = ground_truth[['regulator.gene', 'regulated.gene']].replace(
    list(names['id']), list(names['name']))
ground_truth.to_csv('ground_truth_all_and_noise.csv', index=False)


# matched_truth = ground_truth[ground_truth['regulated.gene'].isin(all_genes)]


# for tf in tf_names:
#     pg = df_true[df_true['regulator.gene']==tf]['regulated.gene']
#     gg = ground_truth[ground_truth['regulator.gene']==tf]['regulated.gene']
#     print(set(pg)-set(gg))
#     print(len(set(pg).intersection(set(gg))))


# ll = []
# for tf in tf_names:
#     print(tf)
#     corder = list(df_pred[df_pred['regulator.gene'] == tf]['regulated.gene'])
#     print(corder)
#     print(len(corder))
#     sub = ground_truth[ground_truth['regulator.gene'] == tf]
#     print(sub)
#     print(set(corder)-set(sub['regulated.gene']))
#     sub['CustomOrder'] = sub['regulated.gene'].apply(lambda x: corder.index(x))
#     sub = sub.sort_values(by='CustomOrder')
#     sub = sub.drop(columns=['CustomOrder'])
#     ll.append(sub)
#     print(sub)
def sort_df(df, col_name, corder: list):
    df['CustomOrder'] = df[col_name].apply(lambda x: corder.index(x))
    df = df.sort_values(by='CustomOrder')
    df = df.drop(columns=['CustomOrder'])
    return df


ll = []
for tf in tf_names:
    tg = sorted(regs[f'{tf}(+)'])
    sub = ground_truth[(ground_truth['regulator.gene'] == tf) & (ground_truth['regulated.gene'].isin(tg))]
    # sort by target gene order
    sub = sort_df(sub, 'regulated.gene', tg)
    ll.append(sub)
matched_truth = pd.concat(ll)


#
pred = df_pred.merge(adj, left_on=['regulator.gene', 'regulated.gene'], right_on=['TF', 'target'], how='left').drop(['TF', 'target'], axis=1)
pred.columns = ['regulator.gene','regulated.gene','regulator.effect']
pred = pred.sort_values(['regulator.gene', 'regulated.gene'], ascending=[True, True])
matched_truth = matched_truth.sort_values(['regulator.gene', 'regulated.gene'], ascending=[True, True])
matched_truth['regulator.effect'] = matched_truth['regulator.effect'].astype('int64')
pred['regulator.effect'] = pred['regulator.effect'].astype('float64')
pred = pred.fillna(int(pred['regulator.effect'].min())-2)
# convert y_true into a binary matrix
matched_truth.loc[matched_truth['regulator.effect'] > 0, 'regulator.effect'] = 1
# ensure two TF-target orders are the same in two dataframe

# ll2 = []
# for tf in tf_names:
#     tg = regs[f'{tf}(+)']
#     sub = df_pred[df_pred['regulator.gene'] == tf]
#     ll2.append(sub[sub['regulated.gene'].isin(tg)])
# matched_truth2 = pd.concat(ll2)


# corder = list(df_pred['regulated.gene'])
# matched_truth2['CustomOrder'] = matched_truth2['regulated.gene'].apply(lambda x: corder.index(x))
# matched_truth2 = matched_truth2.sort_values(by='CustomOrder')
# matched_truth2 = matched_truth2.drop(columns=['CustomOrder'])

prec, recall, thresholds = precision_recall_curve(y_true=matched_truth['regulator.effect'],
                                                  probas_pred=pred['regulator.effect'],
                                                  pos_label=1)
print(prec, recall, thresholds)


def plot_prec_recall(prec, recall, fn='Precision-Recall.png'):
    plt.fill_between(recall, prec)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Train Precision-Recall curve")
    plt.savefig(fn)

new_auc = auc(recall, prec)
