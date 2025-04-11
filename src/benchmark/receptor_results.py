#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 09 Nov 2023 14:11
# @Author: Yao LI
# @File: spagrn/receptor_results.py
import pickle as pk
import pandas as pd
import scanpy as sc


class ReceptorParam:
    def __init__(self, module_fn=None, grn_fn=None, lr_fn=None, name_fn=None, tfs=None, tf_ct=None, data_fn=None):
        if tf_ct is None:
            tf_ct = {'Adf1': 1, 'Aef1': 2, 'grh': 3, 'kn': 4, 'tll': 5}
        if tfs is None:
            tfs = ['grh', 'kn', 'tll', 'Adf1', 'Aef1']
        self.tfs = tfs
        self.tf_ct = tf_ct
        self.data_fn = data_fn
        self.module_fn = module_fn
        self.grn_fn = grn_fn
        self.lr_fn = lr_fn
        self.name_fn = name_fn


def get_module_targets(modules) -> dict:
    d = {}
    for module in modules:
        tf = module.transcription_factor
        tf_mods = [x for x in modules if x.transcription_factor == tf]
        targets = []
        for i, mod in enumerate(tf_mods):
            targets += list(mod.genes)
        d[tf] = list(set(targets))
    return d


def read_gt(fn: str, name_df: pd.DataFrame, keys=('regulated.gene', 'regulator.gene')) -> pd.DataFrame:
    grn_gt = pd.read_csv(fn).astype(str)
    grn_gt[[keys[0], keys[1]]] = grn_gt[[keys[0], keys[1]]].replace(list(name_df.id), list(name_df.name))
    return grn_gt


def get_lr_gt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['ct1', 'ct2']] = df[['ct1', 'ct2']].astype('int64')
    sub = df[abs(df.ct1 - df.ct2) <= 1]
    sub = sub.sort_values(['ct2'])
    return sub


def get_lr_celltype(df: pd.DataFrame, ct: int) -> pd.DataFrame:
    """
    Get ligand-receptor around designated cell type
    :param df:
    :param ct:
    :return:
    """
    df = df.copy()
    df[['ct1', 'ct2']] = df[['ct1', 'ct2']].astype('int64')
    sub = df[~((abs(df.ct1 - ct) >= 2) | (abs(df.ct2 - ct) >= 2))]
    sub = sub.sort_values(['ct2'])
    return sub


def main(params: ReceptorParam):
    modules = pk.load(open(params.module_fn, 'rb'))
    m = get_module_targets(modules)
    name_df = pd.read_csv(params.name_fn).astype(str)
    grn_gt = read_gt(params.grn_fn, name_df)
    lrs = read_gt(params.lr_fn, name_df, keys=('ligand', 'receptor'))
    lr_gt = get_lr_gt(lrs)
    lr_gt = lr_gt[lr_gt.receptor.isin(grn_gt['regulated.gene'])]
    for tf in params.tfs:
        try:
            tgs = m[tf]
        except KeyError:
            print(f'{tf} not found in data.')
            continue
        ct = params.tf_ct[tf]
        ct_gt = get_lr_celltype(lr_gt, ct)
        gt_receptors = set(ct_gt.receptor)
        true_repetor_num = len(gt_receptors)
        tg_repetor_num = len(set(tgs).intersection(gt_receptors))
        print(f'{tf}: cell type {ct}')
        # print(f'cell type {ct} has {true_repetor_num} receptors')
        # # print(ct_gt)
        # print(f'{tf}: find {len(tgs)} targets')
        # print(f'{tg_repetor_num} targets are also receptors for cell type {ct}')
        print(f'{tg_repetor_num} / {true_repetor_num} = {tg_repetor_num / true_repetor_num}')


def main2(params):
    modules = pk.load(open(params.module_fn, 'rb'))
    m = get_module_targets(modules)
    name_df = pd.read_csv(params.name_fn).astype(str)
    grn_gt = read_gt(params.grn_fn, name_df)
    lr_gt = read_gt(params.lr_fn, name_df, keys=('regulator', 'target'))
    total_receptors = set(lr_gt.target).intersection(set(grn_gt['regulated.gene']))
    for tf in params.tfs:
        try:
            tgs = m[tf]
        except KeyError:
            print(f'{tf} not found in data.')
            continue
        # get cell type
        ct = params.tf_ct[tf]
        # find receptors for the cell type
        celltype_grn = grn_gt[grn_gt['regulator.gene'] == tf]
        gt_receptors = set(celltype_grn['regulated.gene']).intersection(set(lr_gt.target))
        # compare with targets
        found_receptor = set(tgs).intersection(gt_receptors)
        other_receptor = (total_receptors - gt_receptors).intersection(set(tgs))

        print('###')
        print(f'{tf}: cell type {ct}')
        # print(f'cell type {ct} has {len(gt_receptors)} receptors')
        # print(f'{tf}: find {len(tgs)} targets')
        # print(f'{len(found_receptor)} targets are also receptors for cell type {ct}')
        print(f'1.1: {len(found_receptor)} / {len(gt_receptors)} = {len(found_receptor) / len(gt_receptors)}')
        print(f'1.2: Also find {len(other_receptor)} receptors belong to other cell types.')


def main3(params):
    try:
        adata = sc.read_h5ad(params.data_fn)
    except FileNotFoundError:
        return

    m = adata.uns['regulon_dict']
    # modules = pk.load(open(params.module_fn, 'rb'))
    # m = get_module_targets(modules)
    name_df = pd.read_csv(params.name_fn).astype(str)
    grn_gt = read_gt(params.grn_fn, name_df)
    lr_gt = read_gt(params.lr_fn, name_df, keys=('regulator', 'target'))
    total_receptors = set(lr_gt.target).intersection(set(grn_gt['regulated.gene']))
    for tf in params.tfs:
        try:
            tgs = m[f'{tf}(+)']
        except KeyError:
            print(f'{tf} not found in data.')
            continue
        ct = params.tf_ct[tf]
        true_targets = set(grn_gt[grn_gt['regulator.gene'] == tf]['regulated.gene'])
        found_targets = set(tgs).intersection(true_targets)
        # print(f'{tf}: cell type {ct}')
        # print(f'cell type {ct} has {len(true_targets)} receptors')
        print(f'2.1: {tf}: find {len(tgs)} targets')
        # print(f'{len(found_targets)} targets are also receptors for cell type {ct}')
        print(f'2.2: {len(found_targets)} / {len(true_targets)} = {len(found_targets) / len(true_targets)}')


if __name__ == '__main__':
    # new 5 replicates:
    for i in list(range(1,6)):
        print(f'dataset {i}')
        params = ReceptorParam(
            module_fn=f'/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver10/data{i}/hotspot/hotspot_modules.pkl',
            grn_fn='/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/GRN_parameter_refined.csv',
            lr_fn='/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/LR_parameter_100.csv',
            name_fn='/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver9/name_df2.csv',
            data_fn=f'/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver10/data{i}/hotspot/hotspot_spagrn.h5ad')
        main2(params)
        main3(params)
        print('---------------------------------------------------------------------------------------------------------')

    # dataset 1 & 5

    # old, which receptor genes are celltype-specific:
    # params2 = ReceptorParam(
    #     module_fn='/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver9/data2/hotspot/hotspot_modules.pkl',
    #     grn_fn='/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/GRN_parameter_refined.csv',
    #     lr_fn='/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/LR_parameter_100.csv',
    #     name_fn='/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver9/name_df2.csv')
    # main2(params2)
