#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Sep 2023 15:21
# @Author: Yao LI
# @File: spagrn/simulation.py

from pyarrow import feather
import pyarrow
import pandas as pd
import scanpy as sc
import sys


tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
tf_names = ['Adf1', 'Aef1', 'grh', 'kn', 'tll', 'disco-r','Med','Dfd','br','so']
motif_names = ['bergman__Adf1', 'bergman__Aef1', 'bergman__grh', 'metacluster_172.20', 'metacluster_140.5', 'flyfactorsurvey__disco-r-Cl1_SANGER_5_FBgn0042650', 'idmmpmm__Med','stark__RATTAMY','bergman__br-Z4','stark__YGATAC']
coor = pd.read_csv('../ver5.2/coord_5types.csv', index_col=0)
ct = pd.read_csv('../ver5.2/celltype_5types.csv', index_col=0)
fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/database/dm6_v10_clust.genes_vs_motifs.rankings.feather'
rdb = pyarrow.feather.read_feather(fn)

# Load in needed data
df1 = pd.read_csv(f'counts_{tfs[0]}.csv', index_col=0)
df_list = [df1]
n = df1.shape[1]
for tf in tfs[1:5]:
    df2 = pd.read_csv(f'counts_{tf}.csv', index_col=0)
    new_cell_num = df2.shape[1]
    new_cell_names = [f'cell{x}' for x in list(range(n+1, n+new_cell_num+1))]
    df2.columns = new_cell_names
    n = n+new_cell_num
    df_list.append(df2)
# merge into one dataframe
df = pd.concat(df_list, axis=1).fillna(0).astype(int)
df3 = pd.read_csv('counts_addition.csv',index_col=0)
df = pd.concat([df,df3]).fillna(0).astype(int)
noise_ids = [i for i in list(df.index) if 'gene' in i]
print(f'{len(noise_ids)} noise genes')
print(noise_ids)


# gene expression matrix
df = df.T
adata = sc.AnnData(df)
adata.obs['cells'] = list(df.index)
adata.var['genes'] = list(df.columns)
adata.obs['celltype'] = ct['cell.type']
adata.obsm['spatial'] = coor.iloc[:int(df.shape[0])]

# Convert gene ids to real gene names
ids_list = []
for tf in tfs[:5]:
    _ids = pd.read_csv(f'GRN_params_{tf}.csv')
    ids_list.append(_ids)
ids_add = pd.read_csv('GRN_params_addition.csv')
ids = pd.concat(ids_list+[ids_add]).drop_duplicates()
dir = {}
for tf in tfs:
    sub = ids[ids['regulator.gene'] == tf]
    sorted_sub = sub.sort_values(by='regulator.effect', ascending=False)
    dir[tf] = list(sorted_sub['regulated.gene'])

tf_motif_dir = dict(zip(tf_names, motif_names))
tf_id_dir = dict(zip(tfs, tf_names))
total_id = []
total_id += tfs
total_name = []
total_name += tf_names
noise_name = []
# tf targets names, assign real names to tf targets. choose top n genes for each TF.
for tf in tfs:
    current_len = len(total_name)  # find current total_name number
    print(f'current number of gene names: {current_len}')
    one_motif = tf_motif_dir[tf_id_dir[tf]]  #
    sub = rdb[rdb.motifs == one_motif].drop('motifs', axis=1)  # remove motif column, only sort gene columns
    sorted_sub = sub.sort_values(by=(sub.index.values[0]), axis=1)  # sort columns (genes) by rank numbers
    top_num = len(dir[tf])
    total_id += dir[tf]
    for tg in sorted_sub.columns:
        #total_name.add(tg)
        # 2023-09-14: sets do not guarantee orders. change back to list
        if tg not in total_name:
            total_name.append(tg)
        if len(total_name) == (current_len + top_num):
            break
# noise names, randomly choose gene names for noise data points
from random import sample
total_rdb_names = set(rdb.columns) - set('motifs')
rest_rdb_names = list(total_rdb_names - set(total_name))
noise_genes_names = sample(rest_rdb_names, k=len(noise_ids)) # !! random.choices returns duplicated items, use random.sample instead
print(type(noise_genes_names))
print(len(noise_genes_names), len(set(noise_genes_names)))
# add noise names into total_name
total_name += noise_genes_names
total_id += noise_ids
print(len(total_id), len(set(total_id)))
print(len(total_name), len(set(total_name)))
name_df = pd.DataFrame({'id': total_id, 'name': list(total_name)}).drop_duplicates(subset='id', keep='first').astype(str)
name_df.to_csv('name_df.csv', index=False)

data_genes = adata.var.copy()
data_genes = data_genes['genes'].replace(list(name_df['id']), list(name_df['name']))
print(data_genes)
data_genes = data_genes.to_frame()
adata.var = data_genes
adata.var_names = adata.var['genes']
adata.write_h5ad('hetero.h5ad')

# second set
coor2 = pd.read_csv('../ver5.2/coord_5types_shuffled.csv', index_col=0)
adata2 = adata.copy()
cell_order = list(coor.index)
ordered_coor2 = coor2.loc[cell_order].iloc[:int(df.shape[0])]
adata2.obsm['spatial'] = ordered_coor2
adata2.write_h5ad('homo.h5ad')

