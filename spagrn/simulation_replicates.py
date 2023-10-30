#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Oct 2023 17:39
# @Author: Yao LI
# @File: spagrn/simulation_replicates.py


import sys
from pyarrow import feather
import pyarrow
import pandas as pd
import scanpy as sc
from random import sample

# Use pre-defined gene name - ID chart
names = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/name_df.csv')

exp_num = sys.argv[1]

tfs = [2, 232, 408, 805, 1006, 1140, 1141, 1142, 1143, 1144]
tf_names = ['Adf1', 'Aef1', 'grh', 'kn', 'tll', 'disco-r', 'Med', 'Dfd', 'br', 'so']
motif_names = ['bergman__Adf1', 'bergman__Aef1', 'bergman__grh', 'metacluster_172.20', 'metacluster_140.5',
               'flyfactorsurvey__disco-r-Cl1_SANGER_5_FBgn0042650', 'idmmpmm__Med', 'stark__RATTAMY', 'bergman__br-Z4',
               'stark__YGATAC']
coor = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver5.2/coord_5types.csv',
                   index_col=0)
ct = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver5.2/celltype_5types.csv',
                 index_col=0)
fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/database/dm6_v10_clust.genes_vs_motifs.rankings.feather'
rdb = pyarrow.feather.read_feather(fn)

# Load in needed data
df1 = pd.read_csv(f'counts_{tfs[0]}_{exp_num}.csv', index_col=0)
df_list = [df1]
n = df1.shape[1]
for num, tf in enumerate(tfs[1:5]):
    df2 = pd.read_csv(f'counts_{tf}_{exp_num}.csv', index_col=0)
    new_cell_num = df2.shape[1]
    new_cell_names = [f'cell{x}' for x in list(range(n + 1, n + new_cell_num + 1))]
    df2.columns = new_cell_names
    n = n + new_cell_num
    df_list.append(df2)
# merge into one dataframe
df = pd.concat(df_list, axis=1).fillna(0).astype(int)
df3 = pd.read_csv(f'counts_addition_{exp_num}.csv', index_col=0)
gene_index = [i if 'gene' not in i else f'{i}1' for i in list(df3.index)]
df3.index = gene_index
df = pd.concat([df, df3]).fillna(0).astype(int)
df['index'] = df.index
df['index'] = df['index'].replace(list(names['id']), list(names['name']))
df = df.set_index('index')
df = df.T
print(df)


adata1 = sc.read_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/07.simulation/ver7/hetero.h5ad')
adata2 = adata1.copy()
adata2.X = df
adata2.obs['cells'] = list(df.index)
adata2.var['genes'] = list(df.columns)
adata2.var_names = adata2.var['genes']
print(adata2)
adata2.write_h5ad(f'hetero_{exp_num}.h5ad')

