#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Sep 2023 15:21
# @Author: Yao LI
# @File: spagrn/simulation.py

from pyarrow import feather
import pyarrow
import pandas as pd
import scanpy as sc

# Load in needed data
df1 = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/counts_1.csv', index_col=0)
df2 = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/counts_2.csv', index_col=0)
new_cell_names = [f'cell{x}' for x in list(range(301, 601))]
df2.columns = new_cell_names
# merge into one dataframe
df = pd.concat([df1, df2], axis=1).fillna(0).astype(int)

coor = pd.read_csv('coord.csv', index_col=0)
ct = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/cell_type.csv', index_col=0)
fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/database/dm6_v10_clust.genes_vs_motifs.rankings.feather'
rdb = pyarrow.feather.read_feather(fn)

# gene expression matrix
df = df.T
adata = sc.AnnData(df)
adata.obs['cells'] = list(df.index)
adata.var['genes'] = list(df.columns)
adata.obs['celltype'] = ct['cell.type']
adata.obsm['spatial'] = coor.iloc[:int(df.shape[0])]

# Convert gene ids to real gene names
ids = pd.read_csv('GRN_params_1139.csv')
tfs = list(set(ids['regulator.gene']))
tf_names = ['Adf1', 'Aef1', 'grh', 'kn', 'tll', 'disco-r']  # ,'Med','Dfd','br','so']
dir = {}
for tf in tfs:
    sub = ids[ids['regulator.gene'] == tf]
    sorted_sub = sub.sort_values(by='regulator.effect', ascending=False)
    dir[tf] = list(sorted_sub['regulated.gene'])

motif_names = ['bergman__Adf1', 'bergman__Aef1', 'bergman__grh', 'metacluster_172.20', 'metacluster_140.5','flyfactorsurvey__disco-r-Cl1_SANGER_5_FBgn0042650']  # ,'idmmpmm__Med','stark__RATTAMY','bergman__br-Z4','stark__YGATAC']
tf_motif_dir = dict(zip(tf_names, motif_names))
tf_id_dir = dict(zip(tfs, tf_names))
total_id = []
total_name = []
total_name += tf_names
total_name = set(total_name)
for tf in tfs:
    current_len = len(total_name)  # find current total_name number
    print(current_len)
    one_motif = tf_motif_dir[tf_id_dir[tf]]  #
    sub = rdb[rdb.motifs == one_motif].drop('motifs', axis=1)  # remove motif column, only sort gene columns
    sorted_sub = sub.sort_values(by=(sub.index.values[0]), axis=1)  # sort columns (genes) by rank numbers
    top_num = len(dir[tf])
    total_id += dir[tf]
    for tg in sorted_sub.columns:
        total_name.add(tg)
        if len(total_name) == (current_len + top_num):
            break

name_df = pd.DataFrame({'id': total_id + tfs, 'name': list(total_name)}).drop_duplicates(subset='id', keep='first')

data_genes = adata.var.copy()
data_genes = data_genes['genes'].replace(list(name_df['id']), list(name_df['name']))
data_genes = data_genes.to_frame()
adata.var = data_genes
adata.var_names = adata.var['genes']

# total_rdb_names = set(rdb.columns)
# rest_rdb_names = total_rdb_names - set(targets)
# rest_rdb_names = list(rest_rdb_names - set('motifs'))
# import random
# rest_genes = random.choices(rest_rdb_names, k=(200 - 62))
# rest_t = set(adata.var['genes']) - set(name)
# data_genes = data_genes['genes'].replace(list(rest_t), list(rest_genes))
# data_genes = data_genes.to_df()
# adata.var = data_genes.to_frame()
# adata.var_names = adata.var['genes']
adata.write_h5ad('hetero.h5ad')

# second set
coor2 = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/liuxiaobin/project/SpaGRN/scMultiSim/parameter/coord_homo.csv',
                    index_col=0)
adata2 = adata.copy()
adata2.obsm['spatial'] = coor2
cell_order = list(coor.index)
ordered_coor2 = coor2.loc[cell_order]
adata.obsm['spatial'] = ordered_coor2
adata2.write_h5ad('homo.h5ad')

# Create Simulation Ranking Database
rdb2 = rdb[rdb.motifs.isin(motif_names)]
rdb3 = rdb2[list(name_df.name) + ['motifs']]
pyarrow.feather.write_feather(rdb3, 'simulation.ranking.feather')

# Create Simulation Motifs Annotation
fn2 = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/motifs/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl'
anno = pd.read_csv(fn2, sep='\t')
anno2 = anno[anno['#motif_id'].isin(motif_names)]
anno3 = anno2[anno2['description'] == 'gene is directly annotated']
anno3.to_csv('motifs_simulation.tbl', sep='\t', index=False)

# Create Simulation TF list
with open('simulation_TF.txt', 'w') as f:
    f.writelines('\n'.join(tf_names))
