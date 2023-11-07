#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 06 Nov 2023 21:28
# @Author: Yao LI
# @File: spagrn/simulation_replicates_subset.py
import sys
import scanpy as sc
from receptor_simulation import Simulator


fn = sys.argv[1]
base_fn = fn.replace('.h5ad', '')
adata = sc.read_h5ad(fn)
sub_adata1 = Simulator.subset(adata, n_samples=200, label='celltype')
sub_adata1.write_h5ad(f'{base_fn}_sub_200.h5ad')

sub_adata2 = Simulator.subset(adata, n_samples=100, label='celltype')
sub_adata2.write_h5ad(f'{base_fn}_sub_100.h5ad')
