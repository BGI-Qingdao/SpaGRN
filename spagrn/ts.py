#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 15 Aug 2023 15:58
# @Author: Yao LI
# @File: spagrn/ts.py

import scanpy as sc
import json
import pandas as pd
import numpy as np
import sys


auc_mtx = pd.read_csv(sys.argv[1], index_col=0)
adata = sc.read_h5ad(sys.argv[2])
receptors = json.load(open(sys.argv[3]))

matrix = adata.to_df()


# drop regulons do not have receptors
receptor_tf = [f'{i}(+)' for i in list(receptors.keys())]
rtf = auc_mtx[list(receptor_tf)]

# multiply sum of the receptor exp values to regulon auc value
np.sum(matrix[receptors[regulon]].loc[cell])  # regulon string does not contain (+)
