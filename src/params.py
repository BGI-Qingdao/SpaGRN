#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 15 Nov 2023 14:32
# @Author: Yao LI
# @File: spagrn/params.py


class InferParam:
    def __init__(self, method=None, rank_threshold=1500, auc_threshold=0.05, motif_similarity_fdr=0.05,
                 c_threshold=-1, layer_key=None, model='danb', latent_obsm_key='spatial', umi_counts_obs_key=None,
                 n_neighbors=30, weighted_graph=False, fdr_threshold=0.05, sigm=15,
                 nes_threshold=None, noweights=False, normalize=False, rho_mask_dropouts=False,
                 **kwargs):
        self.method = None
        self.set_method(method)
        self.rank_threshold = rank_threshold
        self.auc_threshold = auc_threshold
        self.motif_similarity_fdr = motif_similarity_fdr
        self.noweights = noweights
        self.c_threshold = c_threshold
        self.layer_key = layer_key
        self.model = model
        self.latent_obsm_key = latent_obsm_key
        self.umi_counts_obs_key = umi_counts_obs_key
        self.n_neighbors = n_neighbors
        self.weighted_graph = weighted_graph
        self.fdr_threshold = fdr_threshold
        self.sigm = sigm
        self.nes_threshold = nes_threshold
        self.normalize = normalize
        self.rho_mask_dropouts = rho_mask_dropouts

        # Store additional keyword arguments as instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_method(self, method):
        if method in ['spg', 'scc', 'grnboost']:
            self.method = method
        else:
            raise ValueError(f"Invalid method: {method}. Valid options are 'spg', 'scc', 'grnboost'.")

    def get_param(self, keyword, default=None):
        return getattr(self, keyword, default)
