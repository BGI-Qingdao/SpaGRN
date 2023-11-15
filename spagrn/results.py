#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Oct 2023 15:19
# @Author: Yao LI
# @File: spagrn/results.py
import csv

# third party modules
import json
import pandas as pd
from pyscenic.export import export2loom

from .network import Network


def dict_to_df(json_fn):
    """

    :param json_fn:
    :return:
    """
    dic = json.load(open(json_fn))
    df = pd.DataFrame([(key, var) for (key, L) in dic.items() for var in L], columns=['TF', 'targets'])
    df.to_csv(f'{json_fn.strip(".json")}.csv', index=False)


class HandleNetwork(Network):
    # Handle data generate by SpaGRN
    def __init__(self, adata, modules_fn=None, regulons_fn=None):
        super().__init__()
        self.data = adata
        self.load_results(modules_fn=modules_fn, regulons_fn=regulons_fn)

    def regulons_to_csv(self, fn: str = 'regulon_list.csv'):
        """
        Save regulon_list (df2regulons output) into a csv file.
        :param fn:
        :return:
        """
        # self.regulon_dict = self.get_regulon_dict(self.regulon_list)
        # Optional: join list of target genes
        for key in self.regulon_dict.keys(): self.regulon_dict[key] = ";".join(self.regulon_dict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(self.regulon_dict.items())

    def to_loom(self, fn: str = 'output.loom'):
        """
        Save GRN results in one loom file
        :param fn:
        :return:
        """
        export2loom(ex_mtx=self.matrix, auc_mtx=self.auc_mtx,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in self.regulons],
                    out_fname=fn)

    def to_cytoscape(self,
                     tf: str,
                     fn: str = 'cytoscape.txt'):
        """
        Save GRN result of one TF, into Cytoscape format for down stream analysis
        :param tf: one target TF name
        :param fn: output file name
        :return:

        Example:
            grn.to_cytoscape(regulons, adjacencies, 'Gnb4', 'Gnb4_cytoscape.txt')
        """
        # get TF data
        if self.regulons is None:
            raise ValueError("run load_results(regulon_fn) first")
        if self.regulon_dict is None:
            self.regulon_dict = self.get_regulon_dict(self.regulons)
        sub_adj = self.adjacencies[self.adjacencies.TF == tf]
        targets = self.regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')

    def get_cytoscape(self,
                      tf: str):
        """
        Save GRN result of one TF, into Cytoscape format for down stream analysis
        :param tf: one target TF name
        :return:
        Example:
            grn.get_cytoscape(regulons, adjacencies, 'Gnb4')
        """
        tf = tf if '(+)' not in tf else tf.replace('(+)', '')
        # get TF data
        if self.regulons is None:
            raise ValueError("run load_results(regulon_fn) first")
        if self.regulon_dict is None:
            self.regulon_dict = self.get_regulon_dict(self.regulons)
        if f'{tf}(+)' not in self.regulon_dict:
            print(f'{tf}(+) not found in data.')
            return

        sub_adj = self.adjacencies[self.adjacencies.TF == tf]
        targets = self.regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(f'{tf}_cytoscape.txt', index=False, sep='\t')

    def get_metascape(self, tf):
        tf = tf if '(+)' not in tf else tf.replace('(+)', '')
        # get TF data
        if self.regulons is None:
            raise ValueError("run load_results(regulon_fn) first")
        if self.regulon_dict is None:
            self.regulon_dict = self.get_regulon_dict(self.regulons)
        if f'{tf}(+)' not in self.regulon_dict:
            print(f'{tf}(+) not found in data.')
            return
        targets = self.regulon_dict[f'{tf}(+)']
        with open(f'{tf}_metascape.txt', 'w') as f:
            f.writelines('\n'.join(targets))

    def shared_targets(self, reg1, reg2):
        """
        Find shared targets between regulon1 and regulon2, target genes are ranked by their importance value.
        :param reg1:
        :param reg2:
        :return:
        """
        pass

    def get_targets(self, reg):
        """
        Get target genes of a regulon and save them into a text file.
        :param reg:
        :return:
        """
        pass
