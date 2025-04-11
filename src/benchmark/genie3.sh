feather=/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/database/dm6_v10_clust.genes_vs_motifs.rankings.feather
tbl=/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/motifs/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl

pyscenic ctx \
grnboost_adj.csv $feather \
--annotations_fname $tbl \
--expression_mtx_fname hetero.loom  \
--mode "dask_multiprocessing" \
--output grnboost_reg.csv \
--num_workers 20  \
--rank_threshold 1500

