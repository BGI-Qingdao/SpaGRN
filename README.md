# SpaGRN
A comprehensive tool to infer TF-centered, spatial gene regulatory networks for the spatially resolved transcriptomic data.

## Overview
SpaGRN is an open-source Python package for inferring gene regulatory networks (GRNs) based on spatial gene expression data. The model takes into account the spatial proximity of genes and TF binding to infer their regulatory relationships. The package is particularly useful for analyzing spatially resolved gene expression data.

we provide two modules to infer the co-expressed and co-localized gene network:
* spatially-aware model
* spatial-proximity-graph-based model
  
## Examples

* Stereo-seq Mouse Brain
* Stereo-seq *Drosophila* Embryo and Larvae

All input SRT data and related TF database can be downloaded from http://www.bgiocean.com/SpaGRN/

# Installation
To install the latest version of SpaGRN:
```
git clone https://github.com/BGI-Qingdao/SpaGRN.git
cd SpaGRN
python setup.py sdist
pip install dict/spagrn-1.0.0.tar.gz
```
Alternatively, you can install SpaGRN via conda using the following command:
```
conda install spagrn -c bioconda
```
SpaGRN can be imported as:
```
from spagrn import InferRegulatoryNetwork as irn
from spagrn import PlotRegulatoryNetwork as prn
```

# Usage
The package provides functions for loading data, preprocessing data, reconstructing gene network, and visualizing the inferred GRNs. The main functions are:
* Load and process data
* Compute TF-gene similarities
* Create modules
* Perform motif enrichment and determine regulons
* Calculate regulon activity level across cells
* Visualize network and other results

## Example workflow:
```
from spagrn import InferRegulatoryNetwork as irn

# Load data
data = irn.read_file('data.h5ad')

# Preprocess data
data <- irn.preprocess(data)

# Initialize gene regulatory network
grn = irn(data)

grn.main(database_fn,
             motif_anno_fn,
             tfs_fn,
             num_workers=cpu_count(),
             cache=False,
             save=True,
             method=method,
             prefix=prefix,
             noweights=True)
```

## Visualization
SpaGRN offers a wide range of data visualization methods.
### 1. Heatmap
```
# read data from previous analysis
data = irn.read_file('data.h5ad')
data <- irn.preprocess(data)
auc_mtx = pd.read_csv('auc.csv', index_col=0)
# alternative, extract data from the grn object
data = grn.data
auc_mtx = grn.auc_mtx

# plot 
prn.rss_heatmap(data,
            auc_mtx,
            cluster_label='annotation',
            rss_fn='regulon_specificity_scores.txt'),
            topn=5,
            save=True)  
```
<img src="./resource/E14-16h_hotspot_clusters_heatmap_top5.png" width="400">

### 2. Spatial Plots
Plot spatial distribution map of a regulon on a 2D plane:
```
from spagrn import PlotRegulatoryNetwork as prn

prn.plot_2d_reg(data, 'spatial', auc_mtx, reg_name='Egr3')
```
<img src="./resource/Egr3.png" width="300">

If one wants to display their 3D data in a three-dimensional fashion:
```
prn.plot_3d_reg(data, 'spatial', auc_mtx, reg_name='grh', vmin=0, vmax=4, alpha=0.3)
```
<img src="./resource/grh_L3.png" width="300">
