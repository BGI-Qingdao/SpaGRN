# SpaGRN
A comprehensive tool to infer TF-centred, spatial gene regulatory networks for the spatially resolved transcriptomic data.

## Overview
SpaGRN is an open-source Python package for inferring gene regulatory networks (GRNs) based on spatial gene expression data. The model takes into account the spatial proximity of genes to infer their regulatory relationships. The package is particularly useful for analyzing spatially resolved gene expression data.

we provide two modules:
* spatially-aware model
* spatial -proximity-graph-based model
  
## Example Datasets

* Stereo-seq Mourse Brain
* Stereo-seq *Drosophila* embryos and larvae

All input SRT data and related TF database can be download from http://www.bgiocean.com/SpaGRN/

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
spaGRN can be imported as
```
from spagrn import InferRegulatoryNetwork as irn
from spagrn import PlotRegulatoryNetwork as prn
```

# Usage
The package provides functions for loading data, preprocessing data, fitting the XX model, and visualizing the inferred GRNs. The main functions are:
* loading ? preprocessing
* two
* TF gene similarity
* get modules
* cistarget (prune modules)
* AUCell (calculate regulon activity level)
* plot network: Visualization

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
![plot](./resource/E14-16h_hotspot_clusters_heatmap_top5.png)

### 2. Spatial xx
```
from spagrn import PlotRegulatoryNetwork as prn
# plot spatial distribution map of a regulon on a 2D plane 
prn.plot_2d_reg(data, 'spatial', auc_mtx, reg_name='Egr3', vmin=0, vmax=10)
```
![plot](./resource/Egr3.png)
```
prn.plot_3d_reg(data, 'spatial', auc_mtx, reg_name='grh', vmin=0, vmax=10, alpha=0.3)
```
![plot](./resource/grh_L3.png)

# Acknowledgments
