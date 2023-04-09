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

# Installation
To install SpaGRN via Pyxxx
```
pip install spagrn
```
Alternatively, you can install SpaGRN using the following command:
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
from spagrn import PlotRegulatoryNetwork as prn

# Load data
data = irn.read_file('data.h5ad')

# Preprocess data
data <- irn.preprocess(data)

# Initialize gene regulatory network
grn = irn(data)

# load TF list
tfs = irn.load_tfs(tfs_fn)

# load the ranking databases
dbs = irn.load_database(databases)
```

# Acknowledgments
