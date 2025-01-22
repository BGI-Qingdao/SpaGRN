[![stars](https://img.shields.io/github/stars/BGI-Qingdao/SpaGRN?logo=GitHub&color=yellow)](https://img.shields.io/github/stars/BGI-Qingdao/SpaGRN) 
[![docs](https://img.shields.io/static/v1?label=docs&message=spagrn&color=green)](https://spagrn.readthedocs.io/en/latest/index.html)


## **SpaGRN**

<img src="docs/source/_static/spagrn_logo.png" width="30%" height="30%">

A comprehensive tool to infer TF-centered, spatial gene regulatory networks for the spatially resolved transcriptomics (SRT) data.

SpaGRN is an open-source Python package for inferring gene regulatory networks (GRNs) based on spatial gene expression data using GPLv3 license. The model takes into account the spatial proximity of genes and TF binding to infer their regulatory relationships. The package is particularly useful for analyzing spatially resolved gene expression data.
This approach can be applied to various types of spatial transcriptomics data, such as Stereo-seq, Seq-Scope, Pixel-seq, Slide-seq, MERFISH, STARmap, CosMx, ST, 10x Visium, DBiT-seq, and others. Notably, we are still working on the improvement of performance and calculation efficiency.


RRID: SCR_023451

[**Installation**](https://spagrn.readthedocs.io/en/latest/content/00_Installation.html) **-** 
[**Quick start**](https://spagrn.readthedocs.io/en/latest/content/01_Basic_Usage.html) **-** 
[**Tutorial**](https://spagrn.readthedocs.io/en/latest/Tutorials/index.html) **-** 
[**Documentation**](https://spagrn.readthedocs.io/en/latest/index.html) 


<img src="docs/source/_static/mainpipeline.BMP" width="100%" height="100%">

[comment]: <> (![SpaGRN]&#40;./docs/source/_static/mainpipeline.BMP&#41;)

## Discussion 
Please use GitHub [issues](https://github.com/BGI-Qingdao/SpaGRN/issues) tracker for reports and discussions of:
 - Bug reports
 - Document and data issues
 - Feature requirements
 - ...

## Contribution 
**SpaGRN** is in a stage of rapid development so that we will carefully consider all aspects of your proposal. We hope future input will be given by both users and developers.


## File tree
To enhance clarity and prevent any potential confusion, we have organized this GitHub repository into a structured **file tree**, complete with detailed annotation for each code segment, tailored to their respective purposes.

```

.
├── docs
│   ├── requirements.txt      # installation requirements for spagrn & ReadtheDocs.
│   └── source                # folder for storing files for ReadtheDocs building.
│       ├── conf.py                     # configuration file for ReadtheDocs building.
│       ├── content                     # folder for storing documentation files for ReadtheDocs building.
│       │   ├── 00_Installation.rst                          # installation documentation for ReadtheDocs building.
│       │   ├── 01_Basic_Usage.rst                           # basic usage documentation for ReadtheDocs building.
│       │   └── 03_References.rst                            # reference documentation for ReadtheDocs building.
│       ├── index.rst                   # strcture documentation for ReadtheDocs building.
│       ├── _static                     # folder for storing static pictures for ReadtheDocs building.
│       │   ├── E14-16h_hotspot_clusters_heatmap_top5.png    # heatmap for top 5 regulons among cell clusters.
│       │   ├── Egr3.png                                     # spatial distribution for egr3(+).
│       │   ├── grh_L3.png                                   # spatial distribution for grh(+).
│       │   └── mainpipeline.BMP                             # pipeline.
│       └── Tutorials                   # folder for storing tutorial files for ReadtheDocs building.
│           ├── index.rst                                    # strcture documentation in Tutorials section for ReadtheDocs building.
│           ├── Pbx3.png                                     # spatial distribution for pbx3(+).
│           ├── stereo_seq_mouse_brain_hi-res.ipynb          # jupyter notebook for high-resolution (subcellular) spatially resolved transcriptomics platforms.
│           └── stereo_seq_mouse_brain_low-res.ipynb         # jupyter notebook for low-resolution (multicellular) spatially resolved transcriptomics platforms.
├── LICENSE                   # GNU General Public License v3.0.
├── pyproject.toml            # setuptools management ReadtheDocs building.
├── README.md                 # README file for GitHub.
├── requirements.txt          # installation requirements.
├── setup.cfg                 # setuptools configuration file for pip installation.
├── setup.py                  # setuptools python package file for pip installation.
├── simulation                # files for simulation. 
│   ├── GRN_parameter_100.csv            # GRN parameters for simulation.
│   ├── GRN_params_non_spatial.csv       # GRN parameters for non-spatial distribution.
│   ├── GRN_params_spatial.csv           # GRN parameters for spatial distribution.
│   └── LR_parameter_100.csv             # L-R parameters for simulation.
├── spagrn                    # spagrn python script package.
│   ├── auprc.py                         # python script for calculating AUPRC and AUROC.
│   ├── autocor.py                       # python script for calculating spatial autocorrelation.
│   ├── benchmark                        # python script for benchmarking and comparison with other tools.
│   │   ├── genie3.sh                                         # python script for comparing with GENIE3.
│   │   ├── receptor_results.py                               # python script for comparing receptor detection.
│   │   └── run_hotspot.py                                    # python script for comparing with Hotspot.
│   ├── c_autocor.py                     # python script for calculating Geary's C spatial autocorrelation.
│   ├── cli                              # python script for command-line interface
│   │   ├── __init__.py
│   │   └── spagrn_parser.py
│   ├── corexp.py                        # python script for calculating spatial coexpression.
│   ├── g_autocor.py                     # python script for calculating Getis-Ord G* spatial autocorrelation.
│   ├── __init__.py
│   ├── m_autocor.py                     # python script for calculating Moran's I spatial autocorrelation.
│   ├── network.py                       # python script for constructing a basic gene regulatory network.
│   ├── params.py                        # python script for hyperparameters.
│   ├── plot.py                          # python script for plotting.
│   ├── regulatory_network.py            # python script for inferring GRN from input SRT dataset.
│   ├── results.py                       # python script for storing results.
│   ├── simulator.py                     # python script for running simulation.
│   └── spa_logger.py                    # python script for logging.
└── test
    ├── auc.csv                          # example of AUC file.
    ├── old_README.md                    # abandonded README file. 
    └── regulons.json                    # example of regulons, TF: target genes.

```
