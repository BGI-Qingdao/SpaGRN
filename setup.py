#!/usr/bin/env python
import setuptools
import glob
import os
from pathlib import Path

setuptools.setup(
    name="spagrn",
    version="1.0.4",
    author="Yao LI, Lidong GUO, Mengyang XU",
    author_email="liyao1@genomics.cn, guolidong@genomics.cn, xumengyang@genomics.cn",
    url="https://github.com/BGI-Qingdao/SpaGRN",
    #long_description=Path('README.md').read_text('utf-8'),
    python_requires=">=3.7,<3.11",
    packages=setuptools.find_packages(),
    install_requires=[
        "anndata==0.8.0",
        "pandas>=1.3.5",
        "scanpy",
        "seaborn",
        "matplotlib",
        "pyscenic==0.12.1",
        "hotspotsc==1.1.1",
        "scipy",
        #"numpy<1.20.0,>=1.16.6",
        "numpy",
        "dask",
        "arboreto",
        "ctxcore>=0.2.0",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    license="GPL-3.0+",
    description="A comprehensive tool to infer TF-centered, spatial gene regulatory networks for the spatially resolved transcriptomic data.",
    platforms='any',
)

