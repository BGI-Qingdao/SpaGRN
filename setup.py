#!/usr/bin/env python
import setuptools
import glob
import os

setuptools.setup(
    name="spagrn",
    version="1.0.1",
    author="Yao LI, Lidong GUO, Mengyang XU",
    author_email="liyao1@genomics.cn, guolidong@genomics.cn, xumengyang@genomics.cn",
    url="https://github.com/BGI-Qingdao/SpaGRN",
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages(where="spagrn"),
    package_dir={"": "spagrn"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob.glob("spagrn/*.py")
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    license="GPL-3.0+",
    description="A comprehensive tool to infer TF-centered, spatial gene regulatory networks for the spatially resolved transcriptomic data.",
    platforms='any'
)

