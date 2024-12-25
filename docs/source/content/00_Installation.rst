Installation
============

Our tools should be installed on a Linux/Windows system with Python3.8+.

Installing with a 'package manager'
----------------------------------------

We strongly recommend your installation executed in an isolated conda environment, so firstly run:

.. code-block:: 

    conda create --name spagrn python=3.8 # The env name could be set arbitrarily.


Then get into the environment you build:

.. code-block:: 

    conda activate spagrn
	
To install the latest version of SpaGRN via `PyPI`:

.. code-block:: 

	pip install spagrn==1.0.7

Or install by `conda`:

.. code-block:: 

	conda install -c bioconda spagrn
	

Notice: If you install via conda, you will need to install the following dependencies separately:

.. code-block:: 

	pyscenic==0.12.1
	hotspotsc==1.1.1
	arboreto
	ctxcore>=0.2.0

SpaGRN can be imported as:

.. code-block:: 

	from spagrn import InferNetwork as irn
	
	from spagrn import plot as prn

Dependencies:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: 

	anndata==0.8.0
	pandas<2.0.0,>=1.3.4
	scanpy==1.9.1
	seaborn
	matplotlib 
	pyscenic==0.12.1
	hotspotsc==1.1.1
	scipy
	numpy
	dask
	arboreto
	ctxcore>=0.2.0
	scikit-learn

	
Development Mode
--------------------

Use the latest version of dev branch on Github, you need to clone the repository and enter the directory: 

.. code-block:: 

    git clone -b dev https://github.com/BGI-Qingdao/SpaGRN.git

    cd SpaGRN

Install each module of interest via its own instructions


Troubleshooting 
----------------

Possible installation failed due to some factors:

Version of Python
++++++++++++++++++++++++

Make sure you are working on Python3.8.

Conflicts of dependencies
++++++++++++++++++++++++

Find out packages that lead to failures, then create a new requirements.txt of them and run:

.. code-block:: 

    pip install -r requirements.txt

