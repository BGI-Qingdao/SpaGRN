Installation
============

Our tools should be installed on a Linux/Windows system with Python3.8+.

Development Mode
--------------------

We strongly recommend your installation executed in an isolated conda environment, so firstly run:

.. code-block:: 

    conda create --name 4d-bioreconx python=3.8 # The env name could be set arbitrarily.

Then get into the environment you build:

.. code-block:: 

    conda activate 4d-bioreconx

Use the latest version of dev branch on Github, you need to clone the repository and enter the directory: 

.. code-block:: 

    git clone -b dev https://github.com/BGI-Qingdao/4D-BioReconX.git

    cd 4D-BioReconX

Install each module of interest via following its own instructions


Troubleshooting 
----------------

Possible installation failed due to some factors:

    Version of Python

Make sure you are working on Python3.8.

    Conflicts of dependencies

Find out packages that lead to failures, then create a new requirements.txt of them and run:

.. code-block:: 

    pip install -r requirements.txt

Conda installation
-------------------------

.. code-block:: 

    conda create --name 4d-bioreconx python=3.8 

    conda activate 4d-bioreconx

Use the installation command with conda:

.. code-block:: 

    Coming soon...
