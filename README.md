### Deep learning project seed

Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals

The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

 
---

<div align="center">    

# Your HPC Project Name

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/amorehead/deep-learning-hpc-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>

## Description

What it does

## How to run

First, create a Conda environment for the project:

```bash
# Clone project   
git clone https://github.com/YourGithubName/deep-learning-hpc-project-template

# Install project   
cd deep-learning-hpc-project-template

# (If on HPC cluster) Load 'open-ce' module
module load open-ce/0.1-0

# (If on HPC cluster) Clone Conda environment into this directory using provided 'open-ce' environment:
conda create --prefix ./venv --clone open-ce-0.1-0

# (If on HPC cluster - Optional) Create Conda environment in a particular directory using provided 'open-ce' environment:
conda create --prefix MY_VENV_DIR --clone open-ce-0.1-0

# (Else, if on local machine) Set up Conda environment locally
conda env create --prefix ./venv -f environment.yml

# (Else, if on local machine - Optional) Create Conda environment in a particular directory using local 'environment.yml' file:
conda env create --prefix MY-VENV-DIR -f environment.yml

# Activate Conda environment located in the current directory:
conda activate ./venv

# (Optional) Activate Conda environment located in another directory:
conda activate MY-VENV-DIR

# (Optional) Deactivate the currently-activated Conda environment:
conda deactivate

# (If on local machine - Optional) Perform a full update on the Conda environment described in 'environment.yml':
conda env update -f environment.yml --prune

# (Optional) To remove this long prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
```

(If on HPC cluster) Install all project dependencies:

```bash
# Install project as a pip dependency in the Conda environment currently activated:
pip3 install -e .

# Install external pip dependencies in the Conda environment currently activated:
pip3 install -r requirements.txt

# Install pip dependencies used for unit testing in the Conda environment currently activated:
pip3 install -r tests/requirements.txt
 ```

Authenticate with Comet.ml's API to enable advanced logging:
```bash
# Move to module directory
cd project

# Create a hidden config file to house your private Comet.ml API key
touch .comet.config

# Edit this config file with your text editor of choice (i.e. Vim for this example)
vim .comet.config

# Use the following format to declare your private Comet.ml API key inside .comet.config
[comet]
api_key=YOUR-API-KEY
 ```
Then, navigate to any file and run it:

 ```bash
# Module folder
cd project

# Run module (example: MNIST as your main contribution)   
python lit_image_classifier.py    
```

## Imports

This project is set up as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_image_classifier import LitClassifier
from pytorch_lightning import Trainer

# Model
model = LitClassifier()

# Data
train, val, test = mnist()

# Train
trainer = Trainer()
trainer.fit(model, train, val)

# Test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   