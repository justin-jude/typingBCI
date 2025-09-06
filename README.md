
# An intuitive, bimanual, high-throughput QWERTY touch typing neuroprosthesis for people with tetraplegia
Justin J. Jude, Hadar Levi-Aharoni, Alexander J. Acosta, Shane B. Allcroft, Claire Nicolas, Bayardo E. Lacayo, Nicholas S. Card, Maitreyee Wairagkar, David M. Brandman, Sergey D. Stavisky, Francis R. Willett, Ziv M. Williams, John D. Simeral, Leigh R. Hochberg, Daniel B. Rubin

![iBCITyping](iBCITyping.png?raw=true "Typing Decoding")

## Overview

This repository contains code and jupyter notebooks to reproduce figures in our recent [iBCI typing pre-print](https://www.medrxiv.org/content/10.1101/2025.04.01.25324990v1).

Associated data will be uploaded upon manuscript publication.

## Software Requirements

### OS Requirements

This software package has been tested on the Linux operating system:

Linux: Ubuntu 22.04

However, the software is based entirely on Python scientific libraries (numpy, tensorflow etc.) and so should run seamlessly on OSX and Windows.

The hardware used for this work was an Nvidia RTX 4090 and 128 GB of DDR5 system memory.

This code was tested with the following python package versions:

```
numpy==1.26.0
scikit-learn==1.2.2
tensorflow==2.14.0
omegaconf==2.3.0
pyyaml==6.0.1
matplotlib==3.8.4
jupyter==1.0.0
g2p_en==2.1.0
seaborn==0.13.2
scipy==1.11.3
pandas==2.2.2
numba==0.58.1
redis==5.0.1
```

### Installation Guide

Install the required dependencies:

`pip install -r requirements.txt`

This should take around 2 minutes to download and install with a 100mbps internet connection.

Install language modelling software (lm_decoder) using the instructions here: https://github.com/fwillett/speechBCI/tree/main/LanguageModelDecoder

Then run `jupyter notebook` to access and run the notebooks in the repository.


### Data

Place data for each participant in the `Data` folder.

Handwriting data for participant T5 is available here: https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv