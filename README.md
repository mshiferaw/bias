# bias
This is the repository for the bias parameter measurements of IllustrisTNG (TNG) and UniverseMachine (UM), as documented in https://arxiv.org/abs/2412.06886. In addition to making the figures reproducible, we also include the code for running the bias parameter measurements (<code>mpi_ksum.py</code>). As much of the infrastructure for this project was developed for https://arxiv.org/abs/2101.11014, please cite both papers if using any code from this repo.

### notebooks
Start here to reproduce each figure of the paper. All of the necessary data is hosted in the <code>data</code> folder. These notebooks are gemerally self-contained, with the exception of Figures 1 and 2, which rely on TNG and UM data. TNG data can be found at https://www.tng-project.org/data/, while UM data can be made available upon reasonable request to the authors.

We also include a separate notebook for reformatting the bias measurements in such a way that we can set priors on them (<code>Save_data_for_setting_priors.ipynb</code>).

### figures 
The output of each figure in the notebooks.

### data
This includes the bias parameter measurements, which are run using the <code>mpi_ksum.py</code> file, found in the <code>scripts</code> folder. 

Within the <code>data/priors</code> folder, we include these same bias parameter measurements, consolidated in such a way that we were able to set priors. This reformatting of the data is performed in the <code>Save_data_for_setting_priors.ipynb</code> notebook, also found in the <code>notebook</code> folder. 

### batches
The shell script <code>batch_ksum.sh</code> for running the bias parameter measurements can be found here. 

### scripts
This hosts the main <code>mpi_ksum.py</code> which we use to measure the bias parameters in each galaxy sample. This code, as well as the notebook for Figure 1, relies upon the Lagrangian component fields for TNG. These fields were obtained using the code in this repository: https://github.com/kokron/anzu/tree/main. 

This script also produces the data for the consistency check that we perform in Figure 10.

### To be added: 
The code for removing assembly bias.
