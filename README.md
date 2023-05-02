# Quantifying the dissimilarity of texts
## Benjamin Shade and Eduardo Altmann

This repository contains all the code used to produce the numerical results of the paper [Quantifying the Dissimilarity of Texts](https://www.mdpi.com/2078-2489/14/5/271).

A static version of the code is available at https://doi.org/10.5281/zenodo.7861675.

### Repository structure
This repository contains two key directories:
1. gutenberg: where the data from the PG database is stored
2. gutenberg-analysis: where the code for performing analysis on these texts is stored. For more details about the specific files in this folder, see gutenberg-analysis/README.md

### Setting up the data:
1. The data used can be accessed at https://doi.org/10.5281/zenodo.2422560
2. Download and unzip the folder 'SPGC-counts-2018-07-18.zip', rename this folder 'counts', and put it inside the gutenberg/data/ folder in this repository
3. Similarly, Download and unzip the folder 'SPGC-tokens-2018-07-18.zip', rename this folder 'tokens', and put it inside the gutenberg/data/ folder
Note: there are currently three texts in each of the 'counts' and 'tokens' folders - these are there for use in the repository tutorials, see gutenberg-analysis/notebooks_tutorial/. Simply replace these folders with those described in steps 2 and 3 above.

### Questions
For any questions, please email bsha5224@uni.sydney.edu.au


