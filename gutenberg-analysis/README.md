## gutenberg-analysis

This directory contains a number of important files and subdirectories:

### final_analysis
- compute_distance.py: computes the quantity P(X < Y) for all measures except the Jensen-Shannon divergence (JSD)
- FigureX.ipynb: produces the Figure X displayed in our paper
- final_corpus.ipynb: generates the subcorpora used in our analysis
- h_test.py: computes P(X < Y) after computing one text in each pair to a length N, and the other text to length hN, where h > 1
- misc_calculations.ipynb: miscellaneous calculations (eg: identifying optimal alpha, computing p-values)
- optimal_alpha_new.py: computes P(X < Y) using the generalised JSD across a given range of alpha values
- resmaple_test_final.py: computes P(X < Y) after resampling all texts to specified 
lengths

### notebooks_tutorial
Tutorials on how to read in books from the PG database, and how to compute their dissimilarity.

### output_files
Where the results from the files in final_analysis are stored.

### src
Where helper functions are stored for us in the files in final_analysis.
- data_io.py: retrieving books from the PG data
- ent.py: computing generalised entropy
- jsd.py: computing Jensen-Shannon divergence
- metaquery.py: filtering and retrieving information about PG texts from the metadata
- metric_eval.py: resampling a text and computing P(X < Y)
- pretrained_embedding.py: generating vector embeddings of all books in the corpus