Implementation of RB-SBM Model.

# List of Files
1. <tt>model_gibbs.py</tt>: Implementation of inference procedure for RB-SBM model that uses Gibbs sampling.
2. <tt>model_nogibbs.py</tt>: Implementation of inference procedure for RB-SBM model that uses exact computation of gradients for RBM.
3. <tt>model_link_pred.py</tt>: Implementation of inference procedure for RB-SBM model that takes missing links into account. The model uses Gibbs sampling.
4. <tt>exp_community_detection.py</tt>: Executes the experiment on Cora/Citeseer dataset to reproduce the performance scores reported in Table 1.
5. <tt>exp_link_pred.py</tt>: Executes the link prediction experiment on Cora/Citeseer dataset.
6. <tt>/Data/cora/read_cora.py</tt>: Reader for Cora dataset.
7. <tt>/Data/citeseer/read_citeseer.py</tt>: Reader for Citeseer dataset.


# Downloading Datasets
Although we have provided the readers for both Cora and Citeseer dataset, the actual data must be downloaded from the original source.

Cora: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

Citeseer: https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz

Paste the *.cites and *.content files in the respective dataset folders that contain the readers.


# Community Detection Experiments
Edit <tt>exp_community_detection.py</tt> to choose the desired dataset.

Run: <tt>python exp_community_detection.py</tt>

Parse the generated dump file to obtain the performance scores.

# Link Prediction Experiments
Edit <tt>exp_link_pred.py</tt> to choose the desired dataset.

Run: <tt>python exp_link_pred.py</tt>

Parse the generated dump file to obtain the performance scores.