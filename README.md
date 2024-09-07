# CE-GSL
code for "Community-entropy Based Graph Structure Learning for Topology-imbalance"

# Datasets
We experiment on eight open graph benchmark datasets, including 
three citation networks (i.e., `Cora`, `Citeseer`, and `Pubmed`), 
two social networks (i.e., `PT` and `TW`), three website networks 
from WebKB (i.e., `Cornell`, `Texas`, and `Wisconsin`), and a co-occurrence 
network (i.e., `Actor`).

# Baseline and backbone models
We compare CE-GSL with baselines including general GNNs (i.e., 
GCN, GAT, GCNII, Grand) and graph learning/high-order neighborhood 
awareness methods (i.e. MixHop, Dropedge, Geom-GCN, GDC, GEN, 
H2GCN, SE-GSL). Four classic GNNs (GCN, GAT, GraphSAGE, APPNP) 
are chosen as the backbone encoder that CE-GSL works upon.

# Parameter settings
For CE-GSL with various backbones, we uniformly adopt two-layer 
GNN encoders. To avoid over-fitting, We adopt ReLU (ELU for GAT) 
as the activation function and apply a dropout layer with a 
dropout rate of 0.5. The training iteration is set to 10. The 
embedding dimension $d$ is chosen from \{8, 16, 32, 48, 64, 80, 
128, 256\}, and the hyperparameter $\theta$ is tuned among 
\{0.5, 1, 3, 5, 10, 30\}. We adopt the scheme of data split in 
Geom-GCN \cite{pei2020geom} and $\mathrm{H}_{2} \mathrm{GCN}$ 
for all experiments $-$ nodes are randomly divided into the train, 
validation, and test sets, which take up 60\%, 20\%, 20\%, respectively. 
In each iteration, the GNN encoder optimization is carried out for 200 
epochs, using the Adam optimizer, with an initial learning rate 
selected from \{0.1, 0.05, 0.01, 0.005, 0.001\} and a weight decay 
selected from $\{5 e-6,5 e-5,5 e-4\}$. The model with the highest 
accuracy on validation sets is used for further testing and 
reporting.

# Overview
- model: implement of different GNN model
- utils/max1SE.py: unit community-entropy k-selector.
- utils/functional.py: clustering algorithm.
- utils/reshape.py: node-pair sampling.
- utils/utils_data.py, utils/utils.py: data preprocessing code from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn)

# Requirements
The implementation of CE-GSL is tested under Python 3.9.18, with the following packages installed:
* `dgl-cu116==1.1.2`
* `pytorch==1.12.1`
* `numpy==1.26.3`
* `networkx==3.2.1`
* `scipy==1.11.4`
* `cdlib==0.3.0`
