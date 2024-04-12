# CE-GSL
code for "Community-entropy Based Graph Structure Learning for Topology-imbalance"

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
