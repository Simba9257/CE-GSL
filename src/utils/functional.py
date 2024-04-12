import numpy as np
import networkx as nx
import torch
from typing import Sequence
from cdlib import algorithms
from cdlib.utils import convert_graph_formats


def community_detection(name):
    algs = {
        # non-overlapping algorithms  非重叠算法
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        #'edmot': algorithms.edmot,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms  重叠算法
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        #'ego-splitting': algorithms.egonet_splitter,
        #'nnsed': algorithms.nnsed,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]
