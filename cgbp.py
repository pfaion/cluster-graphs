import numpy as np
import pandas as pd
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from functools import reduce

class Graph:
    def __init__(self, nodes = [], edges = []):
        self.nodes = nodes
        self.edges = edges

class MarkovNet:
    def __init__(self, graph = Graph(), factors = []):
        self.graph = graph
        self.factors = factors


class ProbTable:
    def __init__(self):
        self.variables = variables
        i = pd.MultiIndex.from_product([values[v] for v in variables], names = variables)
        self.data = pd.DataFrame(0, index = i, columns = ['value']).reset_index()

    def get_columns(self, assignment):
        column_bools = [self.data[v] == assignment[v] for v in variables if v in assignment.keys()]
        return reduce(lambda x,y: x & y, column_bools)

    def __getitem__(self, assignment):
        return self.data.loc[self.get_columns(assignment), 'value']

    def __setitem__(self, assignment, value):
        self.data.loc[self.get_columns(assignment), 'value'] = value

    def __mul__(self, other):
        return self.data * other.data

t = ProbTable()
t[{'A':0, 'B':0}] = 5
t2 = ProbTable()
t2[{'A':0, 'C':0}] = 10

t * t2







class Cluster:
    def __init__(self, variables, factors):
        self.variables = variables
        self.potential =

class ClusterGraph:
    def __init__(self, markov_net):
        self.markov_net = markov_net
        self.clusters = [f.variables for f in self.markov_net.factors]
        self.potentials = [f.table for f in self.markov_net.factors]
        self.N = len(self.clusters)
        self.edges = np.zeros((self.N,) * 2, dtype = bool)
        edge_idxs = [(i, j) for i in range(self.N) for j in range(self.N) if self.sepset(i, j) != []]
        self.edges[tuple(zip(*edge_idxs))] = True
        self.messages = self.edges.astype(int)

    def sepset(self, i, j):
        return [c for self.clusters[i] if c in self.clusters[j]]

    def update_message(self, i, j):
        incoming_idxs = [idx for idx in range(self.N) if idx != j and cluster_graph.edges[idx, i]]
        incoming_msgs = self.messages[incoming_idxs].prod()
        var_idxs_without_sepset = tuple(idx for idx, var in enumerate(self.clusters[i]) if var not in self.sepsep(i, j))
        marginal_without_sepset = self.potentials[i].sum(var_idxs_without_sepset)



variables = ['A', 'B', 'C', 'D']
values = {
    'A': [0, 1],
    'B': [0, 1],
    'C': [0, 1],
    'D': [0, 1]
}
edges = [
    {'A', 'B'},
    {'A', 'C'},
    {'B', 'D'},
    {'C', 'D'}
]


graph = Graph(variables, edges)

phi1 = Factor(['A', 'B'])
phi1.table[0, 0] = 10
phi1.table[0, 1] = 0.1
phi1.table[1, 0] = 0.1
phi1.table[1, 1] = 10

phi2 = Factor(['A', 'C'])
phi2.table[0, 0] = 5
phi2.table[0, 1] = 0.2
phi2.table[1, 0] = 0.2
phi2.table[1, 1] = 5

phi3 = Factor(['B', 'D'])
phi3.table[0, 0] = 5
phi3.table[0, 1] = 0.2
phi3.table[1, 0] = 0.2
phi3.table[1, 1] = 5

phi4 = Factor(['C', 'D'])
phi4.table[0, 0] = 0.5
phi4.table[0, 1] = 1
phi4.table[1, 0] = 20
phi4.table[1, 1] = 2.5


factors = [phi1, phi2, phi3, phi4]
markov_net = MarkovNet(graph, factors)

cluster_graph = ClusterGraph(markov_net)
