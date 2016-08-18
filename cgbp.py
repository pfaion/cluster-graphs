
import numpy as np

class Graph:
    def __init__(self, nodes = [], edges = []):
        self.nodes = nodes
        self.edges = edges

class MarkovNet:
    def __init__(self, graph = Graph(), factors = []):
        self.graph = graph
        self.factors = factors


class ProbabilityTable:
    def __init__(self, variables = [], values = {}):
        self.variables = variables
        self.values = values
    def __call__(self, assignment):
        idx_tuple = tuple(assignment[v] for v in self.variables)
        return self.values[idx_tuple]





g = Graph()
g.nodes = ['A', 'B', 'C', 'D']
g.edges = [
    {'A', 'B'},
    {'A', 'C'},
    {'B', 'D'},
    {'C', 'D'}
]


m = MarkovNet()
m.graph = g

phi1 = ProbabilityTable(
    variables = ['A', 'B'],
    values = {
        (0, 0): 10,
        (0, 1): 0.1,
        (1, 0): 0.1,
        (1, 1): 10
    }
)

phi2 = ProbabilityTable(
    variables = ['A', 'C'],
    values = {
        (0, 0): 5,
        (0, 1): 0.2,
        (1, 0): 0.2,
        (1, 1): 5
    }
)

phi3 = ProbabilityTable(
    variables = ['B', 'D'],
    values = {
        (0, 0): 5,
        (0, 1): 0.2,
        (1, 0): 0.2,
        (1, 1): 5
    }
)

phi4 = ProbabilityTable(
    variables = ['C', 'D'],
    values = {
        (0, 0): 0.5,
        (0, 1): 1,
        (1, 0): 20,
        (1, 1): 2.5
    }
)

f = {phi1, phi2, phi3, phi4}

m.factors = f



