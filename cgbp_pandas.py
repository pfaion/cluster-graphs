import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from functools import reduce


# ----------------------------- SETUP ----------------------------- #
# This is the only part, where the graph and underlying factors are defined


# First: define some helper functions.

def var_subset(variables, subset_keys):
    return {key: variables[key] for key in subset_keys}

def make_empty_table(variables, subset_keys = None, fill_value = 0):
    if subset_keys:
        variables = var_subset(variables, subset_keys)
    varnames = sorted(variables.keys())
    i = pd.MultiIndex.from_product([variables[var] for var in varnames], names = varnames)
    return pd.DataFrame(fill_value, index = i, columns = ['value']).reset_index()

def column_varnames(column):
    return sorted([var for var in column if var != 'value'])

def table_varnames(tab):
    return column_varnames(tab.columns)

def get_columns(tab, assignment):
    column_bools = [tab[v] == assignment[v] for v in table_varnames(tab) if v in assignment.keys()]
    return reduce(lambda x,y: x & y, column_bools)

def set_assignment(tab, assignment, value):
    tab.loc[get_columns(tab, assignment), 'value'] = value


# Now the actual setup:

# available variables and their possible values
variables = {
    'A': [0, 1],
    'B': [0, 1],
    'C': [0, 1],
    'D': [0, 1]
}

# initialize empty factors over thei variables
phi0 = make_empty_table(variables, ['A', 'B'])
phi1 = make_empty_table(variables, ['A', 'C'])
phi2 = make_empty_table(variables, ['B', 'D'])
phi3 = make_empty_table(variables, ['C', 'D'])

# set values for all factors
for (a, b, val) in [(0, 0, 10),
                    (0, 1, 0.1),
                    (1, 0, 0.1),
                    (1, 1, 10)]:
    set_assignment(phi0, {'A': a, 'B': b}, val)
for (a, c, val) in [(0, 0, 5),
                    (0, 1, 0.2),
                    (1, 0, 0.2),
                    (1, 1, 5)]:
    set_assignment(phi1, {'A': a, 'C': c}, val)
for (b, d, val) in [(0, 0, 5),
                    (0, 1, 0.2),
                    (1, 0, 0.2),
                    (1, 1, 5)]:
    set_assignment(phi2, {'B': b, 'D': d}, val)
for (c, d, val) in [(0, 0, 0.5),
                    (0, 1, 1),
                    (1, 0, 20),
                    (1, 1, 2.5)]:
    set_assignment(phi3, {'C': c, 'D': d}, val)

# list of all factors
factors = [phi0, phi1, phi2, phi3]

# for every cluster: list of corresponding factor indices
clusters = [
    [0],
    [1,2,3]
]





# ----------------------------- COMPUTATION ----------------------------- #
# Actual Cluster Graph Belief Propagation part now that everything is set up.

# Since its too complex for a few lines, do table multiplication in helper function.
def multiply_tables(tab1, tab2):
    varnames_intersect = column_varnames(tab1.columns & tab2.columns)
    varnames_union = column_varnames(tab1.columns | tab2.columns)
    if not varnames_intersect:
        tab1 = tab1.copy()
        tab2 = tab2.copy()
        tab1['mergekey'] = 1
        tab2['mergekey'] = 1
        varnames_intersect = 'mergekey'
    tab3 = pd.merge(tab1, tab2, on = varnames_intersect)
    tab3['value'] = tab3['value_x'] * tab3['value_y']
    tab3 = tab3[varnames_union + ['value']]
    return tab3


# for every cluster: list of actual factors
cluster_factors = [
    [factors[f] for f in c]
    for c in clusters
]

# for every cluster: union set of all variables in contained factors
cluster_vars = [
    set(
        reduce(
            lambda x, y: x + y,
            [table_varnames(f) for f in c]
        )
    )
    for c in cluster_factors
]

# list of all cluster edges (bidirectional) with indices like (0, 1) for edge between clusters 0 and 1
cluster_edges = list({
    (c1, c2)
    for c1, var1 in enumerate(cluster_vars)
    for c2, var2 in enumerate(cluster_vars)
    if var1 & var2 and c1 != c2
})

# for every edge: sepset of variables
edge_sepsets = [
    cluster_vars[c1] & cluster_vars[c2]
    for (c1, c2) in cluster_edges
]

# for every cluster: initial potential as sum of all contained factors
initial_potentials = [
    reduce(multiply_tables, [f for f in c])
    for c in cluster_factors
]

# initialize beliefs: probability tables over all sepsets with value = 1
beliefs = [
    make_empty_table(variables, sepset, fill_value = 1)
    for sepset in edge_sepsets
]

