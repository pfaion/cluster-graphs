import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from functools import reduce



def make_empty_table(variables, subset_keys = None):
    if subset_keys:
        variables = var_subset(variables, subset_keys)
    varnames = sorted(variables.keys())
    i = pd.MultiIndex.from_product([variables[var] for var in varnames], names = varnames)
    return pd.DataFrame(0, index = i, columns = ['value']).reset_index()


def column_varnames(column):
    return sorted([var for var in column if var != 'value'])

def get_columns(tab, assignment):
    column_bools = [tab[v] == assignment[v] for v in column_varnames(tab.columns) if v in assignment.keys()]
    return reduce(lambda x,y: x & y, column_bools)

def get_assignment(tab, assignment):
    return tab.loc[get_columns(tab, assignment)]

def set_assignment(tab, assignment, value):
    tab.loc[get_columns(tab, assignment), 'value'] = value



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



def var_subset(variables, subset_keys):
    return {key: variables[key] for key in subset_keys}

def var_union(vars1, vars2):
    return dict(item for item in d.items() for d in [vars1, vars2])


variables = {
    'A': [0, 1],
    'B': [0, 1],
    'C': [0, 1],
    'D': [0, 1]
}


phi0 = make_empty_table(variables, ['A', 'B'])
phi1 = make_empty_table(variables, ['A', 'C'])
phi2 = make_empty_table(variables, ['B', 'D'])
phi3 = make_empty_table(variables, ['C', 'D'])


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


factors = [phi0, phi1, phi2, phi3]

cluster_factors = [
    [0],
    [1,2,3]
]

initial_potentials = [
    reduce(multiply_tables, [factors[f] for f in cluster])
    for cluster in cluster_factors
]

