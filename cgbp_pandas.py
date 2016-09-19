import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from functools import reduce
import random


# ----------------------------- SETUP ----------------------------- #
# This is the only part, where the graph and underlying factors are defined


# First: define some helper functions.

def var_subset(variables, subset_keys):
    return {key: variables[key] for key in subset_keys}


def make_empty_table(variables, subset_keys = None, fill_value = 0):
    if subset_keys == []:
        return pd.DataFrame(fill_value, index = [0], columns = ['value'])
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

def get_assignment(tab, assignment):
    return tab.loc[get_columns(tab, assignment), 'value']


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
phi1 = make_empty_table(variables, ['B', 'C'])
phi2 = make_empty_table(variables, ['C', 'D'])
phi3 = make_empty_table(variables, ['D', 'A'])

# set values for all factors
for (a, b, val) in [(0, 0, 30),
                    (0, 1, 5),
                    (1, 0, 1),
                    (1, 1, 10)]:
    set_assignment(phi0, {'A': a, 'B': b}, val)
for (b, c, val) in [(0, 0, 100),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 100)]:
    set_assignment(phi1, {'B': b, 'C': c}, val)
for (c, d, val) in [(0, 0, 1),
                    (0, 1, 100),
                    (1, 0, 100),
                    (1, 1, 1)]:
    set_assignment(phi2, {'C': c, 'D': d}, val)
for (d, a, val) in [(0, 0, 100),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 100)]:
    set_assignment(phi3, {'D': d, 'A': a}, val)

# list of all factors
factors = [phi0, phi1, phi2, phi3]

# for every cluster: list of corresponding factor indices
clusters = [
    [0],
    [1],
    [2],
    [3]
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

def marginalize(tab, margin_vars):
    vars_to_keep = [var for var in table_varnames(tab) if var not in margin_vars]
    return (tab.groupby(vars_to_keep).sum())[['value']].reset_index()


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

# initialize messages: probability tables over all sepsets with value = 1
messages = [
    make_empty_table(variables, sepset, fill_value = 1)
    for sepset in edge_sepsets
]







# ----------------------------- SIMULATION ----------------------------- #
# Ok I lied... but now finally the simulation will run!

def pick_edge_for_update(edges):
    idx = random.randint(0, len(edges) - 1)
    return idx, edges[idx]


iterations = 100
coll = []
beliefs_coll = []

for _ in range(iterations):

    # pick some edge/msg to update
    edge_idx, (i, j) = pick_edge_for_update(cluster_edges)
    msg = messages[edge_idx]

    # collect all incoming msgs that do not come from the edge to update
    incoming_msgs = [
        msg
        for (c1, c2), msg in zip(cluster_edges, messages)
        if c2 == i and c1 != j
    ]

    # multiply the tables from all incoming msgs
    initial_table = make_empty_table([], [], fill_value = 1)
    incoming_msgs_prod = reduce(multiply_tables, incoming_msgs, initial_table)

    # update potential
    incoming_with_potential = multiply_tables(initial_potentials[i], incoming_msgs_prod)

    # update message with marginalization over potential
    cluster_vars_without_sepset = cluster_vars[i] - edge_sepsets[edge_idx]
    messages[edge_idx] = marginalize(incoming_with_potential, cluster_vars_without_sepset)

    all_incoming_msgs_prod = [
        reduce(multiply_tables, [msg for (c1, c2), msg in zip(cluster_edges, messages) if c2 == i])
        for i, _ in enumerate(clusters)
    ]

    current_beliefs = [
        multiply_tables(initial_potentials[i], all_incoming_msgs_prod[i])
        for i, _ in enumerate(clusters)
    ]

    beliefs_coll.append(current_beliefs)

    # # save data
    # all_msgs = reduce(multiply_tables, messages)
    # Z = float(all_msgs['value'].sum())
    # plot_msg = float(get_assignment(messages[0], {'A':0}))
    # msg_prob = plot_msg / Z
    # print(msg_prob)

    #coll.append(plot_msg/all_msgs)

# It stays the same and then explodes at one point. Gotta make more plots of more values at once. Not today.

#plt.plot(coll)




sums = [reduce(multiply_tables, beliefs)['value'].sum() for beliefs in beliefs_coll]


for i, _ in enumerate(clusters):
    for val_i in range(len(initial_potentials[i])):
        plt.plot([beliefs[i]['value'][val_i]/sum_ for beliefs, sum_ in zip(beliefs_coll, sums)])








