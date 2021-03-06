

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
from functools import reduce
import random


# ----------------------------- SETUP ----------------------------- #
# This is the only part, where the graph and underlying factors are defined


# First: define some helper functions.

def make_empty_table(variables, subset_keys = None, fill_value = 0):
    """Creates an empty table-dataframe with rows for all variables.

        variables (dict): lists of values for every variable
        subset_keys: list of variables to include in the table
        fill_value: initialization value

        returns: empty table-dataframe with a row for every variable combination
    """

    # if list of subset is set empty: return table with only one entry
    if subset_keys == []:
        return pd.DataFrame(fill_value, index = [0], columns = ['value'])

    # filter variable subset
    if subset_keys:
        variables = {key: variables[key] for key in subset_keys}

    # create a new pandas dataframe
    # one row for every combination of variable values in the subset
    # (by taking the cartesian product)
    varnames = sorted(variables.keys())
    varvalues = [variables[var] for var in varnames]
    i = pd.MultiIndex.from_product(varvalues, names = varnames)
    df = pd.DataFrame(fill_value, index = i, columns = ['value']).reset_index()
    return df




def column_varnames(columns):
    """Extract the variable names from a columns object.

        columns: a dataframe-columns object

        returns: a list of variable names
    """

    return sorted([var for var in columns if var != 'value'])



def table_varnames(tab):
    """Extract the variable names from a table-dataframe.

        tab: a table-dataframe

        returns: a list of variable names
    """

    return column_varnames(tab.columns)




def get_row_bools(tab, assignment):
    """For a given table and assignment, check which rows fit.

        tab: table-dataframe
        assignment (dict): variable assignment

        returns: table with booleans, indicating if the rows fit the assignment
    """

    # look at the table column-wise and check for every column, which rows to
    # select for the given assignment
    row_bools_each_column = [
        tab[v] == assignment[v]
        for v in table_varnames(tab)
        if v in assignment.keys()
    ]

    # reduce the list for all columns with AND, yields the booleans for the
    # complete assignment
    return reduce(lambda x,y: x & y, row_bools_each_column)





def set_assignment(tab, assignment, value):
    """Write values in table for variable assignment.

        tab: the table-dataframe
        assignment (dict): the variable assignment
        value: the value to put into all selected rows
    """
    tab.loc[get_row_bools(tab, assignment), 'value'] = value





# Now the actual setup:

# available variables and their possible values
variables = {
    'A': [0, 1],
    'B': [0, 1],
    'C': [0, 1],
    'D': [0, 1]
}

# initialize empty factors over their variables
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
    """Compute the factor product of two tables.

        tab1: first table
        tab2: second table

        returns: the factor product of tab1 and tab2
    """

    # check intersection and union of the variable sets
    varnames_intersect = column_varnames(tab1.columns & tab2.columns)
    varnames_union = column_varnames(tab1.columns | tab2.columns)

    # if the tables do not intersect, add additional merge key
    if not varnames_intersect:
        # copy to avoid side effects
        tab1 = tab1.copy()
        tab2 = tab2.copy()
        tab1['mergekey'] = 1
        tab2['mergekey'] = 1
        varnames_intersect = 'mergekey'

    # merge tables on intersection-set (or mergekey)
    tab3 = pd.merge(tab1, tab2, on = varnames_intersect)

    # compute value multiplication
    tab3['value'] = tab3['value_x'] * tab3['value_y']

    # select only the variables we need (union + 'Value')
    tab3 = tab3[varnames_union + ['value']]

    return tab3




def marginalize(tab, margin_vars):
    """Compute marginalization of table with respect to variable set.

        tab: table to marginalize
        margin_vars: variable set to marginlize over

        returns: marginalized table
    """

    # check which variables to keep
    vars_to_keep = [var for var in table_varnames(tab) if var not in margin_vars]

    # keep these variables and sum over the value for all other variables
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







# ----------------------------- SIMULATION ----------------------------- #
# Ok I lied... but now finally the simulation will run!

def pick_edge_for_update(edges):
    idx = random.randint(0, len(edges) - 1)
    return idx, edges[idx]

# initialize messages: probability tables over all sepsets with value = 1
messages = [
    make_empty_table(variables, sepset, fill_value = 1)
    for sepset in edge_sepsets
]


iterations = 300
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

    # message normalization
    msg_sum = messages[edge_idx]['value'].sum()
    messages[edge_idx]['value'] /= msg_sum


    all_incoming_msgs_prod = [
        reduce(multiply_tables, [msg for (c1, c2), msg in zip(cluster_edges, messages) if c2 == i])
        for i, _ in enumerate(clusters)
    ]

    current_beliefs = [
        multiply_tables(initial_potentials[i], all_incoming_msgs_prod[i])
        for i, _ in enumerate(clusters)
    ]

    beliefs_coll.append(current_beliefs)



def get_assignment_for_row(table, row):
    return {v: table[v][row] for v in table_varnames(table)}

def normalize_belief(belief):
    b = belief.copy()
    s = b['value'].sum()
    b['value'] /= s
    return b


n_beliefs = len(clusters)
for belief_i in range(n_beliefs):
    plt.figure()


    belief_over_time = [beliefs[belief_i] for beliefs in beliefs_coll]
    normalized_belief_over_time = [normalize_belief(belief) for belief in belief_over_time]



    belief_rows = len(initial_potentials[belief_i])
    for row in range(belief_rows):
        line = [
            normalized_belief_over_time[t]['value'][row]
            for t in range(len(normalized_belief_over_time))
        ]
        assignment = get_assignment_for_row(initial_potentials[belief_i], row)
        plt.plot(line, label = "P({})".format(assignment))
    
    varnames = table_varnames(initial_potentials[belief_i])
    plt.title("Belief #{} over {}".format(belief_i, varnames))
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0.0)

plt.tight_layout()
plt.show()
