import numpy as np
import networkx as nx
AGESIZE = 101

def get_age_matrix(G):
    # function that gets the age-age correlation matrix
    C = np.zeros((AGESIZE,AGESIZE),'float')
    A = np.zeros(AGESIZE,'float')
    for a in G.nodes:
        A[G.nodes[a]['age']] += 1

    for e in G.edges:
        a1 = G.nodes[e[0]]['age']
        a2 = G.nodes[e[1]]['age']
        C[a1][a2] += G.get_edge_data(e[0], e[1])['weight']
    return A, C


def conf2vals(configValue):
    # function that translates input into switch function of different values.
    if configValue == "off" or configValue == "false":
        res = 0
    if configValue == 'on' or configValue == 'true':
        res = 1
    if configValue == 'sb':
        res = 2
    if configValue == 'import':
        res = 3
    if configValue == 'manual':
        res = 4
    return res


def getClusteringCoeff(graph):
    return nx.transitivity(graph)