import networkx as nx
import numpy as np
import scipy.stats as stats
import random


def gen_complete_graph(num_nodes):
    G = nx.complete_graph(num_nodes)
    for e in G.edges():
        G[e[0]][e[1]]['weight']= 1
    return G


def GammaNetGen(num_nodes, degrees, weight):
    fit_alpha, fit_loc, scale = stats.gamma.fit(degrees, floc=0)
    print(f"shape, scale for gamma: {fit_alpha}, {scale}")
    print(f"mean, var of degree dist. : {np.mean(degrees)}, {np.std(degrees)}")
    new_degrees = np.random.gamma(shape=fit_alpha, scale=scale, size=num_nodes)
    # new_degrees = stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=num_nodes)
    G = nx.Graph()
    for (node, deg) in zip(range(num_nodes), new_degrees):
        deg = int(np.ceil(deg/2))
        # generate deg many edges from node
        dests = random.sample(range(num_nodes), deg)
        for i in range(deg):
           G.add_edge(node, dests[i], weight=weight)
    return G


def DegNetGen(num_nodes, degree, weight):
    roundDeg = np.round(np.mean(degree))
    actualWeights = np.mean(degree) * weight / roundDeg
    if num_nodes % 2 != 0 and roundDeg % 2 != 0:
        num_nodes += 1
    G = nx.random_regular_graph(int(roundDeg), int(round(num_nodes)))
    nx.set_edge_attributes(G, values=actualWeights, name='weight')
    return G, actualWeights


def NetGenFromList(num_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G


def randomNet(num_nodes, p, weight):
    G = nx.erdos_renyi_graph(num_nodes, p)
    for edge in G.edges:
        edge.weight = weight
    return G


def importNet(address='test1.gml'):
    G2 = nx.read_gml(address)
    return G2
