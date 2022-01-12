import numpy as np
import networkx as nx
from bisect import bisect_left


def CreateRumor(people, believers_portion=0.3):
    #  initialize beliefs on population, at PEOPLE ARRAY level
    #  prioritizes people furthest from mean age.
    if believers_portion > 1 or believers_portion < 0:
        believers_portion = 0.3

    num_people = len(people)
    to_add = believers_portion * num_people

    ages_vals = [person.age for person in people]
    median_age = np.median(ages_vals)

    order = np.argsort(np.abs(ages_vals - median_age))[::-1]

    ind = 0

    believers_list = []
    while to_add > 0 and ind < num_people:
        believers_list.append(people[order[ind]])
        people[order[ind]].is_believer = True
        to_add -= 1
        ind += 1


def CreateSensitivity(people, age_for_sensitive=65):
    for person in people:
        if person.age > age_for_sensitive:
            person.is_sensitive = True
        else:
            person.is_sensitive = False


def CreateSBFromCorr(people, Cmat):
    probs = np.random.rand(len(people))
    cumCorr = np.cumsum(Cmat)
    SB_cat = [bisect_left(cumCorr, p) for p in probs]
    for ind in range(len(people)):
        if SB_cat[ind] == 0:
            #  a, SNB
            # print('{} snb'.format(ind))
            people[ind].is_sensitive = True
            people[ind].is_believer = False
        elif SB_cat[ind] == 1:
            # b, SB
            # print('{} sb'.format(ind))
            people[ind].is_sensitive = True
            people[ind].is_believer = True
        elif SB_cat[ind] == 2:
            # c, NSNB
            # print('{} nsnb'.format(ind))
            people[ind].is_sensitive = False
            people[ind].is_believer = False
        elif SB_cat[ind] == 3:
            # d, NSB
            # print('{} nsb'.format(ind))
            people[ind].is_sensitive = False
            people[ind].is_believer = True

def CreateGraphSBFromCorr(nodes, Cmat):
    probs = np.random.rand(len(nodes))
    cumCorr = np.cumsum(Cmat)
    SB_cat = [bisect_left(cumCorr, p) for p in probs]
    for ind in range(len(nodes)):
        if SB_cat[ind] == 0:
            #  a, SNB
            nodes[ind]['is_sensitive'] = True
            nodes[ind]['is_believer'] = False
        elif SB_cat[ind] == 1:
            # b, SB
            nodes[ind]['is_sensitive'] = True
            nodes[ind]['is_believer'] = True
        elif SB_cat[ind] == 2:
            # c, NSNB
            nodes[ind]['is_sensitive'] = False
            nodes[ind]['is_believer'] = False
        elif SB_cat[ind] == 3:
            # d, NSB
            nodes[ind]['is_sensitive'] = False
            nodes[ind]['is_believer'] = True


def GraphifyNS(G, Cmat, beta_l, avg_weights):
    #  takes people array and turns into NetworkX graph object.
    #  CreateRumor(people, 0.3)
    #  CreateSensitivity(people, 65)
    CreateGraphSBFromCorr(G.nodes, Cmat)
    EditWeights(G, beta_l)
    # correct weights to accomodate new avergae to be same as structuredGraph
    new_weights = np.mean([b[1] / a[1] for (a,b) in zip(G.degree,G.degree(weight='weight'))])
    f = avg_weights/new_weights
    for a in G.edges():
        G.edges[a[0], a[1]]['weight'] *= f


    '''
    np.mean([G.degree(p,weight='weight') for p in nbeliever_inds])
    nbeliever_inds = [p[0] for p in enumerate(b) if not p[1]]
    b = [p.is_believer for p in people]
    b = [G.nodes[p]['is_believer'] for p in range(len(G))]
    '''
    return G


def EditWeights(G, beta_l):
    #  update Weights according to beliefs
    #  beta_l = np.sqrt(beta_l)  #   important! here we square the connections weight!
    # unnescessary because edges gives multidirectional connections.
    f1 = 0
    for edge in G.edges():
        b1 = G.nodes[edge[0]]['is_believer']
        b2 = G.nodes[edge[1]]['is_believer']
        if b1 and b2:
            # believer - believer
            f = beta_l[0]
        elif bool(b1) != bool(b2):
            # believer - not believer
            f = beta_l[1]
        elif not(b1) and not(b2):
            # nb-nb
            f = beta_l[2]
        else:  # error/ do nothing
            f = 1
        G.edges[edge[0], edge[1]]['weight'] *= f


def Graphify(people, edges, Cmat, beta_l):
    #  takes people array and turns into NetworkX graph object.
    #  CreateRumor(people, 0.3)
    #  CreateSensitivity(people, 65)
    CreateSBFromCorr(people, Cmat)

    node_ids = [person.number for person in people]
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for n in G.nodes():
        for k in vars(people[n]).items():
            G.nodes[n][k[0]] = k[1]

    [G.add_edge(edge[0], edge[1], weight=edge[2]) for edge in edges]
    EditWeights(G, beta_l)
    print(f"sensitives: {sum([G.nodes[n]['is_sensitive'] for n in G.nodes])} out of {G.number_of_nodes()}")
    print(f"believers: {sum([G.nodes[n]['is_believer'] for n in G.nodes])} out of {G.number_of_nodes()}")
    return G
