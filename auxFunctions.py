import numpy as np
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