import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook, show, output_file
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.MultiPlot import MultiPlot
import networkx as nx
import scipy.stats as stats


def plotWeightsOnGraph(C):
    plt.title("weights on graph:")
    plt.imshow(C)
    plt.colorbar()
    plt.show()

def plotDegrees(G,G2):
    degrees2 = sorted([d for n, d in G2.degree()], reverse=True)
    fig, axs = plt.subplots(2)
    bins = axs[0].hist(degrees, bins='auto')
    axs[0].set_title('graph 1 degree dist')

    axs[1].hist(degrees2, bins=bins[1])
    axs[1].set_title('graph 2 degree dist')
    plt.show()

    nx.draw_spring(G)
    plt.title('clustered graph')
    plt.show()

    nx.draw_spring(G2)
    plt.title('random gamma-dist. graph')
    plt.show()


def GraphDegreePlot(gam_e, clust_e):

    fig, axs = plt.subplots(2)

    bins = axs[1].hist(np.real(gam_e), bins='auto')
    axs[1].set_title('graph 2 degree dist')

    bins = axs[0].hist(np.real(clust_e), bins='auto')
    axs[0].set_title('graph 1 spec dist')
    plt.show()


def GephiWrite(G):
    G_print = G.copy()
    '''
    vals = np.sqrt([0.05, 0.15, 1])
    for e in G_print.edges:
        b1 = G_print.nodes[e[0]]['is_believer']
        b2 = G_print.nodes[e[1]]['is_believer']
        if b1 and b2:
             f = vals[0]
        elif b1 ^ b2:
            f = vals[1]
        else:
            f = vals[2]
        G_print.edges[e]['weight'] *= f
    '''
    for n in G_print.nodes:
        G_print.nodes[n]['connectList'] = ""
        G_print.nodes[n]['connectWeightList'] = G_print.degree(weight='weight')[n]
        G_print.nodes[n]['isBeliever'] = G_print.nodes[n]['is_believer']
        G_print.nodes[n]['isSensitive'] = G_print.nodes[n]['is_sensitive']
        del G_print.nodes[n]['is_believer']
        del G_print.nodes[n]['is_sensitive']


    nx.write_gml(G_print, 'test1.gml')



def PlotQn(qns):
    plt.hist(qns, bins=range(51), normed=100)
    plt.title("Qn")
    plt.xlabel("No. of new infectetions")
    plt.ylabel("No. of ")
    plt.show()


def plotAges(ages):
    plt.hist(ages, bins='auto')
    plt.show()


def BokehPlotSEIR(model, trends, model2, trends2, output_name):

    vm = MultiPlot()

    viz = DiffusionTrend(model, trends)
    viz.title = ' - clustered town disease spread: '
    p1 = viz.plot(width=650, height=550)
    vm.add_plot(p1)

    viz2 = DiffusionTrend(model2, trends2)
    viz2.title = ' - random model disease spread'
    p2 = viz2.plot(width=650, height=550)
    vm.add_plot(p2)
    m = vm.plot(ncols=2)
    output_file(output_name + ".html")
    show(m)


# plotting for one graph
def BokehPlotOne(model, trends, output_name):
    vm = MultiPlot()

    viz = DiffusionTrend(model, trends)
    viz.title = ' - clustered town disease spread: '
    p1 = viz.plot(width=650, height=550)
    vm.add_plot(p1)
    m = vm.plot(ncols=1)
    output_file(output_name + ".html")
    show(m)


def getSensitiveFromGraph(myGraph, infectors, verbose=True):
    res = {'snb': 0, 'sb': 0, 'nsnb': 0, 'nsb': 0}
    infs = res.copy()

    if "is_believer" in myGraph.nodes[0].keys():
        for num in myGraph.nodes:
            ind = 0
            if not myGraph.nodes[num]["is_sensitive"]:
                ind += 1
            if myGraph.nodes[num]["is_believer"]:
                ind += 2
            if ind == 0:
                res["snb"] += 1
            if ind == 1:
                res["nsnb"] += 1
            if ind == 2:
                res["sb"] += 1
            if ind == 3:
                res["nsb"] += 1
        
        for num in infectors.keys():
            myInf = infectors[num]
            if myInf == -1:
                continue
            ind = 0
            if not myGraph.nodes[myInf]["is_sensitive"]:
                ind += 1
            if myGraph.nodes[myInf]["is_believer"]:
                ind += 2
            if ind == 0:
                infs["snb"] += 1
            if ind == 1:
                infs["nsnb"] += 1
            if ind == 2:
                infs["sb"] += 1
            if ind == 3:
                infs["nsb"] += 1

    if verbose:
        print(f"number of Nodes: {len(myGraph.nodes)}")
        print(f"portioning: ")
        print(res)
        print(f"portioning: ")
        print(infs)

    return res, infs


def retrieveR0FromIter(iterations, resTable, freq, plot):
    """
    function for calc of R0 from various data after sim.
    input: iterations, trends table as matrix, calc frequency.
    output: R0 calced by:
    1.  valid R0: direct calculation as number of infectees from each new infector.
    2.  R0 as a growth rate of infection.
    3.  R0 as ratio of number of people infected overall before virus dies out.
    """
    Rcalc = 1 / 10
    number_people = sum([resTable[a][0] for a in range(len(resTable))])
    infectionTable = resTable[2]
    time_to_max_inf = np.argmax(infectionTable)
    i_times = iterations[-1]['i_times']

    infectorInds = list(iterations[-1]['tau_IB'].keys())
    #  finds infectors that got infected before critical time
    #  pandemic didn't happen
    if time_to_max_inf < 3:
        reserveTime = resTable.shape[1] / 4
        #  count the first 25% of time towards infections.
    else:
        reserveTime = 0

    TRcalc = max(np.int(time_to_max_inf * Rcalc), reserveTime)
    valid_keys = [a for a in infectorInds if i_times[a] < TRcalc]
    qns = [len(iterations[-1]['tau_IB'][val]) for val in valid_keys]
    if plot:
        PlotQn(qns)

    Rvalid = np.mean(qns)

    try:
        fit = stats.expon.fit(infectionTable[range(2, TRcalc)])
        lmda1 = fit[1]
        #  lmda1 = freq * 1 / fit[1]
    except:
        print("fit error.")
        lmda1 = 0

    R0_growth = lmda1

    end_inf = resTable[0][len(resTable[0]) - 1] / number_people
    R0_ratio = 1 / end_inf

    return (Rvalid, R0_growth, R0_ratio, time_to_max_inf * freq)


