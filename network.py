import numpy as np
import plotting
import os
import networkx as nx
import ndlib.models.ModelConfig as mc
import NetGen
import FamilyNetwork
import SEIRyoel
import time
from scipy.sparse.linalg import eigs, eigsh
import auxFunctions, DEFAULTS
import RumorSpreading as RS

def go(conf):
    print(f"structured graph mode: " + conf.structuredGraphMode)
    print(f"random graph mode: " + conf.randomGraphMode)
    _structGraphMode = auxFunctions.conf2vals(conf.structuredGraphMode)
    plot = conf.plot

    if _structGraphMode:
        G, number_people, avg_weights, degrees = FamilyNetwork.generate(
            conf.num_families, conf.family_dist,
            conf.population_dist,conf.mean_parents,
            conf.employment, conf.size_work,
            conf.size_school,
            conf.random_connections, conf.plot, conf.school_w,
            conf.family_w, conf.work_w, conf.random_w,
            conf.believerSusceptibleCorr, conf.beta_list
            )

        # edges list, avg_weights, number of people, adjacency matrix
        print(f"avg weight: {avg_weights}")

        if plot:
            A, C = auxFunctions.get_age_matrix(G)
            plotting.GephiWrite(G)
            plotting.plotWeightsOnGraph(C)

        A = nx.adjacency_matrix(G)
        cls_str = auxFunctions.getClusteringCoeff(G)
        print(cls_str)
        clust_e, clust_v = eigsh(A, 10, which='LM')
        clust_e = abs(clust_e)
    # generate new graph adversary
    #  NetGen.gen_complete_graph(number_people)
    randomGraphMode = auxFunctions.conf2vals(conf.randomGraphMode)
    if randomGraphMode:
        if randomGraphMode == 4:
            number_people = conf.num_families * DEFAULTS.FAM2PPL
            degrees = DEFAULTS.DREGDEG
            avg_weights = DEFAULTS.AVGWEIGHT
        if randomGraphMode == 3:
            G2 = NetGen.importNet()
        else:
            G2, actual_weights = NetGen.DegNetGen(number_people, np.mean(degrees), avg_weights)
        if randomGraphMode >= 2:
            G2 = RS.GraphifyNS(G2, conf.believerSusceptibleCorr, conf.beta_list, actual_weights)

        A2 = nx.adjacency_matrix(G2)
        A2 = A2.astype(float)
        cls_rnd = auxFunctions.getClusteringCoeff(G2)
        print(f"random clustering: {cls_rnd}, d/n: {degrees / number_people}")
        gam_e, gam_v = eigsh(A2, 5, which='LM')  # eigh(A2.toarray())  #
        gam_e = abs(gam_e)

    if _structGraphMode:
        lines = [
            f"\nclustered graph, size: {number_people}first eig: {clust_e[1]} second eig:{clust_e[0]} spectral gap: {clust_e[1]-clust_e[0]}"
            f"\ngamma fitted graph first eig: {gam_e[1]} second eig: {gam_e[0]} spectral gap: {gam_e[1] - gam_e[0]}"
            f"\n average clustering coefficient structure: {cls_str}, average clustering coef random: {cls_rnd}"
        ]
        f = open('spectrum.txt','a')
        f.writelines(lines)
        f.close()
        Row = f"\n{number_people}, {np.flip(clust_e)},{clust_e[1]-clust_e[0]},{np.flip(gam_e)},{gam_e[1] - gam_e[0]}"
        with open('spectrum.csv', 'a') as f:
            f.write(Row)
        if plot:
            plotting.GraphDegreePlot(gam_e,clust_e)

    # frequency of calculations - number of iterations per day
    freq = conf.freq
    # infection probability, transition to E
    beta = conf.beta / np.mean(degrees)  # (2 * np.mean(degrees) * avg_weights)
    # / (np.mean(degrees)*avg_weights) # np.sqrt(3.4) / avg_weights     0.00125
    # E to I
    alpha = conf.alpha
    if alpha == 0:
        alpha = np.inf
    # I to R
    gamma = conf.gamma
    gammaH = conf.gamma_hospital
    # prob of quarantining upon infection
    q = conf.quarantine
    # quarantining period
    q_t = conf.quarantine_time
    # part of population
    part_infected = conf.part_infected
    # init_inf = {num_families, num_families + 1}
    num_iter = conf.num_iter
    if _structGraphMode:
        # Model selection
        model = SEIRyoel.SEIRQModel(G)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', 1/beta * freq)
    cfg.add_model_parameter('gamma', 1/gamma * freq)
    cfg.add_model_parameter('alpha', 1/alpha * freq)
    cfg.add_model_parameter('hosp_rate', 1/gammaH * freq)
    cfg.add_model_parameter('prob_h', conf.prob_hosp)
    cfg.add_model_parameter('prob_d', conf.prob_dead)
    # cfg.add_model_parameter('Infected', init_inf)
    cfg.add_model_parameter('quarantine', q)
    cfg.add_model_parameter('q_time', q_t * freq)
    cfg.add_model_parameter("fraction_infected", part_infected)
    cfg.add_model_parameter("is_sensitive", True)
    cfg.add_model_parameter("is_believer", True)
    cfg.add_model_parameter("sim_out", conf.output)

    if _structGraphMode:
        model.set_initial_status(cfg)

    if randomGraphMode:
        model2 = SEIRyoel.SEIRQModel(G2)
        if randomGraphMode < 2:
            cfg.add_model_parameter("is_sensitive", False)
            cfg.add_model_parameter("is_believer", False)

        cfg.add_model_parameter("sim_out", conf.output+" rnd")
        model2.set_initial_status(cfg)

    # use my model for iteration
    # model.iteration = SEIRyoel.iteration
    # model2.iteration = SEIRyoel.iteration

    # Simulation execution - model 1
    if _structGraphMode:
        print("starting structured graph\n")

        iterations = model.yoel_iteration_bunch(num_iter, node_status=plot)

        trends = model.build_trends(iterations)
        T1 = np.asarray(list(trends[0]['trends']['node_count'].values()))
        maximun_inf = np.max(T1[2]) / number_people

        if True:  # plot
            (Rvalid, R0_growth, R0_ratio, TtoMax, Qvar) = plotting.retrieveR0FromIter(iterations, T1, freq, plot)
        else:
            Rvalid, R0_growth, R0_ratio, TtoMax, Qvar = 0, 0, 0, 0, 0

        R0str = f"Graph simulation R direct: {Rvalid} Graph Growth rate: {R0_growth} Graph R ratio: {R0_ratio}"
        maxInfStr = f"Graph max inf: {maximun_inf} Graph maxTime: {TtoMax}"
        print(R0str)
        # viz.trends()
        print("finished structured graph\n")

    if randomGraphMode:
        # simulation - model 2 - random model
        print("starting random graph\n")
        iterations2 = model2.yoel_iteration_bunch(num_iter, node_status=plot)
        print("finished random graph\n")

        trends2 = model2.build_trends(iterations2)

        T2 = np.asarray(list(trends2[0]['trends']['node_count'].values()))

        if True:  # plot
            (RvalidRandom, R0_growthRandom, R0_ratioRandom, TtoMax2, QvarRnd) = plotting.retrieveR0FromIter(iterations2, T2, freq, plot)
        else:
            RvalidRandom, R0_growthRandom, R0_ratioRandom, TtoMax2, QvarRnd = 0, 0, 0, 0
        R0strRandom = f"Dreg simulation R direct: {RvalidRandom} Dreg Growth rate: {R0_growthRandom} Dreg R ratio: {R0_ratioRandom}"
        print(R0strRandom)

        maximun_inf2 = np.max(T2[2]) / number_people

        maxInfStr2 = f"Dreg max inf: {maximun_inf2} Dreg maxTime: {TtoMax2}"
        if plot and _structGraphMode:
            print("plotting SEIR results")
            plotting.BokehPlotSEIR(model, trends, model2, trends2, conf.output)

    if bool(randomGraphMode) != bool(_structGraphMode) and plot:
        if _structGraphMode:
            plotting.BokehPlotOne(model, trends, conf.output)
        if randomGraphMode:
            plotting.BokehPlotOne(model2, trends2, conf.output)

    # output stage
    if conf.save:
        localtime = time.asctime(time.localtime(time.time()))
        towrite = []
        if bool(_structGraphMode):
            np.savetxt(
                "israel population graph"+conf.output+".csv",
                T1, fmt='%.13e', delimiter=",")

            towrite.append(["\nLocal current time :" + str(localtime),
                       f"\nmean, var of degree dist. : {np.mean(degrees)}, {np.std(degrees)}",
                       f"\npeople: {number_people}",
                       f"\nclass size: {conf.size_school} office size: {conf.size_work}",
                       f"\nbeta S->E: {conf.beta} gamma E E->I: {alpha} gamma R I->R: {gamma} iter per day: {freq}"
                       f"\nlockdown: {conf.lockdown}"
                       f"\nmean weight: {avg_weights} fraction initially infected: {part_infected}\n",
                       R0str, "\n",
                       maxInfStr,"\n",
                       R0strRandom, "\n",
                       maxInfStr2, "\n",
                       f"Qvar structured: {Qvar}",
                        f""
                       ])

        if randomGraphMode:
            if not bool(_structGraphMode):
                towrite.append(["\nLocal current time :" + str(localtime),
                      f"\nmean, var of degree dist. : {np.mean(degrees)}, {np.std(degrees)}",
                      f"\npeople: {number_people}",
                      R0strRandom, "\n",
                      maxInfStr2, "\n",
                      f"Qvar random: {QvarRnd}"
                      ])
                towrite = ''.join(towrite[0])
            else:
                towrite = ''.join(towrite[0])

            print(os.getcwd() + "random graph" + conf.output + ".csv")
            np.savetxt("random graph" + conf.output + ".csv", T2, fmt='%.13e', delimiter=",")

        f = open("run details.txt", "a")
        f.writelines(towrite)
        f.close()
        f = open(conf.output + ".txt", "w")
        f.writelines(towrite)
        f.close()

    print("simulation complete. goodbye")
