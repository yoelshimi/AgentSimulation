import network
import numpy as np
import distributions
import cProfile


class NetConfig:
    def __init__(self, p, s, n, f_d, p_d, m_p, e, size_w, size_s, r_c, s_c, lckn, f, s_w, w_w, f_w, r_w, frq, b, a, g,
                 q, q_t, p_i, i, o, b_l, g_h, p_h_l, p_d_l, sbc_l, rng, add, stg):
        self.save = s
        self.num_families = n
        self.family_dist = f_d
        self.population_dist = p_d
        self.mean_parents = m_p
        self.employment = e
        self.size_work = size_w
        self.size_school = size_s
        self.random_connections = r_c
        self.school_connections = s_c
        self.lockdown = lckn
        self.factor = f
        self.school_w = s_w
        self.work_w = w_w
        self.family_w = f_w
        self.random_w = r_w
        self.freq = frq
        self.beta = b
        self.alpha = a
        self.gamma = g
        self.quarantine = q
        self.quarantine_time = q_t
        self.part_infected = p_i
        self.num_iter = i
        self.plot = p
        self.output = o
        self.beta_list = b_l
        self.gamma_hospital = g_h
        self.prob_hosp = p_h_l
        self.prob_dead = p_d_l
        self.believerSusceptibleCorr = sbc_l
        self.randomGraphMode = rng
        self.GMLAdress = add
        self.structuredGraphMode = stg


if __name__ == "__main__":
    save = 1
    num_families_list = [10000] # [100*np.power(2, i) for i in range(0,11)]
    # size_family_kids = np.asarray([55, 13.1850, 14.2200, 9.8550, 4.0950, 3.6450])
    # size_family_parents = np.asarray([0, 11.5, 53.4, 18, 11, 6.5])


    mean_parents = 2.1
    employment = 0.9

    size_work = 5

    size_school = 18
    # number_schools = 1 int(np.ceil(1.1 * number_families * mean_dist(size_family_kids)/size_school + 2))

    random_connections = 1.5
    school_connections = 5.3

    #  beta_general = 0.75  # rate: number of new infectants per infectee per day.  # 3.76
    lockdown = 1
    factor = 2.4

    R0 = 3  # new  infected per person sick (overall, before recovery)

    freq = 24*6

    mean_weight_deg = 14.45

    # E to I
    alpha = 3
    # I to R
    gamma = 4
    # infection probability, transition to E
    #  infection rate = R0 * 1/gamma_R * 1/(contacts=degree*intensity=time) * 1/freq=calc_density
    beta = 1  #  1 * mean_weight_deg # / (np.mean(degrees)*avg_weights) # np.sqrt(3.4) / avg_weights     0.00125 R0 *
    # probability of being quarantined upon infection until recovery
    quarantine = 0
    # quarantine time [days]
    q_time = 14
    # part of population
    part_infected = 1 / 100
    #  init_inf = {num_families, num_families + 1}
    num_iter = 60 * freq

    quarantine_list = np.linspace(0,1,11)

    lockdown_list = [1] #  np.arange(0, 1.1, 0.05)
    for num_families in num_families_list:
        for i in range(1):
            factor = 1  #2.4
            school_w = 1 * lockdown  # school_connections / size_school * 2 * lockdown / factor  # 2.3
            work_w = 1 * lockdown / factor  # 1.3
            family_w = 1 / factor  # 2.4
            random_w = 1 * lockdown / factor  # 1
            this_run = NetConfig(p=True,s=save, n=num_families, f_d=distributions.family_dist, p_d=distributions.population_age_dist,
                                 m_p=mean_parents, e=employment, size_w=size_work, size_s=size_school,
                                 r_c=random_connections, s_c=school_connections, lckn=lockdown, f=factor, s_w=school_w,
                                 w_w=work_w, f_w=family_w, r_w=random_w, frq=freq, b=beta, a=alpha, g=gamma, q=quarantine, q_t=q_time,
                                 p_i=part_infected, i=num_iter, o=f"beta={beta} size={num_families}")

            network.go(this_run)
