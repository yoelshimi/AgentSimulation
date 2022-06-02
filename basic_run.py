import argparse
from run import NetConfig
import distributions
import network
import pstats
import cProfile

if __name__ == "__main__":
    pars = argparse.ArgumentParser(description='input for simulation')
    pars.add_argument('-s','--save',default=True,type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=False, help='Save output from simulation?')
    pars.add_argument('-n','--num_families',default=100,type=int, required=True, help='Number of families for simulation')
    pars.add_argument('-p','--is_plot',default=True,type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True, help='plot graphs from simulation? [0/1]')
    pars.add_argument('-l','--lockdown',default=1,type=float, required=False, help='Lockdown implemented weight')
    pars.add_argument('-r_c','--random_connections',default=1.5,type=float, required=False, help='number of random connections')
    pars.add_argument('-s_c','--school_connections',default=5.3,type=float, required=False, help='number of school connections')
    pars.add_argument('-fac','--factor',default=1,type=float, required=False, help='factor to multiply weights')
    pars.add_argument('-sbc_l','--SBC',default=1,nargs='+',type=float, required=False, help='prob. of susceptible believer correlation')
    pars.add_argument('-f', '--frequency',default=24*10,type=int, required=False, help='calculation frequency [calcs/day]')
    pars.add_argument('-b', '--beta',default=1,type=float, required=False, help='S->E infection exponent rate [days]')
    pars.add_argument('-b_l', '--beta_list', nargs='+',type=float, required=False,default=[1,1,1], help='Believer-y,n,Believer infection factor')
    pars.add_argument('-a', '--alpha',default=3,type=float, required=False, help='E->I latency exponent rate [days]')
    pars.add_argument('-g', '--gamma',default=4, type=float, required=False, help='I->R healing/death exponent rate [days]')
    pars.add_argument('-g_h', '--gamma_hospital',default=0.2, type=float, required=False, help='H->R leaving hospital rate [days]')
    pars.add_argument('-p_h_l', '--hospital_list', nargs='+',default=[0.1, 0.2], type=float, required=False, help='probability of hospitalization susctible-NS')
    pars.add_argument('-p_d_l', '--death_list', nargs='+',default=[0.1, 0.2], type=float, required=False, help='probability of death susctible-NS')
    pars.add_argument('-q', '--quarantine',default=0, type=float, required=False, help='binomial probability of being quarantined [0-1]')
    pars.add_argument('-q_t', '--quarantine_time',default=14, type=float, required=False, help='time period of quarantine [days]')
    pars.add_argument('-p_i', '--part_infected',default=1/100, type=float, required=False, help='part of population initially infected [0-1]')
    pars.add_argument('-n_i', '--num_iter',default=80, type=int, required=False, help='number of iterations [days]')
    #  pars.add_argument('-i', '--input',type=str, required=False, help='input names for simulation')
    pars.add_argument('-o','--output',default="simulation output", type=str, required=False, help='output filename for simulation')
    pars.add_argument('-rng', '--RG_mode', default="off", type=str, required=False, help='mode of random graph')
    pars.add_argument('-stg', '--STRG_mode', default="on", type=str, required=False, help='mode of structured graph')
    pars.add_argument('-add', '--import_address', default="off", type=str, required=False, help='path and filename to GML graph file for import')

    args = pars.parse_args()
    print(vars(args))
    [print(f"{k}: {v}") for (k,v) in zip(vars(args).keys(),vars(args).values())]

    mean_parents = 1.8
    employment = 0.8
    size_work = 5
    size_school = 25
    lockdown = args.lockdown
    factor = args.factor
    school_w = 0.7 * lockdown  # school_connections / size_school * 2 * lockdown / factor  # 2.3
    work_w = 0.7 * lockdown / factor  #
    family_w = 2.5 / factor  # 2.4
    random_w = 0.4 * lockdown / factor
    num_iter = args.num_iter * args.frequency

    #  global sim_out
    sim_out = args.output

    config = NetConfig(p=args.is_plot,s=args.save, n=args.num_families, f_d=distributions.family_dist, p_d=distributions.population_age_dist,
                       m_p=mean_parents, e=employment, size_w=size_work, size_s=size_school,
                       r_c=args.random_connections, s_c=args.school_connections, lckn=lockdown, f=factor, s_w=school_w,
                       w_w=work_w, f_w=family_w, r_w=random_w, frq=args.frequency, b=args.beta, a=args.alpha,
                       g=args.gamma, q=args.quarantine, q_t=args.quarantine_time, b_l=args.beta_list, g_h=args.gamma_hospital,
                       p_h_l=args.hospital_list, p_d_l=args.death_list, sbc_l=args.SBC,
                       p_i=args.part_infected, i=num_iter, o=args.output, rng=args.RG_mode, stg=args.stg,
                       add=args.import_address)

    pf = False
    if pf:
        profiler = cProfile.Profile()
        profiler.enable()
        network.go(config)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
    else:
        network.go(config)
