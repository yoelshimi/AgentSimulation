import argparse
from run import NetConfig
from network import go
import networkx
import network

if __name__ == "__main__":
    pars = argparse.ArgumentParser(description='input for simulation')
    pars.add_argument('-s','--save',default=False,type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=False, help='Save output from simulation')
    pars.add_argument('-n','--num_families',default=100,type=int, required=True, help='Number of families for simulation')
    pars.add_argument('-p','--is_plot',default=True,type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True, help='plot graphs from simulation? [0/1]')
    pars.add_argument('-l','--lockdown',default=1,type=float, required=False, help='Lockdown implemented weight')
    pars.add_argument('-r_c','--random_connections',default=1.5,type=float, required=False, help='number of random connections')
    pars.add_argument('-s_c','--school_connections',default=5.3,type=float, required=False, help='number of random connections')
    pars.add_argument('-fac','--factor',default=1,type=float, required=False, help='factor to multiply weights')
    pars.add_argument('-freq','--frequency',default=24*10,type=int, required=False, help='calculation frequency [calcs/day]')
    pars.add_argument('-b','--beta',default=1,type=float, required=False, help='S->E infection exponent rate [days]')
    pars.add_argument('-a','--alpha',default=3,type=float, required=False, help='E->I latency exponent rate [days]')
    pars.add_argument('-g','--gamma',default=4, type=float, required=False, help='I->R healing/death exponent rate [days]')
    pars.add_argument('-q','--quarantine',default=0, type=float, required=False, help='binomial probability of being quarantined [0-1]')
    pars.add_argument('-q_t','--quarantine_time',default=14, type=float, required=False, help='time period of quarantine [days]')
    pars.add_argument('-p_i','--part_inf',default=1/100, type=float, required=False, help='part of population initially infected [0-1]')
    pars.add_argument('-n_i','--num_iter',default=60, type=int, required=False, help='number of iterations [days]')
    #  pars.add_argument('-i','--input',type=str, required=False, help='input names for simulation')
    pars.add_argument('-o','--output',default="simulation output", type=str, required=False, help='output filename for simulation')
    args = pars.parse_args()
    print(vars(args))
    [print(f"{k}: {v}") for (k,v) in zip(vars(args).keys(),vars(args).values())]
