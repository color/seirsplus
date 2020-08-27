import sys
import networkx
import numpy as np
import pandas as pd

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *

# Set the graph network parameters
DEFAULT_FARZ_PARAMS = {'alpha':    5.0,       # clustering param
             'gamma':    5.0,       # assortativity param
             'beta':     0.5,       # prob within community edges
             'r':        1,         # max num communities node can be part of
             'q':        0.0,       # probability of multi-community membership
             'phi':      10,        # community size similarity (1 gives power law community sizes)
             'b':        0,
             'epsilon':  1e-6,
             'directed': False,
             'weighted': False}

def repeat_runs_deescalate(n_repeats, simulation_fxn, save_escalation_time = False):
    """
    A wrapper for repeating the runs, that takes a simulation function defined above.

    NOTE - most of these parameters are defined outside the function.
    """
    output_frames = []
    model_overview = []
    for i in np.arange(0, n_repeats):
        G_baseline, G_quarantine, cohorts, teams = build_farz_graph(num_cohorts = num_cohorts, num_nodes_per_cohort = num_nodes_per_cohort, num_teams_per_cohort = number_teams_per_cohort, pct_contacts_intercohort = pct_contacts_intercohort)

        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = np.random.uniform(R0_COEFFVAR_LOW, R0_COEFFVAR_HIGH))
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                        beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                        gamma_asym=GAMMA,
                                        G_Q=G_quarantine, q=q, beta_Q=BETA_Q, isolation_time=isolation_time,
                                        initE=INIT_EXPOSED, seed = i)
        total_tests = simulation_fxn(model, MAX_TIME)

        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        output_frames.append(thisout)
    return(pd.concat(output_frames))

def baseline_simulation_de(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3)

def weekly_simulation_de(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, initial_days_between_tests=7)


def set_params_deescalate():
    num_cohorts = 10 # number of different groups
    number_teams_per_cohort = 5 # number of teams
    num_nodes_per_cohort = 100 # total number of people per group


    N = num_cohorts*num_nodes_per_cohort
    pct_contacts_intercohort = 0.2
    isolation_time=14
    q = 0

    INIT_EXPOSED = 1
    R0_MEAN = 2.0
    R0_COEFFVAR_HIGH = 2.2
    R0_COEFFVAR_LOW = 0.15
    P_GLOBALINTXN = 0.2
    MAX_TIME = 200

repeats = 10
def main():
    set_params_deescalate()
    baseline = repeat_runs_deescalate(repeats, baseline_simulation_de)
    weekly = repeat_runs_deescalate(repeats, baseline_simulation_de)
