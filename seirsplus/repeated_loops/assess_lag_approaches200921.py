import sys
import networkx
import numpy as np
import pandas as pd

# Probably should not use so many import *
from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *
from repeated_loops.merge_summarize_webapp_runs import *

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
        total_tests, total_intros, cadence_changes, new_intros = simulation_fxn(model, MAX_TIME)

        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        thisout['total_intros'] = total_intros
        thisout['cadence_changes'] = thisout['time'].isin([int(a) for a in cadence_changes])
        thisout['new_intros'] = thisout['time'].isin([int(a) for a in new_intros])

        output_frames.append(thisout)
    return(pd.concat(output_frames))

def weekly_simulation_monthly_intro(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, initial_days_between_tests=7, average_introductions_per_day=1/30, max_day_for_introductions = MAX_INTRO_TIME)

def weekly_simulation_monthly_intro_maxdt(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, initial_days_between_tests=7, average_introductions_per_day=1/30, max_dt=1, max_day_for_introductions = MAX_INTRO_TIME)

def weekly_simulation_monthly_intro_backfill(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, initial_days_between_tests=7, average_introductions_per_day=1/30, backlog_skipped_intervals=True, max_day_for_introductions = MAX_INTRO_TIME)


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
    MAX_INTRO_TIME = 183
    MAX_TIME = 183




def main():
    repeats = 1000

    weekly = repeat_runs_deescalate(repeats, weekly_simulation_monthly_intro)
    weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/200921/weekly_testing.csv'
    weekly.to_csv(weekly_file)

    weekly_maxdt = repeat_runs_deescalate(repeats, weekly_simulation_monthly_intro_maxdt)
    weekly_maxdt_file = '/Users/julianhomburger/Data/covid/seirsplus/200921/weekly_testing_maxdt.csv'
    weekly_maxdt.to_csv(weekly_maxdt_file)

    weekly_backfill = repeat_runs_deescalate(repeats, weekly_simulation_monthly_intro_backfill)
    weekly_backfill_file = '/Users/julianhomburger/Data/covid/seirsplus/200921/weekly_testing_backfill.csv'
    weekly_backfill.to_csv(weekly_backfill_file)

def convert_to_ecdf_tables():
    all_files = [weekly_file, weekly_maxdt_file, weekly_backfill_file ]
    cadence_file_map = {weekly_maxdt_file:'max_dt', weekly_file:'weekly', weekly_backfill_file:'backfill'}

    overall_frames = []
    ecdf_frames = []
    qei_ecdf_frames = []

    for x in all_files:
        # capture_hash = re.search(HASH_RE, x).group(1)


        results_frame = pd.read_csv(x)
        param_dict = { 'cadence' : cadence_file_map[x],
        'tat' : 1,
        'intro_rate' : 30,
        'r0': 2,
        'pop_size': 1000
        }
        overall_frame, ecdf_frame, qei_ecdf_frame = get_aggregate_frame(results_frame, int(param_dict['pop_size']))
        assign_new_cols(overall_frame, param_dict)
        assign_new_cols(ecdf_frame, param_dict)
        assign_new_cols(qei_ecdf_frame, param_dict)

        # file_prefix = f'{cadence}_{pop_size}_{r0}_{tat}_{intro_rate}'
        # overall_frame.to_csv(f'{file_prefix}_overall.csv', index=False, header=False)
        # ecdf_frame.to_csv(f'{file_prefix}_ecdf.csv', index=False, header=False)
        overall_frames.append(overall_frame)
        ecdf_frames.append(ecdf_frame)
        qei_ecdf_frames.append(qei_ecdf_frame)

    ecdf_all = pd.concat(ecdf_frames)
    overall_all = pd.concat(overall_frames)
    qei_all = pd.concat(qei_ecdf_frames)
    ecdf_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200921/lag_ecdf200921.csv', index=False)
    overall_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200921/overall_ecdf200921.csv', index=False)
    qei_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200921/qei_ecdf200921.csv', index=False)
