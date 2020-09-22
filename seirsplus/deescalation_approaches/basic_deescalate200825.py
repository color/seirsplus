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
    return(pd.concat(output_frames), model)

def baseline_simulation_monthly_intro(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, average_introductions_per_day=1/30)

def weekly_simulation_monthly_intro(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, initial_days_between_tests=7, average_introductions_per_day=1/30)

def weekly_simulation_deescalate_monthly_intro(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    # Escalate if more than 5 positives in last 7 days
    cadence_changes = [GreaterThanAbsolutePositivesCadence(7,2,7), LessThanAbsolutePositivesCadence(14,0,7)]
    return run_rtw_adaptive_testing(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3,
                                    initial_days_between_tests=7, average_introductions_per_day=1/30,
                                    cadence_changes = cadence_changes)



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
    MAX_INTRO_TIME = 365
    MAX_TIME = 547

repeats = 200
def main():
    baseline = repeat_runs_deescalate(repeats, baseline_simulation_monthly_intro)
    baseline_file = '/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/baseline.csv'
    baseline.to_csv(baseline_file)
    weekly = repeat_runs_deescalate(repeats, weekly_simulation_monthly_intro)
    weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/weekly_testing.csv'
    weekly.to_csv(weekly_file)
    weekly_de = repeat_runs_deescalate(repeats, weekly_simulation_deescalate_monthly_intro)
    weekly_de_file = '/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/weekly_adaptive_testing.csv'
    weekly_de.to_csv(weekly_de_file)

def get_aggregate_frame(run_data, pop_size):
    run_data['qei'] = run_data['total_e'] + run_data['total_q'] + run_data['total_i']
    overall_infections = run_data.groupby(('seed')).agg({'overall_infections':'max', 'qei': 'max', 'total_tests':'max', 'total_intros':'max', 'time':'max'})
    ecdf_frame = ecdf_from_agg_frame(overall_infections, pop_size)
    qei_ecdf_frame = ecdf_from_agg_frame(overall_infections, pop_size, to_summarize='qei')
    return(overall_infections, ecdf_frame, qei_ecdf_frame)


def convert_to_ecdf_tables():
    all_files = [baseline_file, weekly_file, weekly_de_file]
    cadence_file_map = {baseline_file:'none', weekly_file:'weekly', weekly_de_file:'adaptive'}

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
    ecdf_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/adaptive_ecdf200901.csv', index=False)
    overall_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/overall_ecdf200901.csv', index=False)
    qei_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200828/adaptive/qei_ecdf200901.csv', index=False)
