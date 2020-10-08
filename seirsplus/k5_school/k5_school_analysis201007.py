import sys
import networkx
import numpy as np
import pandas as pd

# Probably should not use so many import *
from extended_models import *
from models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *
from repeated_loops.merge_summarize_webapp_runs import *

## Build a helper function

def set_params_k5_schools():
    # Note, when running in ipython, need to do this outside of fxn for scope reasons, when
    # running as script should also do the same
    # Network parameters:
    num_grades=6
    num_classrooms_per_grade=4
    class_sizes=20
    student_household_connections=True,
    num_staff=24
    num_teacher_staff_communities=3
    teacher_staff_degree=5

    N = num_grades * num_classrooms_per_grade * class_sizes + num_staff + num_grades*num_classrooms_per_grade

    # Set parameters for non-school comparison
    num_cohorts = num_grades*num_classrooms_per_grade
    num_nodes_per_cohort = int(N/num_cohorts)
    number_teams_per_cohort = 1

    pct_contacts_intercohort = 0.2
    isolation_time=14
    q = 0

    INIT_EXPOSED = 1
    R0_MEAN = 2.0
    R0_COEFFVAR_HIGH = 2.2
    R0_COEFFVAR_LOW = 0.15
    P_GLOBALINTXN = 0.2
    MAX_TIME = 365
    repeats = 100

    PERCENT_ASYMPTOMATIC = 0.3
    STUDENT_ASYMPTOMATIC_RATE = 0.8
    STUDENT_SUSCEPTIBILITY = 0.5

    BETA_PAIRWISE_MODE  = 'infected'

def repeat_runs_schools(n_repeats, simulation_fxn, save_escalation_time = False, student_susc = 1.0):
    """
    A wrapper for repeating the runs, that takes a simulation function defined above.

    NOTE - most of these parameters are defined outside the function.
    """
    output_frames = []
    model_overview = []
    for i in np.arange(0, n_repeats):
        (G_baseline,
         grades_studentIDs,
         classrooms_studentIDs,
         classrooms_teacherIDs,
         node_labels) = generate_K5_school_contact_network(num_grades=num_grades,
                                                            num_classrooms_per_grade=num_classrooms_per_grade,
                                                            class_sizes=class_sizes,
                                                            student_household_connections=student_household_connections,
                                                            num_staff=num_staff,
                                                            num_teacher_staff_communities=num_teacher_staff_communities,
                                                            teacher_staff_degree=teacher_staff_degree)
        G_quarantine = networkx.classes.function.create_empty_copy(G_baseline)
        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = np.random.uniform(R0_COEFFVAR_LOW, R0_COEFFVAR_HIGH))

        # For schools, adjust asymptomatic percentage and susceptibility for students:
        PCT_ASYMPTOMATIC = [ STUDENT_ASYMPTOMATIC_RATE if label=="student" else PERCENT_ASYMPTOMATIC for label in node_labels]
        ALPHA = [ student_susc if label=="student" else 1.0 for label in node_labels]

        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                        beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                        gamma_asym=GAMMA,
                                        a=PCT_ASYMPTOMATIC, alpha=ALPHA,
                                        beta_pairwise_mode = BETA_PAIRWISE_MODE,
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

def repeat_runs_basic(n_repeats, simulation_fxn, save_escalation_time = False):
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
                                        gamma_asym=GAMMA, a=PERCENT_ASYMPTOMATIC,
                                        beta_pairwise_mode = BETA_PAIRWISE_MODE,
                                        G_Q=G_quarantine, q=q, beta_Q=BETA_Q, isolation_time=isolation_time,
                                        initE=INIT_EXPOSED, seed = i)
        total_tests, total_intros, cadence_changes, new_intros = simulation_fxn(model, MAX_TIME)

        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        if save_escalation_time:
            print(escalation_time)
            thisout['escalation_time'] = escalation_time
            thisout['escalation_from_screen'] = escalation_from_screen
        output_frames.append(thisout)
    return(pd.concat(output_frames))

def weekly_testing(model, time):
    """
    A simulation with weekly testing
    """
    return run_rtw_adaptive_testing(model=model, T=time, initial_days_between_tests=7, max_dt=0.99, full_time=False)

def no_testing(model, time):
    """
    A simulation with weekly testing
    """
    return run_rtw_adaptive_testing(model=model, T=time, initial_days_between_tests=0, max_dt=0.99, full_time=False)


def main():
    schools_baseline = repeat_runs_schools(repeats, no_testing, student_susc=0.5)
    schools_baseline_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_baseline.csv'
    schools_baseline.to_csv(schools_baseline_file)

    general_baseline = repeat_runs_basic(repeats, no_testing)
    general_baseline_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/general_baseline.csv'
    general_baseline.to_csv(general_baseline_file)

    schools_weekly = repeat_runs_schools(repeats, weekly_testing, student_susc=0.5)
    schools_weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_weekly.csv'
    schools_weekly.to_csv(schools_weekly_file)

    general_weekly = repeat_runs_basic(repeats, weekly_testing)
    general_weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/general_weekly.csv'
    general_weekly.to_csv(general_weekly_file)

    STUDENT_SUSCEPTIBILITY = 1.0 # Let's make the students more susceptible!
    schools_baseline_susc = repeat_runs_schools(repeats, no_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_baseline_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_baseline_susc.csv'
    schools_baseline_susc.to_csv(schools_baseline_susc_file)

    schools_weekly_susc = repeat_runs_schools(repeats, weekly_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_weekly_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_weekly_susc.csv'
    schools_weekly_susc.to_csv(schools_weekly_susc_file)

    STUDENT_SUSCEPTIBILITY = 0.75 # Let's make the students somewhat more susceptible!
    schools_baseline_half_susc = repeat_runs_schools(repeats, no_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_baseline_half_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_baseline_half_susc.csv'
    schools_baseline_half_susc.to_csv(schools_baseline_half_susc_file)

    schools_weekly_half_susc = repeat_runs_schools(repeats, weekly_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_weekly_half_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_weekly_half_susc.csv'
    schools_weekly_half_susc.to_csv(schools_weekly_half_susc_file)


def school_ecdf():
    all_files = [schools_baseline_file, general_baseline_file, schools_weekly_file, general_weekly_file,
        schools_baseline_susc_file, schools_weekly_susc_file]
    cadence_file_map = {schools_baseline_file:'School_No_Testing', general_baseline_file: 'No_Testing',
                        schools_weekly_file:'School_Weekly_Testing',
                        general_weekly_file:'Weekly_Testing',
                        schools_baseline_half_susc_file: 'School_No_Testing_Kids_HSusc',
                        schools_weekly_half_susc_file: 'School_Weekly_Kids_HSusc',
                        schools_baseline_susc_file:'School_No_Testing_Kids_Susc',
                        schools_weekly_susc_file:'School_Weekly_Testing_Kids_Susc'}

    overall_frames = []
    ecdf_frames = []
    qei_ecdf_frames = []

    for x in all_files:
        # capture_hash = re.search(HASH_RE, x).group(1)


        results_frame = pd.read_csv(x)
        param_dict = { 'cadence' : cadence_file_map[x],
        'tat' : 1,
        'intro_rate' : 0,
        'r0': 2,
        'pop_size': 528
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
    ecdf_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_ecdf201008.csv', index=False)
    overall_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_overall201008.csv', index=False)
    qei_all.to_csv('/Users/julianhomburger/Data/covid/seirsplus/schools_201008/schools_qei200108.csv', index=False)
