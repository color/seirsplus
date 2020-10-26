import sys
import networkx
import numpy as np
import pandas as pd

# Probably should not use so many import *
from models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *
from repeated_loops.merge_summarize_webapp_runs import *
import extended_models as ext
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

    # Set parameters for well-mixed population comparison
    num_cohorts = 1
    number_teams_per_cohort = 1
    num_nodes_per_cohort = N
    pct_contacts_intercohort = 0

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

def repeat_runs_basic(n_repeats, simulation_fxn, save_escalation_time = False, set_beta=False, r0_var_low = 0.15, r0_var_high = 2.2):
    """
    A wrapper for repeating the runs, that takes a simulation function defined above.

    NOTE - most of these parameters are defined outside the function.
    """
    output_frames = []
    model_overview = []
    for i in np.arange(0, n_repeats):
        G_baseline, G_quarantine, cohorts, teams = build_farz_graph(num_cohorts = num_cohorts, num_nodes_per_cohort = num_nodes_per_cohort, num_teams_per_cohort = number_teams_per_cohort, pct_contacts_intercohort = pct_contacts_intercohort)

        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = np.random.uniform(R0_COEFFVAR_LOW, R0_COEFFVAR_HIGH))
        r0_coeffvar = np.random.uniform(r0_var_low, r0_var_high)
        print(r0_coeffvar)
        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = r0_coeffvar)

        if set_beta:
            BETA= R0_MEAN/(6.2)
            BETA_Q = R0_MEAN/(6.2)

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

def no_testing_longdt(model, time):
    """
    A simulation with weekly testing
    """
    return run_rtw_adaptive_testing(model=model, T=time, initial_days_between_tests=0, max_dt=100, full_time=False)

def school_temporal_iso_no_testing(model, time, node_labels):
    """
    Simulation of a week of school then no school
    """
    temp_iso_group = [True if label=="student" else False for label in node_labels]
    return run_rtw_adaptive_testing(model=model, T=time, initial_days_between_tests=0, max_dt=0.99, full_time=False, temporal_quarantine=True, temporal_quarantine_nodes = temp_iso_group)

def school_temporal_iso_biweekly_return_testing(model, time, node_labels):
    """
    Simulation of a week of school then no school
    """
    temp_iso_group = [True if label=="student" else False for label in node_labels]
    return run_rtw_adaptive_testing(model=model, T=time, initial_days_between_tests=0, max_dt=0.99, full_time=False, temporal_quarantine=True, temporal_quarantine_nodes = temp_iso_group, cadence_testing_days=[0,14])


def main():
    output_dir = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/'
    schools_baseline = repeat_runs_schools(repeats, no_testing, student_susc=0.5)
    schools_baseline_file = output_dir + 'schools_baseline.csv'
    schools_baseline.to_csv(schools_baseline_file)

    general_baseline = repeat_runs_basic(repeats, no_testing)
    general_baseline_file = output_dir + 'general_baseline.csv'
    general_baseline.to_csv(general_baseline_file)

    # general_baseline_long_dt = repeat_runs_basic(repeats, no_testing_longdt)
    # general_baseline_long_dt_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/general_long_dt_baseline.csv'
    # general_baseline_long_dt.to_csv(general_baseline_long_dt_file)


    # schools_weekly = repeat_runs_schools(repeats, weekly_testing, student_susc=0.5)
    # schools_weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/schools_weekly.csv'
    # schools_weekly.to_csv(schools_weekly_file)

    # general_weekly = repeat_runs_basic(repeats, weekly_testing)
    # general_weekly_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/general_weekly.csv'
    # general_weekly.to_csv(general_weekly_file)

    STUDENT_SUSCEPTIBILITY = 1.0 # Let's make the students more susceptible!
    schools_baseline_susc = repeat_runs_schools(repeats, no_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_baseline_susc_file = output_dir + 'schools_baseline_susc.csv'
    schools_baseline_susc.to_csv(schools_baseline_susc_file)

    # schools_weekly_susc = repeat_runs_schools(repeats, weekly_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    # schools_weekly_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/schools_weekly_susc.csv'
    # schools_weekly_susc.to_csv(schools_weekly_susc_file)

    STUDENT_SUSCEPTIBILITY = 0.75 # Let's make the students somewhat more susceptible!
    schools_baseline_half_susc = repeat_runs_schools(repeats, no_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    schools_baseline_half_susc_file = output_dir + 'schools_baseline_half_susc.csv'
    schools_baseline_half_susc.to_csv(schools_baseline_half_susc_file)

    # schools_weekly_half_susc = repeat_runs_schools(repeats, weekly_testing, student_susc=STUDENT_SUSCEPTIBILITY)
    # schools_weekly_half_susc_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/schools_weekly_half_susc.csv'
    # schools_weekly_half_susc.to_csv(schools_weekly_half_susc_file)

    schools_baseline_25_susc = repeat_runs_schools(repeats, no_testing, student_susc=0.25)
    schools_baseline_25_susc_file = output_dir + 'schools_baseline_25_susc.csv'
    schools_baseline_25_susc.to_csv(schools_baseline_25_susc_file)

    schools_baseline_0_susc = repeat_runs_schools(repeats, no_testing, student_susc=0.0)
    schools_baseline_0_susc_file = output_dir + 'schools_baseline_0_susc.csv'
    schools_baseline_0_susc.to_csv(schools_baseline_0_susc_file)

    schools_baseline_10_susc = repeat_runs_schools(repeats, no_testing, student_susc=0.10)
    schools_baseline_10_susc_file = output_dir + 'schools_baseline_10_susc.csv'
    schools_baseline_10_susc.to_csv(schools_baseline_10_susc_file)

    P_GLOBALINTXN = 0.0
    schools_baseline_0_susc_ng = repeat_runs_schools(repeats, no_testing, student_susc=0.0)
    schools_baseline_0_susc_ng_file = output_dir + 'schools_baseline_0_susc_ng.csv'
    schools_baseline_0_susc_ng.to_csv(schools_baseline_0_susc_ng_file)


    make_ecdf({
    schools_baseline_0_susc_file: 'school_susc0',
    schools_baseline_10_susc_file: 'school_susc10',
    schools_baseline_25_susc_file: 'school_susc25',
    schools_baseline_file: 'school_susc50',
    schools_baseline_half_susc_file:'school_susc75',
    schools_baseline_susc_file: 'school_susc100',
    general_baseline_file: 'no_structure',
    schools_baseline_0_susc_ng_file: 'school_susc0_noglobal'
    },
    output_dir + 'school_susc_compare201014', N)

def baseline_no_structure():
    num_cohorts = 1
    number_teams_per_cohort = 1
    num_nodes_per_cohort = 528
    pct_contacts_intercohort = 0
    baseline_no_struct = repeat_runs_basic(repeats, no_testing)
    baseline_no_struct_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/baseline_no_struct.csv'
    baseline_no_struct.to_csv(baseline_no_struct_file)

def repeat_runs_schools_beta(n_repeats, simulation_fxn, save_escalation_time = False, student_susc = 1.0, set_beta=False, r0_var_low = 0.15, r0_var_high = 2.2, send_labels=False):
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
        r0_coeffvar = np.random.uniform(r0_var_low, r0_var_high)
        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = r0_coeffvar)
        if set_beta:
            BETA= R0_MEAN/(6.2)
            BETA_Q = R0_MEAN/(6.2)
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
        if send_labels:
            total_tests, total_intros, cadence_changes, new_intros = simulation_fxn(model, MAX_TIME, node_labels=node_labels)
        else:
            total_tests, total_intros, cadence_changes, new_intros = simulation_fxn(model, MAX_TIME)

        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        thisout['total_intros'] = total_intros
        thisout['mean_beta'] = np.mean(BETA)
        thisout['sd_beta'] = np.std(BETA)
        thisout['cadence_changes'] = thisout['time'].isin([int(a) for a in cadence_changes])
        thisout['new_intros'] = thisout['time'].isin([int(a) for a in new_intros])

        output_frames.append(thisout)
    return(pd.concat(output_frames))

def compare_beta_dists():
    output_dir = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/'

    constant_beta_no_test = repeat_runs_schools_beta(repeats, no_testing, set_beta=True)
    constant_beta_no_test_file = output_dir + 'schools_baseline_constant_beta.csv'
    constant_beta_no_test.to_csv(constant_beta_no_test_file)

    # well_mixed_pop = repeat_runs_basic(repeats, no_testing)
    # well_mixed_pop_file = '/Users/julianhomburger/Data/covid/seirsplus/schools_201014/well_mixed_pop.csv'
    # well_mixed_pop.to_csv(well_mixed_pop_file)

    similar_beta_no_test = repeat_runs_schools_beta(repeats, no_testing, set_beta=False, r0_var_high=0.2)
    similar_beta_no_test_file = output_dir + 'school_similar_beta.csv'
    similar_beta_no_test.to_csv(similar_beta_no_test_file)

    vary_beta_no_test = repeat_runs_schools_beta(repeats, no_testing, set_beta=False, r0_var_low=1.8)
    vary_beta_no_test_file = output_dir + 'school_vary_beta.csv'
    vary_beta_no_test.to_csv(vary_beta_no_test_file)

    general_constant_beta = repeat_runs_basic(repeats, no_testing, set_beta=True)
    general_constant_beta_file = output_dir + 'general_constant_beta.csv'
    general_constant_beta.to_csv(general_constant_beta_file)

    general_vary_beta = repeat_runs_basic(repeats, no_testing, r0_var_low=1.8)
    general_vary_beta_file = output_dir + 'general_vary_beta'
    general_vary_beta.to_csv(general_vary_beta_file)

    general_similar_beta = repeat_runs_basic(repeats, no_testing, r0_var_high=0.2)
    general_similar_beta_file = output_dir + 'general_sim_beta'
    general_similar_beta.to_csv(general_similar_beta_file)

    make_ecdf({constant_beta_no_test_file : 'school_susc100_constant_beta',
                similar_beta_no_test_file : 'school_susc100_similar_beta',
                vary_beta_no_test_file : 'school_susc100_vary_beta',
                schools_baseline_susc_file: 'school_susc100',
                general_baseline_file: 'no_structure',
                general_constant_beta_file: 'no_structure_constant_beta',
                general_vary_beta_file: 'no_structure_vary_beta',
                general_similar_beta_file: 'no_structure_similar_beta',
    }, output_dir + 'vary_beta201014', 528)

def compare_temp_strategy():
    output_dir = '/Users/julianhomburger/Data/covid/seirsplus/schools_201022/'

    schools_baseline = repeat_runs_schools(repeats, no_testing, student_susc=1.0)
    schools_baseline_file = output_dir + 'schools_baseline_susc100.csv'
    schools_baseline.to_csv(schools_baseline_file)

    general_baseline = repeat_runs_basic(repeats, no_testing)
    general_baseline_file = output_dir + 'general_baseline.csv'
    general_baseline.to_csv(general_baseline_file)

    schools_temporal = repeat_runs_schools_beta(repeats, school_temporal_iso_no_testing, send_labels=True)
    schools_temporal_file = output_dir + 'schools_temporal_q.csv'
    schools_temporal.to_csv(schools_temporal_file)

    schools_temporal_return_test = repeat_runs_schools_beta(repeats, school_temporal_iso_biweekly_return_testing, send_labels=True)
    schools_temporal_return_test_file = output_dir + 'schools_temporal_q_biweekly_test.csv'
    schools_temporal_return_test.to_csv(schools_temporal_return_test_file)

    make_ecdf({
        schools_baseline_file: 'schools_baseline_susc100',
        general_baseline_file: 'no_structure',
        schools_temporal_file: 'school_week_on_off',
        schools_temporal_return_test_file: 'school_week_on_off_return_test'
    }, output_dir + 'schools_temp_q_test', 528)
