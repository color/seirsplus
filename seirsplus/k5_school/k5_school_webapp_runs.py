import sys
import networkx
import numpy as np
import pandas as pd
import itertools
import os

import models as seir_models
import networks as seir_networks
import sim_loops as sim_loops
from helper_functions import *
import k5_school.k5_school_analysis201007 as k5_schools

from repeated_loops.merge_summarize_webapp_runs import *


def write_parallel_inputs(pfilename):
    # Writes a file with arguments for the parallel files
    r0_list = [2.5, 2.0, 1.5]
    student_susceptibility = [0.4, 0.6, 0.8, 1.0]
    introduction_rate = [0, 7, 28] # parameterized in mean days to next introduction
    student_schedule = ['All_5_days','AB_week_5_days', 'AB_day_5_days', 'All_4_days','AB_week_4_days', 'AB_day_4_days']
    quarantine_strategy = ['individual', 'group']
     # test turnaround times
    testing_cadence = ['none', 't_weekly_s_weekly', 't_weekly_s_monthly',
        't_weekly_s_none', 't_twiceweek_s_none', 't_twiceweek_s_twiceweek'] # TODO: update to match school testing cadences

    testing_cadence, introduction_rate, student_schedule, quaratine_strategy, r0

    pfile = open(pfilename, "w")
    for y in itertools.product(testing_cadence, introduction_rate, tats):
        c, ir, t = y
        pfile.write(f'{c},{ir},{t}\n')
    pfile.close()

def get_param_cadences(cadence_type, params_dict):
    """
    Params dict is a dictionary of the param to use where the key
    is the block type
    """
    param_cadences = {
        'AB_week_5_days': [params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all']],
        'AB_day_5_days': [params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],],
        'All_5_days': [params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all']],
        'AB_week_4_days': [params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block1'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block2'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all']],
        'AB_day_4_days': [params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['onsite-block2'],
                   params_dict['onsite-block1'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],],
        'All_4_days': [params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['onsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all'],
                   params_dict['offsite-all']],
    }
    return(param_cadences[cadence_type])


def get_testing_dictionary(testing_code):
    testing_dict = {
      'none': {'teacher':[], 'staff':[], 'student':[]},
      't_weekly_s_weekly': {'teacher':[0,7,14,21], 'staff':[0,7,14,21], 'student':[0,7,14,21]},
      't_weekly_s_monthly': {'teacher':[0,7,14,21], 'staff':[0,7,14,21], 'student':[0]},
      't_weekly_s_none': {'teacher':[0,7,14,21], 'staff':[0,7,14,21], 'student':[]},
      't_twiceweek_s_none': {'teacher':[0, 3, 7, 10, 14, 17, 21, 24], 'staff':[0, 3, 7, 10, 14, 17, 21, 24], 'student':[]},
      't_twiceweek_s_twiceweek': {'teacher':[0, 3, 7, 10, 14, 17, 21, 24], 'staff':[0, 3, 7, 10, 14, 17, 21, 24], 'student':[0, 3, 7, 10, 14, 17, 21, 24]},
    }



def main(n_repeats = 1000):
    testing_cadence, introduction_rate, student_susc, student_block_strategy, quaratine_strategy, r0 = sys.argv[1].split(',')

    # Send parameters to the testing loop
    output_frames = []
    model_overview = []

    R0_MEAN = r0
    R0_COEFFVAR_LOW = 0.15
    R0_COEFFVAR_HIGH = 2.2

    # Parameter set up for school networks
    num_grades=6
    num_classrooms_per_grade=4
    class_sizes=20
    student_household_connections=True,
    num_staff=24
    num_teacher_staff_communities=3
    teacher_staff_degree=5
    if student_block_strategy == 'none':
        num_student_blocks = 1
    else:
        num_student_blocks = 2

    # Total N
    N = num_grades * num_classrooms_per_grade * class_sizes + num_staff + num_grades*num_classrooms_per_grade

    isolation_time=14
    q = 0.0

    INIT_EXPOSED = 1
    P_GLOBALINTXN = 0.2 # Global interaction
    MAX_TIME = 365

    PERCENT_ASYMPTOMATIC = 0.3
    STUDENT_ASYMPTOMATIC_RATE = 0.45
    P_GLOBALINTXN = [0.2] * N
    BETA_PAIRWISE_MODE  = 'infected'

    for i in np.arange(0, n_repeats):
        (networks,
         grades_studentIDs,
         classrooms_studentIDs,
         classrooms_teacherIDs,
         studentIDs_studentBlocks,
         node_labels) = seir_networks.generate_K5_school_contact_network(num_grades=num_grades,
            num_classrooms_per_grade=num_classrooms_per_grade,
            class_sizes=class_sizes,
            num_student_blocks=num_student_blocks,
            block_by_household=True,
            connect_students_in_households=True,
            num_staff=num_staff,
            num_teacher_staff_communities=num_teacher_staff_communities,
            teacher_staff_degree=teacher_staff_degree)

        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = np.random.uniform(R0_COEFFVAR_LOW, R0_COEFFVAR_HIGH))

        # For schools, adjust asymptomatic percentage and susceptibility for students:
        PCT_ASYMPTOMATIC = [ STUDENT_ASYMPTOMATIC_RATE if label=="student" else PERCENT_ASYMPTOMATIC for label in node_labels]
        ALPHA = [ student_susc if label=="student" else 1.0 for label in node_labels]

        # Define the isolation groups
        iso_groups = []
        for classroom, studentIDs in classrooms_studentIDs.items():
            iso_groups.append( studentIDs + [classrooms_teacherIDs[classroom]] )

        # Define p parameter for different sets
        p_paramSets = {}
        p_paramSets['onsite-all']  = [1.0*p for p in P_GLOBALINTXN]
        p_paramSets['offsite-all'] = [0.0*p for p in P_GLOBALINTXN]

        for block in numpy.unique(list(studentIDs_studentBlocks.values())):
            p_paramSets['onsite-block'+str(block)]  = [0.0*p if (i in studentIDs_studentBlocks.keys() and studentIDs_studentBlocks[i]!=block) else 1.0*p for i,p in enumerate(P_GLOBALINTXN)]

        p_cadence = get_param_cadences(student_block_strategy, p_paramSets)
        network_cadence = get_param_cadences(student_block_strategy, networks)
        param_cadence_dict = {'G':network_cadence, 'p':p_cadence}
        group_testing_cadence = get_testing_dictionary(testing_cadence)
        model = seir_models.ExtSEIRSNetworkModel(G=networks['onsite-all'], p=P_GLOBALINTXN,
                                        beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                        gamma_asym=GAMMA,
                                        a=PCT_ASYMPTOMATIC, alpha=ALPHA,
                                        beta_pairwise_mode = BETA_PAIRWISE_MODE,
                                        q=q, beta_Q=BETA_Q, isolation_time=isolation_time,
                                        initE=INIT_EXPOSED, seed = i)
        initial_exposed_group = get_group_initE(model, node_labels)

        total_tests, total_intros, cadence_changes, new_intros = sim_loops.run_rtw_group_testing(
                    model,
                    T=MAX_TIME,
                    symptomatic_selfiso_compliance_rate = 0.2,
                    max_dt = 1.0,
                    node_labels=node_labels,
                    parameter_cadences = param_cadence_dict,
                    group_testing_days = group_testing_cadence,
                    isolation_compliance_positive_groupmate_rate = 1.0
                    )
        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        thisout['total_intros'] = total_intros
        thisout['cadence_changes'] = thisout['time'].isin([int(a) for a in cadence_changes])
        thisout['new_intros'] = thisout['time'].isin([int(a) for a in new_intros])
        group_infections = get_group_final_infections(model, node_labels)
        thisout['teacher_infections'] = group_infections['teacher']
        thisout['student_infections'] = group_infections['student']
        thisout['staff_infections'] = group_infections['staff']
        thisout['initial_exposed_group'] = initial_exposed_group

        output_frames.append(thisout)




if '__name__' == 'main':
    main()
