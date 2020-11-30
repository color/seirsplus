import re
import glob
import pandas as pd
import numpy as np


def inverse_ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = 1 - np.arange(1, n+1) / n
    return(x,y)

def ecdf_from_agg_frame(agg_frame, pop_size, to_summarize='overall_infections'):
    # Let's compute an ECDF for each cadence
    x, y = inverse_ecdf(agg_frame[to_summarize])
    x_scaled = x/pop_size

    # append the ends of the lists so graph is complete
    x_scaled = np.append(x_scaled, 1.0)
    y = np.append(y, 0.0)
    x_scaled = np.insert(x_scaled, 0,0)
    y = np.insert(y, 0, 1.0)
    new_frame = {'prop_infected': x_scaled, 'prop_sims':y}

    return(pd.DataFrame(new_frame))

def get_max_qei(df):
    return(max(df['total_e'] + df['total_q'] + df['total_i']))

def get_aggregate_frame(run_data, pop_size):
    run_data['qei'] = run_data['total_e'] + run_data['total_q'] + run_data['total_i']
    run_data['furthest_time_point'] = (run_data['qei'] > 0) * run_data['time'] # math-magic to get maximum timepoint where qei > 0
    if 'teacher_infections' in run_data.columns:
        overall_infections = run_data.groupby(('seed')).agg({'overall_infections':'max', 'qei': 'max', 'total_tests':'max', 'total_intros':'max',
            'teacher_infections':'max', 'student_infections':'max', 'staff_infections':'max', 'initial_exposed_group': 'first',
            'num_isolated_positive_group':'max', 'num_isolated':'max'})
    else:
        overall_infections = run_data.groupby(('seed')).agg({'overall_infections':'max', 'qei': 'max', 'total_tests':'max', 'furthest_time_point':'max'})
    ecdf_frame = ecdf_from_agg_frame(overall_infections, pop_size)
    qei_ecdf_frame = ecdf_from_agg_frame(overall_infections, pop_size, to_summarize='qei')
    return(overall_infections, ecdf_frame, qei_ecdf_frame)

def assign_new_cols(df, newdict):
    for x in newdict.keys():
        df[x] = newdict[x]

def make_ecdf(cadence_file_map, output_prefix, param_dict):
        overall_frames = []
        ecdf_frames = []
        qei_ecdf_frames = []

        for x in cadence_file_map.keys():
            # capture_hash = re.search(HASH_RE, x).group(1)

            print(x)
            results_frame = pd.read_csv(x)

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
        ecdf_all.to_csv(f'{output_prefix}_ecdf.csv', index=False)
        overall_all.to_csv(f'{output_prefix}_overall.csv', index=False)
        qei_all.to_csv(f'{output_prefix}_qei.csv', index=False)



def main():
    all_files = glob.glob('*_results.csv')

    overall_frames = []
    ecdf_frames = []
    qei_ecdf_frames = []

    for x in all_files:
        # capture_hash = re.search(HASH_RE, x).group(1)


        results_frame = pd.read_csv(x)
        param_dict = {
            'testing_cadence':results_frame.testing_cadence[0],
            'r0':results_frame.r0[0],
            'student_susc':results_frame.student_susc[0],
            'introduction_rate':results_frame.introduction_rate[0],
            'student_block_strategy':results_frame.student_block_strategy[0],
            'quarantine_strategy':results_frame.quarantine_strategy[0],
            'pop_size':528}
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
    ecdf_all.to_csv('k5schools_ecdf_webapp200717.csv', index=False)
    overall_all.to_csv('k5schools_overall_webapp200717.csv', index=False)
    qei_all.to_csv('k5schools_qei_ecdf_webapp200717.csv', index=False)

if __name__=='__main__':
    main()
