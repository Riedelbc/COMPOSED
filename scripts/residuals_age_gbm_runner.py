import os
import sys, traceback, pdb
import pprint
from multiprocessing import Pool
from composed import Composed

logname = 'Brain_Age.log'
data_name = "BrainAge"
base = '/muxley/code/composed/'
results_dir = 'Results_Residuals'

max_combinatorial_array=(16,)
learning_rate_array=(0.12,) # 0.06, 0.08, .1, 0.12, 0.05)
min_zdiff_thresh_array=(1.05,) # 1.02, 1.1)
max_depth_array=(5, ) #8, 10)

max_merges = 8.0
min_tdiff_thresh = 1.08
n_estimators = 80

job_list = []
for _max_combinatorial in max_combinatorial_array:
    for _learning_rate in learning_rate_array:
        for _min_zdiff_thresh in min_zdiff_thresh_array:
            for _max_depth in max_depth_array:
                outdirname = "{}_maxComb{}_learningRate{}_minZDiff{}_maxDepth{}".format(
                    data_name, _max_combinatorial,  _learning_rate, _min_zdiff_thresh, _max_depth
                )
                out_dir = os.path.join(base, 'outputs20180609_80trees_15samsplit_TestTrain_huber-alpha0.1', outdirname)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                job_list.append({'max_combinatorial': _max_combinatorial,
                                 'learning_rate': _learning_rate,
                                 'min_zdiff_thresh': _min_zdiff_thresh,
                                 'max_depth': _max_depth,
                                 'info_table_name': os.path.join(base, 'inputs', '{}_Training.csv'.format(data_name)),
                                 'holdout_table_name': os.path.join(base, 'inputs', '{}_Testing.csv'.format(data_name)),
                                 'random_flag': 1234,
                                 'output_dir': out_dir,
                                 'logname': os.path.join(out_dir, logname),
                                 'num_repeats': 10,
                                 'num_folds': 10,
                                 'classifier_name': "Gradient Boosting Regressor",
                                 'data_type': 'freesurfer',
                                 'n_estimators': n_estimators,
                                 'ncpus': 20,
                                 'data_type_args': {
                                     'prefixes': ["Thk", "SA", "VL"],
                                     'group': 'Age',
                                     'covariate': 'Sex',
                                 },
                                 'c_stat': 'gbm',
                                 'max_merges': max_merges,
                                 # unused b/c groupint by age which isn't categorical 'min_diff_thresh': min_diff_thresh,
                                 'save_holdout_subject_results': True,
                })


def do_job(args):
    mean_perf, validation_perf = Composed(args).run()
    return {'name': outdirname, 'mean_perf': mean_perf, 'validation_perf': validation_perf}

if __name__ == "__main__":
    try:
        for job in job_list:
            if os.path.exists(job['logname']):
                continue
            res = do_job(job)
            print("Job {name} complete. Average results across repeats:"
                  "\n\tmean training R-squared: {mean_perf}"
                  "\n\tmean testing R-squared:  {validation_perf}\n\n"
                  .format(**res))
    except Exception as err:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem()
