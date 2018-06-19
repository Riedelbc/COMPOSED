import os
import sys
import pdb
import traceback

from composed import Composed

logname = 'composed_linear_run.log'
base = os.path.split(__file__)[0]
results_dir = "composed_linear_test"

max_combinatorial = 10.0
max_merges = 8.0
min_diff_thresh = 1.1

args = {'info_table_name': os.path.join(base, 'composed_linear_testing.csv'),
        #'holdout_table_name': os.path.join(base, 'composed_linear_validation.csv'),
        'random_flag': 1234,
        'output_dir': os.path.join('/tmp', results_dir),
        'logname': os.path.join('/tmp', results_dir, logname),
        'num_repeats': 2,
        'num_folds': 5,
        'classifier_name': "Linear_SVM",
        'data_type': 'freesurfer',
        'data_type_args': {
            'prefixes': ["Thk", "SA", "VL"], #,"Sex"],
           'group': "Group",
           },
        'c_stat': 0.1,
        'max_combinatorial': max_combinatorial,
        'max_merges': max_merges,
        'min_zdiff_thresh': min_diff_thresh,
        'min_tdiff_thresh': min_diff_thresh,
        'save_holdout_subject_results': True,
        }
try:
    composed = Composed(args)
    mean_perf, validation_perf = composed.run()

except Exception as err:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem()
