import os
import pdb
import sys
import traceback
from composed import Composed

logname = 'Prostate_Brain_Age.log'
base = '/Users/Brandy/Desktop/COMPOSED'
results_dir = 'Results_Residuals_Test_Sex4'
max_combinatorial = 20.0
max_merges = 8.0
min_diff_thresh = 1.08
min_zdiff_thresh = 1.08

learning_rate = 0.01
max_depth = 6
n_estimators = 300

args = {'info_table_name': os.path.join(base, 'Training_Prostate_Residuals.csv'),
        'holdout_table_name': os.path.join(base, 'Testing_Prostate_Residuals.csv'),
        'random_flag': 1234,
        'output_dir': os.path.join(base, results_dir),
        'logname': os.path.join(base, results_dir, logname),
        'num_repeats': 1,
        'num_folds': 10,
        'classifier_name': "Gradient Boosting Regressor",
        'data_type': 'freesurfer',
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'data_type_args': {
            'prefixes': ["Thk", "SA", "VL"],
            'group': 'Age',
            'covariate': 'Sex',
            },
        'c_stat': 'gbm',
        'max_combinatorial': max_combinatorial,
        'max_merges': max_merges,
        'min_diff_thresh': min_diff_thresh,
        'min_zdiff_thresh': min_zdiff_thresh,
        'save_holdout_subject_results': True,
        }

try:
    mean_perf, validation_perf = Composed(args).run()

except Exception as err:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem()


