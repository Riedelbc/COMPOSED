import os
import pdb
import sys
import traceback

from composed import Composed

# Change this line to the file you want to analyze
base = os.path.split(__file__)[0]
image_file = "composed_basic_testing.csv"
results_dir = "composed_basic_test"
# construct argument dictionary
args = {'info_table_name': os.path.join(base, image_file),
        'random_flag': 0,
        'output_dir': os.path.join('/tmp', results_dir),
        'num_repeats': 5,
        'num_folds': 2,
        'classifier_name': "Linear SVM",
        'data_type': 'freesurfer',
        'data_type_args': {'covariate': 'Sex'},
        }

try:
    # create classification object
    composed = Composed(args)

    # run composed
    composed.run()

except Exception as err:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem()
