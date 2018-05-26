import os
import pdb
import sys
import traceback
import multiprocessing
import glob
from epic import Epic
import numpy as np

class Results():

    def __init__(self):
        self.res = []

    def add_result(self, result_row):
        print("Called against {}".format(result_row))
        self.res.append(result_row)

    def save(self, fname):
        with open(fname, 'w') as f_obj:
            f_obj.write('\n'.join([', '.join(row) for row in self.res]))

def run_epic(file):
    base = os.path.split(__file__)[0]
    fname = os.path.split(file)[1]
    fnam = os.path.splitext(fname)[0]
    results_dir = "epic_test"
    best_per_file = []
    best = -100
    best_combinatorial = None
    best_c = None
    best_merges = None
    best_min_diff_thresh = None
    best_log = None
    for c_stat in np.arange(1,20,1):
        for max_combinatorial in np.arange(8,16,1):
            for max_merges in np.arange(6,8,1):
               for min_diff_thresh in [0.3, 0.4, 0.444444]:
                   logname = 'EPIC_RESULTS_{}_{}_{}_{}.log'.format(fnam, c_stat, max_combinatorial, max_merges, min_diff_thresh)
                   args = {'info_table_name': os.path.join(base, file),
                           'random_flag': 0,
                           'output_dir': os.path.join('/tmp', results_dir),
                           'num_repeats': 5,
                           'num_folds': 2,
                           'classifier_name': "Linear SVM",
                           'data_type': 'freesurfer',
                           'data_args': {'covariate': 'Sex'},
                           'c_stat': c_stat,
                           'max_combinatorial': max_combinatorial,
                           'max_merges': max_merges,
                           'min_diff_thresh': min_diff_thresh,
                           'logname': logname,
                           }
                   res = Epic(args).run()
                   if res > best:
                       best = res
                       best_c = c_stat
                       best_combinatorial = max_combinatorial
                       best_merges = max_merges
                       best_min_diff_thres = min_diff_thresh
                       best_log = logname
                   print("Ran {} {} {} {} on {} and got {}".format(c_stat, max_combinatorial, max_merges, min_diff_thresh, file, best))

    return (file, best, best_combinatorial, best_merges, best_min_diff_thresh, best_log)


# Change this line to the file you want to analyze
# image_file = "EPIC_test_AD.csv"
# image_file = "/ifshome/briedel/Epic_Tools/test/Over21_Berlin_All_Site_Age_Sex_ICV_Comp.csv"
pool = multiprocessing.Pool()
res = Results()
list_of_files = glob.glob('/Users/Brandy/PycharmProjects/Epic_Tools/test/*.csv')
print(list_of_files)
if not list_of_files:
    raise Exception("No Files")
r = pool.map_async(run_epic, list_of_files, callback=res.add_result)
r.wait()
res.save('/ifshome/briedel/full_composed_results.csv')
import pdb; pdb.set_trace()
