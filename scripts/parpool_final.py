import os
import pdb
import sys
import traceback
import multiprocessing
import glob
from composed import Composed
import numpy as np
import fileinput

class Results():

    def __init__(self):
        self.res = []

    def add_result(self, result_row):
        print("Called against {}".format(result_row))
        self.res.append(result_row)

    def save(self, fname):
        with open(fname, 'w') as f_obj:
            f_obj.write('\n'.join([', '.join(row) for row in self.res]))

def run_composed(file):
    base = os.path.split(__file__)[0]
    fname = os.path.split(file)[1]
    fnam = os.path.splitext(fname)[0]
    results_dir = "Test_pred"
    best_per_file = []
    best = -100
    best_validation = -100
    best_combinatorial = None
    best_c = None
    best_merges = None
    best_min_diff_thresh = None
    best_log = None
    max_combinatorial = int(sys.argv[1])
    c_stat = float(sys.argv[2])
    min_diff_thresh = float(sys.argv[3])
    #for c_stat in [0.1,1.0]:
        #for max_combinatorial in np.arange(8,11,1):
    for max_merges in np.arange(7,8,1):
        #for min_diff_thresh in np.arange(0.9,1.6,.2):
        logname = 'EPIC_RESULTS_{}_cstat{}_maxc{}_maxm{}_t{}.log'.format(fnam, c_stat, max_combinatorial, max_merges, min_diff_thresh)
        args = {'info_table_name': os.path.join(base, file),
                'random_flag': 1234,
                'output_dir': os.path.join('/ifs/loni/faculty/thompson/four_d/briedel/MDD', results_dir),
                'num_repeats': 10,
                'num_folds': 5,
                'classifier_name': "Linear SVM",
                'data_type': 'freesurfer',
                'data_args': {},
                'c_stat': c_stat,
                'max_combinatorial': max_combinatorial,
                'max_merges': max_merges,
                'min_diff_thresh': min_diff_thresh,
                'logname': logname,
                }
        mean_perf, validation_perf = Composed(args).run()
        if mean_perf > best:
            best = mean_perf
            best_validation = validation_perf
            best_c = c_stat
            best_combinatorial = max_combinatorial
            best_merges = max_merges
            best_min_diff_thres = min_diff_thresh
            best_log = logname
        print("Ran {} {} {} {} on {} and got {}".format(c_stat, max_combinatorial, max_merges, min_diff_thresh, file, best))

    return (file, best, best_validation, best_combinatorial, best_merges, best_min_diff_thresh, best_log)

#col_headers_list = list(["File_prefix", "File_prefix2", "File_prefix3", "File_prefix4", "File_prefix5", "File_prefix6", "File_prefix7", "File_prefix8", "Balanced_Test_Accuracy", #"Balanced_Prediction_Accuracy", "Max_Combinatorial", "Max_Merges", "Covariate", "Log_Name"])

        
# Change this line to the file you want to analyze
# image_file = "EPIC_test_AD.csv"
# image_file = "/ifshome/briedel/Epic_Tools/test/Over21_Berlin_All_Site_Age_Sex_ICV_Comp.csv"
if __name__ == "__main__":
    pool = multiprocessing.Pool()
    list_of_files = glob.glob('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Duke_controls_may2017_EPIC_test.csv')
    print(list_of_files)
    if not list_of_files:
        raise Exception("No Files")
    res_l = []
    #run_epic(list_of_files[0])
    for res in pool.imap_unordered(run_composed, list_of_files):
        print(res)
        res_l.append(res)
        
    with open('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Test_pred/Results_prediction.csv', 'a') as f_obj:
        for row in res_l:
            f_obj.write('\t'.join(['{}'.format(cell) for cell in row]))
            f_obj.write('\n')

