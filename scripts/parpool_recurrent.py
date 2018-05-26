import os
import pdb
import sys
import traceback
import multiprocessing
import glob
from epic import Epic
import numpy as np

##################################################    
##################################################            
def run_epic(file):
    base = os.path.split(__file__)[0]
    fname = os.path.split(file)[1]
    fnam = os.path.splitext(fname)[0]
    results_dir = "Recurrent_Complete_Fin"
    best_per_file = []
    best = -100
    best_combinatorial = None
    best_c = None
    best_merges = None
    best_min_diff_thresh = None
    best_log = None
    for c_stat in [0.1,1.0,30.0,50.0]:
        for max_combinatorial in np.arange(8,13,1):
            for max_merges in np.arange(7,8,1):
               for min_diff_thresh in np.arange(0.9,1.8,.2):
                   logname = 'EPIC_RESULTS_{}_cstat{}_maxc{}_maxm{}_t{}.log'.format(fnam, c_stat, max_combinatorial, max_merges, min_diff_thresh)
                   args = {'info_table_name': os.path.join(base, file),
                           'random_flag': 1234,
                           'output_dir': os.path.join('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin', results_dir),
                           'num_repeats': 10,
                           'num_folds': 5,
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
if __name__ == "__main__":
    pool = multiprocessing.Pool()
    list_of_files = glob.glob('/ifs/loni/faculty/thompson/four_d/briedel/MDD/All_reg/Complete_Recurrent/BRC*.csv')
    print(list_of_files)
    if not list_of_files:
        raise Exception("No Files")
    res_l = []
    #run_epic(list_of_files[0])
    for res in pool.imap_unordered(run_epic, list_of_files):
        print(res)
        res_l.append(res)
    with open('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Fin_Recurrent_Complete.csv', 'w') as f_obj:
        f_obj.write('\n'.join([', '.join(['{}'.format(cell) for cell in row]) for row in res_l]))
        
##################################################    
##################################################            
def run_epic(file):
    base = os.path.split(__file__)[0]
    fname = os.path.split(file)[1]
    fnam = os.path.splitext(fname)[0]
    results_dir = "Recurrent_Imputed_Fin"
    best_per_file = []
    best = -100
    best_combinatorial = None
    best_c = None
    best_merges = None
    best_min_diff_thresh = None
    best_log = None
    for c_stat in [0.1,30.0]:
        for max_combinatorial in np.arange(8,13,1):
            for max_merges in np.arange(7,8,1):
               for min_diff_thresh in np.arange(0.9,1.8,.2):
                   logname = 'EPIC_RESULTS_{}_cstat{}_maxc{}_maxm{}_t{}.log'.format(fnam, c_stat, max_combinatorial, max_merges, min_diff_thresh)
                   args = {'info_table_name': os.path.join(base, file),
                           'random_flag': 1234,
                           'output_dir': os.path.join('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin', results_dir),
                           'num_repeats': 10,
                           'num_folds': 5,
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
if __name__ == "__main__":
    pool = multiprocessing.Pool()
    list_of_files = glob.glob('/ifs/loni/faculty/thompson/four_d/briedel/MDD/All_reg/Imputed_Recurrent/BRC_Syd_Hou_Grog*.csv')
    print(list_of_files)
    if not list_of_files:
        raise Exception("No Files")
    res_l = []
    #run_epic(list_of_files[0])
    for res in pool.imap_unordered(run_epic, list_of_files):
        print(res)
        res_l.append(res)
    with open('/ifs/loni/faculty/thompson/four_d/briedel/MDD/Fin_Recurrent_Imputed.csv', 'w') as f_obj:
        f_obj.write('\n'.join([', '.join(['{}'.format(cell) for cell in row]) for row in res_l]))
