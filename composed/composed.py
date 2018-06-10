"""
Composed primary python module -- contains the Composed class.
"""
# Gautam Prasad - gprasad@usc.edu - 03.01.15
# modified by Brandalyn Riedel bcriedel@usc.edu 08.15.16

import os
import sys

import numpy as np
from multiprocessing import Pool

from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from composed.data import ComposedDataSource
from composed.util import merge_data, measure_performance, partition_matrix
from composed.consts import (MAX_COMBINATORIAL, MAX_MERGES, MIN_DIFF_THRESH,
                             NUM_PERFORMANCE_METRICS, NUM_GBM_PERFORMANCE_METRICS)

def pool_evaluate(args):
    repeat, num_folds, num_metrics, groups, _features, opt_idx, partitions, is_categorical, classifier, names = args

    print("Repeat {}: Evaluating {} partitions in {} folds"
          .format(repeat, partitions.shape[0], num_folds),
          flush=True
    )
    partitions_performance = np.zeros((num_folds, partitions.shape[0], num_metrics))

    if is_categorical:
        folds = StratifiedKFold(groups, int(num_folds), shuffle=True)
    else:
        kf = KFold(int(num_folds), shuffle=True)
        folds = kf.split(groups)

    for j, (train_index, test_index) in enumerate(folds):
        for k in range(partitions.shape[0]):
            partitioned_features = merge_data(partitions[k,:], _features)
            _, partitions_performance[j, k, :] = measure_performance(
                classifier,
                names,
                partitioned_features[train_index, :],
                groups[train_index],
                partitioned_features[test_index, :],
                groups[test_index]
            )

    # partitions_performance is a 3d matrix, this collapses it into a 2d matrix,
    # taking averages for the first dim, which is used to record results for
    # each fold (j above)
    partition_mean_performance = np.mean(partitions_performance, 0)

    # This finds the best partition (best idx k above) using the optimal row
    # (for gbm this is performance idx 4, which is test R-squared, for all
    # others this is performance idx 5, which is mcc)
    optimal_index = np.argmax(partition_mean_performance[:, opt_idx])

    # This returns the index of the best performing partition as well as its
    # average performance measures over all the folds
    return (optimal_index, partition_mean_performance[optimal_index, :])

class Composed:
    """
    Implements the evolving partitions to improve connectomics (composed)
    algorithm. See doc/about_composed.rst for details on the
    algorithm implementation.
    """

    def __init__(self, args):
        """Initialize Composed by inputting the desired run parameters within the args
        dictionary. Expected parameters are as follows:

        info_table_name: A CSV with your covariates and file locations
           for your image files. No default setting.

           NOTE: Ensure your data is relatively proportional between
           your controls and analysis subjects. Therefore, make sure that:

              num_folds must be < 3x Group==1

           Otherwise, the classifiers will overfit each fold based off
           the control group characteristics.

        holdout_table_name: the filepath to a CSV formatted identically to
           the info_table_name csv. If this argument is supplied, then final
           model holdout will be performed against the data within this
           file. If this argument is NOT supplied, a subset of info_table_name's
           data will be reserved to use for model holdout.

        output_dir: The directory where the log file and output files
            will be saved. Defaults to the current working directory.

        random_flag: Passed into scikit-learn stratifiedKFold:
           Populates the pseudo random number generators initial state

        num_repeats: The number of times to create subfold-groups for a
           partition_features data set. Partioned_features data will be

        num_folds: The number of fold groups to split a
           partitioned_features data set into.

        data_type: The data structure type used to load COMPOSED. Options are
           defined within composed.data. Includes freesurfer and connectomics.

        data_type_args: a dictionary of arguments to pass to the data
           interface specified by data_type. See composed.data classes for details.

        classifier_name: The scikit-learn machine learning algorithm
           used to classify the data. Options include:

           'Linear SVM', 'RBF SVM', 'LDA', or 'LR'

        dont_log: If true, print output to screen, if false, print
           output to a log file.

        c_stat: What type of svm classifier to use

        max_combinatorial: Max number of features to take of a feature type (eg
           VL, Thk, SA) by group (eg DX and if applicable Sex/other covariate)
           when calculating AIC for partition type combinatorial sets - this
           includes positive and negatives each. Only positive are merged with
           positive and negatives with negatives.

        max_merges: Max returned combinatorial of merges over a feature type by
           group (DX/cov) to find the optimal merges within that feature type.

        logname: What to name the logfile in the output dir.

        min_diff_thresh: This threshold is the minimum value taken for
           normalized difference equation results. The formulas are basically
           t-tests, so values around 2 are closer to approaching significant
           (depending on DF's). Often the max combinatorial will choose fewer
           features than the number that meet the min_diff_thresh, so it's
           usually not necessary to change this too much.

        save_holdout_subject_results: Boolean, default False. Whether to
           create a csv file that stores the predictions from the final model
           for each holdout subject.

        """

        self.info_table_name = args['info_table_name']
        if not os.path.exists(self.info_table_name):
            raise ValueError("Cannot load info table, '{}' is not a valid"
                             " filepath!".format(args['info_table_name']))

        self.holdout_table_name = args.get('holdout_table_name', None)
        self.random_flag = int(args.get('random_flag', 0))
        self.output_dir = args.get('output_dir', os.getcwd()).strip()
        self.num_repeats = int(args.get('num_repeats', 1))
        self.num_folds = int(args.get('num_folds', 10))
        self.classifier_name = args.get('classifier_name', 'Linear_SVM').strip()
        self.data_type = args.get('data_type', 'connectomics')
        self.data_type_args = args.get('data_type_args', {})
        self.c_stat = args.get('c_stat', 1)
        self.max_combinatorial = args.get('max_combinatorial', MAX_COMBINATORIAL)
        self.max_merges = args.get('max_merges', MAX_MERGES)
        self.min_tdiff_thresh = args.get('min_diff_thresh', args.get('min_tdiff_thresh', MIN_DIFF_THRESH))
        self.min_zdiff_thresh = args.get('min_zdiff_thresh', MIN_DIFF_THRESH)
        self.logname = args.get('logname', 'COMPOSED_MERGE.log')
        self.dont_log = bool("dont_log" in args)
        self.log = sys.stdout
        self.save_holdout_subject_results = args.get('save_holdout_subject_results', False)

        # GBM specific args
        self.learning_rate = float(args.get('learning_rate', 0.01))
        self.n_estimators = int(args.get('n_estimators', 300))
        self.max_depth = int(args.get('max_depth', 6))

        # Get/create output dir
        self.output_dir = os.path.join(self.output_dir, '')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print('output directory created!')
        else:
            print('output directory exists!')

        # Set during data load:
        self.data_source = None

        # Set during run
        self.partitions = None
        self.partitions_performance = None


    def print_input_arguments(self, logfile=None):
        """
        Prints the arguments that define this run of COMPOSED tools.
        """
        if logfile is None:
            logfile = sys.stdout

        print('COMPOSED Parameters:', file=logfile)
        print('  info_table_name = ' + str(self.info_table_name), file=logfile)
        print('      random_flag = ' + str(self.random_flag), file=logfile)
        print('       output_dir = ' + self.output_dir, file=logfile)
        print('num_repeats = ' + str(self.num_repeats), file=logfile)
        print('  num_folds = ' + str(self.num_folds), file=logfile)
        print('  classifier_name = ' + self.classifier_name, file=logfile)
        if isinstance(self.get_classifier(), GradientBoostingRegressor):
            print('learning_rate = {}'.format(self.learning_rate), file=logfile)
            print('n_estimators = {}'.format(self.n_estimators), file=logfile)
            print('max_depth = {}'.format(self.max_depth), file=logfile)

        print('\n', file=logfile, flush=True)

    def run(self):
        """
        Run the COMPOSED classification algorithm
        """

        mean_perf = None

        if not self.dont_log:
            logname = os.path.join(self.output_dir, self.logname)
            self.log = open(logname, 'w')
            print("Logging to file ... log is located at {}".format(logname))
        try:
            self.print_input_arguments(logfile=self.log)

            print("Loading data ...")
            self.load_data()

            print("Creating partitions ...")
            self.partitions = self.data.create_partitions()

            print("Evaluating {} partitions ...".format(self.partitions.shape[0]))
            _features = self.data.train_x
            groups = self.data.train_y

            clf = self.get_classifier()
            if isinstance(clf, GradientBoostingRegressor):
                num_metrics = NUM_GBM_PERFORMANCE_METRICS
                opt_idx = 4 # Test MSE should work better for young/old age issues
            else:
                num_metrics = NUM_PERFORMANCE_METRICS
                opt_idx = 5 # MCC

            self.best_performances = []
            arglist = [(x,
                        self.num_folds,
                        num_metrics,
                        groups,
                        _features,
                        opt_idx,
                        self.partitions,
                        self.data.is_categorical,
                        self.get_classifier(),
                        self.data.partition_names(x)) for x in range(self.num_folds)]

            with Pool(processes=10) as pool:
                self.best_performances.extend(pool.map(pool_evaluate, arglist))

            # test and compare partition
            mean_perf, validatation_perf = self.validate_optimal_partitions()
        finally:
            print("Done.")
            self.log.close()
        return mean_perf, validatation_perf

    def get_classifier(self):
        if self.classifier_name == 'Linear SVM' or self.classifier_name == 'Linear_SVM':
            clf = svm.LinearSVC(class_weight='balanced', C=self.c_stat, dual=True, tol=0.000001,
                                penalty='l2', random_state=1, loss='hinge')
        elif self.classifier_name == 'Poly SVM' or self.classifier_name == 'Poly_SVM':
            clf = svm.SVC(kernel='poly', class_weight='balanced', degree=4, C=self.c_stat)
        elif self.classifier_name == 'RBF SVM' or self.classifier_name == 'RBF_SVM':
            clf = svm.SVC(kernel='rbf', class_weight='balanced')
        elif self.classifier_name == 'LDA' or self.classifier_name == 'lda':
            clf = LinearDiscriminantAnalysis()
        elif self.classifier_name == 'LR':
            clf = LogisticRegression()
        elif self.classifier_name.lower() == 'gradient boosting regressor' or \
             self.classifier_name.lower() == "gbr":
            # See http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
            clf = GradientBoostingRegressor(loss='huber',
                                            learning_rate=self.learning_rate,
                                            n_estimators=self.n_estimators,
                                            max_depth=self.max_depth, # Needs to be high 5+ for ROI/brain distribution consideratons
                                            criterion='friedman_mse',
                                            min_samples_split=15,
                                            min_samples_leaf=5,
                                            min_weight_fraction_leaf=0,
                                            subsample=0.95, # High to encourage sampling across all ages
                                            max_features=0.65,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0000001,
                                            alpha=0.1,  # Use the alpha% quantile
                                                        # in loss in order to
                                                        # weight towards
                                                        # overpredicting Y.
                                                        # calculates loss based
                                                        # on huber loss of the
                                                        # alpha quantile, where
                                                        # the quantile array is
                                                        # taken from y_actual -
                                                        # y_pred. Therefore this
                                                        # weights towards
                                                        # negative y_actual -
                                                        # y_pred results.
                                            init=None,
                                            verbose=0,
                                            warm_start=False,
                                            random_state=1234,
                                            presort='auto')

        return clf

    def load_data(self):
        """
        1. Initializes self.data_source and grabs features and groups by calling
           the data_source load() method.

        2. Transform and scale features

        3. Reserve a hold-out set from the data
        """
        if not self.data_type in ComposedDataSource.registry:
            raise ValueError("There is no data source interface class '{}' "
                             "defined in composed.data. Cannot load data!"
                             .format(self.data_type))
        data_cls = ComposedDataSource.registry[self.data_type]
        self.data = data_cls(self, **self.data_type_args)

        self.data.load(self.info_table_name, self.holdout_table_name)

        print('Data size:\n\t{}\tparticipants\n\t{}\tfeatures (remaining after filter) per participant'
              .format(self.data.x.shape[0], self.data.x.shape[1]),
              file=self.log, flush=True)

        grps, len_grps = np.unique(self.data.y, return_counts=True)
        if len(grps) > 2:
            assert self.classifier_name.lower() in ('gbr', 'gradient boosting regressor'), \
                "Can only have two subject groups, control==0, and diseased==1" \
                " unless you use a Gradient boosting model"

    def validate_optimal_partitions(self):
        """Run a classifier run against the unadultered data (no partitions), and
        evaluate its baseline performance against the test data. Then run a
        classifier with the optimal partitions and evaluate performance against
        the new holdout data. Then, print out the final results.

        """
        train_features = self.data.train_x
        train_groups = self.data.train_y
        test_features = self.data.test_x
        test_groups = self.data.test_y

        # Baseline performance
        bb_coef, bb_ac = measure_performance(self.get_classifier(),
                                             self.data._x_names,
                                             self.data._train_x,
                                             train_groups,
                                             self.data._test_x,
                                             test_groups)

        # Baseline performance
        b_coef, b_ac = measure_performance(self.get_classifier(),
                                           self.data.x_names,
                                           train_features,
                                           train_groups,
                                           test_features,
                                           test_groups)


        print("\n\n"
              "Base Results\n"
              "============", file=self.log)

        self.save_performance(bb_ac,
                              bb_coef,
                              '\n'
                              'Performance with full data on holdout:\n'
                              '--------------------------------------\n')
        self.save_performance(b_ac,
                              b_coef,
                              '\n'
                              'Performance with filtered data on holdout:\n'
                              '------------------------------------------\n')

        holdout_perfs = []

        for repeat_idx, (optimal_index, mean_performance) in enumerate(self.best_performances):
            print("\n\n"
                  "Repeat {} Results:\n"
                  "=================".format(repeat_idx + 1), file=self.log)

            optimal_partition = self.partitions[optimal_index, :]
            p = partition_matrix(optimal_partition)
            partition_names = self.data.partition_names(optimal_index)

            partitioned_train_features = np.zeros((train_features.shape[0], p.shape[1]))
            for i in range(0, train_features.shape[0]):
                partitioned_train_features[i, :] = train_features[i, :].dot(p)

            partitioned_test_features = np.zeros((test_features.shape[0], p.shape[1]))
            for i in range(0, test_features.shape[0]):
                partitioned_test_features[i, :] = test_features[i, :].dot(p)

            # COMPOSED performance
            e_coef, e_ac = measure_performance(self.get_classifier(),
                                               partition_names,
                                               partitioned_train_features,
                                               train_groups,
                                               partitioned_test_features,
                                               test_groups)

            holdout_perfs.append(e_ac[0][0])

            self.save_performance(mean_performance,
                                  None,
                                  '\n'
                                  'Mean Performance of the runs on train/test data:\n'
                                  '------------------------------------------\n')
            self.save_performance(e_ac,
                                  e_coef,
                                  '\n'
                                  'Performance against new holdout data:\n'
                                  '---------------------------------\n')

            if self.save_holdout_subject_results:
                # TODO: this is where to create the output safe results
                clf = self.get_classifier()
                clf.fit(partitioned_train_features, train_groups)
                train_pred = clf.predict(partitioned_train_features)
                pred = clf.predict(partitioned_test_features)

                fname = "Repeat_{}_predictions_per_subject.csv".format(repeat_idx + 1)
                with open(os.path.join(self.output_dir, fname), 'w') as f_obj:
                    print("Subject ID,Predicted Age,Actual Age", file=f_obj)
                    for idx, subj in enumerate(self.data.test_subj_ids):
                        print("{},{},{}".format(subj, pred[idx], test_groups[idx]), file=f_obj)
                    for idx, subj in enumerate(self.data.train_subj_ids):
                        print("{},{},{}".format(subj, train_pred[idx], train_groups[idx]), file=f_obj)

        mean_perfs = [elem[1][0] for elem in self.best_performances]

        return np.mean(mean_perfs), np.mean(holdout_perfs)

    def save_performance(self, ac_list, coef_list, results_title):

        print(results_title, file=self.log)
        perf_measure_hdr = ["Balanced Accuracy",
                            "Accuracy",
                            "Sensitivity",
                            "Specificity",
                            "F1",
                            "MCC",
                            "PPV"] \
                           if not isinstance(self.get_classifier(), GradientBoostingRegressor) else \
                              ["Training R2 score",
                               "Testing R2 score",
                               "Pearson Coeff",
                               "Pearson P-Value",
                               "Mean Squared Error"]
        print(',\t'.join(perf_measure_hdr), file=self.log)

        row_vals = [str(metric) for metric in ac_list]
        print("{}".format(",\t".join(row_vals)), file=self.log)
        if coef_list is not None:
            print("\nFeature names and classifier coefficients\n"
                  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", file=self.log)
            sorted_coefs = sorted(list(coef_list.items()), key=lambda x: x[1])
            print("{}".format('\n'.join(["{}: {}".format(k, v) for k, v in sorted_coefs])), file=self.log)
