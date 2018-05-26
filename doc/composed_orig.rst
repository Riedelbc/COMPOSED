##################
COMPOSED Algorithm
##################

******
Inputs
******

:info_table_name: A CSV with your covariates and file locations for your image
   files. No default setting.

   NOTE: Ensure your data is relatively proportional between your controls and
   analysis subjects. Therefore, make sure that:

      num_folds must be < 3x Group==1

   Otherwise, the classifiers will overfit each fold based off the control group
   characteristics.

:output_dir: The directory where the log file and output files will be
   saved. Defaults to the current working directory.

:random_flag: Passed into scikit-learn stratifiedKFold: Populates the pseudo
   random number generators initial state

:num_repeats: The number of times to create subfold-groups for a
   partition_features data set. Partioned_features data will be

:num_folds: The number of fold groups to split a partitioned_features data set
   into.

:data_type: The data structure type used to load EPIC. Options are defined
   within epic.data. Includes freesurfer and connectomics.

:data_type_args: a dictionary of arguments to pass to the data interface
   specified by data_type. See epic.data classes for details.

:classifier_name: The scikit-learn machine learning algorithm used to classify
   the data. Options include:

   'Linear SVM', 'RBF SVM', 'LDA', or 'LR'

:dont_log: If true, print output to screen, if false, print output to a log
   file.

:c_stat: What type of svm classifier to use

:max_combinatorial: Max number of features to take of a feature type (eg VL,
   Thk, SA) by group (eg DX and if applicable Sex/other covariate) when
   calculating AIC for partition type combinatorial sets - this includes
   positive and negatives each. Only positive are merged with positive and
   negatives with negatives.

:max_merges: Max returned combinatorial of merges over a feature type by
   group (DX/cov) to find the optimal merges within that feature type.

:logname: What to name the logfile in the output dir.

:min_diff_thresh: This threshold is the minimum value taken for normalized
   difference equation results. The formulas are basically t-tests, so values
   around 2 are closer to approaching significant (depending on DF's). Often the
   max combinatorial will choose fewer features than the number that meet the
   min_diff_thresh, so it's usually not necessary to change this too much.
   print("Creating partitions ...")  self.partitions =
   self.data.create_partitions()


*****************
Overall Algorithm
*****************

#. Run `Load Data Algorithm`_
#. Run `Create Partitions Algorithm`_
#. Split the data returned from Load Data into ``training`` and
   ``validation_testing`` data sets using StratifiedShuffleSplit
#. ``num_repeats`` repeats times do:

   #. Partition the ``training`` set into ``num_folds`` sub_train and sub_test subsets using StratifiedKFold
   #. For each subfold do:

      #. For each merge combination (each row) from the merge table from ``create_partition`` do:

         #. Dimensionally reduce the ``training`` data using the active merge combination
         #. fit an SKF classifier using the reduced train data and only taking
            the ``sub_train`` elements from that data
         #. Evaluate the performance of the classifier by calculating
            :meth:`composed.util.performance_measures` for the classifier
            against the ``sub_test`` elements from the data.

      #. Calculate the average performance for each merge combination across all
         folds it was run against.
      #. Identify the top performing merge combination based on these average
         performance metrics

#. Validate the top performing merge combinations against the reserved
   ``validation_testing`` subjects.


*******************
Load Data Algorithm
*******************

:meth:`composed.data.ComposedDataSource.load`

Loads data from the initial data source(s).

After the data is initialized into memory:

1. Identifies the diagnosis categorical variable in the data set based on the
   ``group`` input argument

2. Optionally identifies a covariate categorical (dichotomous) variable in the data
   set based on the ``covariate`` input argument

3. Independent continuous variables are then identified within the data set and
   categorized based on the ``prefixes`` argument. Each prefix identifies a
   different variable data type. Each variable sharing a common prefix is then
   grouped together. Variable types are necessary in order to identify which
   variables can be merged together. For example, variable prefixes are
   necessary to delineate volume measurements from surface area measurements
   from thickness measurements. This categorization prevents creation of
   non-physical variables (like volume merged with thickness) within the
   variable merges that occur later in the algorithm (see create_partitions).

4. If a covariate was identified, two copies of each category of independent
   variable is created, one for the diagnosis, and one for the covariate. The
   copies are uniquely named by prefixing with 'grp' for the copy associated
   with the diagnosis variable, and 'cov' for those associated with the
   covariate.

5. Each independent variable category is then sorted and split into a positive
   and negative association sub-category based on the ``median_diff``
   equation. The ``median_diff`` equation identifies the variables within each
   category that have the highest positive and negative associations with their
   diagnosis/covariate variable based on the following two equations.

   The diagnosis median diff equation is as follows:

   .. math::
      :label: grp_diff

      \frac{\mu_{grp0} - \mu_{grp1}}{\sqrt{\frac{\sigma_{grp0}^2}{n_{grp0}} + \frac{\sigma_{grp1}^2}{n_{grp1}}}}

   The covariate median diff equation is as follows:

   .. math::
      :label: cov_diff

      \frac{\mu_{cov0_{grp0}} - \mu_{cov0_{grp1}}}{\sqrt{\frac{\sigma_{cov0_{grp0}}^2}{n_{cov0_{grp0}}} + \frac{\sigma_{cov0_{grp1}}^2}{n_{cov0_{grp1}}}}} - \frac{\mu_{cov1_{grp0}} - \mu_{cov1_{grp1}}}{\sqrt{\frac{\sigma_{cov1_{grp0}}^2}{n_{cov1_{grp0}}} + \frac{\sigma_{cov1_{grp1}}^2}{n_{cov1_{grp1}}}}}

   Where :math:`grp0` and :math:`grp1` refer to the two diagnosis variable
   categories and :math:`cov0` and :math:`cov1` refer to the two covariate
   variable categories, :math:`\sigma` is standard deviation, :math:`\mu` is
   arithmetic mean, and :math:`n` is number of available samples.

6. The size of each category is then reduced based on the sort, corresponding min_diff_thresh
   as well as the ``max_combinatorial`` input argument. The filter ensures only the
   most-positively and most negatively associated variables are kept for later
   analysis. For example, given a hypothetical max_combinatorial value of 3, and
   a category of volume variables of size 10, where 5 of those variables are
   negatively associated with the sex covariate, and 5 are positively
   associated, after the filter only the 3 most negatively and 3 most positively
   associated variables will be left in the data set, giving a total size of six
   variables within that category.

7. The categorized data is then stored in memory such that it can be processed
   by the :meth:`composed.data.ComposedDataSource.create_partitions` algorithm.

For example, given a category of independent variables of the same
value type (say there's five volume variables in the input data), then
four merge categories will be created for the set of volume
measurements: 1) the set of volume variables positively associated
with dx, 2) the set of volume variables negatively associated with dx,
3) the set of volume variables positively associated with cov, and
4)the set of volume variables negatively associated with cov.

The length of each set is max 5 in this example (the number of volume
variables) but may be zero as well.

Later, in create_partitions (see below), merge sets will be created
for each of these categories. Merge sets are created through evaluting
aic scores for a candidate merger of volume variables. The max number
of merge sets for each category is limited by max_merges, so if
max_merges is 3, then the maximum number of merge combinations that
will be saved for each of the four categories will be
three. Therefore, if each of the four volume categories had three or
more merge sets which met the aic threshold, then a total of twelve
(four volume categories * three merge sets) would be permuted within
the final merge partition table.


***************************
Create Partitions Algorithm
***************************

:meth:`composed.data.ComposedDataSource.create_partitions`

Takes feature type (VL/SA/Thick) into account while performing merge partitioning.

:param classifier_coefs: the feature coefficients from running a
   classifier on the dataset
:return: train_partitions, a matrix where each row represents a
   unique grouping of features to use to find an effective classifier
   with.

Finds the optimal merge sets for all variable categories then
creates a merge partitions table out of the resulting optimal merge
sets.

For each merge category (merge categories are defined within
:meth:`load`), candidate merge sets are calculated by iterating
through all the binary permutations for a string of the length of the
category. For example, given five positively associated volume measurements in a
volume merge category, there are about :math:`2^5` possible binary values, as
follows::

   00000
   00011
   00101
   01001
   10001
   ...
   11101
   11110
   11111

This general formula for each category and each positive/negative association 
filtered set is:
   .. math::
      :label: merge_formula

      P = 2^N - N
     
Where P is the formula for number of permutations, and N is the number of features
selected for that subcategory and feature type. 


A candidate merge set is composed of: 1) original values for all
volume variables with indices that map to a 0 in the permutation, plus
2) a merged value corresponding to the average value for all indices
that map to a 1 in the permutation.

Once a merge set candidate is calculated, it is evaluated by comparing
the Akiaike Information Criterion (aic) from an Ordinary Least Squares
(OLS) model generated from the candidate merge values versus the
baseline AIC -- calculated from an OLS model generated using the
un-merged variable set. If the candidate merge set has a lower AIC
than the baseline, it is kept as a merge group for later processing.

AIC is an information theoretic measure of the quality of a statistical
model. As such, AIC is calculated by first training an ordinary least
squares linear model on the variables (using `statsmodels.OLS`_), then
returning the following value:

.. math::
   :label: aic

   2k - 2ln(L)

Where :math:`k` is the number of independent variables of the model, and
:math:`ln(L)` is the log likelihood function calculated from
statsmodels.GLS.loglike_.

.. _statsmodels.OLS: http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels-regression-linear-model-ols
.. _statsmodels.GLS.loglike: http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.loglike.html#statsmodels.regression.linear_model.GLS.loglike

Once AIC is calculated for the baseline model vs a candidate merge
model, if the merge model's aic is less than the baseline (i.e. better), then that
merge set is saved for use in the merge partition.

This process is repeated three times for each feature type and for
both positive and negative association features in order to create
more refined merge models. The merge sets identified within each
feature type will go into making the final merge list and thus
creating the optimal merge partitions.


The number of permutations for the first AIC run is equal
to :math:`2^{l_m} - l_m` where :math:`l_m` is the initial number of features in
the category. The subsequent two AIC runs look for new features to merge, 
while maintaining those identified through the first AIC run. 
The number of permutations for the second AIC run is equal
to :math:`R*2^{l_r}` where :math:`R` is the number of saved merge sets
from the first run, and :math:`l_r` is the number of values in a saved
merge category. The number of permutation for the next run is the same
as the second, except :math:`R` is the number of merge sets generated from the
previous run only.

An example second AIC run (starting with 6 features) might look like this::

   002211
   022011
   022101
   002121

And a third corresponding AIC run might then look like this::

   332211
   322311
   322131

During each AIC run, the merge sets that are better than baseline are retained, 
to be added to the final merge partition, with all subcategories for each 
feature type. 

The per-feature type optimal merge partitions are then combined into a
full-dataset merge table by taking all permutations between each
categories final merge lists. This combined merge partition table then
serves as the input to the principle composed SVM algorithm.

***************************
**TLDR Basic Summary**
***************************

   The first step of COMPOSED is feature selection, this is defined as follows:

   .. math::
      :label: feature_selection
      
      {f_1...f_N} > |t_{thresh}| ≤ max_{c}
         
   Where f are the original MRI measures within a given feature type (i.e. volume/thickness/surface area), 
   separated by subcategory (i.e. features with a positive/negative association), that are greater 
   than the user specified test statistic threshold. Max_c determines the maximum number of subcategory 
   features to be used for each feature type, is user defined, and helps reduce computational complexity.   
      
   The second step is a data reduction process completed by merging features. Given N identified features within 
   a subcategory from step 1, up to N features will be combined together by averaging; this defines a merge, creating 
   one merge feature in place of the original N features. There can be up to three separate non-overlapping 
   merges within a merge set of N features. However, fewer than N features can also be merged, leaving the other features 
   in the set as their original form. These merge sets are determined from the features remaining after 
   step 1 and defined as:
   
   .. math::
      :label: merge_def
      
      set_{M_1...M_N} \rightarrow \!\, ϕ  ◦ P_{merge}(n,r)_{f_1...f_N} 
      
   Where set M are the different merge sets, identified by permuting through merge combinations of the features 
   selected through step 1.   
   
   Only the optimal merge sets are retained for disease classification. Optimal merge sets are defined as follows:

   .. math::
      :label: merge_set_selection
      
      \forall \!\,\text{ }set_{M_1...M_N} \text{ where }AIC\text{ }set_{M_N} < AIC\text{ }f_{N}
   
   Where each merge set within a subcategory is compared to the baseline AIC for that subcategory using only
   the original features (i.e. no merging has occured). Diagnosis is the dependent variable. Lower AIC values 
   indicate better model fit, and are kept for the partition matrix. 
      
   Finally, all subcategory optimal merge sets for each feature are then combined into a merge table by taking 
   all permutations between each subcategories final merge list. These combinations across feature merge listss 
   are used to find the highest performing combination, as measured by test accuracy, through a linear SVM classifier. 

   .. math::
      :label: merge_set_to_partition

      P(n,r) = F \subseteq \!\,set_{M1...M_N} \text{} \rightarrow \!\, ϕ  ◦{f_1...f_N}
   
   Here, P represents all permutations across all features F. These are composed of all optimal merge sets M, which are
   transformations from the original features within each category f.   
