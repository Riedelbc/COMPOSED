NUM_PERFORMANCE_METRICS = 7
"""
Performance measures are:
0. balanced_accuracy
1. accuracy
2. sensitivity
3. specificity
4. f1
5. mcc
6. ppv
"""

NUM_GBM_PERFORMANCE_METRICS = 5
"""
Performance measures are:
0. The coefficient of determination R^2 of the prediction against the training data
1. The coefficient of determination R^2 of the prediction against the testing data
2. Pearson correlation coefficient
3. Pearson correlation coefficeint's two-tailed p-value
4. mean_squared_error
"""

MAX_COMBINATORIAL = 10
"""Max number of features to take of a feature type (eg VL, Thk, SA) by group
(eg DX and if applicable Sex/other covariate) when calculating AIC for partition
type combinatorial sets - this includes positive and negatives each. Only
positive are merged with positive and negatives with negatives.

"""

MAX_MERGES = 7
"""Max returned combinatorial of merges over a feature type by group (DX/cov) to find the
optimal merges within that feature type.
"""

MIN_DIFF_THRESH = 2.0
"""TBD. This threshold is the minimum value taken for normalized difference
equation results. The formulas are basically t-tests, so values around 2 are
closer to approaching significant (depending on DF's). Often the max
combinatorial will choose fewer features than the number that meet the
min_diff_thresh, so it's usually not necessary to change this too much.

"""

PREDICTION_HOLDOUT_RATIO=0.3
"""Reserve this percentage of the samples to validate final models against. This
should change depending on your sample size.

"""

BIG_COMBO = 15
"""Final combinatorial limiter. Ensures there arent over BIG_COMBO**(2)
total partition rows.
The squared comes from the fact there's group features, and covariate (usually
sex) features that have their own partition rows. These are combinatorially
expanded, limited by this BIG_COMBO constant.
"""
