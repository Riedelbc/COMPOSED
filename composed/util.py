from math import sqrt

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS

from composed.consts import NUM_PERFORMANCE_METRICS, NUM_GBM_PERFORMANCE_METRICS


def partition_matrix(partition):
    """
    Create a partition matrix for the dataset using the provided partition row
    """

    labels, labels_index = np.unique(partition, return_inverse=True)
    p = np.zeros((len(partition), len(labels)))
    p[np.arange(0, len(partition)), labels_index] = 1

    for i in range(0, len(labels)):
        p[:, i] = p[:, i] / np.sum(p[:, i])

    return p


def merge_data(partition, dataset):
    """
    Create a merged dataset from dataset using merge_rule
    """
    p = partition_matrix(partition)
    mergeset = np.zeros((dataset.shape[0], p.shape[1]))
    for i in range(0, dataset.shape[0]):
        mergeset[i, :] = dataset[i, :].dot(p)

    return mergeset


def m_sort(x):
    if len(x):
        return x[0]
    return False


def tdiff(dataset, feat_col, dx_col, sex_col=None):
    """
    Compute t score for a feature wrt a categorical diagnosis.

    See :meth:`composed.data.ComposedDataSource.load`
    """

    m = 0
    f = 1
    cn = 0
    dx = 1

    cn_feats = dataset[dataset[:, dx_col] == cn]
    cn_sigma = np.std(cn_feats[:, feat_col])
    cn_mean = np.mean(cn_feats[:, feat_col])
    cn_n = cn_feats.shape[0]

    dx_feats = dataset[dataset[:, dx_col] == dx]
    dx_sigma = np.std(dx_feats[:, feat_col])
    dx_mean = np.mean(dx_feats[:, feat_col])
    dx_n = dx_feats.shape[0]

    dx_tdiff = (cn_mean - dx_mean)/sqrt(cn_sigma**2/cn_n + dx_sigma**2/dx_n)
    sex_tdiff = None

    if sex_col is not None:
        m_feats = dataset[dataset[sex_col] == m]
        m_cn_feats = m_feats[m_feats[:, dx_col] == cn]
        m_cn_mean = np.mean(m_cn_feats[:, feat_col])
        m_cn_sigma = np.std(m_cn_feats[:, feat_col])
        m_cn_n = m_cn_feats.shape[0]
        m_dx_feats = m_feats[m_feats[:, dx_col] == dx]
        m_dx_mean = np.mean(m_dx_feats[:, feat_col])
        m_dx_sigma = np.std(m_dx_feats[:, feat_col])
        m_dx_n = m_dx_feats.shape[0]

        f_feats = dataset[dataset[sex_col] == f]
        f_cn_feats = f_feats[f_feats[:, dx_col] == cn]
        f_cn_mean = np.mean(f_cn_feats[:, feat_col])
        f_cn_sigma = np.std(f_cn_feats[:, feat_col])
        f_cn_n = f_cn_feats.shape[0]
        f_dx_feats = f_feats[f_feats[:, dx_col] == dx]
        f_dx_mean = np.mean(f_dx_feats[:, feat_col])
        f_dx_sigma = np.std(f_dx_feats[:, feat_col])
        f_dx_n = f_dx_feats.shape[0]

        sex_tdiff = (m_cn_mean - m_dx_mean)/ \
            sqrt(m_cn_sigma**2/m_cn_n + m_dx_sigma**2/m_dx_n) -  \
            (f_cn_mean - f_dx_mean) / \
            sqrt(f_cn_sigma**2/f_cn_n + f_dx_sigma**2/f_dx_n)

    return dx_tdiff, sex_tdiff


def zdiff(dataset, feat_col, dx_col, sex_col=None):
    """
    Compute distance correlation for a feature wrt a continuous diagnosis.
    """
    m = 0
    f = 1

    grp_diff = distcorr(dataset[:, feat_col], dataset[:, dx_col])
    z = 1 / (1 - grp_diff)
    sexz = None
    if sex_col is not None:
        mdata = dataset[dataset[sex_col] == m]
        mz = distcorr(mdata[:, feat_col], mdata[:, dx_col])
        mn = mdata.shape[0]

        fdata = dataset[dataset[sex_col] == f]
        fz = distcorr(fdata[:, feat_col], fdata[:, dx_col])
        fn = fdata.shape[0]

        sexz = (mz - fz) / sqrt(1/(mn - 3) + 1/(fn - 3))

    return z, sexz


def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def aic(group, train):
    """Akaike Information Criterion

    See :doc:`/composed` for details.
    """
    ols = OLS(group, train).fit()
    k = train.shape[1]
    return 2*k - 2*ols.llf


def measure_performance(classifier, x_names, train_x, train_y, test_x, test_y):

    classifier.fit(train_x, train_y)
    if isinstance(classifier, GradientBoostingRegressor):
        performance = np.zeros((1, NUM_GBM_PERFORMANCE_METRICS))
        performance[0, 0] = classifier.score(train_x, train_y)
        performance[0, 1] = classifier.score(test_x, test_y)
        p_coef, p_p = pearsonr(classifier.predict(test_x), test_y)
        performance[0, 2] = p_coef
        performance[0, 3] = p_p
        performance[0, 4] = mean_squared_error(test_y, classifier.predict(test_x))

        coef = dict(zip(x_names, classifier.feature_importances_))
    else:
        y_pred = classifier.predict(test_x)
        y_true = test_y
        tp = np.count_nonzero((y_pred == 1) & (y_true == 1))
        tn = np.count_nonzero((y_pred == 0) & (y_true == 0))
        fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
        fn = np.count_nonzero((y_pred == 0) & (y_true == 1))

        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0

        accuracy = (tp + tn) / (tp + fn + tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2

        f1 = (2 * tp) / (2 * tp + fp + fn)
        if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) == 0:
            mcc = 0
        else:
            mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        ppv = bool(tp + fp) and (tp)/(tp + fp) or 0

        performance = np.zeros((1, NUM_PERFORMANCE_METRICS))
        performance[0, 0] = balanced_accuracy
        performance[0, 1] = accuracy
        performance[0, 2] = sensitivity
        performance[0, 3] = specificity
        performance[0, 4] = f1
        performance[0, 5] = mcc
        performance[0, 6] = ppv

        coef = dict(zip(x_names, classifier.coef_[0, :]))

    return coef, performance
