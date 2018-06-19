import copy
import itertools
import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from composed.consts import PREDICTION_HOLDOUT_RATIO, BIG_COMBO
from composed.util import aic, distcorr, tdiff, zdiff, merge_data

class DataMetaInterface(type):
    """
    Provides a way for COMPOSED to grab all known data source interfaces.
    """
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "registry"):
            cls.registry = {}
        if cls.source is None:
            return
        elif cls.source in cls.registry:
            raise NameError("An COMPOSED data source with name {} is already "
                            "registered! First registration: '{}', Second: '{}'"
                            .format(cls.source, cls.registry[cls.source], cls))
        else:
            cls.registry[cls.source] = cls

class ComposedDataSource(object, metaclass=DataMetaInterface):
    """
    Base class for composed data sources. Simply inherit from this and create a
    load function and a create_partition_matrix function, and a unique
    source_name for your source. Then you will be able to specify
    data_source="your_source_name" within the composed input args and everything
    should be peachy.
    """

    registry = {}

    source = None

    def __init__(self, composed, **kwargs):
        self.composed = composed
        self.prefixes = kwargs.get('prefixes', ["Thk", "SA", "VL", "LIThk", "LISA", "LIVL", "Sex", "BMI", "ICV"])
        self.group = kwargs.get('group', "Group")
        self.subj_id = kwargs.get('subject_id', "RID")
        self.cov = kwargs.get('covariate', None)
        self.kwargs = kwargs

        self.filepath = None
        self._x_names = None
        self.x_names = None

        self._train_x = None
        self.train_x = None
        self.train_y = None

        self._test_x = None
        self.test_x = None
        self.test_y = None

        self.train_subj_ids = None
        self.test_subj_ids = None

        self.x_sets = {'grp': {}}
        if self.cov:
            self.x_sets['cov'] = {}

        self.is_categorical = None

        self.partitions = None

    def loadfile(self, info_table_name):
        # load csv file
        _features = pd.read_csv(info_table_name)
        return _features

    def load(self, info_table_name, holdout_table_name):
        """
        Loads data from the initial data source(s).

        See :doc:`/composed` for details.
        """
        _features = self.loadfile(info_table_name)

        # Ensure the loaded data matches our expectations of grp, cov, and
        # feature prefixes
        expected = set(self.prefixes) | set([self.group, self.subj_id])
        if self.cov:
            expected.add(self.cov)
        assert expected.issubset([name.split("_")[0] for name in _features.columns])

        # Downselect data to the features starting with the feature prefixes
        _prefixes = tuple(["{}_".format(pref) for pref in self.prefixes])
        var_names = [n for n in _features.columns if n.startswith(_prefixes)]

        # Always include the covariate in our vars
        if self.cov:
            var_names = [self.cov] + var_names

        _x = pd.DataFrame(
            preprocessing.RobustScaler().fit_transform(_features[var_names]),
            columns=var_names
        )
        self._x_names = list(_x.columns.values)
        _x = _x.as_matrix()

        _y = np.array(_features[self.group])

        # Ensure data is split into a train and test/holdout
        # dataset. Test/holdout is used for final validation of the trained
        # model.
        if holdout_table_name is None:
            # Split the dataset into training and holdout/testing
            ss = StratifiedShuffleSplit(n_splits=1,
                                        test_size=PREDICTION_HOLDOUT_RATIO,
                                        random_state=self.composed.random_flag)
            idxs = []
            for train_idx, test_idx in ss.split(np.zeros(_x.shape[0]), _y):
                idxs.append((train_idx, test_idx))
            assert len(idxs) == 1
            _train, _test = idxs[0]

            self._train_x = _x[_train, :]
            self.train_y = _y[_train]

            self._test_x = _x[_test, :]
            self.test_y = _y[_test]

            _subj_ids = _features[self.subj_id].ravel()
            self.test_subj_ids = _subj_ids[_test]
            self.train_subj_ids = _subj_ids[_train]
        else:
            self._train_x = _x
            self.train_y = _y
            self.train_subj_ids = _features[self.subj_id].ravel()

            # Load the supplied test features
            _test_features = self.loadfile(holdout_table_name)
            self.test_y = np.array(_test_features[self.group])
            self.test_subj_ids = _test_features[self.subj_id].ravel()

            self._test_x = pd.DataFrame(
                preprocessing.RobustScaler().fit_transform(_test_features[var_names]),
                columns=var_names
            )
            assert self._x_names == list(self._test_x.columns.values)
            self._test_x = self._test_x.as_matrix()

        self.is_categorical = len(set(self.train_y.flatten())) <= 10
        corr_func = tdiff if self.is_categorical else zdiff
        diff_threshold = self.composed.min_tdiff_thresh if self.is_categorical else \
            self.composed.min_zdiff_thresh
        print(
            "It appears the group variable is {}, so using {} to calculate correlated features"
            .format(
                "categorical" if self.is_categorical else "continuous",
                "tdiff" if self.is_categorical else "zdiff")
        )

        if self.cov:
            _idx = self._x_names.index(self.cov) if self.cov else None
            cov_dat = np.column_stack([self.train_y, self._train_x[:, _idx]])
        else:
            cov_dat = self.train_y

        _group_idx = 0
        _cov_idx = 1 if self.cov else None
        _incr = 1
        if self.cov:
            _incr += 1
        _featsets = {}
        for prefix in _prefixes:
            feats = [featname for featname in self._x_names
                     if featname.startswith(prefix)]

            feat_idxs = [idx for idx, name in enumerate(self._x_names) if name in feats]

            _featsets[prefix] = (feats, np.column_stack([cov_dat, self._train_x[:, feat_idxs]]))

        # Analyze features in order to create filtered dataset. The filter is
        # based on correlation strength between features and the self.group
        # variable.
        #
        # Need to ensure this filtered dataset is only calculated
        # based on the training data!
        self.x_names = []
        if self.cov:
            self.x_names.append(self.cov)

        for prefix in _featsets:
            _fdata = {}
            for grp in self.x_sets:
                self.x_sets[grp][prefix] = {}
                _fdata[grp] = []

            featnames, featdat = _featsets[prefix]
            assert len(featnames) + _incr == featdat.shape[1]
            for feat_idx, featname in enumerate(featnames):
                grp_corr, cov_corr = corr_func(featdat, feat_idx + _incr, _group_idx, _cov_idx)
                _fdata['grp'].append((featname, grp_corr))
                if self.cov:
                    _fdata['cov'].append((featname, cov_corr))

            title = "Covariate Diff Results"
            print(title)
            print("-" * len(title))
            print("")
            print("")
            for grp, _fdats in _fdata.items():
                _fdats = sorted(_fdats, key=lambda x: x[1])
                for _name, diff in _fdats:
                    print("{}_{}\t\t{}".format(grp, _name, diff))
            print("\n\n")

            # Filter the features based on high positive and negative correlations
            for grp in self.x_sets:
                _fdata[grp] = sorted(_fdata[grp], key=lambda x: x[1])

                pos_diff = [t[0] for t in _fdata[grp] if t[1] >= diff_threshold]
                if len(pos_diff) > self.composed.max_combinatorial:
                    pos_diff = pos_diff[-int(self.composed.max_combinatorial):]
                if pos_diff:
                    self.x_names.extend(pos_diff)
                    self.x_sets[grp][prefix]['pos'] = pos_diff

                neg_diff = [t[0] for t in _fdata[grp] if t[1] <= - diff_threshold]
                if len(neg_diff) > self.composed.max_combinatorial:
                    neg_diff = neg_diff[:int(self.composed.max_combinatorial)]
                if neg_diff:
                    self.x_names.extend(neg_diff)
                    self.x_sets[grp][prefix]['neg'] = neg_diff


        # Downselect train_x and test_x to the filtered features
        _train_indices = [idx for idx, name in enumerate(self._x_names) if name in self.x_names]
        self.train_x = self._train_x[:, _train_indices]
        _test_indices = [idx for idx, name in enumerate(self._x_names) if name in self.x_names]
        self.test_x = self._test_x[:, _test_indices]

        # Ensure training and test/holdout have the same columns
        assert self.train_x.shape[1] and self.train_x.shape[1] == self.test_x.shape[1]


    def create_partitions(self):
        """Takes feature type into account while performing merge partitioning.

        :param classifier_coefs: the feature coefficients from running a
           classifier on the dataset
        :return: train_partitions, a matrix where each row represents a
           unique grouping of features to use to find an effective classifier
           with.

        See :doc:`/composed` for details.
        """
        train_features = self.train_x
        groups = {'grp': self.train_y}
        if self.cov:
            cov_idx = [idx for idx, name in enumerate(self.x_names) if name == self.cov]
            groups['cov'] = self.train_x[:, cov_idx]

        fullperm = []
        for grp in self.x_sets:
            grp_perm = []
            for prefix in self.x_sets[grp]:
                for _dir in self.x_sets[grp][prefix]:
                    featlist = self.x_sets[grp][prefix][_dir]
                    if not featlist:
                        continue
                    feat_idxs = [idx for idx, name in enumerate(self.x_names) if name in featlist]
                    t_feats = train_features[:,feat_idxs]
                    base_aic = aic(groups[grp], t_feats)
                    min_aic = base_aic
                    mscore = []
                    featopts = [(-1, k) for k in range(t_feats.shape[1])]
                    if len(featopts) > self.composed.max_combinatorial:
                        raise Exception("Nope!, I'm not going to make ~4^{}"
                                        " different merge combos!"
                                        .format(len(featopts)))
                    for mset in itertools.product(*featopts):
                        m1 = [i for i, k in enumerate(mset) if k == -1]
                        if len(m1) < 2:
                            # No merges, equivalent to base case
                            continue
                        mdata = merge_data(mset, t_feats)
                        m_aic = aic(groups[grp], mdata)
                        if m_aic <= min_aic:
                            mscore.append((m_aic, mset))
                    mscore = sorted(mscore, key=lambda x: x[0])
                    if len(mscore) > self.composed.max_merges:
                        mscore = mscore[:int(self.composed.max_merges)]
                    print("Group {}_{}{}: mscore1 len: {}"
                          .format(grp, prefix, _dir, len(mscore) or 1), flush=True)
                    min_aic = mscore[0][0] if mscore else min_aic
                    # Second round
                    mscore2 = []
                    _seensets = []
                    for _aic, _mset in mscore:
                        featopts2 = [k == -1 and tuple([k]) or (-2, k) for k in _mset]
                        for mset2 in itertools.product(*featopts2):
                            m2 = tuple([j for j, k in enumerate(mset2) if k == -2])
                            if len(m2) < 2:
                                continue
                            m1 = tuple([i for i, k in enumerate(mset2) if k == -1])
                            _orderless = tuple(sorted([m1, m2], key=lambda x: x[0]))
                            if _orderless in _seensets:
                                continue
                            _seensets.append(_orderless)
                            mdata2 = merge_data(mset2, t_feats)
                            m2_aic = aic(groups[grp], mdata2)
                            if m2_aic <= min_aic:
                                mscore2.append((m2_aic, mset2))
                    mscore2 = sorted(mscore2, key=lambda x: x[0])
                    if len(mscore2) > self.composed.max_merges:
                        mscore2 = mscore2[:int(self.composed.max_merges)]
                    print("Group {}_{}{}: mscore2 len: {}"
                          .format(grp, prefix, _dir, len(mscore2)), flush=True)
                    mscore3 = []
                    _seensets = []
                    for _aic, _mset in mscore2:
                        featopts3 = [k in (-1, -2) and tuple([k]) or (-3, k) for k in _mset]
                        for mset3 in itertools.product(*featopts3):
                            m3 = tuple([j for j, k in enumerate(mset3) if k == -3])
                            if len(m3) < 2:
                                continue
                            m2 = tuple([j for j, k in enumerate(mset3) if k == -2])
                            m1 = tuple([i for i, k in enumerate(mset3) if k == -1])
                            _orderless = tuple(sorted([m1, m2, m3], key=lambda x: x[0]))
                            if _orderless in _seensets:
                                continue
                            _seensets.append(_orderless)
                            mdata3 = merge_data(mset3, t_feats)
                            m3_aic = aic(groups[grp], mdata3)
                            if m3_aic < min_aic:
                                mscore3.append((m3_aic, mset3))
                    if len(mscore3) > self.composed.max_merges:
                        mscore3 = mscore3[:int(self.composed.max_merges)]
                    print("Group {}_{}{}: mscore3 len: {}"
                          .format(grp, prefix, _dir, len(mscore3)), flush=True)
                    if not mscore:
                        mscore.append((base_aic, list(range(t_feats.shape[1]))))
                    mscore.extend(mscore2)
                    mscore.extend(mscore3)
                    print("Group {}_{}{}: mscore total len: {}"
                          .format(grp, prefix, _dir, len(mscore)), flush=True)
                    grp_perm.append(((grp, prefix, _dir), mscore))
            print("Finding optimal {} groups".format(grp), flush=True)
            x_combos = []
            # Use itertools.product to get merge combos
            combo_sets = [range(len(fp[1])) for fp in grp_perm]
            combos = list(itertools.product(*combo_sets))
            grp_feats = [name
                       for pref in self.x_sets[grp]
                       for __dir in self.x_sets[grp][pref]
                       for name in self.x_sets[grp][pref][__dir]]
            feat_sels = []
            grp_base_idxs = [idx for idx, name in enumerate(self.x_names) if name in grp_feats]
            grp_base = train_features[:, grp_base_idxs]
            grp_base_aic = aic(groups['grp'], grp_base)
            _merge_sel = np.zeros((1, len(grp_feats)))
            _merge_sel[0, :] = grp_base_idxs
            feat_sels.append((grp_base_aic, _merge_sel[0, :]))
            for combo in combos:
                _merge_sel = np.zeros((1, len(grp_feats)))
                _grp_base_sel = np.zeros((1, len(grp_feats)))
                for j, idx in enumerate(combo):
                    (x, prefix, _dir), msets = grp_perm[j]
                    _aic, mset = msets[idx]
                    m1 = [m for m, k in enumerate(mset) if k == -1]
                    m2 = [m for m, k in enumerate(mset) if k == -2]
                    m3 = [m for m, k in enumerate(mset) if k == -3]
                    featlist = self.x_sets[grp][prefix][_dir]
                    x_name_idxs = [idx for idx, name in enumerate(self.x_names) if name in featlist]
                    _merge_idxs = copy.copy(x_name_idxs)
                    if m1:
                        m_idx = x_name_idxs[m1[0]]
                        for _idx in m1:
                            _merge_idxs[_idx] = m_idx
                    if m2:
                        m_idx = x_name_idxs[m2[0]]
                        for _idx in m2:
                            _merge_idxs[_idx] = m_idx
                    if m3:
                        m_idx = x_name_idxs[m3[0]]
                        for _idx in m3:
                            _merge_idxs[_idx] = m_idx
                    for __idx, _x_name_idx in enumerate(_merge_idxs):
                        fname = featlist[__idx]
                        _mname = self.x_names[_x_name_idx]
                        _grp_feat_idx = grp_feats.index(fname)
                        _grp_mname_idx = grp_feats.index(_mname)
                        _grp_base_sel[0, _grp_feat_idx] = _grp_mname_idx
                        _merge_sel[0, _grp_feat_idx] = _x_name_idx
                mdats = merge_data(_grp_base_sel[0, :], grp_base)
                maic = aic(groups['grp'], mdats)
                if maic < grp_base_aic:
                    feat_sels.append((maic, _merge_sel[0, :]))
            feat_sels = sorted(feat_sels, key=lambda x: x[0])
            if len(feat_sels) > BIG_COMBO:
                print("The number of merge partitions exceeding the AIC cutoff"
                      " is greater than the permutation cutoff for grp={}."
                      " Limiting len(feat_sels) from {} to {}"
                      .format(grp, len(feat_sels), BIG_COMBO), flush=True)
                feat_sels = feat_sels[:BIG_COMBO]
            fullperm.append((grp, grp_feats, feat_sels))

        combined_combo_sets = [list(range(len(x[2]))) for x in fullperm]
        combo_sets = list(itertools.product(*combined_combo_sets))
        partitions = np.zeros((len(combo_sets), train_features.shape[1]))
        if self.cov:
            # Always include the covariate on its own in the permutations
            partitions[:, cov_idx] = cov_idx
        for i, combo in enumerate(combo_sets):
            for j, row in enumerate(combo):
                _, grp_feats, feat_sels = fullperm[j]
                _, sel = feat_sels[row]
                x_name_idxs = [idx for idx, name in enumerate(self.x_names) if name in grp_feats]
                partitions[i, x_name_idxs] = sel
        self.partitions = partitions

        return partitions

    def partition_names(self, partition_idx):
        partition = self.partitions[partition_idx, :]
        names_idx, lookup = np.unique(partition.astype(int),
                                      return_inverse=True)
        merge_names = [[] for idx in range(names_idx.shape[0])]
        for i in range(lookup.shape[0]):
            merge_names[lookup[i]].append(self.x_names[i])

        for i in range(len(merge_names)):
            merge_names[i] = "+".join(merge_names[i])

        return merge_names


class Connectomics(ComposedDataSource):

    source = "connectomics"

    def loadfile(self, info_table_name):
        vector_info = pd.read_csv(info_table_name)

        first_col = int(self.kwargs.get("first_col", 1))
        last_col = int(self.kwargs.get("last_col", 69))
        example_data = np.loadtxt(vector_info.Features[0],
                                  delimiter=',',
                                  usecols=range(first_col, last_col))
        mlen = example_data.shape[0] - 1
        assert mlen == example_data.shape[1]
        f_len = int((mlen**2 - mlen)/2)
        features = np.zeros((vector_info.Features.size, f_len))

        feature_names = []
        for j in range(mlen):
            for k in range(j+1, mlen):
                feature_names.append("{}_X_{}".format(example_data[0, j],
                                                      example_data[0, k]))

        # load twins data
        for i in range(0, vector_info.Features.size):
            subj_data = pd.read_csv(vector_info.Features[i],
                                    delimiter=',',
                                    skiprows=1,
                                    usecols=range(first_col, last_col))
            row = np.zeros((f_len))
            idx = 0
            for j in range(mlen):
                for k in range(j+1, mlen):
                    row[idx] = subj_data[i,j]
                    idx += 1

        np.savetxt(os.path.join(info_table_name, 'connectomics_combined.csv'), self.train_x, delimiter=', ', header=self.x_names)
        _feats = pd.DataFrame(features, columns = feature_names)
        _feats = pd.concat([_feats, vector_info.Group], axis=1)
        return _feats

class FreeSurfer(ComposedDataSource):
    source = 'freesurfer'
