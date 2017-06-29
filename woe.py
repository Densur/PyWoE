import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score

__author__ = 'Denis Surzhko'


# ver 0.1.6
# default woe replcement for labels not observable in traning dataset addded in transform method
# self.plot optimized for matplotlib version 2
# self.type to self.v_type renamed (for convenience)
class WoE:
    """
    Basic functionality for WoE bucketing of continuous and discrete variables
    :param self.bins: DataFrame WoE transformed variable and all related statistics
    :param self.iv: Information Value of the transformed variable
    """

    def __init__(self, qnt_num=16, min_block_size=16, spec_values=None, v_type='c', bins=None, t_type='b'):
        """
        :param qnt_num: Number of buckets (quartiles) for continuous variable split
        :param min_block_size: min number of obs in bucket (continuous variables), incl. optimization restrictions
        :param spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        :param v_type: 'c' for continuous variable, 'd' - for discrete
        :param bins: Predefined bucket borders for continuous variable split
        :t_type : Binary 'b' or continous 'c' target variable
        :return: initialized class
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.v_type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type  # Type of target variable
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}

    def fit(self, x, y):
        """
        Fit WoE transformation
        :param x: continuous or discrete predictor
        :param y: binary target variable
        :return: WoE class
        """
        # Data quality checks
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not x.size == y.size:
            raise Exception("Y size don't match Y size")
        # Calc total good bad ratio in the sample
        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample")
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")
        # setting discrete values as special values
        if self.v_type == 'd':
            sp_values = {i: 'd_' + str(i) for i in x.unique()}
            if len(sp_values) > 100:
                raise type("DiscreteVarOverFlowError", (Exception,),
                           {"args": ('Discrete variable with too many unique values (more than 100)',)})
            else:
                if self.spec_values:
                    sp_values.update(self.spec_values)
                self.spec_values = sp_values
        # Make data frame for calculations
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        # Separating NaN and Special values
        df_sp_values, df_cont = self._split_sample(df)
        # # labeling data
        df_cont, c_bins = self._cont_labels(df_cont)
        df_sp_values, d_bins = self._disc_labels(df_sp_values)
        # getting continuous and discrete values together
        self.df = df_sp_values.append(df_cont)
        self.bins = d_bins.append(c_bins)
        # calculating woe and other statistics
        self._calc_stat()
        # sorting appropriately for further cutting in transform method
        self.bins.sort_values('bins', inplace=True)
        # returning to original observation order
        self.df.sort_values('order', inplace=True)
        self.df.set_index(x.index, inplace=True)
        return self

    def fit_transform(self, x, y):
        """
        Fit WoE transformation
        :param x: continuous or discrete predictor
        :param y: binary target variable
        :return: WoE transformed variable
        """
        self.fit(x, y)
        return self.df['woe']

    def _split_sample(self, df):
        if self.v_type == 'd':
            return df, None
        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        return df_sp_values, df_cont

    def _disc_labels(self, df):
        df['labels'] = df['X'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        d_bins = pd.DataFrame({"bins": df['X'].unique()})
        d_bins['labels'] = d_bins['bins'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        return df, d_bins

    def _cont_labels(self, df):
        # check whether there is a continuous part
        if df is None:
            return None, None
        # Max buckets num calc
        self.qnt_num = int(np.minimum(df['X'].unique().size / self._min_block_size, self.__qnt_num)) + 1
        # cuts - label num for each observation, bins - quartile thresholds
        bins = None
        cuts = None
        if self._predefined_bins is None:
            try:
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False)
            except ValueError as ex:
                if ex.args[0].startswith('Bin edges must be unique'):
                    ex.args = ('Please reduce number of bins or encode frequent items as special values',) + ex.args
                    raise
            bins = np.append((-float("inf"),), bins[1:-1])
        else:
            bins = self._predefined_bins
            if bins[0] != float("-Inf"):
                bins = np.append((-float("inf"),), bins)
            cuts = pd.cut(df['X'], bins=np.append(bins, (float("inf"),)),
                          labels=np.arange(len(bins)).astype(str))
        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        return df, c_bins

    def _calc_stat(self):
        # calculating WoE
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        stat = self.df.groupby("labels")['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()
        if self.t_type != 'b':
            stat['bad'] = stat['mean'] * stat['obs']
        stat['good'] = stat['obs'] - stat['bad']
        t_good = np.maximum(stat['good'].sum(), 0.5)
        t_bad = np.maximum(stat['bad'].sum(), 0.5)
        stat['woe'] = stat.apply(self._bucket_woe, axis=1) + np.log(t_bad / t_good)
        iv_stat = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
        self.iv = iv_stat.sum()
        # adding stat data to bins
        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
        label_woe = self.bins[['woe', 'labels']].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, left_on=['labels'], right_on=['labels'])

    def transform(self, x, manual_woe=None, replace_missing=None):
        """
        Transforms input variable according to previously fitted rule
        :param x: input variable
        :param manual_woe: one can change fitted woe with manual values by providing dict {label: new_woe_value}
        :replace_missing: replace woe for labels not observable in traning dataset by this value
        :return: DataFrame with transformed with original and transformed variables
        """
        if not isinstance(x, pd.Series):
            raise TypeError("pandas.Series type expected")
        if self.bins is None:
            raise Exception('Fit the model first, please')
        df = pd.DataFrame({"X": x, 'order': np.arange(x.size)})
        # splitting to discrete and continous pars
        df_sp_values, df_cont = self._split_sample(df)
        # Replacing original with manual woe
        tr_bins = self.bins[['woe', 'labels']].copy()
        if manual_woe:
            if not type(manual_woe) == dict:
                TypeError("manual_woe should be dict")
            else:
                for key in manual_woe:
                    tr_bins['woe'].mask(tr_bins['labels'] == key, manual_woe[key], inplace=True)

        if replace_missing is not None:
            tr_bins = tr_bins.append({'labels': 'd__transform_missing_replacement__', 'woe': replace_missing},
                                     ignore_index=True)

        # function checks existence of special values, raises error if sp do not exist in training set
        def get_sp_label(x_):
            if x_ in self.spec_values.keys():
                return self.spec_values[x_]
            else:
                str_x = 'd_' + str(x_)
                if str_x in list(self.bins['labels']):
                    return str_x
                else:
                    if replace_missing is not None:
                        return 'd__transform_missing_replacement__'
                    else:
                        raise ValueError('Value {} does not exist in the training set'.format(str_x))

        # assigning labels to discrete part
        df_sp_values['labels'] = df_sp_values['X'].apply(get_sp_label)
        # assigning labels to continuous part
        c_bins = self.bins[self.bins['labels'].apply(lambda z: not z.startswith('d_'))]
        if self.v_type != 'd':
            cuts = pd.cut(df_cont['X'], bins=np.append(c_bins["bins"], (float("inf"),)), labels=c_bins["labels"])
            df_cont['labels'] = cuts.astype(str)
        # Joining continuous and discrete parts
        df = df_sp_values.append(df_cont)
        # assigning woe
        df = pd.merge(df, tr_bins[['woe', 'labels']].drop_duplicates(), left_on=['labels'], right_on=['labels'])
        # returning to original observation order
        df.sort_values('order', inplace=True)
        return df.set_index(x.index)

    def merge(self, label1, label2=None):
        """
        Merge of buckets with given labels
        In case of discrete variable, both labels should be provided. As the result labels will be marget to one bucket.
        In case of continous variable, only label1 should be provided. It will be merged with the next label.
        :param label1: first label to merge
        :param label2: second label to merge
        :return:
        """
        spec_values = self.spec_values.copy()
        c_bins = self.bins[self.bins['labels'].apply(lambda x: not x.startswith('d_'))].copy()
        if label2 is None and not label1.startswith('d_'):  # removing bucket for continuous variable
            c_bins = c_bins[c_bins['labels'] != label1]
        else:
            if not (label1.startswith('d_') and label2.startswith('d_')):
                raise Exception('Labels should be discrete simultaneously')
            for i in self.bins[self.bins['labels'] == label1]['bins']:
                spec_values[i] = label1 + '_' + label2
            bin2 = self.bins[self.bins['labels'] == label2]['bins'].iloc[0]
            spec_values[bin2] = label1 + '_' + label2
        new_woe = WoE(self.__qnt_num, self._min_block_size, spec_values, self.v_type, c_bins['bins'], self.t_type)
        return new_woe.fit(self.df['X'], self.df['Y'])

    def plot(self, sort_values=True, labels=False):
        """
        Plot WoE transformation and default rates
        :param sort_values: whether to sort discrete variables by woe, continuous by labels
        :param labels: plot labels or intervals for continuous buckets
        :return: plotting object
        """
        bar_width = 0.8
        woe_fig = plt.figure()
        plt.title('Number of Observations and WoE per bucket')
        ax = woe_fig.add_subplot(111)
        ax.set_ylabel('Observations')
        plot_data = self.bins[['labels', 'woe', 'obs', 'bins']].copy().drop_duplicates()

        # creating plot labels
        if self.v_type != 'd':
            cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
            plot_data['plot_bins'] = plot_data['bins'].apply(lambda x: '{:0.2g}'.format(x))
            temp_data = plot_data[cont_labels].copy()

            right_bound = temp_data['plot_bins'].iloc[1:].append(pd.Series(['Inf']))
            temp_data['plot_bins'] = temp_data['plot_bins'].add(' : ').add(list(right_bound))

            plot_data = temp_data.append(plot_data[~cont_labels])
            cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
            plot_data['plot_bins'] = np.where(cont_labels, plot_data['plot_bins'], plot_data['labels'])
        else:
            plot_data['plot_bins'] = plot_data['bins']

        # sorting
        if sort_values:
            if self.v_type != 'd':
                cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
                temp_data = plot_data[cont_labels].sort_values('bins')
                plot_data = temp_data.append(plot_data[~cont_labels].sort_values('labels'))
            else:
                plot_data.sort_values('woe', inplace=True)

        # start plotting
        index = np.arange(plot_data.shape[0])
        plt.xticks(index, plot_data['labels'] if labels else plot_data['plot_bins'])
        plt.bar(index, plot_data['obs'], bar_width, color='b', label='Observations')
        ax2 = ax.twinx()
        ax2.set_ylabel('Weight of Evidence')
        ax2.plot(index, plot_data['woe'], 'bo-', linewidth=4.0, color='r', label='WoE')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels)
        woe_fig.autofmt_xdate()
        return woe_fig

    def optimize(self, criterion=None, fix_depth=None, max_depth=None, cv=3, scoring=None, min_samples_leaf=None):
        """
        WoE bucketing optimization (continuous variables only)
        :param criterion: binary tree split criteria
        :param fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        :param max_depth: maximum tree depth for a optimum cross-validation search
        :param cv: number of cv buckets
        :param scoring: scorer for cross_val_score
        :param min_samples_leaf: minimum number of observations in each of optimized buckets
        :return: WoE class with optimized continuous variable split
        """
        if self.t_type == 'b':
            tree_type = tree.DecisionTreeClassifier
        else:
            tree_type = tree.DecisionTreeRegressor
        m_depth = int(np.log2(self.__qnt_num)) + 1 if max_depth is None else max_depth
        cont = self.df['labels'].apply(lambda z: not z.startswith('d_'))
        x_train = np.array(self.df[cont]['X'])
        y_train = np.array(self.df[cont]['Y'])
        x_train = x_train.reshape(x_train.shape[0], 1)
        if not min_samples_leaf:
            min_samples_leaf = self._min_block_size
        start = 1
        cv_scores = []
        if fix_depth is None:
            for i in range(start, m_depth):
                if criterion is None:
                    d_tree = tree_type(max_depth=i, min_samples_leaf=min_samples_leaf)
                else:
                    d_tree = tree_type(criterion=criterion, max_depth=i, min_samples_leaf=min_samples_leaf)
                scores = cross_val_score(d_tree, x_train, y_train, cv=cv, scoring=scoring)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
        else:
            best = fix_depth
        final_tree = tree_type(max_depth=best, min_samples_leaf=min_samples_leaf)
        final_tree.fit(x_train, y_train)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]
        opt_bins = np.sort(opt_bins)
        new_woe = WoE(self.__qnt_num, self._min_block_size, self.spec_values, self.v_type, opt_bins, self.t_type)
        return new_woe.fit(self.df['X'], self.df['Y'])

    @staticmethod
    def _bucket_woe(x):
        t_bad = x['bad']
        t_good = x['good']
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log(t_good / t_bad)


# Examples
if __name__ == "__main__":
    # Set target type: 'b' for default/non-default, 'c' for continous pd values
    t_type_ = 'c'
    # Set sample size
    N = 300
    # Random variables
    x1 = np.random.rand(N) - 0.5
    x2 = np.random.rand(N) - 0.5
    if t_type_ == 'b':
        y_ = np.where(np.random.rand(N, ) + x1 + x2 > 2, 1, 0)
    else:
        y_ = np.random.rand(N) + x1 + x2
        y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) / 2
    # Inserting special values
    x1[0:20] = float('nan')
    x1[30:50] = float(0)
    x1[60:80] = float(1)
    x2[0:10] = float('nan')
    x2[10:20] = float(1)
    x2[30:40] = float(0)
    # Initialize WoE object
    woe_def = WoE()
    woe = WoE(7, 30, spec_values={0: '0', 1: '1'}, v_type='c', t_type=t_type_)
    # Transform x1
    woe.fit(pd.Series(x1), pd.Series(y_))
    # Transform x2 using x1 transformation rules
    woe.transform(pd.Series(x2))
    fig = woe.plot()
    plt.show(fig)
    # refit to check nan replacement by default value
    x1[0:20] = 0
    woe.fit(pd.Series(x1), pd.Series(y_))
    # Optimize x1 transformation using tree with maximal depth = 5 (optimal depth is chosen by cross-validation)
    woe2 = woe.optimize(max_depth=5, min_samples_leaf=int(N / 3))
    woe2 = woe.optimize(max_depth=3, scoring='r2')
    # Merge discrete buckets
    woe3 = woe.merge('d_0', 'd_1')
    # Merge 2 and 3 continuous buckets
    woe4 = woe3.merge('2')
    print('Transformation with manual woe')

    print(woe4.transform(pd.Series(x2), manual_woe={'d_0_d_1': -1, '2': 1}, replace_missing=0)
          [['labels', 'woe']].drop_duplicates())
    # Print Statistics
    print(woe.bins)
    print(woe4.bins)
    # Plot and show WoE graph
    fig = woe2.plot()
    plt.show(fig)
    fig = woe4.plot(labels=True)
    plt.show(fig)
    # Discrete target variable test
    data = pd.DataFrame({'sex': ['Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male'],
                         'sex2': ['Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male', 'Unknown']},)
    defaults = pd.Series([0, 1, 1, 1, 1, 0, 0, 1])
    d_var = WoE(t_type='d', v_type='d')
    d_var.fit(data['sex'], defaults)
    # replace missing 'Unknown' gender by default value in transform
    print(d_var.transform(data['sex2'], replace_missing=0))
    fig = d_var.plot()
    plt.show(fig)
