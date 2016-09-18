import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation

__author__ = 'Denis Surzhko'


class WoE:
    """
    Basic functionality for WoE bucketing of continuous and discrete variables
    :param self.bins: DataFrame WoE transformed variable and all related statistics
    :param iv: Information Value of the transformed variable
    """
    def __init__(self, qnt_num=16, min_block_size=16, spec_values=None, v_type='c', bins=None, t_type='b'):
        """
        :param qnt_num: Number of buckets (quartiles) for continuous variable split
        :param min_block_size: minimum number of observation in each bucket (continuous variables)
        :param spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        :param v_type: 'c' for continuous variable, 'd' - for discrete
        :param bins: Predefined bucket borders for continuous variable split
        :t_type : Binary 'b' or continous 'c' target variable
        :return: initialized class
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of goods and bads in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type # Type of target variable
        if type(spec_values) == dict: # Parsing special values to dict for cont variables
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
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise TypeError("pandas.Series type expected")
        if not x.size == y.size:
            raise Exception("Y size don't match Y size")
        # Calc total good bad ratio in the sample
        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample")
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")
        # setting discrete values as special values
        if self.type == 'd':
            sp_values = {i: 'd_' + str(i) for i in x.unique()}
            if len(sp_values) > 100:
                raise type("DiscreteVarOverFlowError", (Exception,),
                           {"args": ('Discrete variable with too many unique values (more than 100)',)})
            else:
                self.spec_values = sp_values if self.spec_values is None else sp_values.update(self.spec_values)
        # Make data frame for calcs
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
        Fit WoE transformation :param x: continuous or discrete predictor :param y: binary target variable :return:
        WoE transformed variable
        """
        self.fit(x, y)
        return self.df['woe']

    def _split_sample(self, df):
        if self.type == 'd':
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
        if self._predefined_bins is None:
            try:
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False)
            except ValueError as ex:
                if ex.args[0].startswith('Bin edges must be unique'):
                    ex.args = ('Please reduce number of bins or encode frequent items as special values',) + ex.args
                    raise
            bins = np.append(-float("inf"), bins[1:-1])
        else:
            bins = self._predefined_bins
            if bins[0] != float("-Inf"):
                bins = np.append(-float("inf"), bins)
            cuts = pd.cut(df['X'], bins=np.append(bins, float("inf")),
                          labels=np.arange(len(bins)).astype(str))
        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        return df, c_bins

    def _calc_stat(self):
        # calculating WoE
        stat = self.df.groupby("labels")['Y'].agg({'mean': np.mean, 'bad': np.count_nonzero, 'obs': np.size}).copy()
        if self.t_type != 'b':
            stat['bad'] = stat['mean'] * stat['obs']
        stat['good'] = stat['obs'] - stat['bad']
        t_good = np.maximum(stat['good'].sum(), 0.5)
        t_bad = np.maximum(stat['bad'].sum(), 0.5)
        stat['woe'] = stat.apply(self._bucket_woe, axis=1) + np.log(t_good / t_bad)
        iv_stat = (stat['bad'] / t_bad - stat['good'] / t_good) * stat['woe']
        self.iv = iv_stat.sum()
        # adding stat data to bins
        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
        label_woe = self.bins[['woe', 'labels']].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, left_on=['labels'], right_on=['labels'])

    def transform(self, x):
        """
        Transforms input variable according to previously fitted rule
        :param x: input variable
        :return: DataFrame with transformed with original and transformed variables
        """
        if not isinstance(x, pd.Series):
            raise TypeError("pandas.Series type expected")
        if self.bins is None:
            raise Exception('Fit the model first, please')
        df = pd.DataFrame({"X": x, 'order': np.arange(x.size)})
        # splitting to discrete and continous pars
        df_sp_values, df_cont = self._split_sample(df)
        # assigning labels to discrete part
        df_sp_values['labels'] = df_sp_values['X'].apply(
                 lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        # assigning labels to continuous part
        c_bins = self.bins[self.bins['labels'].apply(lambda x: not x.startswith('d_'))]
        if not self.type == 'd':
            cuts = pd.cut(df_cont['X'], bins=np.append(c_bins["bins"], float("inf")), labels=c_bins["labels"])
            df_cont['labels'] = cuts.apply(str)
        # Joining continuous and discrete parts
        df = df_sp_values.append(df_cont)
        # assigning woe
        df = pd.merge(df, self.bins[['woe', 'labels']], left_on=['labels'], right_on=['labels'])
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
        if label2 is None and not label1.startswith('d_'): # removing bucket for continuous variable
            c_bins = c_bins[c_bins['labels'] != label1]
        else:
            if not (label1.startswith('d_') and label2.startswith('d_')):
                raise Exception('Labels should be discrete simultaneously')
            bin1 = self.bins[self.bins['labels'] == label1]['bins'].iloc[0]
            bin2 = self.bins[self.bins['labels'] == label2]['bins'].iloc[0]
            spec_values[bin1] = label1 + '_' + label2
            spec_values[bin2] = label1 + '_' + label2
        woe = WoE(self.__qnt_num, self._min_block_size, spec_values, self.type, c_bins['bins'], self.t_type)
        return woe.fit(self.df['X'], self.df['Y'])

    def plot(self):
        """
        Plot WoE transformation and default rates
        :return: plotting object
        """
        index = np.arange(self.bins.shape[0])
        bar_width = 0.8
        fig = plt.figure()
        plt.title('Number of Observations and WoE per bucket')
        ax = fig.add_subplot(111)
        ax.set_ylabel('Observations')
        plt.xticks(index + bar_width / 2, self.bins['labels'])
        p1 = plt.bar(index, self.bins['obs'], bar_width, color='b', label='Observations')
        ax2 = ax.twinx()
        ax2.set_ylabel('Weight of Evidence')
        p2 = ax2.plot(index + bar_width / 2 , self.bins['woe'], 'bo-', linewidth=4.0, color='r', label='WoE')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels)
        fig.autofmt_xdate()
        return fig

    def optimize(self, criterion=None, fix_depth=None, max_depth=None, cv=3):
        """
        WoE bucketing optimization (continuous variables only)
        :param criterion: binary tree split criteria
        :param fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        :param max_depth: maximum tree depth for a optimum cross-validation search
        :param cv: number of cv buckets
        :return: WoE class with optimized continuous variable split
        """
        if self.t_type == 'b':
            tree_type = tree.DecisionTreeClassifier
        else:
            tree_type = tree.DecisionTreeRegressor
        m_depth = int(np.log2(self.__qnt_num))+1 if max_depth is None else max_depth
        cont = self.df['labels'].apply(lambda x: not x.startswith('d_'))
        X = np.array(self.df[cont]['X'])
        y = np.array(self.df[cont]['Y'])
        X = X.reshape(X.shape[0],1)
        start = 1
        cv_scores = []
        if fix_depth is None:
            for i in range(start, m_depth):
                if criterion is None:
                    d_tree = tree_type(max_depth=i)
                else:
                    d_tree = tree_type(criterion=criterion, max_depth=i)
                scores = cross_validation.cross_val_score(d_tree, X, y,  cv=cv)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
        else:
            best = fix_depth
        final_tree = tree_type(max_depth=best)
        final_tree.fit(X, y)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.threshold > 0]
        opt_bins = np.sort(opt_bins)
        new_woe = WoE(self.__qnt_num, self._min_block_size, self.spec_values, self.type, opt_bins, self.t_type)
        return new_woe.fit(self.df['X'], self.df['Y'])

    @staticmethod
    def _bucket_woe(x):
        t_bad = x['bad']
        t_good = x['good']
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log(t_bad / t_good)


# Examples
if __name__ == "__main__":
    # Set target type: 'b' for default/non-default, 'c' for continous pd values
    t_type = 'c'
    # Set sample size
    N = 300
    # Random variables
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    if t_type == 'b':
        y = np.where(np.random.rand(N) + x1 + x2 > 2, 1, 0)
    else:
        y = np.random.rand(N) + x1 + x2
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) / 2
    # Inserting special values
    x1[0:20] = float('nan')
    x1[30:50] = float(0)
    x1[60:80] = float(1)
    # Initialize WoE object
    woe_def = WoE()
    woe = WoE(7, 30, spec_values={0: '0', 1: '1'}, v_type='c',  t_type=t_type)
    # Transform x1
    woe.fit(pd.Series(x1), pd.Series(y))
    # Transform x2 using x1 transformation rules
    woe.transform(pd.Series(x2))
    # Optimize x1 transformation using tree with maximal depth = 5 (optimal depth is chosen by cross-validation)
    woe2 = woe.optimize(max_depth=5)
    # Merge discrete bickets
    woe3 = woe.merge('d_0', 'd_1')
    # Merge 2 and 3 continuous buckets
    woe4 = woe3.merge('2')
    # Print Statistics
    print(woe.bins)
    # print(woe2.bins)
    # print(woe3.bins)
    # print(woe4.bins)
    # Plot and show WoE graph
    fig = woe.plot()
    plt.show(fig)
    fig = woe2.plot()
    plt.show(fig)
