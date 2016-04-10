__author__ = 'Denis Surzhko'

import pandas as pd
import numpy as np

inp_data = pd.read_csv("/home/den/PycharmProjects/woe/woe.csv", decimal=".")
max_qnt_num = 20
min_bck_size = 50.0


class WoE:
    def __init__(self, x, y, gb_ratio = None):
        # Data quality checks
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise TypeError("pandas.Series type expected")
        if not x.size == y.size:
            raise Exception("Y size don't match Y size")
        # Calc total good bad ratio in the samplef
        if gb_ratio is None:
            t_bad = np.sum(y)
            if t_bad == 0 or t_bad == y.size:
                raise ValueError("There should be BAD and GOOD observations in the sample")
            self._gb_ratio = t_bad / (y.size - t_bad)
        else:
            self._gb_ratio = gb_ratio
        # Max buckets num calc
        self.q_num = int(np.minimum(x.unique().size / min_bck_size, min_bck_size))
        # Make data frame for calcs
        self.df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        # Calculate continuous WoE
        self._calc_cont_woe()

    def _calc_cont_woe(self):
        cuts, bins = pd.qcut(self.df["X"], self.q_num, retbins=True, labels=False)
        self.df["cuts"] = cuts
        woe = self.df.groupby("cuts")['Y'].apply(self._bucket_woe) + np.log(self._gb_ratio)
        self.bins = pd.DataFrame({"bins": np.append(-float("inf"), bins[1:-1]), "labels": np.arange(len(bins)-1)})
        self.bins = pd.merge(self.bins, pd.DataFrame(woe), left_on=['labels'], right_index=True, how='inner')
        self.bins.rename(columns={"Y": "woe"}, inplace=True)
        self.df = pd.merge(self.df, self.bins, left_on=['cuts'], right_index=True)
        self.df.sort_values('order', inplace=True)
        # self.df.to_csv("c:\PyData\out.csv")
        # print(self.df)

    def assign_woe(self, x):
        cuts =  pd.cut(x, bins=np.append(self.bins["bins"], float("inf")), labels=self.bins["labels"])
        new_df = pd.DataFrame({"X": x, 'cuts': cuts, 'order': np.arange(x.size)})
        new_df = pd.merge(new_df, self.bins, left_on=['cuts'], right_on=['labels'])
        new_df.sort_values('order', inplace=True)
        return new_df[['X', 'cuts', 'woe']]


    @staticmethod
    def _bucket_woe(x):
        t_bad = np.sum(x)
        t_good = x.size - t_bad
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        woe = np.log(t_good / t_bad)
        return(np.log(t_good / t_bad))


a = WoE(inp_data["X2"], inp_data["Y"])
a.df.to_csv("out1.csv")
a.assign_woe(inp_data["X2"]).to_csv("out2.csv")
# self.inp_data = pd.DataFrame({"bins": self.q_cut, "Y": y})
# a = self.inp_data.groupby('bins').transform(_bucket_WoE)
        # print(self.inp_data.groupby('bins').transform(lambda x: print(x)))
              # transform(lambda x: x.mean()))
        # print(self._gb_ratio)





w = WoE(inp_data["X1"], inp_data["Y"])
