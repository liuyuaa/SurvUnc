import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch import nn
import torch

np.random.seed(1234)
_ = torch.manual_seed(123)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Dataset:
    def __init__(self, dataset):

        df_train = pd.read_csv('./dataset/'+dataset+'/df_train.csv')
        df_val = pd.read_csv('./dataset/'+dataset+'/df_val.csv')
        df_test = pd.read_csv('./dataset/'+dataset+'/df_test.csv')

        if dataset == 'support':
            cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
            cols_leave = ["x1", "x2", "x3", "x4", "x5", "x6"]
        elif dataset == 'seer_bc':
            cols_standardize = ["Regional nodes examined (1988+)", "CS tumor size (2004-2015)",
                                "Total number of benign/borderline tumors for patient",
                                "Total number of in situ/malignant tumors for patient", ]
            cols_leave = ["Sex", "Year of diagnosis", "Race recode (W, B, AI, API)", "Histologic Type ICD-O-3",
                          "Laterality", "Sequence number", "ER Status Recode Breast Cancer (1990+)",
                          "PR Status Recode Breast Cancer (1990+)", "Summary stage 2000 (1998-2017)",
                          "RX Summ--Surg Prim Site (1998+)", "Reason no cancer-directed surgery",
                          "First malignant primary indicator", "Diagnostic Confirmation",
                          "Median household income inflation adj to 2019"]
        elif dataset == 'sac3':
            cols_standardize = ['x' + str(i) for i in range(0, 45)]
            cols_leave = []
        elif dataset == 'flchain':
            cols_standardize = ['kappa', 'lambda', 'creatinine', ]
            cols_leave = ['sex', 'age', 'flc.grp', 'mgus', ]
            df_train['duration'] = df_train['duration'] + 1e-10
            df_val['duration'] = df_val['duration'] + 1e-10
            df_test['duration'] = df_test['duration'] + 1e-10

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)

        self.x_train = x_mapper.fit_transform(df_train).astype('float32')
        self.x_val = x_mapper.transform(df_val).astype('float32')
        self.x_test = x_mapper.transform(df_test).astype('float32')

        get_target = lambda df: (df['duration'].values, df['event'].values)
        self.durations_train, self.events_train = get_target(df_train)
        self.durations_val, self.events_val = get_target(df_val)
        self.durations_test, self.events_test = get_target(df_test)
        self.et_train = np.array([(self.events_train[i], self.durations_train[i]) for i in range(len(self.events_train))], dtype=[('e', bool), ('t', float)])
        self.et_val = np.array([(self.events_val[i], self.durations_val[i]) for i in range(len(self.events_val))], dtype=[('e', bool), ('t', float)])
        self.et_test = np.array([(self.events_test[i], self.durations_test[i]) for i in range(len(self.events_test))], dtype=[('e', bool), ('t', float)])

