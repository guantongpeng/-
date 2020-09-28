# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:42:39 2020

@author: GuanTongpeng
"""

# %% import packages
import pandas as pd
from help_function import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# %% read data
data_path = 'mimic_total.csv'
origin_data = pd.read_csv(data_path,) 
origin_cols = origin_data.columns.values.tolist() 
#print(origin_cols)
# %% data process
new_data = origin_data.drop(['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 'charttime', 'charttimenext', 'ethnicity'],axis=1)
new_data['gender'].replace(['M','F'], [0,1], inplace=True)
new_data['ICU type'].replace(['CCU', 'CSRU', 'MICU', 'SICU', 'TSICU'], [0, 1, 2, 3 ,4], inplace=True)

new_data['age'].replace([new_data[new_data['age']>90]['age']], 95, inplace=True)
cols = new_data.columns.values.tolist() 
# %% Exploratory Data Analysis
new_data.info()
data_describe = new_data.describe()
# 缺失值可视化
missing = new_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
miss_rate = missing/len(new_data)
miss_rate.plot.bar()

miss_index = missing[miss_rate>0.4].index
new_data = new_data.drop(miss_index,axis=1)
na = new_data['na2']
# 统计低钠、正常、高钠样本
origin_low_na_sum = new_data[na<135]
origin_normal_na_sum = new_data[na>=135][na<=145]
origin_high_na_sum = new_data[na>145]
#%% 多重插补填补缺失值，增大迭代次数速度太慢
# import numpy as np
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(max_iter=10, random_state=42,sample_posterior=True,)
# imp.fit(new_data)  
# new_data1 = imp.transform(new_data)
# new_data1 = pd.DataFrame(new_data1,columns=new_data.columns.values)
#%% 中位数填补缺失值
new_data.fillna(new_data.median(),inplace=True)
#%% 异常值处理
cols = new_data.columns.values.tolist()
#set(column).issubset(set(cols))
for i in column:
    cols.remove(i)

new_data,_,_ = box_plot_outliers(new_data, new_data, 4, cols)
new_data.to_csv(save_path+'\data_clean.csv',index=False)
#%%

# 打印数据分析报告
# import pandas_profiling
# data_eda = pandas_profiling.ProfileReport(new_data)
# data_eda.to_file('data_eda.html')

origin_data = new_data
classify = []
na = origin_data['na2']
for i in range(len(na)):
    if na[i]>145:
        classify.append(2)
    elif na[i]<135:
        classify.append(0)
    else:
        classify.append(1)
classify = pd.DataFrame(classify,columns=['class'])
data = pd.concat([origin_data,classify],axis=1)

data.to_csv(save_path+'\data_class.csv',index=False)
