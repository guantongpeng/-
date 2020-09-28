# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:42:50 2020

@author: GuanTongpeng
"""

#%% import bags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from help_function import *
import warnings
warnings.filterwarnings('ignore')
#%% import data
data_path = save_path+'\data_class.csv'
origin_data = pd.read_csv(data_path,)

dataset = origin_data.drop(useless_columns, axis=1)  # 数据
dataset_cols = dataset.columns.values.tolist()
#数据平衡化处理
dataset1 = dataset.drop(['na2'], axis=1)  # 数据
data_cols = dataset1.columns.values.tolist()

class0 = origin_data["class"].values
rus=RandomUnderSampler(ratio={0:4007,1:4200,2:4201},random_state=42)
x_rus,class_rus=rus.fit_sample(dataset1,class0)
x_rus = pd.DataFrame(x_rus,columns=data_cols)

# x_low = x_rus[x_rus['class']<2]
# x_low.drop(['ICU type'],axis=1)
# x_low.to_csv(save_path+'lowna.csv')
# x_high = x_rus[x_rus['class']>0]
# x_high.drop(['ICU type'],axis=1)
# x_high.to_csv(save_path+'highna.csv')
# x_lh = x_rus[x_rus['class']!=1]
# x_lh.to_csv(save_path+'highlowna.csv')

dataset = x_rus.drop(['class'], axis=1)
dataset_cols = dataset.columns.values.tolist()
label = x_rus['class'].values
#%% 标准化
scale = StandardScaler()
dataset = scale.fit_transform(dataset)
dataset_new = pd.DataFrame(dataset,columns=dataset_cols)
#%% 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    dataset_new, label, test_size=0.25, random_state=42)
#定义模型评价指标
def model_scores(model_name, y_pred, y_ture,):
    accuracy = accuracy_score(y_ture, y_pred,)
    precision = precision_score(y_ture, y_pred,average="macro")
    recall = recall_score(y_ture, y_pred,average="macro")
    f1 = f1_score(y_ture, y_pred,average="macro")
    confusionmatrix = confusion_matrix(y_ture, y_pred)
    print(model_name + " model accuracy_score:%s" % accuracy)
    print(model_name + " model precision_score:%s" % precision)
    print(model_name + " model recall_score:%s" % recall)
    print(model_name + " model f1_score:%s" % f1)
    return accuracy,precision,recall,f1,confusionmatrix
#%%  逻辑回归模型
lr_model = LogisticRegression(max_iter=300,)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
model_scores('lr',lr_pred, y_test)

#%% 支持向量机模型
svm_model = SVC(C=1, gamma='scale',)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
model_scores('svm',svm_pred, y_test)

#%% 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
model_scores('rf',rf_pred, y_test)

#%% xgboost
xgb_model = XGBClassifier(n_estimators=200)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
model_scores('xgboost',xgb_pred, y_test)

#%% lightgbm
lgb_model = LGBMClassifier(n_estimators=200)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
model_scores('lightgbm',lgb_pred, y_test)

#%%
cat_model = CatBoostClassifier(n_estimators=200,
                               early_stopping_rounds=20,
                               silent=True)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
model_scores('catboost',cat_pred, y_test)

