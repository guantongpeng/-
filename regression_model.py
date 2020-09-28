# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:42:39 2020

@author: GuanTongpeng
"""
# %% import packages
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from help_function import *
warnings.filterwarnings('ignore')
# %% read data
data_path = save_path+'\data_class.csv'
origin_data = pd.read_csv(data_path,)  
cols = origin_data.columns.values.tolist()
# %% 

dataset = origin_data.drop(useless_columns, axis=1)  # 数据
dataset_cols = dataset.columns.values.tolist()
#label = origin_data["na2"].values  # 标签 
# 统计低钠、正常、高钠样本
na = dataset['na2']
origin_low_na_sum = dataset[na<135].shape[0]
origin_normal_na_sum = dataset[na>=135][na<=145].shape[0]
origin_high_na_sum = dataset[na>145].shape[0]
print('低钠血症：%s\n正常血钠：%s\n高钠血症：%s'%(origin_low_na_sum,origin_normal_na_sum,origin_high_na_sum))
#平衡化
from imblearn.under_sampling import RandomUnderSampler
dataset1 = dataset.drop(['class'], axis=1)  # 数据
data_cols = dataset1.columns.values.tolist()

class0 = origin_data["class"].values
rus=RandomUnderSampler(ratio={0:4007,1:4200,2:4201},random_state=42)
x_rus,class_rus=rus.fit_sample(dataset1,class0)
x_rus = pd.DataFrame(x_rus,columns=data_cols)
dataset = x_rus.drop(['na2'], axis=1)
dataset_cols = dataset.columns.values.tolist()
label = x_rus['na2'].values
#%%
#标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit_transform(dataset)
dataset_new = pd.DataFrame(ss,columns=dataset_cols)
#%% 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    dataset_new, label, test_size=0.25, random_state=42)
# print(X_train.dtypes)
#num_col = findnum(X_train,dataset_cols)
#%%
def model_score(model, model_name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(model_name + " model R^2:%s" % r2)
    print(model_name + " model RMSE:%s" % rmse)
    print(model_name + " model MAE:%s" % mae)
    return r2,rmse,mae

#%% lasso回归做特征选择，十折交叉验证
from sklearn.linear_model import Lasso, LassoCV
alphas = 10 ** np.linspace(-5, 5, 100)
lasso_cofficients = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, normalize=True, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_cofficients.append(lasso.coef_)
# alpha & coeff of LASSO regression
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# ploting style
plt.style.use('ggplot')
plt.plot(alphas, lasso_cofficients)
plt.xscale('log')
plt.axis('tight')
plt.title('alpha系数与LASSO回归系数的关系')
plt.xlabel('Log Alpha')
plt.ylabel('Cofficients')
plt.grid(True)
plt.show()

# %%

# cross validation of LASSO regression
lasso_cv = LassoCV(alphas=alphas, normalize=True, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)
# optimal alpha
lasso_best_alpha = lasso_cv.alpha_
print("LASSO回归最佳alpha:%s" % lasso_best_alpha)
# building model based on optimal alpha
lasso = Lasso(alpha=lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X_train, y_train)

# lasso.coef_
# print('LASSO回归系数:%s' %lasso.coef_)
# predicting
lasso_predict = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, lasso_predict)
# score = lasso.score(X_test,y_test)
print('LASSO回归R^2:%s' % r2_score_lasso)
# metrics
RMSE = np.sqrt(mean_squared_error(y_test, lasso_predict))
print('LASSO回归RMSE:%s' % RMSE)
MAE = mean_absolute_error(y_test, lasso_predict)
print('LASSO回归MAE:%s' % MAE)

ranks = {}

def rank_to_dict(ranks, names, order=1):
    #minmax = MinMaxScaler()
    #ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 3), ranks)
    return dict(zip(names, ranks))

ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), X_test.columns.values)
# import operator
# sorted(d.items(), key=operator.itemgetter(1))
rank = sorted(ranks["Lasso"].items(), key=lambda item: item[1], reverse=True)
# print(rank)
# true and predicting
# plt.figure(figsize=(20, 10))
# plt.figure()
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')
# plt.plot(range(len(y_test)), y_test, color='red', label='true_label')
# plt.plot(range(len(y_test)), lasso_predict, color='black', label='predict_label')
# plt.legend(loc='upper left')
# plt.xlabel('sample_point')
# plt.ylabel('curve_comparision')
# plt.show()

# feature selection
import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np

keys = []
values = []
for i in range(len(rank)):
    keys.append(rank[i][0])
    values.append(rank[i][1])
    
count = np.sum(np.abs(lasso.coef_)>0.001)
label = keys[:count]
label = label[::-1]
print(label)
x = sorted(values[:count])
print(x)
idx = np.arange(len(x))
#plt.figure(figsize=(20, 10))
#plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.style.use('ggplot')
plt.barh(idx, x)
plt.yticks(idx + 0.2, label,fontsize=10)
plt.grid(axis='x')
# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.xlabel('The absolute value of regression coefficient')
#plt.ylabel('variables')
plt.title('LASSO regression variables performance',fontsize=12)
plt.savefig(save_path+'\lasso.png',dpi=300,bbox_inches = 'tight')
plt.show()

print("Lasso model: ", pretty_print_linear(lasso.coef_, X_test.columns.values, sort = True))

#%%
count = np.sum(np.abs(lasso.coef_)>0)
useless_feature = label[:-count]
use = label[-count:]
X_train = X_train[use]
X_test = X_test[use]
feature_cols = X_train.columns.values.tolist()

#%%
# 建立随机森林模型
rf_model = RandomForestRegressor(max_depth=18, max_features=0.271,
                      n_estimators=145, random_state=42)
rf_model.fit(X_train, y_train)
model_score(rf_model, "random-forest")
feature_importances_rf = rf_model.feature_importances_
index = X_train.columns.values.tolist()
print(sorted(zip(map(lambda x: round(x, 15), rf_model.feature_importances_), dataset_cols), reverse=True))

#%%
lgb_model = LGBMRegressor(learning_rate=0.071, max_depth=11,
              max_features=0.147, min_samples_split=18,
              n_estimators=203, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_predict = lgb_model.predict(X_test)
model_score(lgb_model, "lightgbm")
lgb_model.feature_importances_
# explainer = shap.TreeExplainer(lgb_model)
# shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.summary_plot(shap_values, X_test)

#%% 建立xgboost模型
xgb_model = XGBRegressor(learning_rate=0.049, max_depth=10,
             max_features=0.332, min_samples_split=2,
             n_estimators=175, 
             # reg_alpha=0.785,
             # reg_lambda=0.752,
             random_state=42)
xgb_model.fit(X_train, y_train)
model_score(xgb_model, "xgboost")
xgb_model.feature_importances_

# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决作图中文不显示的问题
# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.summary_plot(shap_values, X_test)
# from xgboost import plot_importance
# plot_importance(xgb_model)
# plt.show()

#%% 建立catboost模型
cat_model = CatBoostRegressor(n_estimators=199,
                              silent=True,
                              learning_rate=0.08,
                              max_depth=13,
                              random_state=42)
cat_model.fit(X_train, y_train)
model_score(cat_model, "catboost")
cat_model.feature_importances_

#%% 一致性评价
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    #mean = np.mean([data1, data2], axis=0)
    mean = data2
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference
    print(md,sd)
    plt.scatter(mean, diff,c='b',s=16)
    plt.axhline(md, color='k', linestyle='-')
    plt.axhline(md + 1.96*sd,linestyle='--')
    plt.axhline(md - 1.96*sd, linestyle='-.')
    plt.xlabel('real serum sodium(mmol/l)')
    plt.ylabel('difference between real and \npredictive value(mmol/l)')
    # plt.text(160,-15,'—— mean')
    # plt.text(160,-17,'-- mean+1.96SD')
    # plt.text(160,-19,'-. mean-1.96SD')
    # plt.text(160,-21,'. differences')
    plt.title(*args)
    plt.show()
    
import random
resultList=random.sample(range(0,len(y_test)),500); # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。

bland_altman_plot(rf_model.predict(X_test)[resultList],y_test[resultList],'random forest')
bland_altman_plot(xgb_model.predict(X_test)[resultList],y_test[resultList],'XGBoost')
bland_altman_plot(lgb_model.predict(X_test)[resultList],y_test[resultList],'LightGBM')
bland_altman_plot(cat_model.predict(X_test)[resultList],y_test[resultList],'CatBoost')

# #%% 画shap图,xgboost、lightgbm可以立马计算出结果，而GBDT和随机森林需要较长时间
# models = [lgb_model,xgb_model,cat_model,rf_model]
# models_name = ['LightGBM','XGBoost','CatBoost','Random forest']
# for i in range(len(models)):
#     explainer = shap.TreeExplainer(models[i])
#     shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
#     #shap.summary_plot(shap_values, X_test, plot_type="bar")
#     plt.figure()
#     #plt.rcParams['figure.dpi'] = 300 #分辨率
#     plt.title(models_name[i]+' model shap value')
#     shap.summary_plot(shap_values, X_test, show=False)#show=False才可以保存图，因为保存图要在show之前
#     plt.savefig(save_path+'\shap'+models_name[i]+'.png',dpi=300,bbox_inches = 'tight')

#%% print the score and the feature importance
rf_r2,rf_rmse,rf_mae = model_score(rf_model, "random-forest")
xgb_r2,xgb_rmse,xgb_mae = model_score(xgb_model, "xgboost")
lgb_r2,lgb_rmse,lgb_mae = model_score(lgb_model, "lightgbm")
cat_r2,cat_rmse,cat_mae = model_score(cat_model, "catboost")
rf_score = [rf_r2,rf_rmse,rf_mae]
xgb_score = [xgb_r2,xgb_rmse,xgb_mae]
lgb_score = [lgb_r2,lgb_rmse,lgb_mae]
cat_score = [cat_r2,cat_rmse,cat_mae]
all_score = pd.DataFrame(np.array([rf_score,xgb_score,lgb_score,cat_score]).T,
                         index=['R2','RMSE','MAE'],
                         columns=['Random forest','XGBoost','LightGBM','CatBoost'])
print(all_score)

#%%

rf_feature = sorted(zip(map(lambda x: round(x, 3), rf_model.feature_importances_), feature_cols), reverse=True)
xgb_feature = sorted(zip(map(lambda x: round(x, 3), xgb_model.feature_importances_), feature_cols), reverse=True)
lgb_feature = sorted(zip(map(lambda x: round(x, 3), lgb_model.feature_importances_), feature_cols), reverse=True)
cat_feature = sorted(zip(map(lambda x: round(x, 3), cat_model.feature_importances_), feature_cols), reverse=True)

rf_fea,xgb_fea,lgb_fea,cat_fea = [],[],[],[]
for i in range(len(rf_feature)):
    rf_fea.append(rf_feature[i][1])
    xgb_fea.append(xgb_feature[i][1])
    lgb_fea.append(lgb_feature[i][1])
    cat_fea.append(cat_feature[i][1])
lasso_feature = label[:count]
lasso_feature.reverse()
all_feature_importances = pd.DataFrame(np.array([lasso_feature, rf_fea,xgb_fea,lgb_fea,cat_fea]).T,
                                       columns=['LASSO','Random forest','XGBoost','LightGBM','CatBoost'])
#%%
X = pd.concat([X_train, X_test],axis=1)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
plt.title('xgboost model shap value')
shap.summary_plot(shap_values, X_test, show=False)#show=False才可以保存图，因为保存图要在show之前
#%%
shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
if len(shap_values) == 2:
    shap_values = shap_values[1]
#shap.summary_plot(shap_values, X_test, plot_type="bar")
#plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.savefig(save_path+'\shap.png',dpi=300,bbox_inches = 'tight')
#%%
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test,show=False)
plt.savefig(save_path+'\shap2.tif',dpi=300,bbox_inches = 'tight')
#%%
#from collections import Counter
im1 = all_feature_importances.values[:5,:].flatten()
im2 = all_feature_importances.values[:10,:].flatten()
im3 = all_feature_importances.values[:15,:].flatten()
im4 = all_feature_importances.values.flatten()
# out = Counter(im)
# print(out)
rank = pd.DataFrame([pd.value_counts(im1),pd.value_counts(im2),pd.value_counts(im3),pd.value_counts(im4)],index=['rank5_count','rank10_count','rank15_count','all_count']).T

all_score.to_csv(save_path+r'\all_score.csv')
all_feature_importances.to_csv(save_path+r'\all_feature_importances.csv')
rank.to_csv(save_path+r'\rank.csv')

# =============================================================================
# #%% stocking 模型融合
# from sklearn.linear_model import LinearRegression
# 
# p1 = rf_model.predict(X_train)
# p2 = xgb_model.predict(X_train)
# p3 = lgb_model.predict(X_train)
# p4 = cat_model.predict(X_train)
# 
# t1 = rf_model.predict(X_test)
# t2 = xgb_model.predict(X_test)
# t3 = lgb_model.predict(X_test)
# t4 = cat_model.predict(X_test)
# 
# p_train = pd.DataFrame([p1,p2,p3,p4]).T
# t_test = pd.DataFrame([t1,t2,t3,t4]).T
# 
# linear_model = LinearRegression(fit_intercept =False)
# linear_model.fit(p_train,y_train)
# y_pred = linear_model.predict(t_test)
# print("linear model R^2:%s" % r2_score(y_test, y_pred))
# print("linear model RMSE:%s" % np.sqrt(mean_squared_error(y_test, y_pred)))
# print("linear model MAE:%s" % mean_absolute_error(y_test, y_pred))
# print(linear_model.coef_)
# =============================================================================
#%% eicu database verify
import pandas as pd
use_cols = ['hematocrit', 'na_avg', 'Nacl', 'nephrotoxic drug', 
            'creatinine', 'ICU type', 'heart rate', 'respiratory rate', 
            'PO2', 'ICU first weight', 'diuretic drug', 'WBC count', 
            'age', 'sedative drug', 'gcs score', 'na_max', 'calcium', 
            'ptt', 'calculated total CO2', 'potassium', 'urine output', 
            'urea nitrogen', 'magnesium', 'glucose', 'chloride', 'na1','na2']

eicu_data = pd.read_csv(r'C:\Users\Administrator\Desktop\2020ICU血钠\eicu.csv',)  
eicu_data['ICU type'].replace(['CCU-CTICU', 'Cardiac ICU','CSICU','CTICU', 'MICU', 'SICU', 'Neuro ICU','Med-Surg ICU'], [0, 1,1, 1, 2, 3 ,4,3], inplace=True)
eicu_cols = eicu_data.columns.values.tolist()
verify_data_all = eicu_data[use_cols]


# 缺失值可视化
missing = verify_data_all.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
miss_rate = missing/len(verify_data_all)
miss_rate.plot.bar()
#删除特征数量小于25的行
verify_data_new = verify_data_all.dropna(axis=0, thresh=26)
verify_data_new.fillna(verify_data_new.median(),inplace=True)

eicu_y = verify_data_new['na2']
verify_data = verify_data_new[use_cols[:-1]]
verify_data_cols = verify_data.columns.values.tolist()

#标准化
stand_data = dataset[use]
stand = StandardScaler().fit(stand_data)
verify_data = stand.transform(verify_data)
#转化成dataframe数据格式
verify_data = pd.DataFrame(verify_data,columns=verify_data_cols)

def eicu_score(model, model_name):
    y_pred = model.predict(verify_data)
    r2 = r2_score(eicu_y, y_pred)
    rmse = np.sqrt(mean_squared_error(eicu_y, y_pred))
    mae = mean_absolute_error(eicu_y, y_pred)
    print(model_name + " model R^2:%s" % r2)
    print(model_name + " model RMSE:%s" % rmse)
    print(model_name + " model MAE:%s" % mae)
    return r2,rmse,mae

eicu_score(rf_model, "RF")
eicu_score(xgb_model, "xgboost")
eicu_score(lgb_model, "lightgbm")
eicu_score(cat_model, "catboost")
rf_pred = rf_model.predict(verify_data)
#直接计算na1和na2差距
eicu_x = verify_data_new['na1']
r2 = r2_score(eicu_y, eicu_x)
rmse = np.sqrt(mean_squared_error(eicu_y, eicu_x))
mae = mean_absolute_error(eicu_y, eicu_x)
print('no' + " model R^2:%s" % r2)
print('no' + " model RMSE:%s" % rmse)
print('no' + " model MAE:%s" % mae)
  
bland_altman_plot(rf_model.predict(verify_data),eicu_y,'random forest')
bland_altman_plot(xgb_model.predict(verify_data),eicu_y,'XGBoost')
bland_altman_plot(lgb_model.predict(verify_data),eicu_y,'LightGBM')
bland_altman_plot(cat_model.predict(verify_data),eicu_y,'CatBoost')
bland_altman_plot(eicu_x,eicu_y,'differences between na1 and na2')
