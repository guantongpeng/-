# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:15:27 2020

@author: GuanTongpeng
"""

#%% 贝叶斯调参寻找最优参数
from bayes_opt import BayesianOptimization
init_points=3
n_iter=200

def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    rf_model = RandomForestRegressor(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=42)
    val = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    return val

rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': [100, 400],
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 20),
       
        }
    )

rf_bo.maximize(init_points=init_points,n_iter=n_iter)
#init_points您要执行多少步随机探索。随机探索可以通过使探索空间多样化而有所帮助。，n_iter代表迭代次数（即采样数）
print (rf_bo.max)#可以通过该属性访问找到的参数和目标值的最佳组合
para = rf_bo.max['params']
n_estimators = para['n_estimators']
min_samples_split = para['min_samples_split']
max_features = para['max_features']
max_depth = para['max_depth']

rf_model = RandomForestRegressor( n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=42)


def lgb_cv(n_estimators, min_samples_split, max_features, max_depth, learning_rate,reg_alpha,reg_lambda):
    lgb_model = LGBMRegressor(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            random_state=42)
    val = cross_val_score(lgb_model, X_train, y_train, cv=5).mean()
    return val

lgb_bo = BayesianOptimization(
        lgb_cv,
        {'n_estimators': [10, 250],
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15),
        'learning_rate': (0.01, 0.4),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1),
        }
    )
lgb_bo.maximize(init_points=init_points,n_iter=n_iter)
#init_points您要执行多少步随机探索。随机探索可以通过使探索空间多样化而有所帮助。，n_iter代表迭代次数（即采样数）
print (lgb_bo.max)#可以通过该属性访问找到的参数和目标值的最佳组合

para = lgb_bo.max['params']
n_estimators = para['n_estimators']
min_samples_split = para['min_samples_split']
max_features = para['max_features']
max_depth = para['max_depth']
learning_rate = para['learning_rate']

lgb_model = LGBMRegressor(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            random_state=42)

def xgb_cv(n_estimators, min_samples_split, max_features, max_depth, learning_rate,reg_alpha,reg_lambda):
    xgb_model = XGBRegressor(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            random_state=42)
    val = cross_val_score(xgb_model, X_train, y_train, cv=5).mean()
    return val

xgb_bo = BayesianOptimization(
        xgb_cv,
        {'n_estimators': [100, 400],
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 25),
        'learning_rate': (0.01, 0.4),
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1],
        }
    )
xgb_bo.maximize(init_points=init_points,n_iter=n_iter)
#init_points您要执行多少步随机探索。随机探索可以通过使探索空间多样化而有所帮助。，n_iter代表迭代次数（即采样数）
print (xgb_bo.max)#可以通过该属性访问找到的参数和目标值的最佳组合

para = xgb_bo.max['params']
n_estimators = para['n_estimators']
min_samples_split = para['min_samples_split']
max_features = para['max_features']
max_depth = para['max_depth']
learning_rate = para['learning_rate']

xgb_model = XGBRegressor(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            random_state=42)

def cat_cv(n_estimators, max_depth, learning_rate):
    cat_model = CatBoostRegressor(
            n_estimators=int(n_estimators),
            silent=True,
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            random_state=42)
    val = cross_val_score(cat_model, X_train, y_train, cv=5).mean()
    return val

cat_bo = BayesianOptimization(
        cat_cv,
        {'n_estimators': [100, 300],
        'max_depth': (5, 20),
        'learning_rate': (0.01, 0.4),
        }
    )
cat_bo.maximize(init_points=init_points,n_iter=init_points)
#init_points您要执行多少步随机探索。随机探索可以通过使探索空间多样化而有所帮助。，n_iter代表迭代次数（即采样数）
print (cat_bo.max)#可以通过该属性访问找到的参数和目标值的最佳组合

para = cat_bo.max['params']
n_estimators = para['n_estimators']

max_depth = para['max_depth']
learning_rate = para['learning_rate']

cat_model = CatBoostRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate = learning_rate,
            silent=True,
            random_state=2)

print(rf_model)
print(lgb_model)
print(xgb_model)
print('catboost params:')
print(cat_model.get_params())
