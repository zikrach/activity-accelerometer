import numpy as np
import pandas as pd
from hyperopt.pyll import scope

from sklearn.model_selection import (cross_val_score, train_test_split,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb

import os
import logging
# Let OpenMP use 4 threads to evaluate models - may run into errors
# if this is not set. Should be set before hyperopt import.
os.environ['OMP_NUM_THREADS'] = '32'

import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# -----------------------------------------------------
#                       SETUP
# -----------------------------------------------------

SEED = 42  # Fix the random state to the ultimate answer in life.
# Initialize logger
#logging.basicConfig(filename="xgb_hyperopt.log", level=print)


# -----------------------------------------------------
#                       HYPEROPT
# -----------------------------------------------------

def score(params):
    print("Training with params: ")
    print(params)
    # Delete 'n_estimators' because it's only a constructor param
    # when you're using  XGB's sklearn API.
    # Instead, we have to save 'n_estimators' (# of boosting rounds)
    # to xgb.cv().
    
    num_boost_round = int(params['n_estimators'])
    del params['n_estimators']

    
    
    dtrain = xgb.DMatrix(X_train_ho.values, 
                         label = y_train_ho)
    # As of version 0.6, XGBoost returns a dataframe of the following form:
    # boosting iter | mean_test_err | mean_test_std | mean_train_err | mean_train_std
    # boost iter 1 mean_test_iter1 | mean_test_std1 | ... | ...
    # boost iter 2 mean_test_iter2 | mean_test_std2 | ... | ...
    # ...
    # boost iter n_estimators

    score_history = xgb.cv(params, dtrain, num_boost_round,
                           nfold=5, #stratified=True,
                           early_stopping_rounds=50,
                           verbose_eval=20, )
    # Only use scores from the final boosting round since that's the one
    # that performed the best.
    #print(score_history.columns)
    mean_final_round = score_history['test-rmse-mean'].values[-1]
    std_final_round = score_history['test-rmse-std'].values[-1]
    print("\tMean Score: {0}\n".format(mean_final_round))
    print("\tStd Dev: {0}\n\n".format(std_final_round))
    # score() needs to return the loss (1 - score)
    # since optimize() should be finding the minimum, and AUC
    # naturally finds the maximum.
    #loss = 1 - mean_final_round
    return {'loss': mean_final_round, 'status': STATUS_OK}


def optimize(
    # trials,
        random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here),
    finds the best hyperparameters.
    """

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 400, 10)),
        'eta': hp.quniform('eta', 0.025, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
        'max_depth':  scope.int(hp.choice('max_depth', np.arange(1, 14, dtype=int))),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' :  hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'booster': 'gbtree',
        'tree_method': 'hist',
        'seed': random_state,
        
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        
        'n_jobs': -1,
        'nthread' : 12
            }
    
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                # trials=trials,
                max_evals=100)
    return best


best_hyperparams = optimize(
    # trials
)
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)


