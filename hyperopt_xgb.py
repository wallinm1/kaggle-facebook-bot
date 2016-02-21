import pandas as pd
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os

def score(params):
    global df_scores
    params['n_estimators'] = int(params['n_estimators'])
    print "Training with params : "
    print params
    sel_pct = int(params['sel_pct'])
    del params['sel_pct']
    clf = xgb.XGBClassifier()
    clf.set_params(**params)
    pipeline = Pipeline([('selector', SelectPercentile(chi2, sel_pct)),
                         ('clf', clf)])
    scores = cross_val_score(pipeline, xtrain, y, scoring = 'roc_auc',cv = kf)
    score = scores.mean()
    print "\tScore {0}\n\n".format(score)
    row = [score, params['n_estimators'], params['learning_rate'],
           params['max_depth'], params['min_child_weight'],
           params['subsample'], params['gamma'],
           params['colsample_bytree'], sel_pct]
    df_scores.loc[len(df_scores.index)] = row
    df_scores.sort(columns = 'score', ascending = False, inplace = True)
    df_scores.to_csv(fname, index = False)
    return {'loss': score, 'status': STATUS_OK}

def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 5, 1000, 1),
             'learning_rate' : hp.quniform('learning_rate', 0.001, 0.5, 0.001),
             'max_depth' : hp.quniform('max_depth', 1, 13, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.4, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.4, 1, 0.05),
             'sel_pct' : hp.quniform('sel_pct', 1, 100, 1),
             'objective' : 'binary:logistic',
             'silent' : 1
             }
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=500)
    print best

xtrain = joblib.load('data/xtrain.pkl')
y = joblib.load('data/y.pkl')
nf = 4
kf = StratifiedKFold(y, n_folds = nf, random_state = 42, shuffle = True)
fname = 'hyperopt_xgb.csv'
if os.path.isfile(fname):
    df_scores = pd.read_csv(fname)
else:
    df_scores = pd.DataFrame(columns = ('score', 'n_estimators','learning_rate',
                                     'max_depth', 'min_child_weight',
                                     'subsample', 'gamma',
                                     'colsample_bytree', 'sel_pct'))
trials = Trials()
optimize(trials)