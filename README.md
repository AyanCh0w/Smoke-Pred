# Smoke-Pred
Notebook for: Binary Prediction of Smoker Status using Bio-Signals

### imports
```{python}
import numpy as np
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import xgboost as xgb
import optuna
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
```

### data proccessing
```{python}
train = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")
```

```{python}
smoking = train["smoking"]
train = train.drop("smoking", axis=1)

def cleanData(dataset):
    ids = dataset["id"]
    dataset = dataset.drop("id", axis=1)
    
    catData = dataset[['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']]
    catData = pd.get_dummies(catData, columns = catData.columns)
    
    dataset = dataset.drop(['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'], axis=1)
    
    
    
    nDataset = (dataset - dataset.mean()) / dataset.std()
    
    nDataset = nDataset.join(catData)
    
    return nDataset, ids

nTrain, trainIds = cleanData(train)
nTest, testIds = cleanData(test)
```
### fine tuning
```{python}
# def objective_lgb(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators',500,1000),
#         "max_depth":trial.suggest_int('max_depth',3,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.25, log=True),
#         "min_child_weight" : trial.suggest_float('min_child_weight', 0.5,4),
#         "min_child_samples" : trial.suggest_int('min_child_samples',1,250),
#         "subsample" : trial.suggest_float('subsample', 0.2, 1),
#         "subsample_freq" : trial.suggest_int('subsample_freq',0,5),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         'num_leaves' : trial.suggest_int('num_leaves', 2, 128),
#     }
#     lgbmmodel_optuna = LGBMClassifier(**params,random_state=6)
#     cv = cross_val_score(lgbmmodel_optuna, nTrain, smoking, cv = 4, scoring='roc_auc').mean()
#     return cv

# study = optuna.create_study(direction='maximize')
# study.optimize(objective_lgb, n_trials=100,timeout=2000)

# def objective_xgb(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators',500,750),
#         'max_depth':  trial.suggest_int('max_depth',3,50),
#         'min_child_weight': trial.suggest_float('min_child_weight', 2,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.2,log=True),
#         'subsample': trial.suggest_float('subsample', 0.2, 1),
#         'gamma': trial.suggest_float("gamma", 1e-4, 1.0),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
#         "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1),
#     }
#     xgbmodel_optuna = XGBClassifier(**params,random_state=6,tree_method = "gpu_hist",eval_metric= "auc")
#     cv = cross_val_score(xgbmodel_optuna, nTrain, smoking, cv = 4,scoring='roc_auc').mean()
#     return cv

# study = optuna.create_study(direction='maximize')
# study.optimize(objective_xgb, n_trials=120,timeout=2000)
```

```{python}
perams_lgb = {'n_estimators': 835, 'max_depth': 35, 'learning_rate': 0.031055009498070017, 'min_child_weight': 2.1835638592136943, 'min_child_samples': 220, 'subsample': 0.9298705560281547, 'subsample_freq': 5, 'colsample_bytree': 0.34162312179431537, 'num_leaves': 125}
perams_xgb = {'n_estimators': 551, 'max_depth': 13, 'min_child_weight': 49.846543999570784, 'learning_rate': 0.047465437274911496, 'subsample': 0.7827674748408318, 'gamma': 0.8378942824941107, 'colsample_bytree': 0.5997681524662475, 'colsample_bylevel': 0.8678362118325031, 'colsample_bynode': 0.7988100778098284}
```

### make, set, and predict model
```{python}
model_lgb = lgb.LGBMRegressor(**perams_lgb)
model_xgb = xgb.XGBRegressor(**perams_xgb)

model_xgb.fit(nTrain, smoking)
model_lgb.fit(nTrain, smoking)

xgb_pred = model_xgb.predict(nTest)
lgb_pred = model_lgb.predict(nTest)

preds = (xgb_pred*0.5) + (lgb_pred*0.5)

def sigmoid(x):
    return 1 / (1 + np.e**(-x))

preds = sigmoid(preds)
```

### submit csv
```{python}
sub = pd.DataFrame()
sub['id'] = testIds
sub['smoking'] = preds
sub.to_csv('submission.csv',index=False)
```
