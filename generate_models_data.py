import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
import shap
import os
import sklearn

from joblib import dump, load
from tabulate import tabulate

import pickle
import os
from treeinterpreter import treeinterpreter as ti
from anchor import anchor_tabular
import sys
sys.path.append('..')
from utils import transform_feat_v2, transform, squeeze_dim_v2, squeeze_dim
from sklearn.datasets import fetch_openml

def transform_target(dataset_id: int, ames: dict) -> np.ndarray:
    """AI is creating summary for transform_target

    Args:
        dataset_id (int): [description]
        ames (dict): [description]

    Returns:
        np.ndarray: [description]
    """

    if dataset_id == 151:
        ames.target = ames.target.replace('DOWN', 0)
        ames.target = ames.target.replace('UP', 1)
        target = ames.target.astype(int)
    elif dataset_id == 293:
        ames.target[ames.target == '-1'] = 0
        ames.target[ames.target == '1'] = 1
        target = ames.target.astype(int)
    elif dataset_id ==  722 or dataset_id ==  821:
        ames.target = ames.target.replace('N', 0)
        ames.target = ames.target.replace('P', 1)
        target = ames.target.astype(int)
    elif dataset_id ==  41150 or dataset_id ==  42769 or dataset_id == 41168 \
        or dataset_id == 44 or dataset_id==37 or dataset_id==61 or dataset_id==15 or dataset_id==1462 \
        or dataset_id == 1510 or dataset_id ==54 or dataset_id == 1494 or dataset_id==59 or dataset_id==44124 \
        or dataset_id == 1120 or dataset_id ==  1461 or dataset_id ==  1489 or dataset_id == 1504 or dataset_id == 1464 \
        or dataset_id == 1046:
        target = ames.target.values.codes
    elif dataset_id ==  1044:
        tmp = np.array(list(ames.target)).astype(int)
        tmp[tmp > 0] = 1
        target = tmp

    if  scipy.sparse.issparse(target):
        target = target.toarray()
                
    if not isinstance(target, np.ndarray):
        target = target.values
    
    return target

def drop_columns(dataset_id, data) -> object:
    if dataset_id == 151:
        data = data.drop('day', axis=1)
    elif dataset_id == 1044:
        data = data.drop(['P1stFixation', 'P2stFixation', 'nextWordRegress'], axis=1)
    elif dataset_id == 1046:
        data = data.drop(['id'], axis=1)
    return data

def data_exists(DATA_PATH):
    data_file_exists = os.path.isfile(f'{DATA_PATH}/X_train.npy') & os.path.isfile(f'{DATA_PATH}/X_train.npy') \
                & os.path.isfile(f'{DATA_PATH}/X_train.npy') & os.path.isfile(f'{DATA_PATH}/X_train.npy') 
    return data_file_exists

def model_exists(MODEL_PATH):
    model_file_exists = os.path.isfile(f'{MODEL_PATH}/rf.joblib') & os.path.isfile(f'{MODEL_PATH}/gbc.joblib')
    return model_file_exists


def tree_shap_exp(instances, x_train, model_obj, x_test, exp_type):
    if exp_type == 'interventional':
        shap_explainer = shap.TreeExplainer(model_obj, x_train,  
                                            feature_perturbation="interventional", 
                                            model_output='raw')
    elif exp_type == 'observational': 
        shap_explainer = shap.TreeExplainer(model_obj, feature_perturbation="observational", 
                                            model_output='raw')
    else: 
        raise Exception('Type not supported')

    print(type(instances))
    shap_values = shap_explainer.shap_values(instances, check_additivity=False)
    shap_values = np.array(shap_values)

    return shap_values


if __name__ == '__main__':
    #dataset_ids = [   59,  1510,    15,    37,  1494,  1462,    44, 44124,  1489,
    #        1044,   722,  1120,   821,   151, 41168, 41150,   293, 42769]
    #dataset_names = ['ionosphere', 'wdbc', 'breast-w', 'diabetes', 'qsar-biodeg',
    #       'banknote-authentication', 'spambase', 'kdd_ipums_la_97-small',
    #       'phoneme', 'eye_movements', 'pol', 'MagicTelescope', 'house_16H',
    #       'electricity', 'jannis', 'MiniBooNE', 'covertype', 'Higgs']
    dataset_names = ['ionosphere', 'wdbc', 'breast-w', 'diabetes', 'qsar-biodeg',
           'banknote-authentication', 'spambase', 'kdd_ipums_la_97-small',
           'phoneme', 'eye_movements', 'pol', 'MagicTelescope', 'house_16H',
           'electricity', 'MiniBooNE', 'covertype', 'Higgs', 'steel-plates-fault', 
                     'blood-transfusion-service-center', 'mozilla4']
    dataset_ids = [   59,  1510,    15,    37,  1494,  1462,    44, 44124,  1489,
            1044,   722,  1120,   821,   151 , 41150,   293, 42769,  1504, 1464, 1046]
    
    RANDOM_STATE = 40
    BASE_PATH = '/home/amir/code/tree_exp'

    for i in range(19, len(dataset_ids)):
        DATA_PATH = f'{BASE_PATH}/data/{dataset_names[i]}'
        DATA_DIR_EXISTS = os.path.exists(DATA_PATH)
        print(i, dataset_ids[i], dataset_names[i])
        if not DATA_DIR_EXISTS: 
            os.makedirs(DATA_PATH)
                    
        if data_exists(DATA_PATH):
            X_train = np.load(f'{DATA_PATH}/X_train.npy')
            y_train = np.load(f'{DATA_PATH}/y_train.npy')
            X_test = np.load(f'{DATA_PATH}/X_test.npy')
            y_test = np.load(f'{DATA_PATH}/y_test.npy')
        else: 
            dset_id = dataset_ids[i]
            ames = fetch_openml(data_id = dset_id, as_frame='auto', parser='auto')
            data = ames.data
            data = drop_columns(dset_id, data)
        
            if scipy.sparse.issparse(data):
                data = data.toarray()
        
            if not isinstance(data, np.ndarray):
                data = data.values
        
            target = transform_target(dset_id, ames)

            print(data)
            if np.sum(np.isnan(data)):
                feat_col_means = np.nanmean(data, axis=0)
                data = np.where(np.isnan(data), feat_col_means, data)

            if dataset_names[i] in ['MiniBooNE', 'covertype', 'Higgs']:
                s_idx = np.random.randint(0, data.shape[0], 20000)
                data = data[s_idx]
                target = target[s_idx]

            #print(np.unique(target, return_counts=True))

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, 
                                                         shuffle=True, random_state=RANDOM_STATE)
    
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            np.save(f'{DATA_PATH}/X_train.npy', X_train)
            np.save(f'{DATA_PATH}/y_train.npy', y_train)
            np.save(f'{DATA_PATH}/X_test.npy', X_test)
            np.save(f'{DATA_PATH}/y_test.npy', y_test)
            np.save(f'{DATA_PATH}/f_names.npy', ames.feature_names)
    
        MODEL_PATH = f'{BASE_PATH}/models/default/{dataset_names[i]}'
        MODEL_DIR_EXISTS = os.path.exists(MODEL_PATH)
    
        if not MODEL_DIR_EXISTS:
            os.makedirs(MODEL_PATH)
            
        EXP_PATH = f'{BASE_PATH}/explanations/default/{dataset_names[i]}'
        EXP_DIR_EXISTS = os.path.exists(EXP_PATH)
    
        if not EXP_DIR_EXISTS:
            os.makedirs(EXP_PATH)
            
        if not model_exists(MODEL_PATH):
            print('Training GB')
            gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
            gb.fit(X_train, y_train)
            
            dump(gb, f'{MODEL_PATH}/gb.joblib')
    
            print('SHAP GB')
            
            exp_gb_obs = tree_shap_exp(X_test, X_train, gb, X_test, exp_type='observational')
            pickle.dump( exp_gb_obs, open( f'{EXP_PATH}/tree_shap_obs_gb.p', "wb" ) )
            
            exp_gb_int = tree_shap_exp(X_test, X_train, gb, X_test, exp_type='interventional')
            pickle.dump( exp_gb_int, open( f'{EXP_PATH}/tree_shap_inter_gb.p', "wb" ) )
            print('Training RF')
            
            rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=10)
            rf.fit(X_train, y_train)
            dump(rf, f'{MODEL_PATH}/rf.joblib')
            
            exp_rf_obs = tree_shap_exp(X_test, X_train, gb, X_test, exp_type='observational')
            pickle.dump( exp_rf_obs, open( f'{EXP_PATH}/tree_shap_obs_rf.p', "wb" ) )
            exp_rf_int = tree_shap_exp(X_test, X_train, gb, X_test, exp_type='interventional')
            pickle.dump( exp_rf_int, open( f'{EXP_PATH}/tree_shap_inter_rf.p', "wb" ) )
            