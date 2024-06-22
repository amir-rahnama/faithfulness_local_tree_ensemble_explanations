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
import argparse
from joblib import dump, load
from tabulate import tabulate

import pickle
import os
from treeinterpreter import treeinterpreter as ti
from anchor import anchor_tabular
import sys

sys.path.append('.')
from utils import transform_feat_v2, transform, squeeze_dim_v2, squeeze_dim
#from all_tree_explanations_v2 import tree_shap_exp, mdi_exp, anchors_exp, lime_exp, shap_exp, lpi_exp, random_exp, local_mdi_exp
from sklearn.datasets import fetch_openml

def get_robustness(explanation, instance_explained, train_data, model, type_robust, selection_type='abs'):    
    cutoffs = np.linspace(0.05, 0.5, 10)

    robust_result = []
    for cutoff in cutoffs:
        instance_explained = instance_explained.reshape(1, -1)
        
        explained_class = model.predict(instance_explained)[0]
        base_pred = model.predict_proba(instance_explained)[0][explained_class]
        
        threshold = int(np.round(cutoff * explanation.shape[0]))
        if selection_type == 'abs': 
            feat_selected = np.abs(explanation).argsort()[::-1][:threshold]
        elif selection_type == 'normal': 
            feat_selected = explanation.argsort()[::-1][:threshold]
        
        if type_robust == 'insertion':
            tmp_feat_selected = feat_selected
            feat_selected = np.setxor1d(np.arange(explanation.shape[0]), tmp_feat_selected)
            
        copy_instance_explained = instance_explained.copy().reshape(1, -1)
        for f in feat_selected:
            copy_instance_explained[:, f] = np.random.choice(train_data[:, f], 1)[0]

        new_pred = model.predict_proba(copy_instance_explained)[0][explained_class]
        
        robust_result.append(np.abs(new_pred - base_pred))

    return robust_result

def get_robustness_sample(explanation, instance_explained, train_data, 
                          model, type_robust, selection_type='abs', sample_size=20):    
    cutoffs = np.linspace(0.05, 0.5, 10)
    instance_explained = instance_explained.reshape(1, -1)
    explained_class = model.predict(instance_explained)[0]
    base_pred = model.predict_proba(instance_explained)[0][explained_class]
    
    robust_result = []
    for cutoff in cutoffs:
        threshold = int(np.round(cutoff * explanation.shape[0]))
        
        if selection_type == 'abs': 
            feat_selected = np.abs(explanation).argsort()[::-1][:threshold]
        elif selection_type == 'normal':
            feat_selected = explanation.argsort()[::-1][:threshold]
            
        if type_robust == 'insertion':
            tmp_feat_selected = feat_selected
            feat_selected = np.setxor1d(np.arange(explanation.shape[0]), tmp_feat_selected)

        copy_instance_explained = np.tile(instance_explained, reps=sample_size).reshape(-1, instance_explained.shape[1])
        
        for f in feat_selected: 
            uniq_vals = np.unique(train_data[:, f])
            if len(uniq_vals) < sample_size: 
                random_sel_vals = np.random.choice(uniq_vals, size=sample_size, replace=True)
            else: 
                random_sel_vals = np.random.choice(uniq_vals, size=sample_size, replace=False)
            copy_instance_explained[:, f] = random_sel_vals
                
        new_pred = model.predict_proba(copy_instance_explained)[:, explained_class]
        
        robust_result.append(np.mean(np.abs(new_pred - base_pred)))
        
    return robust_result
    '''for t in range(sample_size):
            change = []
            copy_instance_explained = instance_explained.copy().reshape(1, -1)
            
            for f in feat_selected:
                #unique_vals = np.unique(train_data[:, f])
                copy_instance_explained[:, f] = np.random.choice(unique_vals, 1)[0]
                new_pred = model.predict_proba(copy_instance_explained)[0][explained_class]
                
            change.append(np.abs(new_pred - base_pred))
        robust_result.append(np.mean(change))

    return robust_result'''
    

def get_auc(result):
    cutoffs = np.linspace(0.05, 0.5, 10)
    temp = np.array(result).mean(axis=0)
    auc_ = 0
    for k in range(1, len(cutoffs) - 1):
        x = cutoffs[k] - cutoffs[k - 1]
        y = temp[k] + temp[k-1]
        auc = y / ( 2 * x)
    
    return auc

if __name__ == '__main__':
    BASE_PATH = '/home/amir/code/tree_exp'
    e_path = f'{BASE_PATH}/explanations/default'
    d_path = f'{BASE_PATH}/data'
    m_path = f'{BASE_PATH}/models/default'

    dataset_names = ['ionosphere', 'wdbc', 'breast-w', 'diabetes', 'qsar-biodeg',
           'banknote-authentication', 'spambase', 'kdd_ipums_la_97-small',
           'phoneme', 'eye_movements', 'pol', 'MagicTelescope', 'house_16H',
           'electricity', 'MiniBooNE', 'covertype', 'Higgs', 'steel-plates-fault', 
                     'blood-transfusion-service-center', 'mozilla4']
    
    #m_name = 'rf'
    exp_names = ['lime', 'kernel_shap', 'lpi', 'tree_shap_obs', 'tree_shap_inter', 'local_mdi', 'saabas',  'random', 'global']
    
    robust_vals = {'abs': {'insertion': {}, 'deletion': {}}, 'normal': {'insertion': {}, 'deletion': {}}}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', '-model_name', dest='model_name', required=True, type=str, help="the name of the explanation technique")
    args = parser.parse_args()
    m_name = args.model_name
    
    '''
    for d_name in dataset_names:
        print(d_name)
        train_data = np.load("{}/{}/X_train.npy".format(d_path, d_name))
        test_data = np.load("{}/{}/X_test.npy".format(d_path, d_name))
        model = load(f'{m_path}/{d_name}/{m_name}.joblib')

        robust_vals['abs']['insertion'][d_name] = {}
        robust_vals['normal']['insertion'][d_name] = {}
        
        robust_vals['normal']['deletion'][d_name] = {}
        robust_vals['abs']['deletion'][d_name] = {}
        
        for e_name in exp_names:    
            robust_vals['abs']['insertion'][d_name][e_name] = []
            robust_vals['normal']['insertion'][d_name][e_name] = []
            
            robust_vals['normal']['deletion'][d_name][e_name] = []
            robust_vals['abs']['deletion'][d_name][e_name] = []
            
            temp_in_abs = []
            temp_del_abs = []
            
            temp_in = []
            temp_del = []

            temp = pickle.load( open( "{}/{}/{}_{}.p".format(e_path, d_name, e_name, m_name), "rb" ) )
            if e_name in ['kernel_shap', 'local_mdi']: 
                temp = np.squeeze(temp)
            if e_name in ['tree_shap_obs', 'tree_shap_inter']:
                temp = np.array(temp)[:100]
            exp_res = temp
            
            for idx in range(100):
                exp_example = exp_res[idx]
                instance_explained = test_data[idx]
                
                temp_in_abs.append(get_robustness(exp_example, instance_explained, train_data, 
                                                  model, type_robust='insertion', selection_type='abs'))
                temp_in.append(get_robustness(exp_example, instance_explained, train_data, 
                                              model, type_robust='insertion', selection_type='normal'))
                temp_del_abs.append(get_robustness(exp_example, instance_explained, train_data, 
                                                   model, type_robust='deletion', selection_type='abs'))
                temp_del.append(get_robustness(exp_example, instance_explained, train_data, 
                                               model, type_robust='deletion', selection_type='normal'))

            robust_vals['abs']['insertion'][d_name][e_name] = temp_in_abs
            robust_vals['normal']['insertion'][d_name][e_name] = temp_in
            
            robust_vals['abs']['deletion'][d_name][e_name] = temp_del_abs
            robust_vals['normal']['deletion'][d_name][e_name] = temp_del 
    
    pickle.dump( robust_vals, open( "{}/evaluation/robust_vals_{}.p".format(BASE_PATH, m_name), "wb" ) )
    '''

    s_size = [1, 10, 20, 50]
    
    robust_vals_sample = {}

    for s in s_size:
        print(s)
        robust_vals_sample[s] = {'abs': {'insertion': {}, 'deletion': {}}, 
                                 'normal': {'insertion': {}, 'deletion': {}}}
        eps = np.finfo(float).eps

        for d_name in dataset_names:
            print(d_name)
            train_data = np.load("{}/{}/X_train.npy".format(d_path, d_name))
            test_data = np.load("{}/{}/X_test.npy".format(d_path, d_name))
            model = load(f'{m_path}/{d_name}/{m_name}.joblib')

            robust_vals_sample[s]['abs']['insertion'][d_name] = {}
            robust_vals_sample[s]['normal']['insertion'][d_name]= {}
            robust_vals_sample[s]['normal']['deletion'][d_name]= {}
            robust_vals_sample[s]['abs']['deletion'][d_name] = {}
            
            for e_name in exp_names:
                robust_vals_sample[s]['abs']['insertion'][d_name][e_name] = []
                robust_vals_sample[s]['normal']['insertion'][d_name][e_name] = []
                
                robust_vals_sample[s]['normal']['deletion'][d_name][e_name] = []
                robust_vals_sample[s]['abs']['deletion'][d_name][e_name] = []
                
                temp_in_abs_s = []
                temp_in_s = []
                
                temp_del_abs_s = []
                temp_del_s = []
    
                temp = pickle.load( open( "{}/{}/{}_{}.p".format(e_path, d_name, e_name, m_name), "rb" ) )
                
                if e_name in ['kernel_shap', 'local_mdi']: 
                    temp = np.squeeze(temp)
                if e_name in ['tree_shap_obs', 'tree_shap_inter']:
                    temp = np.array(temp)[:100]
                exp_res = temp
                
                for idx in range(100):
                    exp_example = exp_res[idx]
                    instance_explained = test_data[idx]
                    temp_in_abs_s.append(get_robustness_sample(exp_example, instance_explained, train_data, model, 
                                                             type_robust='insertion', selection_type='abs',sample_size =s ))
                    temp_in_s.append(get_robustness_sample(exp_example, instance_explained, train_data, model, 
                                                         type_robust='insertion', selection_type='normal', sample_size =s))
                    
                    temp_del_abs_s.append(get_robustness_sample(exp_example, instance_explained, train_data, model, 
                                                              type_robust='deletion', selection_type='abs', sample_size =s))
                    temp_del_s.append(get_robustness_sample(exp_example, instance_explained, train_data, model, 
                                                          type_robust='deletion', selection_type='normal', sample_size =s))
                
                robust_vals_sample[s]['abs']['insertion'][d_name][e_name] = temp_in_abs_s
                robust_vals_sample[s]['normal']['insertion'][d_name][e_name] = temp_in_s
                
                robust_vals_sample[s]['abs']['deletion'][d_name][e_name] = temp_del_abs_s
                robust_vals_sample[s]['normal']['deletion'][d_name][e_name] = temp_del_s
     
    pickle.dump( robust_vals_sample, open( "{}/evaluation/robust_vals_sample_{}.p".format(BASE_PATH, m_name), "wb" ) )
