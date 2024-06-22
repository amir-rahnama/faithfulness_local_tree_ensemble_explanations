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
#from xgboost import XGBClassifier
import shap
import os
import concurrent.futures

from sklearn.base import TransformerMixin
import sklearn
#import logging
import lime
from joblib import dump, load
#from tabulate import tabulate
import logging
import pickle
import os
from treeinterpreter import treeinterpreter as ti
from anchor import anchor_tabular
import sys
sys.path.append('..')
from sklearn.datasets import fetch_openml
import argparse
import collections
import psutil
import re
np.random.seed(45)

def compute_mdi_local_tree(tree, X, vimp_view):
    impurity_view = tree.impurity
    threshold_view = tree.threshold
    children_left_view = tree.children_left
    children_right_view = tree.children_right
    feature_view = tree.feature

    nsamples = X.shape[0]

    for i in range(nsamples):
        node = 0
        oldvimp = impurity_view[node]
        for _ in range(len(children_left_view)):
            ifeat = feature_view[node]
            if X[i, ifeat] <= threshold_view[node]:
                node = children_left_view[node]
            else:
                node = children_right_view[node]
            newvimp = impurity_view[node]
            vimp_view[i, ifeat] += oldvimp - newvimp
            oldvimp = newvimp
    return vimp_view
 
def local_mdi_exp(instance_explained, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer=None):
    nsamples, nfeatures = instance_explained.shape
    vimp = np.zeros((nsamples, model_obj.n_features_in_), dtype='float64')

    for i in range(len(model_obj.estimators_)):
        estimator = model_obj.estimators_[i]
        if model_name == 'rf':
            vimp =+ compute_mdi_local_tree(estimator.tree_, instance_explained, vimp)
        else: 
            vimp =+ compute_mdi_local_tree(estimator[0].tree_, instance_explained, vimp)

    vimp /= model_obj.n_estimators
    return vimp

def tree_shap_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer):
    
    #background_size = 100
    #background_data = shap.kmeans(X_train, background_size).data
    
    shap_values = explainer.shap_values(instance, check_additivity=False)
    shap_values = np.array(shap_values)
    
    is_rfc =  type(model_obj) == sklearn.ensemble._forest.RandomForestClassifier
    
    if is_rfc:
        if isinstance(explained_class, collections.abc.Sequence):
            shap_values_ = []
            for i in range(len(explained_class)):
                shap_values_.append(shap_values[explained_class[i], i, :])
            shap_values = shap_values_
            shap_values = np.array(shap_values)
        else: 
            shap_values = np.array(shap_values[explained_class, :])

    return shap_values


def anchors_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer):
    instance = instance.reshape(1, -1)
    predict_fn = model_obj.predict
    fake_feat_names = [] 
    
    for i in range(instance.shape[1]):
        f_name = 'feature_id_{}'.format(i)
        fake_feat_names.append(f_name)
    
    exp = explainer.explain_instance(instance, predict_fn, threshold=0.95)
    rules = exp.names()

    return rules

def saabas_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer=None):
    is_boosting = type(model_obj) == sklearn.ensemble._gb.GradientBoostingClassifier
    is_rf = type(model_obj) == sklearn.ensemble._forest.RandomForestClassifier

    instance = instance.reshape(1, -1)
    
    if is_boosting:
        cont = []
        for t in model_obj.estimators_:
            _, _, contrib = ti.predict(t[0], instance.reshape(1, -1))
            cont.append(contrib)
        contributions = np.sum(cont, axis=0)[0]
    elif is_rf or is_dt :
        prediction, bias, contributions = ti.predict(model_obj, instance.reshape(1, -1))
        contributions = contributions.squeeze()[:, explained_class]
        
    return contributions


def lpi_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer=None):
    instance = instance.reshape(1, -1)
    importance  = np.zeros(instance.shape[1])    

    predict_fn = model_obj.predict_proba
    
    base_pred = predict_fn(instance)[:, explained_class]

    for i in range(0, instance.shape[1]):    
        all_feat_values = np.unique(x_train[:, i])
        new_instance = np.tile(instance, (len(all_feat_values), 1))
        
        for j in range(new_instance.shape[0]):
            new_instance[j, i] = all_feat_values[j] 

        pred = predict_fn(new_instance)[:, explained_class]
        importance[i] = np.mean(np.abs(pred - base_pred))
    
    return importance

def kernel_shap_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer):
    instance = instance.reshape(1, -1)
    
    predict_fn = model_obj.predict_proba
            
    #background = x_train[:100]
    '''shap_explainer = shap.KernelExplainer(predict_fn, background)'''
    shap_values = explainer.shap_values(instance, nsamples=sample_size)[explained_class]
    shap_values = np.array(shap_values)
    
    return shap_values

def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    
    if len(res) > 0:
        return int(res[0])
    
def transform_lime_exp(exp, feature_size):
    transform_exp = np.zeros(feature_size)

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp

def lime_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer):
    sample_size = 5000
    
    predict_fn = model_obj.predict_proba
    instance = instance.flatten()
    exp_lime = explainer.explain_instance(instance, predict_fn, labels= (explained_class,), top_labels=2, num_samples=sample_size)
    trans_lime = transform_lime_exp(exp_lime.as_list(), instance.shape[0])

    return trans_lime
    
def random_exp(instance, x_train, model_obj, x_test, model_name, explained_class, sample_size, explainer=None):
    instance = instance.reshape(1, -1)    
    random_exp = np.random.uniform(-1, 1, instance.shape)
    
    return random_exp.flatten()


def get_explainer(exp_name, model, X_train):
    fake_feat_names = [] 
    for i in range(X_train.shape[1]):
        f_name = 'feature_id_{}'.format(i)
        fake_feat_names.append(f_name)
            
    if exp_name == 'tree_shap': 
        explainer = shap.TreeExplainer(model, X_train, 
                                        feature_perturbation="interventional",
                                       model_output='probability')
    elif exp_name == 'kernel_shap':
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
    elif exp_name == 'lime':
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names = fake_feat_names, verbose=False)
        
    return explainer

def global_exp(instance_explained, X_train, model, X_test, model_name, explained_class, exp_sample_size, explainer):
    return model.feature_importances_
    

def process_explanation(exp_name, model, X_train, X_test, model_name, exp_sample_size):
    exp_function = {
        'kernel_shap': kernel_shap_exp,
        'local_mdi': local_mdi_exp,
        'lime': lime_exp,
        'lpi': lpi_exp,
        'random': random_exp,
        'saabas': saabas_exp,
        'global': global_exp
    }
    
    results = []

    explainer = None
    if exp_name in ['tree_shap', 'kernel_shap', 'lime']: 
        explainer = get_explainer(exp_name, model, X_train)

    for i in range(100):
        instance = X_test[i]
        instance_explained = instance.reshape(1, -1)
        explained_class = model.predict(instance_explained)[0]
        result = exp_function[exp_name](instance_explained, X_train, model, X_test, model_name, explained_class, exp_sample_size, explainer)
        results.append(result)

    return np.array(results)


if __name__ == '__main__':
    BASE_PATH = os.getcwd()
    random_state = 45 

    exp_function = {
        'kernel_shap': kernel_shap_exp,
        #'tree_shap': tree_shap_exp, 
        'local_mdi': local_mdi_exp, 
        'lime': lime_exp,
        'lpi': lpi_exp,
        'random': random_exp,
        'saabas': saabas_exp
    }

    #exp_names = ['lime', 'kernel_shap', 'lpi', 'local_mdi', 'random', 'saabas']
    exp_names = ['global']

    logger = logging.getLogger('spam_application')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('exp.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    #parser.add_argument('--e', '-exp_name', required=True, dest='exp_name', type=str, help="the name of the explanation technique")
    parser.add_argument('--s', '-exp_sample_size',  default=5000, type=int, dest='exp_sample_size',help="explanation sample size")
    parser.add_argument('--m', '-model_name', dest='model_name', required=True, type=str, help="the name of the explanation technique")
    

    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    #d_name = args.datasets
    exp_sample_size = args.exp_sample_size
    
    dataset_names = ['ionosphere', 'wdbc', 'breast-w', 'diabetes', 'qsar-biodeg',
           'banknote-authentication', 'spambase', 'kdd_ipums_la_97-small',
           'phoneme', 'eye_movements', 'pol', 'MagicTelescope', 'house_16H',
           'electricity', 'MiniBooNE', 'covertype', 'Higgs', 'steel-plates-fault', 
                     'blood-transfusion-service-center', 'mozilla4']
    
    dataset_ids = [   59,  1510,    15,    37,  1494,  1462,    44, 44124,  1489,
            1044,   722,  1120,   821,   151 , 41150,   293, 42769,  1504, 1464, 1046]
    
    
    exp_path = '{}/explanations/default/'.format(BASE_PATH)

    for i in range(len(dataset_names)):
        d_name = dataset_names[i]
        data_path = '{}/data/{}'.format(BASE_PATH, d_name)
        model_path = '{}/models/default/{}/{}.joblib'.format(BASE_PATH, d_name, model_name)
        
        X_test = np.load('{}/X_test.npy'.format(data_path, d_name))
        X_train = np.load('{}/X_train.npy'.format(data_path, d_name))
        y_train = np.load('{}/y_train.npy'.format(data_path, d_name))
    
        max_runs = X_test.shape[0]
        
        model = load(model_path) 
    
        exp_path_dset = '{}/{}'.format(exp_path, d_name)
        exp_path_dset_exists = os.path.exists(exp_path_dset)
        
        if not exp_path_dset_exists:
            os.makedirs(exp_path_dset)
        
        logger.info('Dataset: {}'.format(d_name))
        for exp_name in exp_names:
            logger.info('EXP: {}'.format(exp_name))
            if exp_name in ['tree_shap', 'shap', 'lime']: 
                explainer = get_explainer(exp_name, model, X_train)
            else:
                explainer = None

            with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each explanation method
                futures = {executor.submit(process_explanation, exp_name, model, X_train, X_test, model_name, exp_sample_size): exp_name for exp_name in exp_names}
    
                # Iterate through the results as they become available
                for future in concurrent.futures.as_completed(futures):
                    exp_name = futures[future]
                    try:
                        exp_results = future.result()
                    except Exception as exc:
                        print(f'Error occurred while processing {exp_name}: {exc}')
                    else:
                        #np.save('{}/{}_{}.npy'.format(exp_path_dset, exp_name, model_name), exp_results)
                        pickle.dump( exp_results, open( '{}/{}_{}.p'.format(exp_path_dset, exp_name, model_name), "wb" ) )
