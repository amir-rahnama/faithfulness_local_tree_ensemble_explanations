import pickle
import numpy as np
import os 
import scipy 
from joblib import dump, load
from sklearn.metrics import accuracy_score
import pandas as pd
#BASE_PATH = os.getcwd()

def transform_feat_v2(instances, encoder, all_feat, cat_idx, dataset_id):
    x_cat_new = np.zeros((instances.shape[0], len(cat_idx)))
    #encoder = pickle.load(open( "{}/generalization/encoder_{}.p".format(BASE_PATH, dataset_id), "rb"))
    n_bins = encoder.n_bins
    
    for cat_feat_idx in range(len(cat_idx)):
        idx_found = []
        for j in range(n_bins):
            criteria = np.argwhere('x{}_{}.0'.format(int(cat_feat_idx), j) == all_feat)[0][0]
            idx_found.append(criteria)
            print('idx_found', idx_found)
        
        x_cat_new[:, cat_feat_idx] = np.sum(instances[:, idx_found], axis=1)
       
    return x_cat_new


def squeeze_dim_v2(exps, encoder, feature_info, dataset_id):
    #encoder = pickle.load(open( "{}/generalization/encoder_{}.p".format(BASE_PATH, dataset_id), "rb"))
    
    exp_num = exps[:, 0: len(feature_info['numerical_feature_names'])]
    
    exp_cat = exps[:, len(feature_info['numerical_feature_names']):]
    
    all_feat = encoder.get_feature_names_out()
    cat_info = feature_info['categorical_feature_names']
    
    
    exp_new_cat = transform_feat_v2(exp_cat, encoder, all_feat, cat_info, dataset_id)
    exps = np.concatenate((exp_num, exp_new_cat), axis=1)

    return exps


def squeeze_dim(exps, encoder, feature_info, dataset_id):
    #encoder = pickle.load(open( "{}/generalization/encoder_{}.p".format(BASE_PATH, dataset_id), "rb"))
    
    exp_num = exps[:, 0: len(feature_info['numerical_feature_names'])]
    
    exp_cat = exps[:, len(feature_info['numerical_feature_names']):]
    
    all_feat = encoder.get_feature_names_out()
    cat_info = feature_info['categorical_feature_names']
    
    
    exp_new_cat = transform_feat(all_feat, exp_cat, cat_info)
    exps = np.concatenate((exp_num, exp_new_cat), axis=1)

    return exps



def transform(instance, encoder, dataset_id, feature_info):
    #encoder = pickle.load(open( "{}/generalization/encoder_{}.p".format(BASE_PATH, dataset_id), "rb"))
    #feature_info = pickle.load(open( "{}/data/{}/feature_info.p".format(BASE_PATH, dataset_name), "rb"))
    x_num = instance[:, 0: len(feature_info['numerical_feature_names'])]
    x_cat = instance[:, len(feature_info['numerical_feature_names']): ]
    x_cat_encoded = encoder.transform(x_cat)
    if scipy.sparse.issparse(x_cat_encoded):
        x_cat_encoded = x_cat_encoded.toarray()

    x = np.concatenate((x_num, x_cat_encoded), axis=1)

    return x

    
def transform_feat(all_feat, x_cat, cat_info):
    x_cat_new = np.zeros((x_cat.shape[0], len(cat_info)))
    
    x_cat_copy = x_cat.copy()
    
    for cat_feat_idx in range(len(cat_info)):
        val_cat_feat = np.array(cat_info[cat_feat_idx])
        print(val_cat_feat)
        idx_found = []
        for c_f_i in val_cat_feat:
            criteria = np.argwhere('x{}_{}.0'.format(int(cat_feat_idx), int(c_f_i)) == all_feat)[0][0]
            idx_found.append(criteria)
        idx_found = np.array(idx_found)

        x_cat_new[:, cat_feat_idx] = np.sum(x_cat_copy[:, idx_found], axis=1)
       
    return x_cat_new

def get_meta_data(): 
    dataset_names = ['ionosphere', 'wdbc', 'breast-w', 'diabetes', 'qsar-biodeg',
           'banknote-authentication', 'spambase', 'kdd_ipums_la_97-small',
           'phoneme', 'eye_movements', 'pol', 'MagicTelescope', 'house_16H',
           'electricity', 'MiniBooNE', 'covertype', 'Higgs', 'steel-plates-fault', 
                     'blood-transfusion-service-center', 'mozilla4']
    
    meta = []
    BASE_PATH = '/home/amir/code/tree_exp'

    
    for d_name in dataset_names: 
        data_path = '{}/data/{}'.format(BASE_PATH, d_name)
        
        X_test = np.load('{}/X_test.npy'.format(data_path, d_name))
        X_train = np.load('{}/X_train.npy'.format(data_path, d_name))
        y_test = np.load('{}/y_test.npy'.format(data_path, d_name))
    
        max_runs = X_test.shape[0]
    
        gb_path = '{}/models/default/{}/gb.joblib'.format(BASE_PATH, d_name)
        gb = load(gb_path) 
        acc_gb = accuracy_score(y_test, gb.predict(X_test))
    
        rf_path = '{}/models/default/{}/rf.joblib'.format(BASE_PATH, d_name)
        rf = load(rf_path)
        acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
        meta.append([X_train.shape[0], X_test.shape[0], X_train.shape[1], 
                     np.round(acc_gb, 2), np.round(acc_rf, 2)])
    
    meta = pd.DataFrame(meta, index=dataset_names, 
                        columns = ['Training Size', 'Test Size', 'Features', 'GB', 'RF'])
    return meta 
