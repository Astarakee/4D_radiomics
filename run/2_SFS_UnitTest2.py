import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('../../4D_radiomics/')
import os
import json
import argparse 
import numpy as np
from sklearn import preprocessing
from utilities.json_stuff import load_json
from utilities.feature_tools import  data_shuffling
from data_loader.feature_loader import load_feature_set, flat_list
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from learning_algorithms.models import random_forest, decicion_tree, adab_tree, lda, qda, naive

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", 
                    help="Config path")
args = parser.parse_args()
configs = load_json(args.config)

# data pafrom utilities.json_stuff import prepare_summary, save_json, summary_dictionary, load_jsonth settings
feature_set1_path = configs['Feature_Train_Path']['feature_set1_path']
feature_set2_path = configs['Feature_Train_Path']['feature_set2_path']
feature_set3_path = configs['Feature_Train_Path']['feature_set3_path']
feature_set4_path = configs['Feature_Train_Path']['feature_set4_path']
feature_set5_path = configs['Feature_Train_Path']['feature_set5_path']  
feature_set_test = None 

# training settings
seed_val = configs['Training_Params']['seed_val']
n_fold_split = configs['Training_Params']['n_fold_split']
kf_repeat = configs['Training_Params']['kf_repeat']
data_normalization = configs['Training_Params']['data_normalization']

# models params
n_estimators = configs['Model_Params']['n_estimators']
depth_max = configs['Model_Params']['depth_max']

# forward feature selectoin setting
n_selecting_features = configs['Feature_Selection_Params']['n_selecting_features']


experiment_name = 'UnitTest2_LungCancer_'
 

feature_set_all = [feature_set1_path, feature_set2_path,
                   feature_set3_path, feature_set4_path, feature_set5_path]

features_4D, subject_name, subject_class, features_sqz = load_feature_set(feature_set_all)

features_orig = flat_list(features_4D)
subject_id = flat_list(subject_name)
label_set = np.array(flat_list(subject_class))
features_set = np.array(flat_list(features_sqz))
features_set = features_set[:,14:]


# Feature Normalization
if data_normalization == True:
    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    features_set = min_max_norm.fit_transform(features_set)


# shuffling the orders
features_set, label_set = data_shuffling(features_set, label_set, seed_val)

# Compiling Learning Algorithms
qda_model = qda()
lda_model = lda()
naive_model = naive()
dt_model =  decicion_tree(criterion='gini', max_depth=depth_max, class_weight=None)
adab_model =  adab_tree(max_depth=depth_max, criterion='gini', class_weight=None, n_estimators=n_estimators)
rf_model = random_forest(n_estimators=n_estimators, criterion='gini', max_depth=depth_max, class_weight=None)


all_algorithms = {}
all_algorithms['dt'] = dt_model
all_algorithms['rf'] = rf_model
all_algorithms['lda'] = lda_model
all_algorithms['qda'] = qda_model
all_algorithms['adab'] = adab_model
all_algorithms['naive'] = naive_model


# Sequential feature selection (SFS) for each of the lerning algorithm

all_results = {}
for key, value in all_algorithms.items():
    metrics = {}
    for n_features in n_selecting_features: 
        print('working on algorithm {} with {} features.'.format(key, n_features))
        sffs_model = SFS(value,
                         k_features = n_features,
                         forward = True,
                         floating = False,
                         scoring = 'roc_auc',
                         n_jobs = -1,
                         cv= 5)
    
        sffs = sffs_model.fit(features_set, label_set)
        
        sffs_features = sffs.subsets_
        sffs_features = sffs_features[n_features]
        cv_scores = sffs_features['cv_scores']
        cv_average_score = np.mean(cv_scores)
        cv_std_score = np.std(cv_scores)
        feature_names = sffs_features['feature_idx']
        metric_name = 'features:'+str(n_features)
        metric_name_mean = metric_name+'_cv_mean'
        metric_name_std = metric_name+'_cv_std'
        metric_name_features = metric_name+'_feature_names'
        metrics[metric_name_mean] = cv_average_score
        metrics[metric_name_std] = cv_std_score
        metrics[metric_name_features] = feature_names

    all_results[key] = metrics
    

# Saving the params and results in a dictionary-json file.
jason_name = experiment_name+'_.json'
json_dir = os.path.join('../selected_features/', jason_name)
with open(json_dir, 'w') as fp:
    json.dump(all_results, fp, indent = 4)
