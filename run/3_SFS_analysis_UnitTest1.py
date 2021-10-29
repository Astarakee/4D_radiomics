import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../../4D_radiomics/')
import copy
import argparse 
import numpy as np
import pandas as pd
from datetime import datetime
from training.model_trainer import learning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from data_loader.feature_loader import load_feature_set, flat_list
from utilities.feature_tools import data_shuffling,feature_normalization
from utilities.json_stuff import prepare_summary, save_json, summary_dictionary,load_json
from learning_algorithms.models import random_forest, decicion_tree, adab_tree, lda, qda, naive

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", 
                    help="Config path")
args = parser.parse_args()
configs = load_json(args.config)

# data pafrom utilities.json_stuff import prepare_summary, save_json, summary_dictionary, load_jsonth settings
feature_set1_path = None 
feature_set2_path = None 
feature_set3_path = None 
feature_set4_path = None 
feature_set5_path = None 
feature_set_test = None 
    
# training settings
seed_val = configs['Training_Params']['seed_val']
n_fold_split = configs['Training_Params']['n_fold_split']
kf_repeat = configs['Training_Params']['kf_repeat']
data_normalization = configs['Training_Params']['data_normalization']

# models params
n_estimators = configs['Model_Params']['n_estimators']
depth_max = configs['Model_Params']['depth_max']

# training model
rf_model = configs['Learning_Algorithm']['rf_model']
dt_model = configs['Learning_Algorithm']['dt_model']
adab_model = configs['Learning_Algorithm']['adab_model']
lda_model = configs['Learning_Algorithm']['lda_model']
qda_model = configs['Learning_Algorithm']['qda_model']
naive_model = configs['Learning_Algorithm']['naive_model']

#make sure the same model as SFS is used
selected_features_model = configs['Analysis_After_SFS']['model_with_SFS']
n_selected_features = configs['Analysis_After_SFS']['n_top_features']
selected_feature_json_file = configs['Analysis_After_SFS']['feature_selection_log_name']

# set saving names
experiment_name = 'experiment_'+selected_features_model+'_'+str(n_selected_features)+'_TopFeatures'
n_selected_feature_name = 'features:'+str(n_selected_features)+'_feature_names'
selected_feature_json_path = os.path.join('../selected_features', selected_feature_json_file)


features_set, label_set = load_breast_cancer(return_X_y=True)

# shuffling the orders
features_set, label_set = data_shuffling(features_set, label_set, seed_val)


# recalling the selected feature indices
selected_features_summary = pd.read_json(selected_feature_json_path)
selected_features = selected_features_summary[selected_features_model][n_selected_feature_name]



features_set_copy = copy.deepcopy(features_set)
top_features = selected_features
features_set_selected = features_set_copy[:, top_features]



kf = RepeatedStratifiedKFold(n_splits=n_fold_split, n_repeats=kf_repeat, random_state=seed_val)

temp_auc_val = {}
temp_acc_val = {}
temp_sen_val = {}
temp_spc_val = {}
n_run = 0
for train_index, val_index in kf.split(features_set_selected, label_set):
    n_run += 1
    run_name = 'KFRepeat_'+str(n_run)
    
    x_train, x_val = features_set_selected[train_index], features_set_selected[val_index]
    y_train, y_val = label_set[train_index], label_set[val_index]
    
    if data_normalization:
        x_train, x_val, _ = feature_normalization(x_train, x_val, x_test = None)
        
    if rf_model:
        clf = random_forest(n_estimators=n_estimators, criterion='gini', max_depth=depth_max, class_weight=None)
    elif dt_model:
        clf= decicion_tree(criterion='gini', max_depth=depth_max, class_weight=None)
    elif adab_model:
        clf = adab_tree(max_depth=depth_max, criterion='gini', class_weight=None, n_estimators=n_estimators)
    elif lda_model:
        clf = lda()
    elif qda_model:
        clf = qda()
    elif naive_model:
        clf = naive()
    else:
        raise Exception('The learning algorithm has not imported properly! Please note only one model has to be compiled!')
    
    auc_val, acc_val, sen_val, spc_val, _ = learning(clf, x_train, y_train, x_val, y_val, x_test=None)
    
    temp_auc_val[run_name] = auc_val
    temp_acc_val[run_name] = acc_val
    temp_sen_val[run_name] = sen_val
    temp_spc_val[run_name] = spc_val


performance_summary = summary_dictionary(temp_auc_val, temp_acc_val, temp_sen_val, temp_spc_val)
model_name  = str(clf)
report_summary = prepare_summary(feature_set1_path,feature_set2_path,
                                 feature_set3_path,feature_set4_path,
                                 feature_set5_path,feature_set_test,
                                 model_name, seed_val,n_fold_split,
                                 kf_repeat, data_normalization,
                                 n_estimators, depth_max,
                                 performance_summary)

time = datetime.now().strftime("%y%m%d-%H%M")
exp_name_json = experiment_name+'_'+time+'.json'
save_json('../logs/', exp_name_json, report_summary)
print(performance_summary['Performance_Summary']['mean_AUC_Val'])