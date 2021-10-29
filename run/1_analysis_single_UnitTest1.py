import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../../4D_radiomics/')
import argparse 
import numpy as np
from datetime import datetime
from training.model_trainer import learning
from sklearn.datasets import load_breast_cancer
from data_loader.feature_loader import load_single_set, flat_list
from utilities.json_stuff import prepare_summary, save_json, load_json
from sklearn.model_selection import RepeatedStratifiedKFold
from utilities.feature_tools import data_shuffling,feature_normalization,kfold_stats
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
           
# set savig name
experiment_name = 'exp_fullfeature_'


features_set, label_set = load_breast_cancer(return_X_y=True)




# shuffling the orders
features_set, label_set = data_shuffling(features_set, label_set, seed_val)


kf = RepeatedStratifiedKFold(n_splits=n_fold_split, n_repeats=kf_repeat, random_state=seed_val)

temp_auc_val = {}
temp_acc_val = {}
temp_sen_val = {}
temp_spc_val = {}
n_run = 0
for train_index, val_index in kf.split(features_set, label_set):
    n_run += 1
    run_name = 'KFRepeat_'+str(n_run)
    
    x_train, x_val = features_set[train_index], features_set[val_index]
    y_train, y_val = label_set[train_index], label_set[val_index]
    
    if data_normalization:
        x_train, x_val, _ = feature_normalization(x_train, x_val, x_test=None)
        
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
    
performance_summary = {}
performance_summary['AUC_Val'] = temp_auc_val
performance_summary['ACC_Val'] = temp_acc_val
performance_summary['SEN_Val'] = temp_sen_val
performance_summary['SPC_Val'] = temp_spc_val
mean_acc, mean_auc, _, _, max_acc, max_auc= kfold_stats(performance_summary) 
performance_statistic = {}
performance_statistic['mean_AUC'] = mean_auc
performance_statistic['Max_AUC'] = max_auc
performance_statistic['mean_ACC'] = mean_acc
performance_statistic['Max_ACC'] = max_acc
performance_summary['Metrics_Average'] = performance_statistic  
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
