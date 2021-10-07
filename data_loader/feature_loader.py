import sys
sys.path.append('../../4D_radiomics/')
import pandas as pd
from utilities.feature_tools import feature_set_details, feature_squeeze
import itertools

def load_feature_set(feature_set_all):
    
    features_4D = []
    subject_name = []
    subject_class = []
    features_sqz = []
    features_mean = []
    for item in feature_set_all:
        if item is not None:
            feature_set = pd.read_excel(item, sheet_name=None)
            feature_list, subject_ids, subject_label = feature_set_details(feature_set)
            mean_feature, _, _, feature_set_squeeze = feature_squeeze(feature_list)
            features_4D.append(feature_list)
            subject_name.append(subject_ids)
            subject_class.append(subject_label)
            features_sqz.append(feature_set_squeeze)
            features_mean.append(mean_feature)
            
    return features_4D, subject_name, subject_class, features_mean, features_sqz



def flat_list(nested_list):

    merged = list(itertools.chain(*nested_list))
    
    return merged

def load_unseen_set(path_to_unseen):
    
    feature_set = pd.read_excel(path_to_unseen, sheet_name=None)
    feature_list, subject_ids, subject_label = feature_set_details(feature_set)
    mean_feature, _, _, feature_set_squeeze = feature_squeeze(feature_list)
    
    return feature_list, subject_ids, subject_label, mean_feature, feature_set_squeeze