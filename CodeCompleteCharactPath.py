# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:36:22 2022

@author: yazdianp
"""
import graphviz
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pydot
import pydotplus
import random
import seaborn as sns
import shap

from sklearn.inspection import permutation_importance
from collections import Counter
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTEN
from imblearn.under_sampling import CondensedNearestNeighbour
from itertools import cycle
from scipy import interp
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, make_scorer)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier
from sklearn.base import clone
os.environ["GRAPHVIZ_DOT"] = "/usr/local/Anaconda/pkgs/graphviz-2.40.1-h0dab3d1_0/bin/dot"

def plot_shap_values(model, X_test,  model_index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Create a summary plot and save it as a PNG file
    plt.figure(figsize=(20, 20))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(f'shap_values_{model_index + 1}.png')
    plt.clf()

def make_seed(seedStr):
    return int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)

def get_actual_class(tumor_code, actual_classes):
    return actual_classes.get(tumor_code)
def optimize_thresholds(y_test, y_pred_proba):
    best_mcc = -1
    best_thresholds = (0, 0)

    for t_low in np.linspace(0, 1, num=100):
        for t_high in np.linspace(t_low, 1, num=100):
            y_pred = np.zeros_like(y_test)
            y_pred[y_pred_proba[:, 0] > t_low] = 0
            y_pred[y_pred_proba[:, 1] > t_high] = 1

            mcc = matthews_corrcoef(y_test, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresholds = (t_low, t_high)
                print (best_mcc)
    return best_thresholds

def plot_multiclass_precision_recall(y_test, y_score, n_classes, title='Precision-Recall curve', filename='precision_recall_curve.png'):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    precision = dict()
    recall = dict()
    average_precision = dict()
    plt.clf()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    classes = ['AML', 'pRCC', 'ccRCC', 'oncocytoma', 'chromophobe']
    
    plt.figure()
    for i, color, class_name in zip(range(n_classes), colors, classes):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='{0} (average precision = {1:0.2f})'.format(class_name, average_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close

def plot_multiclass_roc(y_test, y_score, n_classes, title='ROC curve', filename='ROC_curve.png'):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    classes = ['AML', 'pRCC', 'ccRCC', 'oncocytoma', 'chromophobe']
    plt.clf()
    plt.figure()
    for i, color, class_name in zip(range(n_classes), colors, classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'.format(class_name, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close
def visualize_tree(model, feature_names, class_names, output_file):
    dot_data = export_graphviz(model,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True,
                               rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(output_file)

def make_seed(seedStr):
    return int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)

def plot_confusionmatrix(y_train_pred,y_train,dom, filename):
    plt.clf()
    print(f'{dom} Confusion matrix')
    cf = confusion_matrix(y_train_pred,y_train)
    sns.heatmap(cf,annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close


def plot_multiclass_precision(y_test, y_score, n_classes, title='Precision curve', filename='precision_curve.png'):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])


    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    classes = ['AML', 'pRCC', 'ccRCC', 'oncocytoma', 'chromophobe']
    plt.clf()
    plt.figure()
    for i, color, class_name in zip(range(n_classes), colors, classes):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=class_name)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="best")

    plt.savefig(filename)
    plt.close
def plot_multiclass_recall(y_test, y_score, n_classes, title='Recall curve', filename='Recall_curve.png'):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])

    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    classes = ['AML', 'pRCC', 'ccRCC', 'oncocytoma', 'chromophobe']
    
    plt.figure()

    for i, color, class_name in zip(range(n_classes), colors, classes):
        plt.plot(precision[i], recall[i], color=color, lw=2, label=class_name)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close

def rename_duplicates(old_columns):
    seen = set()
    new_columns = []
    for column in old_columns:
        new_column = column
        count = 1
        while new_column in seen:
            new_column = f"{column}_{count}"
            count += 1
        seen.add(new_column)
        new_columns.append(new_column)
    return new_columns

def safe_division(numerator, denominator, default=0):
    return numerator / denominator if denominator else default

def plot_permutation_importance(clf, X, y, ax, seed, top_n_features=30):
    result = permutation_importance(clf, X, y, n_repeats=30, random_state=seed, n_jobs=2)
    mdi_importances = pd.Series(clf.feature_importances_, index=X.columns)
    sorted_indices = mdi_importances.argsort()[::-1][:top_n_features]  # Select top features
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T[:, sorted_indices],
        vert=False,
        labels=X.columns[sorted_indices],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax

#data = pd.read_csv(r"O:\AMprj\Projects\Ongoing Projects\Original Research\Renal Decision Tree (SAFA)\RDT - Malayeri (Combined ccRCC).csv")'''
data1 = pd.read_csv("Reader2.csv")
data2 = pd.read_csv("Reader1.csv")
data = pd.concat([data2, data1], axis=1)

labelclassnum2 = np.array(data['Pathology'])
data = data.drop(['Pathology', 'RPC'], axis=1)
data = data.fillna('')

one_hot_data = pd.get_dummies(data, columns=['Intravoxel', 'BulkFat', 'CentralScar', 'T1SigIntensity',
       'RestrictedDiffusion', 'ContrastEnhancement', 'VascularInvasion',
       'Capsule', 'Infiltration', 'PerinephricInvasion', 'T2SigIntensity',
       'T2SigHeterogeneity', 'DynamicCharacteristics?', 'TumorLocalization',
       'PeakEnhancement', 'PercentEnhancement', 'Intravoxel', 'BulkFat',
       'CentralScar', 'T1SigIntensity', 'RestrictedDiffusion',
       'ContrastEnhancement', 'VascularInvasion', 'Capsule', 'Infiltration',
       'PerinephricInvasion', 'T2SigIntensity', 'T2SigHeterogeneity',
       'DynamicCharacteristics?', 'TumorLocalization', 'PeakEnhancement',
       'PercentEnhancement'])



one_hot_data = one_hot_data.drop([ 'Intravoxel_', 'Intravoxel_', 'BulkFat_',  'BulkFat_', 
        'CentralScar_', 'CentralScar_','T1SigIntensity_','T1SigIntensity_',
        'RestrictedDiffusion_', 'RestrictedDiffusion_',
        'ContrastEnhancement_', 'ContrastEnhancement_', 'VascularInvasion_', 
        'VascularInvasion_', 'Capsule_', 'Capsule_', 'Infiltration_', 'Infiltration_', 
        'PerinephricInvasion_','PerinephricInvasion_','T2SigIntensity_','T2SigIntensity_', 
        'T2SigHeterogeneity_', 'T2SigHeterogeneity_', 
        'DynamicCharacteristics?_', 'DynamicCharacteristics?_', 
        'TumorLocalization_', 'TumorLocalization_', 'PeakEnhancement_', 'PeakEnhancement_', 
        'PercentEnhancement_', 'PercentEnhancement_', 'Intravoxel_', 'Intravoxel_', 
        'BulkFat_', 'BulkFat_',  'CentralScar_', 'CentralScar_', 'T1SigIntensity_', 
        'T1SigIntensity_', 'RestrictedDiffusion_', 'RestrictedDiffusion_', 
        'ContrastEnhancement_',  'ContrastEnhancement_', 'VascularInvasion_', 
        'VascularInvasion_', 'Capsule_', 'Capsule_', 'Infiltration_',
        'Infiltration_',  'PerinephricInvasion_',  'PerinephricInvasion_', 'T2SigIntensity_', 
        'T2SigIntensity_', 'T2SigHeterogeneity_','T2SigHeterogeneity_', 
        'DynamicCharacteristics?_', 'DynamicCharacteristics?_', 'TumorLocalization_', 
        'TumorLocalization_','PeakEnhancement_','PeakEnhancement_', 'PercentEnhancement_','PercentEnhancement_'], axis=1)

one_hot_data.columns = rename_duplicates(one_hot_data.columns)
#print (one_hot_data.columns.tolist())
feature_names = one_hot_data.columns
accuracy = np.zeros([100])
'''
output_file = "all_reports_stack96.txt"
with open(output_file, "w") as f:
    f.write("")
    '''
misclassified_count = {}
actual_classes = {}
best_model = None
best_score = 0
best_thresholds = (0, 0)

for i in range(15, 17):

    seed1 = make_seed(f"growth1{i}")
    seed2 = make_seed(f"growth2{i}")
    seed3 = make_seed(f"growth3{i}")
    seed4 = make_seed(f"growth4{i}")

    np.random.seed(seed3)
    random.seed(seed4)
    print (f'Start {i+1}')

    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(one_hot_data, labelclassnum2, data.index,
                                                                                 test_size=0.35, random_state=seed1,
                                                                                 stratify=labelclassnum2)

    #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    #weights = dict(zip(np.unique(y_train), class_weights))
    #scale_pos_weight = weights[1] / weights[0]
    sampler = SMOTEN(random_state=seed1)
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    '''
    xgb_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    xgb_param_grid = {
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 3],
        'gamma': [0, 0.1, 0.2],
        'colsample_bytree': [0.6, 0.8, 1.0], 
        'subsample': [0.6, 0.8, 1.0],
    }
 
    # Define a scorer that uses 'weighted' average for the F1 score
    f1_weighted_scorer = make_scorer(f1_score, average='weighted')
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, scoring=f1_weighted_scorer, cv=5, n_jobs=-1)
    xgb_grid_search.fit(x_train, y_train)
    print ('DONE XGB')
    '''
    
    # Grid search for Random Forest
    rf_model = RandomForestClassifier()
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    print ('DONE RF')

    # Define a scorer that uses 'weighted' average for the F1 score
    f1_weighted_scorer = make_scorer(f1_score, average='weighted')
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, scoring=f1_weighted_scorer, cv=5, n_jobs=-1)
    rf_grid_search.fit(x_train, y_train)
    #best_xgb_model = xgb_grid_search.best_estimator_
    #best_xgb_model.fit(x_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    best_rf_model.fit(x_train, y_train)
    '''
    stacking_clf = StackingClassifier(
        estimators=[('xgb', best_xgb_model), ('rf', best_rf_model)],
        final_estimator=LogisticRegression(),
        cv=5
    )
        '''
    #stacking_clf.fit(x_train, y_train)

    #stacked_y_pred = stacking_clf.predict(x_test)
    #stacked_y_pred_proba = stacking_clf.predict_proba(x_test)
    #stacked_accuracy = accuracy_score(y_test, stacked_y_pred)
    #print("Stacking Classifier Accuracy: %.2f%%" % (stacked_accuracy * 100.0))

    #ensemble_report = classification_report(y_test, stacked_y_pred) 
    #print(ensemble_report)
    # Initialize the Tree explainer for the Random Forest model
    explainer = shap.TreeExplainer(best_rf_model)
    
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", plot_size=(15, 10))

    plt.savefig(f'rf_shap_summary_plot {i+1}.png', )
    plt.close()

    shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:])
    shap.save_html(f'rf_shap_force_plot {i+1}.html', shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:], feature_names=x_test.columns.tolist()))
    '''
    classes = ['AML', 'pRCC', 'ccRCC', 'oncocytoma', 'chromophobe']
    n_classes = 5
    plot_multiclass_roc(y_test, stacked_y_pred_proba, n_classes, 'Multiclass ROC curve',f'ROC_curve_2_{i}.png')
    stacked_y_pred = np.argmax(stacked_y_pred_proba, axis=1)
    plot_confusionmatrix(stacked_y_pred, y_test, dom='Test', filename=f'confusionmatrix_2_{i}')


    with open(output_file, "a") as f:
        f.write(f"Iteration {i+1}\n")
        f.write(ensemble_report)
        f.write("\n")

    print("Shape of the data: ", x_test.shape)
    print("Data type of the data: ", type(x_test))
    '''

    #with open("feature_names.pkl", "wb") as f:
        #pickle.dump(feature_names, f)

    #with open("finalized_model.sav", "wb") as f:
        #pickle.dump(stacking_clf, f)




#with open("best_model.pkl", "wb") as f:
    #pickle.dump((best_model, best_thresholds), f)

# To load the model and the thresholds later, use the following code:
# with open("best_model.pkl", "rb") as f:
#     loaded_model, loaded_thresholds = pickle.load(f)
#average_accuracy = np.mean(accuracy)
#print("Average accuracy: %.2f%%" % (average_accuracy * 100.0))

