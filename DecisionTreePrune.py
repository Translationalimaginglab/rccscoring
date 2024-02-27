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
from collections import Counter

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
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



def plot_precision_recall(y_true, y_scores, title='Precision-Recall Curve', filename='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
    plt.figure()
    plt.plot(recall, precision, lw=2, label='Binary Classification')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def plot_roc(y_true, y_scores, title='ROC Curve', filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 0])
    roc_auc = roc_auc_score(y_true, y_scores[:, 0])
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'Binary Classification (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


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

def plot_confusion_matrix(y_true, y_pred, classes=['Benign', 'Malignant'], filename='confusion_matrix.png'):
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, yticklabels=classes, xticklabels=classes, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_multiclass_precision(y_test, y_score, n_classes, title='Precision curve', filename='precision_curve.png'):
    y_test_bin = label_binarize(y_test, classes=[0, 1])
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])

    colors = cycle(['blue', 'red'])
    classes = ['Benign', 'Malignant']
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
    y_test_bin = label_binarize(y_test, classes=[0, 1])
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])

    colors = cycle(['blue', 'red'])
    classes = ['Benign', 'Malignant']
    
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

#data = pd.read_csv(r"O:\AMprj\Projects\Ongoing Projects\Original Research\Renal Decision Tree (SAFA)\RDT - Malayeri (Combined ccRCC).csv")'''
data1 = pd.read_csv("Reader2.csv")
data2 = pd.read_csv("Reader1.csv")
data = pd.concat([data2, data1], axis=1)

labelclassnum1 = np.array(data['Pathology'])
labelclassnum2 = np.array(data['BenignMalig'])
data = data.drop(['Pathology', 'RPC','BenignMalig'], axis=1)
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

output_file = "all_reports_stack55-57.txt"
with open(output_file, "w") as f:
    f.write("")

misclassified_count = {}
actual_classes = {}
best_model = None
best_score = 0
best_thresholds = (0, 0)


params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [2, 3],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30, 40],
    
}
for i in range(55,57):
    seed1=make_seed(f"growth1{i}")
    seed2=make_seed(f"growth2{i}")
    seed3=make_seed(f"growth3{i}")
    seed4=make_seed(f"growth4{i}")
    np.random.seed(seed3)
    random.seed(seed4)
    data_list = np.array(one_hot_data.columns)
    #RandomOverSampler(sampling_strategy='minority')
    #oversample = SMOTE(sampling_strategy='not majority', k_neighbors = 5)

    x_train, x_test, y_train, y_test = train_test_split(one_hot_data, labelclassnum2, test_size = 0.35, random_state = seed1, stratify=labelclassnum2 )  

    #cnn = CondensedNearestNeighbour(random_state=42)
    #x_train, y_train = cnn.fit_resample(x_train, y_train)
    
    #smote = RandomOverSampler(sampling_strategy='minority')
    #x_train, y_train = smote.fit_resample(x_train, y_train)

    #x_train, y_train = oversample.fit_resample(x_train, y_train)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    #k_best = SelectKBest(score_func=f_classif, k=10)
    #x_train = k_best.fit_transform(x_train, y_train)
    #x_test = k_best.transform(x_test)

    clf = tree.DecisionTreeClassifier()
    gcv = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='accuracy')
    gcv.fit(x_train,y_train)
    model = gcv.best_estimator_
    model.fit(x_train,y_train, sample_weight=weights)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    path = clf.cost_complexity_pruning_path(x_train, y_train, sample_weight=weights)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    best_params = gcv.best_params_
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(**best_params, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train, sample_weight=weights)
        clfs.append(clf)
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]

    train_acc = []
    test_acc = []
    for c in clfs:
        y_train_pred = c.predict(x_train)
        y_test_pred = c.predict(x_test)
        train_acc.append(accuracy_score(y_train_pred,y_train))
        test_acc.append(accuracy_score(y_test_pred,y_test))
    classes = ['Benign', 'Malignant']

    best_tree_index = np.argmax(test_acc)
    best_tree = clfs[best_tree_index]
    y_train_pred = best_tree.predict(x_train)
    y_test_pred = best_tree.predict(x_test)

    train_classification_report = classification_report(y_train, y_train_pred, target_names=classes)
    test_classification_report = classification_report(y_test, y_test_pred, target_names=classes)

    #print(f"Train Classification Report (Iteration {i}):\n{train_classification_report}")
    print(f"Test Classification Report (Iteration {i}):\n{test_classification_report}")

    n_classes = 2
    y_score = clf.predict_proba(x_test)
    plot_precision_recall(y_test, y_score, filename='precision_recall_curve.png')
    plot_roc(y_test, y_score, filename='roc_curve.png')
    
    plt.clf()
    plt.figure(figsize=(100, 100))
    tree.plot_tree(best_tree,feature_names=feature_names,class_names=classes,filled=True)
    plt.title((f'trainset{i}'),fontsize=12)
    plt.savefig(f'TREE Number_{i}')
    plt.close
    
    with open(output_file, "a") as f:
        f.write(f"Iteration {i}:\n")
        f.write(test_classification_report)
        f.write("\n\n")

    plt.clf() 
    explainer = shap.TreeExplainer(best_tree)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, x_test)
    plt.savefig(f'shap_DT_summary_plot_2_{i}.png', bbox_inches='tight')
    plt.close()           

print (acc.mean())
print (acc.max())
