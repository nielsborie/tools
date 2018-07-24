import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix,average_precision_score,classification_report,precision_recall_curve,f1_score

from .confusion_matrix import plot_cm
from .density import plot_density
from .roc_auc import plot_auc
from .precision_recall import plot_precision_recall


###############################################################################################################

##########################              Generate Classification report               ##########################  

###############################################################################################################
def get_classification_report(ytrain,y_train_pred,ytest,y_test_pred,threshold,target_names = ['Genuine', 'Fraud']):
    
    # --- Find prediction applying threshold
    pred_train = pd.Series(y_train_pred).apply(lambda x: 1 if x > threshold else 0)
    pred_test = pd.Series(y_test_pred).apply(lambda x: 1 if x > threshold else 0)
    
    # --- Plot confusion matrix + ROC curve
    fig,ax = plt.subplots(1,3)
    fig.set_size_inches(15,5)
    plot_cm(ax[0],  ytrain, y_train_pred, [0,1], 'Confusion matrix (TRAIN)', threshold)
    plot_cm(ax[1],  ytest, y_test_pred,   [0,1], 'Confusion matrix (TEST)', threshold)
    plot_auc(ax[2], ytrain, y_train_pred, ytest, y_test_pred, threshold)
    plt.tight_layout()
    plt.show()
    
    # ---  Density curve plot 
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    plot_density(ax[0],ytrain,y_train_pred,"Density plot (TRAIN)",threshold)
    plot_density(ax[1],ytest,y_test_pred,"Density plot (TEST)",threshold)
    plt.tight_layout()
    plt.show()
    
    # --- Plot recall/precision curve
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    plot_precision_recall(ax[0],ytrain,y_train_pred,"Precision Recall (TRAIN)",threshold)
    plot_precision_recall(ax[1],ytest,y_test_pred,"Precision Recall (TEST)",threshold)
    plt.tight_layout()
    plt.show()
    
    # --- Get classification report on train & test set
    print("----------------------------------------------")
    print()
    print("|  ** Train  **   |  Accuracy : {0:.6f}   |   AUC : {0:.6f}   |  AUPRC :  {0:.6f}   |".format(accuracy_score(ytrain, pred_train),roc_auc_score(ytrain, pred_train),average_precision_score(ytrain, pred_train)))
    print("|  ** Test   **   |  Accuracy : {0:.6f}   |   AUC : {0:.6f}   |  AUPRC :  {0:.6f}   |".format(accuracy_score(ytest, pred_test),roc_auc_score(ytest, pred_test),average_precision_score(ytest, pred_test)))
    print()
    print("----------------------------------------------")
    print()
    print("Classification report Train  |  Test")
    print()
    print(classification_report(ytrain, pred_train, target_names=target_names))
    print(classification_report(ytest, pred_test, target_names=target_names))
    print()
    print("----------------------------------------------")
    print()
