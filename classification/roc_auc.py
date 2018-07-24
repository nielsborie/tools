import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score

###############################################################################################################

##########################                    Plot ROC AUC curve                     ##########################  

###############################################################################################################
def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):
    """
    This function prints and plots the roc auc curve of a train and a test set.
    
    Parameters
    ----------
    ax : axes of the current figure.
    
    y_train : array, shape = [n_samples] 
              Ground truth (correct) target values of the train set.
    
    y_train_pred : array, shape = [n_samples] 
                   Estimated probability (for a specific class) returned by a classifier for the train set.
    
    y_test : array, shape = [n_samples]
             Ground truth (correct) target values of the train set.
             
    y_test_pred : array, shape = [n_samples]
                  Estimated probability (for a specific class) returned by a classifier for the test set.
        
    th : float in (0., 1),  optional (default=0.5)
         Probability threshold for assigning observations to a given class. 
    
    
    Returns
    -------
    None

    """
    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])
    