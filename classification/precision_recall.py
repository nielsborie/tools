import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,f1_score

###############################################################################################################

##########################                    Plot Precision/Recall curve            ##########################  

###############################################################################################################

def plot_precision_recall(ax,y_true,y_pred,title,th=0.5):
    """
    This function plots the precision/recall curve of a classifier.
    
    Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, precision     is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
    
    Parameters
    ----------
    ax : axes of the current figure.
    
    y_true : array, shape = [n_samples] 
             Ground truth (correct) target values of the train set.
    
    y_pred : array, shape = [n_samples] 
             Estimated probability (for a specific class) returned by the classifier.
        
    title : str
            Text of the figure.
            
    th : float in (0., 1),  optional (default=0.5)
         Probability threshold for assigning observations to a given class. 
    
    
    Returns
    -------
    None

    """
    y_pred_labels = (y_pred>th).astype(int)
    
    prec, rapp, seuil = precision_recall_curve(y_true == 1, y_pred.ravel())
    f1 = [f1_score(y_true[y_pred >= s].ravel(),
                   y_pred_labels[y_pred >= s]) for s in seuil.ravel()]

    ax.plot(seuil, prec[1:], label="Precision")
    ax.plot(seuil, rapp[1:], label="Recall")
    ax.plot(seuil, f1, label="F1")
    ax.plot([th, th], [0, 1], "--")
    ax.set_title(title)
    ax.legend()