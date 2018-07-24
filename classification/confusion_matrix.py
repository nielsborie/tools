import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

###############################################################################################################

##########################                    Plot confusion matrix                  ##########################  

###############################################################################################################

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix to evaluate the accuracy of a classification.
    
    Parameters
    ----------
    ax : axes of the current figure.
    
    
    y_true : array, shape = [n_samples] 
             Ground truth (correct) target values.
    
    
    y_pred : array, shape = [n_samples]
             Estimated probability (for a specific class) returned by the classifier.
    
    
    classes : list, len() = number of classes
              list defining the lables of classes to predict.
        
    title : str
            Text of the figure/confusion matrix.
        
    th : float in (0., 1),  optional (default=0.5)
         Probability threshold for assigning observations to a given class. 
    
    
    Returns
    -------
    None

    """
    y_pred_labels = (y_pred>th).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')