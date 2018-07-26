import numpy as np
import matplotlib.pyplot as plt

###############################################################################################################

##########################                    Plot Density curve                     ##########################  

###############################################################################################################

def plot_density(ax, y_true, y_pred, title, th=0.5, target_names = ['Class0', 'Class1']):
    """
    This function plots the density curve of a classifier.
    
    
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
    
    target_names : list, optional (default=['Class0', 'Class1'])
                   List defining the lables of classes to predict.
    
    
    Returns
    -------
    None

    """
    ax.hist(y_pred[y_true == 0], color="r",
               label=target_names[0], alpha=0.5, bins=50)
    ax.hist(y_pred[y_true == 1], color="b",
               label=target_names[1], alpha=0.5, bins=50)
    ax.set_title(title)
    ax.plot([th, th], [0, 800], "--")
    ax.set_yscale("log", nonposy='clip')
    ax.legend()
