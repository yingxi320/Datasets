from matplotlib import pyplot as plt
from scipy.optimize import minimize
from telstra_data import multiclass_log_loss
import numpy as np

def reliability_curve(label, predicted_prob, bins = 10):
    x = np.linspace(0,1,bins+1)
    predicted_proba = np.zeros(bins)
    fraction_positives = np.zeros(bins)
    for i in range(len(x)-1):
        ind = np.logical_and(predicted_prob >= x[i], predicted_prob <= x[i+1])
        predicted_proba[i] = np.mean(predicted_prob[ind])
        fraction_positives[i] = np.sum(label[ind]==1) / np.sum(ind)
    return predicted_proba, fraction_positives

def plot_reliability_curves(labels, predicted_probs, bins = 10):
    legend = []
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(predicted_probs.shape[1]):
        curve = reliability_curve(labels==i,predicted_probs[:,i],bins=bins)
        ax.plot(*curve,'o-')
        legend.append(str(i))
    ax.plot([0,1],[0,1],'k--')
    ax.grid()
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(legend,loc='best')
    return ax

class PlattScaler(object):
    """Performs Platt Scaling on predicted probabilities """
    def __init__(self):
        pass
    def _platt_transform(self, f, A, B):
        return 1. / (1. + np.exp(f*A + B))
    def _transform_probs(self, pred, A, B):
        prednew = np.zeros_like(pred)
        for i in range(pred.shape[1]):
            prednew[:,i] = self._platt_transform(pred[:,i],A[i],B[i])
        return prednew
    def fit(self, predicted_probs, labels):
        def optimization_objective(x, pred, labels):
            A = x[0:3]
            B = x[3:6]
            newprobs = self._transform_probs(pred, A, B)
            return multiclass_log_loss(labels, newprobs)
        x0 = np.array([-10.,-10,-10,4,4,4])
        res = minimize(optimization_objective, x0,args = (predicted_probs, labels))
        self.A = res.x[0:3]
        self.B = res.x[3:6]
    def transform(self, predicted_probs):
        pred_cal = self._transform_probs(predicted_probs, self.A, self.B)
        # eps = 1e-15
        # pred_cal = np.clip(pred_cal, eps, 1 - eps)
        # normalize row sums to 1
        pred_cal /= pred_cal.sum(axis=1)[:, np.newaxis]
        return pred_cal
