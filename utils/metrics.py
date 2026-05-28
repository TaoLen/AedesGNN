import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score, 
    matthews_corrcoef,
    average_precision_score,
    balanced_accuracy_score,
    r2_score, 
    mean_squared_error, 
    mean_absolute_error
    )


class ClassificationMetrics:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        denom = tn + fp
        return tn / denom if denom != 0 else np.nan

    @staticmethod
    def ppv(y_true, y_pred):
        return precision_score(
            y_true, y_pred, zero_division=0)

    @staticmethod
    def npv(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        denom = tn + fn
        return tn / denom if denom != 0 else np.nan

    @staticmethod
    def g_mean(sensitivity, specificity):
        return np.sqrt(sensitivity * specificity)

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred)

    @staticmethod
    def mcc(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def auc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob)

    @staticmethod
    def prauc(y_true, y_prob):
        return average_precision_score(y_true, y_prob)

    def calculate_metrics(
        self, 
        probabilities, 
        y_true, 
        threshold=0.5):
        
        metrics = []
        num_tasks = y_true.shape[1
                ] if y_true.ndim > 1 else 1
        for i in range(num_tasks):
            mask = ~np.isnan(y_true[:, i]
                ) if num_tasks > 1 else ~np.isnan(y_true)
            y_true_task = y_true[mask
                ] if num_tasks == 1 else y_true[:, i][mask]
            y_pred_task = (probabilities[mask] > 
                threshold).astype(int) if num_tasks == 1 else (
                probabilities[:, i][mask] > threshold).astype(int)
            y_prob_task = probabilities[mask
                ] if num_tasks == 1 else probabilities[:, i][mask]
            if len(y_true_task) == 0:
                continue
            accuracy = self.accuracy(y_true_task, y_pred_task)
            recall = self.recall(y_true_task, y_pred_task)
            specificity = self.specificity(y_true_task, y_pred_task)
            ppv = self.ppv(y_true_task, y_pred_task)
            npv = self.npv(y_true_task, y_pred_task)
            g_mean = self.g_mean(recall, specificity)
            f1 = self.f1(y_true_task, y_pred_task)
            mcc = self.mcc(y_true_task, y_pred_task)
            prauc = self.prauc(y_true_task, y_prob_task)
            auc = self.auc(y_true_task, y_prob_task)
            metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'recall': recall,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'g_mean': g_mean,
                'f1': f1,
                'mcc': mcc,
                'prauc': prauc,
                'auc': auc
            })
        return metrics
    
    def calibration(
        self, 
        threshold, 
        probabilities, 
        y_true):

        return self.calculate_metrics(
            probabilities, y_true, threshold
            )

class RegressionMetrics:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def pearson(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    def calculate_metrics(self, predictions, y_true):
        tasks = y_true.shape[1] if len(y_true.shape) > 1 else 1
        metrics = []
        for i in range(tasks):
            if tasks > 1:
                mask = ~np.isnan(y_true[:, i])
                y_true_task = y_true[:, i][mask]
                y_pred_task = predictions[:, i][mask]
            else:
                mask = ~np.isnan(y_true)
                y_true_task = y_true[mask]
                y_pred_task = predictions[mask]
            if len(y_true_task) == 0:
                continue
            metrics.append({
                'r2': self.r2(y_true_task, y_pred_task),
                'pearson': self.pearson(y_true_task, y_pred_task),
                'mse': self.mse(y_true_task, y_pred_task),
                'rmse': self.rmse(y_true_task, y_pred_task),
                'mae': self.mae(y_true_task, y_pred_task)}
                )
        return metrics


class MulticlassMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def balanced_accuracy_macro(y_true, y_pred):
        return balanced_accuracy_score(y_true, y_pred)

    @staticmethod
    def recall_macro(y_true, y_pred):
        return recall_score(
            y_true, y_pred, average='macro', zero_division=0
            )

    @staticmethod
    def f1_macro(y_true, y_pred):
        return f1_score(
            y_true, y_pred, average='macro', zero_division=0
            )

    @staticmethod
    def mcc(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def prauc_macro(y_true, y_prob):
        y_true = np.asarray(y_true, dtype=int)
        n_classes = y_prob.shape[1]
        y_bin = np.zeros((len(y_true), n_classes), dtype=int)
        y_bin[np.arange(len(y_true)), y_true] = 1
        per_class = []
        for i in range(n_classes):
            if np.any(y_bin[:, i] == 1):
                per_class.append(average_precision_score(
                    y_bin[:, i], y_prob[:, i]))
        return float(np.mean(per_class)) if per_class else np.nan

    @staticmethod
    def auc_ovr(y_true, y_prob):
        return roc_auc_score(
            y_true, y_prob, multi_class='ovo', average='macro'
            )

    def calculate_metrics(self, probabilities, y_true):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(probabilities).argmax(axis=1)
        metrics = {
            'accuracy': self.accuracy(y_true, y_pred),
            'balanced_accuracy_macro': self.balanced_accuracy_macro(
                y_true, y_pred),
            'recall_macro': self.recall_macro(y_true, y_pred),
            'f1_macro': self.f1_macro(y_true, y_pred),
            'mcc': self.mcc(y_true, y_pred),
            'prauc_macro': np.nan,
            'auc_ovr': np.nan
            }
        if len(np.unique(y_true)) > 1:
            try:
                metrics['prauc_macro'] = self.prauc_macro(
                    y_true, probabilities)
                metrics['auc_ovr'] = self.auc_ovr(
                    y_true, probabilities)
            except ValueError:
                metrics['prauc_macro'] = np.nan
                metrics['auc_ovr'] = np.nan
        return metrics
