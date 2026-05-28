import numpy as np
import torch
from IPython.display import (
    Markdown, 
    display
    )

from utils import task_inference
from metrics import (
    ClassificationMetrics, 
    RegressionMetrics,
    MulticlassMetrics
    )
from predictor import predict
from save import save_thresholds


def threshold_moving(
    calculator, 
    y_pred_train, 
    y_true_train, 
    y_pred_val, 
    y_true_val):

    threshold_range = np.arange(0.0, 1.0, 0.01)
    train_results = []
    val_results = []

    for t in threshold_range:
        train_results.append(
            calculator.calibration(
                t, y_pred_train, y_true_train)
                )
        val_results.append(
            calculator.calibration(
                t, y_pred_val, y_true_val)
                )
    return train_results, val_results


def best_thresholds(
    train_results, 
    val_results, 
    num_tasks):

    train_metrics, val_metrics = [], []
    avg_thresholds = {'train': [], 'val': []}
    for task_idx in range(num_tasks):
        best_train = None
        best_val = None
        for train_res, val_res in zip(train_results, val_results):
            train_task = train_res[task_idx]
            val_task = val_res[task_idx]
            if best_train is None or train_task[
                'g_mean'] > best_train['g_mean']:
                best_train = train_task
            if best_val is None or val_task[
                'g_mean'] > best_val['g_mean']:
                best_val = val_task
        avg_thresholds['train'].append(
            best_train['threshold'])
        avg_thresholds['val'].append(
            best_val['threshold'])
        train_metrics.append({
            'task': task_idx + 1,
            'best_threshold': best_train['threshold'],
            'best_metrics': best_train}
            )
        val_metrics.append({
            'task': task_idx + 1,
            'best_threshold': best_val['threshold'],
            'best_metrics': best_val}
            )
    return train_metrics, val_metrics, avg_thresholds


def apply_thresholds(
    test_pred, 
    test_true, 
    val_thresholds, 
    calculator):

    test_metrics = []
    for task_idx, threshold in enumerate(val_thresholds):
        mask = ~np.isnan(test_true[:, task_idx])
        task_true = test_true[:, task_idx][mask]
        task_pred = (test_pred[:, task_idx
                ] > threshold).astype(int)[mask]
        task_prob = test_pred[:, task_idx][mask]
        metrics = {
            'accuracy': calculator.accuracy(
                task_true, task_pred),
            'recall': calculator.recall(
                task_true, task_pred),
            'specificity': calculator.specificity(
                task_true, task_pred),
            'ppv': calculator.ppv(
                task_true, task_pred),
            'npv': calculator.npv(
                task_true, task_pred),
            'g_mean': calculator.g_mean(
                calculator.recall(
                    task_true, task_pred),
                calculator.specificity(
                    task_true, task_pred)),
            'f1': calculator.f1(
                task_true, task_pred),
            'mcc': calculator.mcc(
                task_true, task_pred),
            'prauc': calculator.prauc(
                task_true, task_prob),
            'auc': calculator.auc(
                task_true, task_prob)
                }
        test_metrics.append({
            'task': task_idx + 1,
            'best_threshold': threshold,
            'best_metrics': metrics}
            )
    return test_metrics


def standard_threshold(
    train_pred, 
    train_true, 
    val_pred, 
    val_true, 
    test_pred, 
    test_true, 
    threshold, 
    calculator):

    train_metrics = calculator.calculate_metrics(
        train_pred, train_true, threshold
        )
    val_metrics = calculator.calculate_metrics(
        val_pred, val_true, threshold
        )
    test_metrics = calculator.calculate_metrics(
        test_pred, test_true, threshold
        )
    train_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(train_metrics)
            ]
    val_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(val_metrics)
            ]
    test_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(test_metrics)
            ]
    return train_metrics, val_metrics, test_metrics


def classification_markdown(
    train_metrics, 
    val_metrics, 
    test_metrics,
    global_train_metrics, 
    global_val_metrics, 
    global_test_metrics,
    avg_train_threshold, 
    avg_val_threshold, 
    avg_test_threshold,
    task_indices=None):

    def format_row(label, set_name, threshold, metrics):
        return (
            f"{label} | {set_name:<10} | {threshold:<9.4f} | "
            f"{metrics['accuracy']:<8.4f} | {metrics['recall']:<6.4f} | "
            f"{metrics['specificity']:<11.4f} | {metrics['ppv']:<6.4f} | "
            f"{metrics['npv']:<6.4f} | {metrics['f1']:<6.4f} | "
            f"{metrics['g_mean']:<6.4f} | {metrics['mcc']:<6.4f} | "
            f"{metrics['prauc']:<6.4f} | {metrics['auc']:<4.4f}\n"
            )
    output = (
        "Task  | Set   | Threshold | Accuracy | Recall | Specificity | PPV  | NPV  | F1   | G-mean | MCC  | PRAUC | AUC\n"
        "------|-------|-----------|----------|--------|-------------|------|------|------|--------|------|-------|-----\n"
        )
    for idx, (train, val, test) in enumerate(zip(
        train_metrics, 
        val_metrics, 
        test_metrics)):
       
        label_idx = task_indices[idx
            ] if task_indices is not None else (idx + 1)
        task_label = f"Task {label_idx}"
        output += format_row(
            task_label, 'Training',  
            train['best_threshold'], 
            train['best_metrics']
            )
        output += format_row(
            task_label, 'Validation',
            val['best_threshold'],   
            val['best_metrics']
            )
        output += format_row(
            task_label, 'Test', 
            test['best_threshold'], 
            test['best_metrics']
            )
    output += format_row(
        'Global', 'Training',  
        avg_train_threshold, 
        global_train_metrics
        )
    output += format_row(
        'Global', 'Validation',
        avg_val_threshold,    
        global_val_metrics
        )
    output += format_row(
        'Global', 'Test', 
        avg_test_threshold, 
        global_test_metrics
        )
    display(Markdown(output))


def regression_markdown(
    train_metrics, 
    val_metrics, 
    test_metrics, 
    global_train_metrics, 
    global_val_metrics, 
    global_test_metrics,
    task_indices=None):

    def format_row(label, set_name, metrics):
        return (f"{label} | {set_name:<10} | {metrics['r2']:<5.4f} | {metrics['pearson']:<7.4f} | "
                f"{metrics['mse']:<7.4f} | {metrics['rmse']:<7.4f} | {metrics['mae']:<7.4f}\n"
                )
    output = (
        "Task   | Set        | R^2   | Pearson | MSE     | RMSE    | MAE    \n"
        "-------|------------|-------|---------|---------|---------|--------\n"
        )
    for i, (train, val, test) in enumerate(zip(
        train_metrics, 
        val_metrics, 
        test_metrics)):

        label_idx = task_indices[i
            ] if task_indices is not None else (i + 1)
        task_label = f"Task {label_idx}"
        if isinstance(train, list): train = train[0]
        if isinstance(val, list): val = val[0]
        if isinstance(test, list): test = test[0]
        output += format_row(
            task_label, 'Training', train)
        output += format_row(
            task_label, 'Validation', val)
        output += format_row(
            task_label, 'Test', test)

    output += format_row(
        'Global', 'Training', 
        global_train_metrics
        )
    output += format_row(
        'Global', 'Validation', 
        global_val_metrics
        )
    output += format_row(
        'Global', 'Test', 
        global_test_metrics
        )
    display(Markdown(output))


def multiclass_markdown(
    train_metrics,
    val_metrics,
    test_metrics,
    global_train_metrics,
    global_val_metrics,
    global_test_metrics,
    task_indices=None):

    def format_row(label, set_name, metrics):
        return (
            f"{label} | {set_name:<10} | {metrics['accuracy']:<8.4f} | "
            f"{metrics['balanced_accuracy_macro']:<8.4f} | "
            f"{metrics['f1_macro']:<8.4f} | "
            f"{metrics['mcc']:<6.4f} | "
            f"{metrics['prauc_macro']:<8.4f} | "
            f"{metrics['auc_ovr']:<8.4f}\n"
            )
    output = (
        "Task  | Set   | Accuracy | BalAcc | F1     | MCC   | PRAUC | AUC(OVO)\n"
        "------|-------|----------|--------|--------|-------|-------|---------\n"
        )
    for idx, (m_tr, m_va, m_te) in enumerate(zip(
        train_metrics, val_metrics, test_metrics)):
        label_idx = task_indices[idx
            ] if task_indices is not None else (idx + 1)
        task_label = f"Task {label_idx}"
        output += format_row(task_label, "Training", m_tr)
        output += format_row(task_label, "Validation", m_va)
        output += format_row(task_label, "Test", m_te)
    output += format_row("Global", "Training", global_train_metrics)
    output += format_row("Global", "Validation", global_val_metrics)
    output += format_row("Global", "Test", global_test_metrics)
    display(Markdown(output))


class MixedEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cls_calc = ClassificationMetrics(model, device)
        self.reg_calc = RegressionMetrics(model, device)
        self.mc_calc = MulticlassMetrics()
        self.best_thresholds = {}

    def _split_by_type(self, y_pred, y_true):
        mask = ~np.isnan(y_true)
        y_true_t = np.nan_to_num(y_true, nan=0.0)
        yt = torch.tensor(y_true_t, dtype=torch.float32)
        mk = torch.tensor(mask, dtype=torch.bool)
        task_type = task_inference(
            yt, mk).cpu().numpy()
        idx_bin = np.where(task_type == 1)[0].tolist()
        idx_mc = np.where(task_type == 2)[0].tolist()
        idx_reg = np.where(task_type == 0)[0].tolist()

        def take_cols(a, cols):
            return a[:, cols] if (
                a is not None and len(cols) > 0) else None

        return (
            idx_bin, idx_mc, idx_reg,
            take_cols(y_pred, idx_bin), 
            take_cols(y_true, idx_bin),
            take_cols(y_pred, idx_reg), 
            take_cols(y_true, idx_reg),
            )
    def _evaluate_classification_block(
        self,
        y_pred_tr, y_true_tr,
        y_pred_va, y_true_va,
        y_pred_te, y_true_te,
        idx_cls, 
        calibration=True):

        if calibration:
            train_results, val_results = threshold_moving(
                self.cls_calc, y_pred_tr, 
                y_true_tr, y_pred_va, y_true_va
                )
            train_metrics, val_metrics, avg_thresholds = best_thresholds(
                train_results, val_results, y_true_tr.shape[1]
                )
            test_metrics = apply_thresholds(
                y_pred_te, y_true_te, avg_thresholds['val'], self.cls_calc
                )
            avg_thresholds['test'] = avg_thresholds['val']
        else:
            default_thresh = 0.5
            train_metrics, val_metrics, test_metrics = standard_threshold(
                y_pred_tr, y_true_tr, y_pred_va, 
                y_true_va, y_pred_te, y_true_te,
                default_thresh, self.cls_calc
                )
            avg_thresholds = {
                'train': [default_thresh] * y_true_tr.shape[1],
                'val':   [default_thresh] * y_true_va.shape[1],
                'test':  [default_thresh] * y_true_te.shape[1],
                }
        self.best_thresholds.update(avg_thresholds)
        save_thresholds(self.best_thresholds)

        def flat_valid(pred, true):
            mk = ~np.isnan(true)
            return pred[mk].reshape(-1, 1), true[mk].reshape(-1, 1)

        trp_g, trt_g = flat_valid(y_pred_tr, y_true_tr)
        vap_g, vat_g = flat_valid(y_pred_va, y_true_va)
        tep_g, tet_g = flat_valid(y_pred_te, y_true_te)

        tr_avg = self.cls_calc.calculate_metrics(
            trp_g, trt_g, np.mean(avg_thresholds['train']))[0]
        va_avg = self.cls_calc.calculate_metrics(
            vap_g, vat_g, np.mean(avg_thresholds['val']))[0]
        te_avg = self.cls_calc.calculate_metrics(
            tep_g, tet_g, np.mean(avg_thresholds['test']))[0]

        classification_markdown(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            global_train_metrics=tr_avg,
            global_val_metrics=va_avg,
            global_test_metrics=te_avg,
            avg_train_threshold=np.mean(avg_thresholds['train']),
            avg_val_threshold=np.mean(avg_thresholds['val']),
            avg_test_threshold=np.mean(avg_thresholds['test']),
            task_indices=[i + 1 for i in idx_cls]
            )
    def _evaluate_regression_block(
        self,
        y_pred_tr, y_true_tr,
        y_pred_va, y_true_va,
        y_pred_te, y_true_te,
        idx_reg):

        tr = self.reg_calc.calculate_metrics(
            y_pred_tr, y_true_tr)
        va = self.reg_calc.calculate_metrics(
            y_pred_va, y_true_va)
        te = self.reg_calc.calculate_metrics(
            y_pred_te, y_true_te)

        tr_g = self.reg_calc.calculate_metrics(
            y_pred_tr, y_true_tr)[0]
        va_g = self.reg_calc.calculate_metrics(
            y_pred_va, y_true_va)[0]
        te_g = self.reg_calc.calculate_metrics(
            y_pred_te, y_true_te)[0]

        regression_markdown(
            train_metrics=tr,
            val_metrics=va,
            test_metrics=te,
            global_train_metrics=tr_g,
            global_val_metrics=va_g,
            global_test_metrics=te_g,
            task_indices=[i + 1 for i in idx_reg]
            )

    def _evaluate_multiclass_block(
        self,
        y_prob_tr, y_true_tr,
        y_prob_va, y_true_va,
        y_prob_te, y_true_te,
        idx_mc):

        train_metrics = []
        val_metrics = []
        test_metrics = []
        label_vals = getattr(self.model, "mc_label_values", None)
        for j, task_idx in enumerate(idx_mc):
            def slice_task(y_prob, y_true):
                if y_prob is None or y_true is None:
                    return None, None
                if isinstance(y_prob, dict):
                    probs = y_prob.get(task_idx)
                else:
                    probs = y_prob[:, task_idx, :]
                if probs is None:
                    return None, None
                true = y_true[:, task_idx]
                mask = ~np.isnan(true)
                if not mask.any():
                    return None, None
                labels = np.rint(true[mask]).astype(int)
                mapping_vals = None
                if label_vals is not None:
                    mapping_vals = label_vals[task_idx]
                if mapping_vals is None:
                    mapping_vals = sorted(np.unique(labels).tolist())
                mapping = {int(v): i for i, v in enumerate(mapping_vals)}
                mapped = []
                for v in labels.tolist():
                    if int(v) not in mapping:
                        raise ValueError(
                            "Unseen multiclass label in evaluation.")
                    mapped.append(mapping[int(v)])
                mapped = np.asarray(mapped, dtype=int)
                if probs.shape[1] != len(mapping_vals):
                    raise ValueError(
                        "Multiclass probabilities size mismatch.")
                return probs[mask], mapped

            probs_tr, labels_tr = slice_task(
                y_prob_tr, y_true_tr)
            probs_va, labels_va = slice_task(
                y_prob_va, y_true_va)
            probs_te, labels_te = slice_task(
                y_prob_te, y_true_te)

            if probs_te is None or labels_te is None:
                continue
            if probs_tr is not None and labels_tr is not None:
                train_metrics.append(self.mc_calc.calculate_metrics(
                    probs_tr, labels_tr))
            else:
                train_metrics.append(self.mc_calc.calculate_metrics(
                    probs_te, labels_te))
            if probs_va is not None and labels_va is not None:
                val_metrics.append(self.mc_calc.calculate_metrics(
                    probs_va, labels_va))
            else:
                val_metrics.append(self.mc_calc.calculate_metrics(
                    probs_te, labels_te))
            test_metrics.append(self.mc_calc.calculate_metrics(
                probs_te, labels_te))

        if not test_metrics:
            print("No multiclass tasks detected.")
            return

        def avg_metric(metrics_list, key):
            vals = [m[key] for m in metrics_list]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan

        global_train_metrics = {
            'accuracy': avg_metric(train_metrics, 'accuracy'),
            'balanced_accuracy_macro': avg_metric(
                train_metrics, 'balanced_accuracy_macro'),
            'f1_macro': avg_metric(train_metrics, 'f1_macro'),
            'mcc': avg_metric(train_metrics, 'mcc'),
            'prauc_macro': avg_metric(train_metrics, 'prauc_macro'),
            'auc_ovr': avg_metric(train_metrics, 'auc_ovr')
            }
        global_val_metrics = {
            'accuracy': avg_metric(val_metrics, 'accuracy'),
            'balanced_accuracy_macro': avg_metric(
                val_metrics, 'balanced_accuracy_macro'),
            'f1_macro': avg_metric(val_metrics, 'f1_macro'),
            'mcc': avg_metric(val_metrics, 'mcc'),
            'prauc_macro': avg_metric(val_metrics, 'prauc_macro'),
            'auc_ovr': avg_metric(val_metrics, 'auc_ovr')
            }
        global_test_metrics = {
            'accuracy': avg_metric(test_metrics, 'accuracy'),
            'f1_macro': avg_metric(test_metrics, 'f1_macro'),
            'balanced_accuracy_macro': avg_metric(
                test_metrics, 'balanced_accuracy_macro'),
            'mcc': avg_metric(test_metrics, 'mcc'),
            'prauc_macro': avg_metric(test_metrics, 'prauc_macro'),
            'auc_ovr': avg_metric(test_metrics, 'auc_ovr')
            }

        multiclass_markdown(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            global_train_metrics=global_train_metrics,
            global_val_metrics=global_val_metrics,
            global_test_metrics=global_test_metrics,
            task_indices=[i + 1 for i in idx_mc]
            )
    def evaluate(self, train_loader, 
        val_loader, test_loader, calibration=True):

        tt = getattr(self.model, "task_type", None)
        mlv = getattr(self.model, "mc_label_values", None)
        y_pred_tr, y_true_tr, _, y_prob_tr = predict(
            self.model, train_loader, self.device,
            return_mc_probs=True, task_type=tt,
            mc_label_values=mlv)
        y_pred_va, y_true_va, _, y_prob_va = predict(
            self.model, val_loader, self.device,
            return_mc_probs=True, task_type=tt,
            mc_label_values=mlv)
        y_pred_te, y_true_te, _, y_prob_te = predict(
            self.model, test_loader, self.device,
            return_mc_probs=True, task_type=tt,
            mc_label_values=mlv)

        (idx_bin, idx_mc, idx_reg,
         ypt_c, ytt_c, ypt_r, ytt_r
         ) = self._split_by_type(y_pred_tr, y_true_tr)

        def slice_cols(arr, cols):
            return arr[:, cols] if (
                arr is not None and len(cols) > 0) else None
        ypv_c = slice_cols(y_pred_va, idx_bin)
        ytv_c = slice_cols(y_true_va, idx_bin)
        yps_c = slice_cols(y_pred_te, idx_bin)
        yts_c = slice_cols(y_true_te, idx_bin)
        ypv_r = slice_cols(y_pred_va, idx_reg)
        ytv_r = slice_cols(y_true_va, idx_reg)
        yps_r = slice_cols(y_pred_te, idx_reg)
        yts_r = slice_cols(y_true_te, idx_reg)

        if ypt_c is not None and ytt_c is not None and ypt_c.shape[1] > 0:
            self._evaluate_classification_block(
                y_pred_tr=ypt_c, y_true_tr=ytt_c,
                y_pred_va=ypv_c, y_true_va=ytv_c,
                y_pred_te=yps_c, y_true_te=yts_c,
                idx_cls=idx_bin,
                calibration=calibration
                )
        else:
            print("No classification tasks detected.")

        if y_prob_tr is not None and idx_mc:
            self._evaluate_multiclass_block(
                y_prob_tr=y_prob_tr, y_true_tr=y_true_tr,
                y_prob_va=y_prob_va, y_true_va=y_true_va,
                y_prob_te=y_prob_te, y_true_te=y_true_te,
                idx_mc=idx_mc
                )
        elif idx_mc:
            print("No multiclass probabilities available.")

        if ypt_r is not None and ytt_r is not None and ypt_r.shape[1] > 0:
            self._evaluate_regression_block(
                y_pred_tr=ypt_r, y_true_tr=ytt_r,
                y_pred_va=ypv_r, y_true_va=ytv_r,
                y_pred_te=yps_r, y_true_te=yts_r,
                idx_reg=idx_reg
                )
        else:
            print("No regression tasks detected.")


def ModelEvaluator(
    model,
    device,
    train_loader,
    val_loader,
    test_loader,
    calibration=True):

    evaluator = MixedEvaluator(model, device)
    evaluator.evaluate(
        train_loader, 
        val_loader, 
        test_loader, 
        calibration=calibration
        )
