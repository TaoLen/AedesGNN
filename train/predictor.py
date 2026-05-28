import numpy as np
import torch

from utils import task_inference


def predict(
    model,
    data_loader,
    device,
    return_embeddings=False,
    return_mc_probs=False,
    task_type=None,
    mc_label_values=None):

    model.eval()
    torch.set_grad_enabled(False)
    all_predictions = []
    all_labels = []
    embeddings = []
    all_mc_probs = {} if return_mc_probs else None

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = batch.get(
                    "supervised",
                    batch.get("original"),
                )
            if batch is None:
                continue
            inputs = batch.to(device)

            if return_embeddings:
                emb = model(inputs, return_embeddings=True)
                embeddings.extend(emb.detach().cpu().numpy())

            outputs = model(inputs)
            outputs_scalar = None
            mc_logits = None
            if isinstance(outputs, dict):
                outputs_scalar = outputs.get("scalar")
                mc_logits = outputs.get("mc_logits", {})
            else:
                if outputs.dim() == 3:
                    mc_logits = outputs
                    outputs_scalar = outputs[..., 0]
                else:
                    outputs_scalar = outputs
            if outputs_scalar is None:
                raise ValueError(
                    "Model must return scalar logits for all tasks.")

            if hasattr(inputs, "y") and inputs.y is not None:
                y_true = inputs.y
                mask = ~torch.isnan(y_true)
                y_true_z = torch.nan_to_num(y_true, nan=0.0)
                tt = task_type
                if tt is None:
                    tt = getattr(model, "task_type", None)
                if tt is None:
                    tt = task_inference(
                        y_true_z, mask
                        ).to(outputs_scalar.device)
                else:
                    tt = tt.to(outputs_scalar.device)
                is_bin = (tt == 1).view(1, -1)
                is_mc = (tt == 2).view(1, -1)
                preds = torch.where(
                    is_bin, 
                    torch.sigmoid(outputs_scalar),
                    outputs_scalar
                    )
                if is_mc.any() and mc_logits is not None:
                    label_vals = mc_label_values
                    if label_vals is None:
                        label_vals = getattr(
                            model, "mc_label_values", None)
                    preds = preds.clone()
                    for j in torch.where(is_mc)[0].tolist():
                        logits_j = (
                            mc_logits.get(j)
                            if isinstance(mc_logits, dict)
                            else mc_logits[:, j, :]
                            )
                        if logits_j is None:
                            continue
                        probs_j = torch.softmax(logits_j, dim=-1)
                        class_idx = probs_j.argmax(dim=-1)
                        if label_vals is not None and label_vals[j] is not None:
                            values = torch.tensor(
                                label_vals[j],
                                device=class_idx.device,
                                dtype=preds.dtype)
                            mapped = values[class_idx]
                            preds[:, j] = mapped
                        else:
                            preds[:, j] = class_idx.to(preds.dtype)
                labels = y_true.detach().cpu().numpy()
                all_labels.extend(labels)
            else:
                preds = outputs_scalar

            all_predictions.extend(preds.detach().cpu().numpy())
            if return_mc_probs and mc_logits is not None:
                if isinstance(mc_logits, dict):
                    for k, v in mc_logits.items():
                        probs = torch.softmax(v, dim=-1)
                        if k not in all_mc_probs:
                            all_mc_probs[k] = []
                        all_mc_probs[k].extend(
                            probs.detach().cpu().numpy())
                else:
                    probs = torch.softmax(mc_logits, dim=-1)
                    if "__tensor__" not in all_mc_probs:
                        all_mc_probs["__tensor__"] = []
                    all_mc_probs["__tensor__"].extend(
                        probs.detach().cpu().numpy())

    all_predictions = np.asarray(all_predictions)
    all_labels = np.asarray(all_labels
                ) if len(all_labels) else None
    embeddings = np.asarray(embeddings
                ) if return_embeddings else None

    if return_mc_probs:
        mc_probs_payload = None
        if all_mc_probs:
            if "__tensor__" in all_mc_probs:
                mc_probs_payload = np.asarray(
                    all_mc_probs["__tensor__"])
            else:
                mc_probs_payload = {
                    k: np.asarray(v) for k, v in all_mc_probs.items()
                    }
        return all_predictions, all_labels, embeddings, mc_probs_payload

    return all_predictions, all_labels, embeddings
