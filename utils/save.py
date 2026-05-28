import os
import json
import torch

from utils import task_inference


def save_model(
    model, 
    out_path, 
    filename):

    os.makedirs(out_path, exist_ok=True)
    if not filename.endswith(".pth"):
        filename += ".pth"
    filepath = os.path.join(out_path, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def save_params(
    params,
    out_path,
    filename,
    device,
    data_loader=None):

    if ("is_cls" not in params) and (data_loader is not None):
        ys, ms = [], []
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = batch.get(
                    "supervised",
                    batch.get("original"),
                )
            if batch is None:
                continue
            if not hasattr(batch, "y") or batch.y is None:
                continue
            y = batch.y.to(device)
            m = ~torch.isnan(y)
            yz = torch.nan_to_num(y, nan=0.0)
            ys.append(yz)
            ms.append(m)
        if ys:
            Y = torch.cat(ys, dim=0)
            M = torch.cat(ms, dim=0)
            is_cls = task_inference(Y, M)
            params["is_cls"] = is_cls.detach().cpu().tolist()

    if not filename.endswith(".json"):
        filename += ".json"
    filepath = os.path.join(out_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Best parameters saved to {filepath}")


def save_thresholds(
    best_thresholds, 
    out_path='../output/calibration/thresholds.json'):

    try:
        dir_path = os.path.dirname(out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        absolute_path = os.path.abspath(out_path)
        with open(absolute_path, 'w') as file:
            json.dump(best_thresholds, file, indent=4)
    except Exception as e:
        print(f"Failed to save best thresholds: {e}")


def save_embeddings(
    model,
    data_loader,
    epoch,
    out_path="../output/embeddings",
    is_cls=None):
    
    os.makedirs(out_path, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    all_embeddings = []
    all_labels = []
    ys_for_infer, ms_for_infer = [], []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = batch.get(
                    "supervised",
                    batch.get("original"),
                )
            if batch is None:
                continue
            batches = batch if isinstance(
            batch, list) else [batch]

            for b in batches:
                b = b.to(device)
                try:
                    emb = model(b, save_embeddings=False, 
                        return_penultimate=True).cpu()
                except TypeError:
                    emb = model(b).cpu()
                all_embeddings.append(emb)

                if hasattr(b, 'y') and b.y is not None:
                    y = b.y
                    all_labels.append(y.cpu())
                    m = ~torch.isnan(y)
                    yz = torch.nan_to_num(y, nan=0.0)
                    ys_for_infer.append(yz)
                    ms_for_infer.append(m)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    payload = {'embeddings': embeddings_tensor}

    if all_labels:
        labels_tensor = torch.cat(all_labels, dim=0)
        payload['labels'] = labels_tensor
    is_cls_tensor = None
    if is_cls is not None:
        is_cls_tensor = torch.as_tensor(
            is_cls, dtype=torch.bool)
    elif ys_for_infer:
        Y = torch.cat(ys_for_infer, dim=0)
        M = torch.cat(ms_for_infer, dim=0)
        is_cls_tensor = task_inference(Y, M).to(torch.bool)
    if is_cls_tensor is not None:
        payload['is_cls'] = is_cls_tensor
    filename = os.path.join(
        out_path, f"embeddings_epoch_{epoch+1}.pt")
    torch.save(payload, filename)
