import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device(
    "cuda" if torch.cuda.is_available() 
           else "cpu"
           )


def one_hot(value, categories):
    return [1 if i == categories.index(value) 
            else 0 for i in range(len(categories))
           ]


def decode_one_hot(one_hot_vector, categories):
    if isinstance(one_hot_vector, list):
        index = one_hot_vector.index(1)
    else:
        index = one_hot_vector.argmax().item()
    return categories[index]


def task_inference(true, mask, tol=1e-4):
    device = true.device
    num_tasks = true.size(1)
    task_type = torch.zeros(num_tasks, 
            dtype=torch.long, device=device)

    diff_in01 = torch.where(mask, (
        true - true.clamp(0.0, 1.0)).abs(), 
        torch.zeros_like(true))
    cond_in01 = diff_in01.amax(dim=0) <= tol

    y_round01 = (true >= 0.5).float()
    diff_near01 = torch.where(mask, (
        true - y_round01).abs(), torch.zeros_like(true))
    cond_near01 = diff_near01.amax(dim=0) <= tol

    has_valid = mask.any(dim=0)
    is_binary = cond_in01 & cond_near01 & has_valid
    task_type[is_binary] = 1

    candidates = (~is_binary) & has_valid
    cand_idx = torch.where(candidates)[0]

    for j in cand_idx.tolist():
        vals = true[mask[:, j], j]
        if vals.numel() == 0:
            continue
        if (vals - vals.round()).abs().max() > tol:
            continue
        labels = vals.round().to(torch.long)
        if labels.unique().numel() >= 3:
            task_type[j] = 2

    return task_type


def clip_gradients(
    model, 
    max_norm, 
    norm_type=2, 
    method='norm'):
    
    parameters = [p for p in model.parameters() 
                  if p.grad is not None
                 ]
    if method == 'norm':
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 
            max_norm, 
            norm_type=norm_type
            )
        return total_norm
    elif method == 'value':
        torch.nn.utils.clip_grad_value_(
            parameters, max_norm
            )
        return None
    else:
        raise ValueError(
            f"Unsupported clipping method: {method}"
            )
