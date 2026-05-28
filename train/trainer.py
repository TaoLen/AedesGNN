import copy
import torch
import optuna
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils import (
    device, 
    clip_gradients
    )

from loss import (
    MaskedLoss,
    ContrastiveAuxiliaryLoss,
    prepare_masked_data,
    build_mc_label_maps,
    compute_loss_matrix,
    reduce_task_mean,
    )

from save import save_embeddings


def get_loss(
    model,
    num_tasks,
    use_uncertainty,
    task_type=None,
    mc_label_values=None):

    if not use_uncertainty:
        loss_fn = MaskedLoss(
            num_tasks=num_tasks,
            task_type=task_type,
            mc_label_values=mc_label_values,
            )
        loss_fn._task_type = task_type
        loss_fn._mc_label_values = mc_label_values
        loss_fn._uses_uncertainty = False
        return loss_fn
    mc_label_maps = build_mc_label_maps(mc_label_values)
    def loss_fn(y_pred, y_true):
        pred_scalar = (
            y_pred.get("scalar")
            if isinstance(y_pred, dict)
            else y_pred
            )
        y_t, y_p, mask = prepare_masked_data(
            y_true, pred_scalar)
        return model.uncertainty(
            y_pred, y_t, mask,
            task_type=task_type,
            mc_label_maps=mc_label_maps
            )

    loss_fn._task_type = task_type
    loss_fn._mc_label_values = mc_label_values
    loss_fn._uses_uncertainty = True
    return loss_fn



def loss_stats(model, y_pred, y_true, loss_fn):
    task_type = getattr(loss_fn, "_task_type", None)
    mc_label_values = getattr(loss_fn, "_mc_label_values", None)
    loss_mat, mask, task_type = compute_loss_matrix(
        y_pred, y_true,
        task_type=task_type,
        mc_label_values=mc_label_values
        )

    if getattr(loss_fn, "_uses_uncertainty", False):
        if not hasattr(model, "uncertainty"):
            raise ValueError("Missing uncertainty module on model.")
        alphas = model.uncertainty.log_vars.view(1, -1).to(loss_mat.device)
        is_cls = (task_type != 0).view(1, -1).to(loss_mat.device)
        w_cls = torch.exp(-alphas) * loss_mat + alphas
        w_reg = 0.5 * torch.exp(-alphas) * loss_mat + 0.5 * alphas
        loss_mat = torch.where(is_cls, w_cls, w_reg)

    task_means, valid = reduce_task_mean(
        loss_mat, mask.float())

    is_bin = (task_type == 1)
    is_mc = (task_type == 2)
    is_reg = (task_type == 0)

    def sum_count(type_mask):
        valid_type = valid & type_mask
        return task_means[valid_type].sum(), valid_type.sum()

    total_sum = task_means[valid].sum()
    total_count = valid.sum()
    bin_sum, bin_count = sum_count(is_bin)
    mc_sum, mc_count = sum_count(is_mc)
    reg_sum, reg_count = sum_count(is_reg)

    return {
        "total_sum": total_sum,
        "total_count": total_count,
        "bin_sum": bin_sum,
        "bin_count": bin_count,
        "mc_sum": mc_sum,
        "mc_count": mc_count,
        "reg_sum": reg_sum,
        "reg_count": reg_count,
    }


def get_contrastive_loss(contrastive_config=None):
    if not contrastive_config:
        return None
    if not contrastive_config.get("enabled", True):
        return None
    return ContrastiveAuxiliaryLoss(
        beta_aug=contrastive_config.get("beta_aug", 0.0),
        alpha_global=contrastive_config.get("alpha_global", 0.0),
        alpha_local=contrastive_config.get("alpha_local", 0.0),
        global_temperature=contrastive_config.get(
            "global_temperature", 0.1),
        local_temperature=contrastive_config.get(
            "local_temperature", 0.1),
        tanimoto_lambda=contrastive_config.get(
            "tanimoto_lambda", 0.5),
        fp_radius=contrastive_config.get("fp_radius", 2),
        fp_size=contrastive_config.get("fp_size", 2048),
        use_cosine_similarity=contrastive_config.get(
            "use_cosine_similarity", True),
    )


def _unwrap_supervised_batch(data):
    if isinstance(data, dict):
        return data.get("supervised", data.get("original"))
    return data


def _require_contrastive_batch(data):
    if not isinstance(data, dict):
        raise ValueError(
            "contrastive_config is active, but the data loader is not "
            "returning contrastive batch dictionaries."
        )

    required_keys = (
        "supervised",
        "view_i",
        "view_j",
        "smiles",
        "fragment_atom_index",
        "fragment_index",
        "num_fragments",
    )
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(
            "contrastive_config is active, but the contrastive batch is "
            f"missing required keys: {missing}"
        )


def _init_contrastive_stats():
    return {
        "total_sum": 0.0,
        "global_sum": 0.0,
        "local_sum": 0.0,
        "count": 0.0,
    }


def _update_contrastive_stats(stats, aux_losses, weight):
    stats["total_sum"] += float(aux_losses["total"].item()) * weight
    stats["global_sum"] += float(aux_losses["global"].item()) * weight
    stats["local_sum"] += float(aux_losses["local"].item()) * weight
    stats["count"] += float(weight)


def _finalize_contrastive_stats(stats):
    if stats["count"] <= 0:
        return {
            "contrastive_total": float("nan"),
            "contrastive_global": float("nan"),
            "contrastive_local": float("nan"),
            "contrastive_count": 0.0,
        }
    denom = stats["count"]
    return {
        "contrastive_total": stats["total_sum"] / denom,
        "contrastive_global": stats["global_sum"] / denom,
        "contrastive_local": stats["local_sum"] / denom,
        "contrastive_count": denom,
    }


def _attach_total_loss(stats, supervised_loss):
    contrastive_total = stats.get("contrastive_total", 0.0)
    if not torch.isfinite(torch.tensor(contrastive_total)):
        contrastive_total = 0.0
    supervised_aug_loss = stats.get("supervised_aug_loss", 0.0)
    if not torch.isfinite(torch.tensor(supervised_aug_loss)):
        supervised_aug_loss = 0.0
    beta_aug = stats.get("beta_aug", 0.0)
    supervised_block = supervised_loss + beta_aug * supervised_aug_loss
    stats["supervised_base_loss"] = supervised_loss
    stats["supervised_loss"] = supervised_block
    stats["total_loss"] = supervised_block + contrastive_total
    return stats


def _current_lr(optimizer):
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def train_epoch(
    model, 
    optimizer, 
    data_loader, 
    loss_fn, 
    contrastive_loss_fn=None,
    max_grad_norm=1.0, 
    clip_method='norm',
    return_stats=False):

    model.train()
    total_loss = 0.0
    total_count = 0.0
    bin_sum = 0.0
    bin_count = 0.0
    mc_sum = 0.0
    mc_count = 0.0
    reg_sum = 0.0
    reg_count = 0.0
    aug_supervised_sum = 0.0
    aug_supervised_count = 0.0
    contrastive_stats = _init_contrastive_stats()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        if contrastive_loss_fn is not None:
            _require_contrastive_batch(data)
        supervised_batch = _unwrap_supervised_batch(data)
        if isinstance(supervised_batch, Batch):
            supervised_batch = supervised_batch.to(device)
            labels = supervised_batch.y.to(device
                ) if hasattr(supervised_batch, 'y'
                ) and supervised_batch.y is not None else None
            out = model(supervised_batch)
            loss = loss_fn(out, labels)
            if contrastive_loss_fn is not None and isinstance(data, dict):
                view_i = data["view_i"].to(device)
                view_j = data["view_j"].to(device)
                view_i_repr = model(
                    view_i, return_representations=True)
                view_j_repr = model(
                    view_j, return_representations=True)
                beta_aug = getattr(
                    contrastive_loss_fn, "beta_aug", 0.0)
                if beta_aug > 0.0:
                    view_i_pred = view_i_repr["predictions"]
                    view_j_pred = view_j_repr["predictions"]
                    view_i_loss = loss_fn(view_i_pred, labels)
                    view_j_loss = loss_fn(view_j_pred, labels)
                    loss = loss + beta_aug * 0.5 * (
                        view_i_loss + view_j_loss
                    )
                aux_losses = contrastive_loss_fn(
                    model,
                    view_i_repr,
                    view_j_repr,
                    data,
                )
                _update_contrastive_stats(
                    contrastive_stats,
                    aux_losses,
                    len(data.get("smiles", [])),
                )
                loss = loss + aux_losses["total"]
        else:
            continue

        loss.backward()
        clip_gradients(model, 
            max_grad_norm, 
            method=clip_method
            )
        optimizer.step()
        with torch.no_grad():
            stats = loss_stats(model, out, labels, loss_fn)
        total_loss += float(stats["total_sum"].item())
        total_count += float(stats["total_count"].item())
        bin_sum += float(stats["bin_sum"].item())
        bin_count += float(stats["bin_count"].item())
        mc_sum += float(stats["mc_sum"].item())
        mc_count += float(stats["mc_count"].item())
        reg_sum += float(stats["reg_sum"].item())
        reg_count += float(stats["reg_count"].item())
        if contrastive_loss_fn is not None:
            beta_aug = getattr(
                contrastive_loss_fn, "beta_aug", 0.0)
            if beta_aug > 0.0:
                with torch.no_grad():
                    view_i_stats = loss_stats(
                        model,
                        view_i_repr["predictions"],
                        labels,
                        loss_fn,
                    )
                    view_j_stats = loss_stats(
                        model,
                        view_j_repr["predictions"],
                        labels,
                        loss_fn,
                    )
                aug_supervised_sum += 0.5 * (
                    float(view_i_stats["total_sum"].item())
                    + float(view_j_stats["total_sum"].item())
                )
                aug_supervised_count += 0.5 * (
                    float(view_i_stats["total_count"].item())
                    + float(view_j_stats["total_count"].item())
                )

    avg_supervised_loss = (
        total_loss / total_count if total_count > 0 else float("nan")
    )
    avg_aug_supervised_loss = (
        aug_supervised_sum / aug_supervised_count
        if aug_supervised_count > 0 else 0.0
    )
    if not return_stats:
        if contrastive_loss_fn is None:
            return avg_supervised_loss
        ctr_stats = _finalize_contrastive_stats(contrastive_stats)
        beta_aug = getattr(contrastive_loss_fn, "beta_aug", 0.0)
        return (
            avg_supervised_loss
            + beta_aug * avg_aug_supervised_loss
            + ctr_stats["contrastive_total"]
        )
    stats = {
        "bin": bin_sum / bin_count if bin_count > 0 else float("nan"),
        "mc": mc_sum / mc_count if mc_count > 0 else float("nan"),
        "reg": reg_sum / reg_count if reg_count > 0 else float("nan"),
        "bin_count": bin_count,
        "mc_count": mc_count,
        "reg_count": reg_count,
        "total_count": total_count,
        "supervised_aug_loss": avg_aug_supervised_loss,
        "beta_aug": (
            getattr(contrastive_loss_fn, "beta_aug", 0.0)
            if contrastive_loss_fn is not None else 0.0
        ),
    }
    if contrastive_loss_fn is not None:
        stats.update(_finalize_contrastive_stats(contrastive_stats))
        stats = _attach_total_loss(stats, avg_supervised_loss)
        return stats["total_loss"], stats
    stats = _attach_total_loss(stats, avg_supervised_loss)
    return avg_supervised_loss, stats


def evaluate(
    model, 
    data_loader, 
    loss_fn,
    contrastive_loss_fn=None,
    return_stats=False):

    model.eval()
    total_loss = 0.0
    total_count = 0.0
    bin_sum = 0.0
    bin_count = 0.0
    mc_sum = 0.0
    mc_count = 0.0
    reg_sum = 0.0
    reg_count = 0.0
    aug_supervised_sum = 0.0
    aug_supervised_count = 0.0
    contrastive_stats = _init_contrastive_stats()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if data is None:
                continue

            supervised_batch = _unwrap_supervised_batch(data)
            if contrastive_loss_fn is not None:
                _require_contrastive_batch(data)
            if isinstance(supervised_batch, Batch): 
                supervised_batch = supervised_batch.to(device)
                labels = supervised_batch.y.to(device
                    ) if hasattr(supervised_batch, 'y'
                    ) and supervised_batch.y is not None else None
                out = model(supervised_batch)
                if contrastive_loss_fn is not None and isinstance(data, dict):
                    view_i = data["view_i"].to(device)
                    view_j = data["view_j"].to(device)
                    view_i_repr = model(
                        view_i, return_representations=True)
                    view_j_repr = model(
                        view_j, return_representations=True)
                    beta_aug = getattr(
                        contrastive_loss_fn, "beta_aug", 0.0)
                    if beta_aug > 0.0:
                        view_i_stats = loss_stats(
                            model,
                            view_i_repr["predictions"],
                            labels,
                            loss_fn,
                        )
                        view_j_stats = loss_stats(
                            model,
                            view_j_repr["predictions"],
                            labels,
                            loss_fn,
                        )
                        aug_supervised_sum += 0.5 * (
                            float(view_i_stats["total_sum"].item())
                            + float(view_j_stats["total_sum"].item())
                        )
                        aug_supervised_count += 0.5 * (
                            float(view_i_stats["total_count"].item())
                            + float(view_j_stats["total_count"].item())
                        )
                    aux_losses = contrastive_loss_fn(
                        model,
                        view_i_repr,
                        view_j_repr,
                        data,
                    )
                    _update_contrastive_stats(
                        contrastive_stats,
                        aux_losses,
                        len(data.get("smiles", [])),
                    )
            else:
                continue

            stats = loss_stats(model, out, labels, loss_fn)
            total_loss += float(stats["total_sum"].item())
            total_count += float(stats["total_count"].item())
            bin_sum += float(stats["bin_sum"].item())
            bin_count += float(stats["bin_count"].item())
            mc_sum += float(stats["mc_sum"].item())
            mc_count += float(stats["mc_count"].item())
            reg_sum += float(stats["reg_sum"].item())
            reg_count += float(stats["reg_count"].item())

    avg_supervised_loss = (
        total_loss / total_count if total_count > 0 else float("nan")
    )
    avg_aug_supervised_loss = (
        aug_supervised_sum / aug_supervised_count
        if aug_supervised_count > 0 else 0.0
    )
    if not return_stats:
        if contrastive_loss_fn is None:
            return avg_supervised_loss
        ctr_stats = _finalize_contrastive_stats(contrastive_stats)
        beta_aug = getattr(contrastive_loss_fn, "beta_aug", 0.0)
        return (
            avg_supervised_loss
            + beta_aug * avg_aug_supervised_loss
            + ctr_stats["contrastive_total"]
        )
    stats = {
        "bin": bin_sum / bin_count if bin_count > 0 else float("nan"),
        "mc": mc_sum / mc_count if mc_count > 0 else float("nan"),
        "reg": reg_sum / reg_count if reg_count > 0 else float("nan"),
        "bin_count": bin_count,
        "mc_count": mc_count,
        "reg_count": reg_count,
        "total_count": total_count,
        "supervised_aug_loss": avg_aug_supervised_loss,
        "beta_aug": (
            getattr(contrastive_loss_fn, "beta_aug", 0.0)
            if contrastive_loss_fn is not None else 0.0
        ),
    }
    if contrastive_loss_fn is not None:
        stats.update(_finalize_contrastive_stats(contrastive_stats))
        stats = _attach_total_loss(stats, avg_supervised_loss)
        return stats["total_loss"], stats
    stats = _attach_total_loss(stats, avg_supervised_loss)
    return avg_supervised_loss, stats


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    patience,
    delta,
    window_size,
    best_model=True,
    plateau_patience=3,
    eta_min=0.001,
    enable_pruning=True,
    contrastive_loss_fn=None,
    warm_up_epochs=None):

    if warm_up_epochs is not None:
        plateau_patience = warm_up_epochs

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=plateau_patience,
        min_lr=eta_min
        )
    best_val_loss = float('inf')
    min_val_loss = float('inf')
    best_model_state = None
    best_epoch = None
    epochs_no_improve = 0
    val_loss_window = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if contrastive_loss_fn is not None:
            avg_train_loss, train_stats = train_epoch(
                model,
                optimizer, 
                train_loader, 
                loss_fn,
                contrastive_loss_fn=contrastive_loss_fn,
                return_stats=True,
            )
            avg_val_loss, val_stats = evaluate(
                model,
                val_loader,
                loss_fn,
                contrastive_loss_fn=contrastive_loss_fn,
                return_stats=True,
            )
        else:
            avg_train_loss = train_epoch(
                model,
                optimizer, 
                train_loader, 
                loss_fn,
                contrastive_loss_fn=contrastive_loss_fn
                )
            avg_val_loss = evaluate(
                model,
                val_loader,
                loss_fn,
                contrastive_loss_fn=contrastive_loss_fn,
            )
            train_stats = None
            val_stats = None
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_loss_window.append(avg_val_loss)
        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)
        avg_val_loss_window = (sum(val_loss_window)
            / len(val_loss_window)
            )
        if contrastive_loss_fn is not None:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss_total: {avg_train_loss:.4f} - "
                f"Val_global: {val_stats['contrastive_global']:.4f} - "
                f"Val_local: {val_stats['contrastive_local']:.4f} - "
                f"Val_supervised: {val_stats['supervised_loss']:.4f} - "
                f"Val_total: {avg_val_loss:.4f} - "
                f"Win_total: {avg_val_loss_window:.4f} - "
                f"LR: {_current_lr(optimizer):.2e}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_train_loss:.4f} - "
                f"Val: {avg_val_loss:.4f} - "
                f"Win: {avg_val_loss_window:.4f} - "
                f"LR: {_current_lr(optimizer):.2e}"
            )
        if (avg_val_loss_window
            < best_val_loss - delta):

            best_val_loss = avg_val_loss_window
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(
                model.state_dict()
                )
            best_epoch = epoch + 1
        if best_model:
            save_embeddings(
                model,
                train_loader,
                epoch,
                "../output/embeddings"
                )
        if (enable_pruning
            and val_losses
            and val_losses[0] < 0.70):
            print(f"Pruned: low initial loss: "
                f"{val_losses[0]:.4f}"
                )
            raise optuna.exceptions.TrialPruned()
        scheduler.step(avg_val_loss)

    if best_model_state is not None:
        print(f"Restoring best model from epoch "
            f"{best_epoch} with val_total "
            f"{min_val_loss:.4f}"
            )
        model.load_state_dict(best_model_state)

    return (
        best_val_loss,
        min_val_loss,
        train_losses,
        val_losses
        )
