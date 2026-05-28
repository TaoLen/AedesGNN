import torch.optim as optim

from utils import device, set_seed
from model import network

from hyperparams import (
    configure_optimizer,
    configure_mpnn,
)
from resets import mpnn_resets
from trainer import (
    get_loss,
    get_contrastive_loss,
    train_model,
)
from loaders import infer_task_metadata
from loss import attach_contrastive_heads


def _validate_architecture(architecture_type):
    if architecture_type != "mpnn":
        raise ValueError(
            "This repository now supports only architecture_type='mpnn'."
        )


contrastive_config = {
    "enabled": True,
    "beta_aug": 0.25,
    "alpha_global": 0.1,
    "alpha_local": 0.1,
    "global_temperature": 0.1,
    "local_temperature": 0.1,
    "tanimoto_lambda": 0.5,
    "projection_dim": 128,
    "global_hidden_dim": 128,
    "local_hidden_dim": 128,
    "fp_radius": 2,
    "fp_size": 2048,
}


def objective(
    trial,
    node_dim,
    edge_dim,
    train_loader,
    val_loader,
    num_tasks=None,
    architecture_type="mpnn",
    use_uncertainty=False,
    lr=None,
    task_type=None,
    mc_class_counts=None,
    mc_label_values=None,
    contrastive_config=None,
):
    _validate_architecture(architecture_type)
    set_seed(42)

    if task_type is None or mc_class_counts is None or mc_label_values is None:
        inferred_type, inferred_counts, inferred_labels = (
            infer_task_metadata(train_loader)
        )
        if task_type is None:
            task_type = inferred_type
        if mc_class_counts is None:
            mc_class_counts = inferred_counts
        if mc_label_values is None:
            mc_label_values = inferred_labels

    model, projection_dim = configure_mpnn(
        trial,
        node_dim,
        edge_dim,
        num_tasks,
        mc_class_counts=mc_class_counts,
    )

    if contrastive_config and contrastive_config.get("enabled", True):
        attach_contrastive_heads(
            model,
            projection_dim=projection_dim,
            global_hidden_dim=projection_dim,
            local_hidden_dim=projection_dim,
        )
    model.to(device)
    model.task_type = task_type
    model.mc_label_values = mc_label_values

    mpnn_resets(model)

    loss_fn = get_loss(
        model,
        num_tasks,
        use_uncertainty,
        task_type=task_type,
        mc_label_values=mc_label_values,
    )
    contrastive_loss_fn = get_contrastive_loss(contrastive_config)
    optimizer = configure_optimizer(trial, model, lr=lr)
    _, min_val_loss, _, _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        num_epochs=2000,
        patience=10,
        delta=0.01,
        window_size=5,
        best_model=False,
        enable_pruning=True,
        contrastive_loss_fn=contrastive_loss_fn,
    )
    return min_val_loss


def retrain(
    best_params,
    node_dim,
    edge_dim,
    train_loader,
    val_loader,
    num_tasks,
    architecture_type="mpnn",
    use_uncertainty=False,
    lr=None,
    task_type=None,
    mc_class_counts=None,
    mc_label_values=None,
    contrastive_config=None,
):
    _validate_architecture(architecture_type)
    set_seed(42)

    agg_hidden_dims = [
        best_params[f"agg_hidden_dim_{i+1}"]
        for i in range(best_params["num_agg_layers"])
    ]
    lin_hidden_dims = [
        best_params[f"lin_hidden_dim_{i+1}"]
        for i in range(best_params["num_lin_layers"])
    ]

    if task_type is None or mc_class_counts is None or mc_label_values is None:
        inferred_type, inferred_counts, inferred_labels = (
            infer_task_metadata(train_loader)
        )
        if task_type is None:
            task_type = inferred_type
        if mc_class_counts is None:
            mc_class_counts = inferred_counts
        if mc_label_values is None:
            mc_label_values = inferred_labels

    edge_heads = best_params.get("heads_edge", 4)
    node_heads = best_params.get("heads_node", 4)

    model = network(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        best_params["num_agg_layers"],
        lin_hidden_dims,
        best_params["num_lin_layers"],
        best_params["activation"],
        best_params["dropout_rate"],
        num_tasks,
        mc_class_counts=mc_class_counts,
        edge_heads=edge_heads,
        node_heads=node_heads,
    )

    if contrastive_config and contrastive_config.get("enabled", True):
        projection_dim = best_params.get(
            "projection_dim",
            contrastive_config.get("projection_dim", 128),
        )
        attach_contrastive_heads(
            model,
            projection_dim=projection_dim,
            global_hidden_dim=projection_dim,
            local_hidden_dim=projection_dim,
        )
    model.to(device)
    model.task_type = task_type
    model.mc_label_values = mc_label_values

    mpnn_resets(model)

    optimizer = getattr(optim, best_params["optimizer"])(
        model.parameters(),
        lr=lr,
        weight_decay=best_params["weight_decay"],
    )
    loss_fn = get_loss(
        model,
        num_tasks,
        use_uncertainty,
        task_type=task_type,
        mc_label_values=mc_label_values,
    )
    contrastive_loss_fn = get_contrastive_loss(contrastive_config)
    best_val_loss, min_val_loss, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        num_epochs=2000,
        patience=10,
        delta=0.01,
        window_size=5,
        best_model=True,
        enable_pruning=False,
        contrastive_loss_fn=contrastive_loss_fn,
    )

    return model, best_val_loss, min_val_loss, train_losses, val_losses
