import json
import os

import pandas as pd
import torch

try:
    import optuna
except ImportError:
    optuna = None

try:
    from loss import attach_contrastive_heads
    from model import network
except ImportError:
    from train.loss import attach_contrastive_heads
    from components.model import network


def load_data(file_path):
    df = pd.read_csv(
        file_path,
        delimiter=",",
        low_memory=False,
    )
    smiles = df.iloc[:, 1].values
    targets = df.iloc[:, 2:].values
    return smiles, targets


def initialize_optuna():
    if optuna is None:
        raise ImportError(
            "optuna is required only for optimization workflows."
        )
    output_dir = "../output/optimization/"
    db_file = os.path.join(output_dir, "optuna_study.db")
    storage_name = f"sqlite:///{db_file}"
    os.makedirs(output_dir, exist_ok=True)
    study_name = "optimization_study"

    try:
        existing_studies = optuna.study.get_all_study_summaries(
            storage=storage_name
        )
        study_names = [summary.study_name for summary in existing_studies]
        if study_name in study_names:
            print(f"Study '{study_name}' found in the database")
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_name,
            )
        else:
            print("Creating a new study")
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=storage_name,
            )
    except Exception as exc:
        print(f"Error occurred while accessing the study: {exc}")
        raise

    return study


def load_params(hyperparams_path):
    with open(hyperparams_path, "r") as file:
        hyperparameters = json.load(file)
    return hyperparameters


def load_model(
    model_path,
    architecture_type="mpnn",
    params=None,
    node_dim=None,
    edge_dim=None,
    num_tasks=None,
    mc_class_counts=None,
    task_type=None,
    mc_label_values=None,
    contrastive_config=None,
):
    if (
        isinstance(architecture_type, dict)
        and params is not None
        and node_dim is not None
        and edge_dim is not None
        and num_tasks is None
    ):
        num_tasks = edge_dim
        edge_dim = node_dim
        node_dim = params
        params = architecture_type
        architecture_type = "mpnn"

    if params is None or node_dim is None or edge_dim is None or num_tasks is None:
        raise TypeError(
            "load_model requires params, node_dim, edge_dim and num_tasks."
        )

    if architecture_type != "mpnn":
        raise ValueError(
            "This repository now supports only architecture_type='mpnn'."
        )

    if mc_class_counts is None and mc_label_values is not None:
        mc_class_counts = torch.zeros(num_tasks, dtype=torch.long)
        for i, vals in enumerate(mc_label_values):
            if vals is None:
                continue
            mc_class_counts[i] = len(vals)

    agg_hidden_dims = [
        params[f"agg_hidden_dim_{i+1}"]
        for i in range(params["num_agg_layers"])
    ]
    lin_hidden_dims = [
        params[f"lin_hidden_dim_{i+1}"]
        for i in range(params["num_lin_layers"])
    ]
    edge_heads = params.get("heads_edge", 4)
    node_heads = params.get("heads_node", 4)
    if contrastive_config is None:
        contrastive_config = params.get("contrastive_config", None)

    model = network(
        node_dim=node_dim,
        edge_dim=edge_dim,
        agg_hidden_dims=agg_hidden_dims,
        num_agg_layers=params["num_agg_layers"],
        lin_hidden_dims=lin_hidden_dims,
        num_lin_layers=params["num_lin_layers"],
        activation=params["activation"],
        dropout_rate=params["dropout_rate"],
        num_tasks=num_tasks,
        mc_class_counts=mc_class_counts,
        edge_heads=edge_heads,
        node_heads=node_heads,
    )

    if contrastive_config and contrastive_config.get("enabled", True):
        projection_dim = params.get(
            "projection_dim",
            contrastive_config.get("projection_dim", 128),
        )
        attach_contrastive_heads(
            model,
            projection_dim=projection_dim,
            global_hidden_dim=projection_dim,
            local_hidden_dim=projection_dim,
        )

    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(
            model_path,
            weights_only=True,
            **load_kwargs,
        )
    except TypeError:
        checkpoint = torch.load(model_path, **load_kwargs)

    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint),
        )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported checkpoint format. Expected a state_dict."
        )
    if checkpoint and all(
        str(key).startswith("module.") for key in checkpoint.keys()
    ):
        checkpoint = {
            str(key)[7:]: value for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint, strict=False)

    model.task_type = task_type
    model.mc_label_values = mc_label_values
    model.architecture_type = architecture_type
    model.eval()
    return model


def load_embeddings(directory_path, epoch):
    filename = f"embeddings_epoch_{epoch+1}.pt"
    filepath = os.path.join(directory_path, filename)
    if os.path.exists(filepath):
        data = torch.load(filepath)
        embeddings = data["embeddings"]
        labels = data.get("labels", None)
        is_cls = data.get("is_cls", None)
        return embeddings, labels, is_cls

    raise FileNotFoundError(
        f"No file found for epoch {epoch+1} in {directory_path}"
    )
