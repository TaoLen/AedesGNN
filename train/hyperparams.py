import optuna
import torch.optim as optim

from model import network


def configure_optimizer(trial, model, lr):
    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["RAdam", "Adam", "RMSprop", "SGD"],
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-8,
        1e-4,
    )
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def configure_mpnn(
    trial,
    node_dim,
    edge_dim,
    num_tasks,
    mc_class_counts=None,
):
    heads_node = trial.suggest_categorical(
        "heads_node",
        [1, 2, 3, 4, 6],
    )
    heads_edge = trial.suggest_categorical(
        "heads_edge",
        [1, 2, 3, 4, 6],
    )
    num_agg_layers = trial.suggest_int('num_agg_layers', 2, 6)
    agg_hidden_dims = [
        trial.suggest_categorical(f'agg_hidden_dim_{i+1}', 
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 
         144, 180, 192, 240, 300, 324, 348, 372, 
         396, 420, 444, 456, 480, 516]) 
        for i in range(num_agg_layers)
        ]
    num_lin_layers = trial.suggest_int('num_lin_layers', 2, 4)
    lin_hidden_dims = [
        trial.suggest_int(f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(num_lin_layers)
        ]
    if any(dim % heads_edge != 0 for dim in agg_hidden_dims):
        raise optuna.exceptions.TrialPruned(
            "Message-passing hidden dimensions must be divisible by "
            f"heads_edge={heads_edge}."
        )
    if sum(agg_hidden_dims) % heads_node != 0:
        raise optuna.exceptions.TrialPruned(
            "sum(agg_hidden_dims) must be divisible by "
            f"heads_node={heads_node}."
        )
    projection_dim = trial.suggest_categorical(
        "projection_dim",
        [32, 64, 128, 256],
    )
    activation_choice = trial.suggest_categorical(
        "activation",
        ["relu", "leakyrelu", "elu", "gelu", "selu"],
    )
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)

    model = network(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation_choice,
        dropout_rate,
        num_tasks,
        mc_class_counts=mc_class_counts,
        edge_heads=heads_edge,
        node_heads=heads_node,
    )
    return model, projection_dim