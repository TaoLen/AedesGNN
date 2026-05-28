from initialization import mpnn_weights
from utils import set_seed


def _reset_module_tree(module):
    if module is None:
        return
    for submodule in module.modules():
        if submodule is module:
            continue
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()


def _reset_projection_heads(model):
    for attr in ("global_projection", "local_projection"):
        head = getattr(model, attr, None)
        _reset_module_tree(head)


def _reset_virtual_nodes(model):
    for emb in getattr(model, "virtualnode_embedding", []):
        emb.data.zero_()
    for mlp in getattr(model, "virtualnode_mlp", []):
        mpnn_weights(mlp)


def _reset_uncertainty(model):
    uncertainty = getattr(model, "uncertainty", None)
    if uncertainty is not None and hasattr(
        uncertainty,
        "reset_parameters",
    ):
        uncertainty.reset_parameters()


def mpnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        mpnn_weights(layer)
    _reset_virtual_nodes(model)
    mpnn_weights(model.node_readout)
    for lin_seq in model.lin_layers:
        mpnn_weights(lin_seq)
    mpnn_weights(model.embedding_layer)
    mpnn_weights(model.output_layer)
    for head in model.mc_heads.values():
        mpnn_weights(head)
    _reset_uncertainty(model)
    _reset_projection_heads(model)
