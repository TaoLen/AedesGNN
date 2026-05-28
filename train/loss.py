import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from fingerprints import calc_fp
from utils import task_inference


def _build_projection_head(
    input_dim,
    output_dim=None,
    hidden_dim=None):

    if output_dim is None:
        output_dim = input_dim
    if hidden_dim is None:
        hidden_dim = input_dim

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def attach_contrastive_heads(
    model,
    projection_dim=None,
    global_hidden_dim=None,
    local_hidden_dim=None):

    if not hasattr(model, "embedding_dim"):
        raise ValueError(
            "Model must expose embedding_dim to attach contrastive heads."
        )
    if not hasattr(model, "node_rep_dim"):
        raise ValueError(
            "Model must expose node_rep_dim to attach contrastive heads."
        )

    model.global_projection = _build_projection_head(
        input_dim=model.embedding_dim,
        output_dim=projection_dim,
        hidden_dim=global_hidden_dim,
    )
    model.local_projection = _build_projection_head(
        input_dim=model.node_rep_dim,
        output_dim=projection_dim,
        hidden_dim=local_hidden_dim,
    )
    model.contrastive_projection_dim = (
        model.embedding_dim
        if projection_dim is None
        else projection_dim
    )
    return model


class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        use_cosine_similarity=True):

        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity
        )
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        return torch.tensordot(
            x.unsqueeze(1),
            y.T.unsqueeze(0),
            dims=2,
        )

    def _cosine_simililarity(self, x, y):
        return self._cosine_similarity(
            x.unsqueeze(1),
            y.unsqueeze(0),
        )

    def _similarity(self, x, y):
        if self.use_cosine_similarity:
            return self._cosine_simililarity(x, y)
        return self._dot_simililarity(x, y)

    @staticmethod
    def _get_correlated_mask(batch_size, device):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(
            2 * batch_size,
            2 * batch_size,
            k=-batch_size,
        )
        l2 = np.eye(
            2 * batch_size,
            2 * batch_size,
            k=batch_size,
        )
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask.to(device)

    def forward(self, z1, z2):
        if z1.size(0) != z2.size(0):
            raise ValueError("NT-Xent inputs must share the same batch size.")
        batch_size = z1.size(0)
        if batch_size == 0:
            return z1.new_tensor(0.0)

        representations = torch.cat([z2, z1], dim=0)
        similarity_matrix = self.similarity_function(
            representations,
            representations,
        )

        positives = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ],
            dim=0,
        ).view(2 * batch_size, 1)

        neg_mask = self._get_correlated_mask(batch_size, z1.device)
        negatives = similarity_matrix[neg_mask].view(
            2 * batch_size,
            -1,
        )

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(
            2 * batch_size,
            device=z1.device,
            dtype=torch.long,
        )
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


class WeightedNTXentLoss(NTXentLoss):
    def __init__(
        self,
        temperature=0.1,
        use_cosine_similarity=True,
        lambda_1=0.5,
        fp_radius=2,
        fp_size=2048):

        super().__init__(
            temperature=temperature,
            use_cosine_similarity=use_cosine_similarity,
        )
        self.lambda_1 = lambda_1
        self.fp_radius = fp_radius
        self.fp_size = fp_size

    def _negative_weights(self, smiles, device):
        batch_size = len(smiles)
        fp_score = np.zeros((batch_size, max(batch_size - 1, 0)))
        fps = [
            calc_fp(
                smi,
                fp_size=self.fp_size,
                radius=self.fp_radius,
            )
            for smi in smiles
        ]
        fp_counts = [fp.sum() for fp in fps]

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                intersection = float(np.dot(fps[i], fps[j]))
                union = float(fp_counts[i] + fp_counts[j] - intersection)
                fp_sim = intersection / union if union > 0 else 0.0
                fp_score[i, j - 1] = fp_sim
                fp_score[j, i] = fp_sim

        fp_score = 1 - self.lambda_1 * torch.tensor(
            fp_score,
            dtype=torch.float32,
            device=device,
        )
        return fp_score.repeat(2, 2)

    def forward(self, z1, z2, smiles):
        if z1.size(0) != z2.size(0):
            raise ValueError(
                "Weighted NT-Xent inputs must share the same batch size."
            )
        batch_size = z1.size(0)
        if len(smiles) != batch_size:
            raise ValueError(
                "Number of SMILES must match the embedding batch size."
            )
        if batch_size == 0:
            return z1.new_tensor(0.0)

        representations = torch.cat([z2, z1], dim=0)
        similarity_matrix = self.similarity_function(
            representations,
            representations,
        )

        positives = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ],
            dim=0,
        ).view(2 * batch_size, 1)

        neg_mask = self._get_correlated_mask(batch_size, z1.device)
        negatives = similarity_matrix[neg_mask].view(
            2 * batch_size,
            -1,
        )

        fp_score = self._negative_weights(smiles, z1.device)
        negatives = negatives * fp_score

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(
            2 * batch_size,
            device=z1.device,
            dtype=torch.long,
        )
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


class ContrastiveAuxiliaryLoss(nn.Module):
    def __init__(
        self,
        beta_aug=0.0,
        alpha_global=0.0,
        alpha_local=0.0,
        global_temperature=0.1,
        local_temperature=0.1,
        tanimoto_lambda=0.5,
        fp_radius=2,
        fp_size=2048,
        use_cosine_similarity=True):

        super().__init__()
        self.beta_aug = beta_aug
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.global_loss = WeightedNTXentLoss(
            temperature=global_temperature,
            use_cosine_similarity=use_cosine_similarity,
            lambda_1=tanimoto_lambda,
            fp_radius=fp_radius,
            fp_size=fp_size,
        )
        self.local_loss = NTXentLoss(
            temperature=local_temperature,
            use_cosine_similarity=use_cosine_similarity,
        )

    @staticmethod
    def _pool_fragments(
        node_embeddings,
        fragment_atom_index,
        fragment_index,
        num_fragments):

        if num_fragments <= 0 or fragment_atom_index.numel() == 0:
            return node_embeddings.new_zeros(
                (0, node_embeddings.size(-1))
            )

        pooled = node_embeddings.new_zeros(
            (num_fragments, node_embeddings.size(-1))
        )
        pooled.index_add_(
            0,
            fragment_index,
            node_embeddings[fragment_atom_index],
        )

        counts = node_embeddings.new_zeros(num_fragments)
        counts.index_add_(
            0,
            fragment_index,
            torch.ones_like(
                fragment_index,
                dtype=node_embeddings.dtype,
            ),
        )
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        return pooled

    def forward(
        self,
        model,
        view_i_repr,
        view_j_repr,
        batch_payload):

        device = view_i_repr["graph_embeddings"].device

        global_i = model.global_projection(
            view_i_repr["graph_embeddings"]
        )
        global_j = model.global_projection(
            view_j_repr["graph_embeddings"]
        )
        global_loss = self.global_loss(
            global_i,
            global_j,
            batch_payload["smiles"],
        )

        fragment_atom_index = batch_payload[
            "fragment_atom_index"
        ].to(device)
        fragment_index = batch_payload[
            "fragment_index"
        ].to(device)
        num_fragments = int(batch_payload["num_fragments"])

        if num_fragments > 0 and fragment_atom_index.numel() > 0:
            local_i = self._pool_fragments(
                view_i_repr["node_embeddings"],
                fragment_atom_index,
                fragment_index,
                num_fragments,
            )
            local_j = self._pool_fragments(
                view_j_repr["node_embeddings"],
                fragment_atom_index,
                fragment_index,
                num_fragments,
            )
            local_i = model.local_projection(local_i)
            local_j = model.local_projection(local_j)
            local_loss = self.local_loss(local_i, local_j)
        else:
            local_loss = global_i.new_tensor(0.0)

        total = (
            self.alpha_global * global_loss
            + self.alpha_local * local_loss
        )
        return {
            "total": total,
            "global": global_loss,
            "local": local_loss,
        }


def build_mc_label_maps(mc_label_values):
    if mc_label_values is None:
        return None
    maps = []
    for vals in mc_label_values:
        if vals is None:
            maps.append(None)
            continue
        maps.append({int(v): i for i, v in enumerate(vals)})
    return maps


def prepare_masked_data(
    y_true, y_pred):

    if isinstance(y_true, (Batch, Data)):
        y_true = y_true.y
    if isinstance(y_pred, (Batch, Data)):
        y_pred = y_pred.y
    if not torch.is_tensor(y_true):
        raise ValueError("y_true must be a tensor")
    if not torch.is_tensor(y_pred):
        raise ValueError("y_pred must be a tensor")

    if y_true.dim() != 2:
        raise ValueError("y_true must be 2D [batch, num_tasks]")

    if y_pred.dim() == 2:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must match shape"
            )
    elif y_pred.dim() == 3:
        if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
            raise ValueError(
                "y_true and y_pred must match in first two dims"
            )
    else:
        raise ValueError("y_pred must be 2D or 3D")

    mask = ~torch.isnan(y_true)

    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)

    return y_true, y_pred, mask


def _split_pred(y_pred):
    if isinstance(y_pred, dict):
        pred_scalar = y_pred.get("scalar")
        pred_logits = y_pred.get("mc_logits", {})
    elif y_pred.dim() == 2:
        pred_scalar = y_pred
        pred_logits = None
    elif y_pred.dim() == 3:
        pred_scalar = y_pred[..., 0]
        pred_logits = y_pred
    else:
        raise ValueError("y_pred must be 2D, 3D, or dict")
    if pred_scalar is None:
        raise ValueError("y_pred must contain scalar logits")
    return pred_scalar, pred_logits


def map_multiclass_targets(targets, mapping, num_classes):
    targets = targets.to(torch.long)
    if mapping is None:
        raise ValueError("Missing label mapping for multiclass task.")

    mapped = []
    for v in targets.tolist():
        if int(v) not in mapping:
            if int(v) < 0 or int(v) >= num_classes:
                raise ValueError(
                    "Unseen multiclass label encountered.")
            mapped.append(int(v))
        else:
            mapped.append(mapping[int(v)])
    return torch.tensor(
        mapped, device=targets.device, dtype=torch.long)


def reduce_task_mean(loss_mat, mask):
    task_counts = mask.sum(dim=0)
    task_sums = (loss_mat * mask).sum(dim=0)
    valid = task_counts > 0
    task_means = torch.zeros_like(task_sums)
    task_means[valid] = task_sums[valid] / task_counts[valid]
    return task_means, valid


def compute_loss_matrix(
    y_pred,
    y_true,
    task_type=None,
    mc_label_values=None,
    mc_label_maps=None):

    pred_scalar, pred_logits = _split_pred(y_pred)
    y_true, y_pred_scalar, mask = prepare_masked_data(
        y_true, pred_scalar)

    if task_type is None:
        task_type = task_inference(
            y_true, mask).to(y_true.device)
    else:
        task_type = task_type.to(y_true.device)

    if mc_label_maps is None:
        mc_label_maps = build_mc_label_maps(mc_label_values)

    loss_cls = F.binary_cross_entropy_with_logits(
        y_pred_scalar, y_true, reduction='none')
    loss_reg = F.mse_loss(
        y_pred_scalar, y_true, reduction='none')

    is_bin = (task_type == 1).to(y_pred_scalar.device)
    is_mc = (task_type == 2).to(y_pred_scalar.device)

    loss = torch.where(is_bin.view(
        1, -1), loss_cls, loss_reg)

    if is_mc.any():
        if pred_logits is None:
            raise ValueError(
                "Multiclass task detected but logits are missing. "
                "Model must output per-task multiclass logits."
            )

        B, T = y_pred_scalar.shape
        loss_mc = torch.zeros(
            (B, T), device=y_pred_scalar.device,
            dtype=y_pred_scalar.dtype)

        for j in torch.where(is_mc)[0].tolist():
            valid = mask[:, j]
            if not valid.any():
                continue

            targets = y_true[valid, j].round().to(torch.long)
            mapping = None
            if mc_label_maps is not None:
                mapping = mc_label_maps[j]
            if mapping is None:
                raise ValueError(
                    "Missing label mapping for multiclass task.")

            if isinstance(pred_logits, dict):
                logits = pred_logits.get(j)
            else:
                logits = pred_logits[:, j, :]
            if logits is None:
                raise ValueError(
                    "Missing logits for multiclass task.")
            if logits.size(1) != len(mapping):
                raise ValueError(
                    "Multiclass logits size does not match label mapping.")
            mapped = map_multiclass_targets(
                targets, mapping, logits.size(1))
            l = F.cross_entropy(logits[valid], mapped, reduction='none')

            tmp = torch.zeros(
                B, device=y_pred_scalar.device,
                dtype=y_pred_scalar.dtype)
            tmp[valid] = l
            loss_mc[:, j] = tmp

        loss = torch.where(is_mc.view(1, -1), loss_mc, loss)

    return loss, mask, task_type


class SupervisedUncertainty(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def reset_parameters(self):
        with torch.no_grad():
            self.log_vars.zero_()

    def compute_task_loss(
        self, pred, true, mask=None, task_type=None, mc_label_maps=None):

        if mask is None:
            mask = ~torch.isnan(true)
        mask = mask.to(device=true.device, dtype=torch.bool)
        true = torch.nan_to_num(true, nan=0.0)

        if task_type is None:
            task_type = task_inference(true, mask).to(true.device)
        else:
            task_type = task_type.to(true.device)

        pred_logits = None
        if isinstance(pred, dict):
            pred_scalar = pred.get("scalar")
            pred_logits = pred.get("mc_logits", {})
        elif pred.dim() == 2:
            pred_scalar = pred
        elif pred.dim() == 3:
            pred_scalar = pred[..., 0]
            pred_logits = pred
        else:
            raise ValueError("y_pred must be 2D or 3D")

        loss_cls = F.binary_cross_entropy_with_logits(
            pred_scalar, true, reduction='none')
        loss_reg = F.mse_loss(
            pred_scalar, true, reduction='none')

        is_bin = (task_type == 1)
        is_mc = (task_type == 2)

        loss = torch.where(is_bin.view(1, -1),
                loss_cls, loss_reg
                )

        if is_mc.any():
            if pred_logits is None:
                raise ValueError(
                    "Multiclass task detected but logits are missing. "
                    "Model must output per-task multiclass logits."
                )

            B, T = pred_scalar.shape
            loss_mc = torch.zeros(
                (B, T), device=pred_scalar.device,
                dtype=pred_scalar.dtype)

            for j in torch.where(is_mc)[0].tolist():
                valid = mask[:, j]
                if not valid.any():
                    continue

                targets = true[valid, j].round().to(torch.long)
                mapping = None
                if mc_label_maps is not None:
                    mapping = mc_label_maps[j]
                if mapping is None:
                    raise ValueError(
                        "Missing label mapping for multiclass task.")

                if isinstance(pred_logits, dict):
                    logits = pred_logits.get(j)
                else:
                    logits = pred_logits[:, j, :]
                if logits is None:
                    raise ValueError(
                        "Missing logits for multiclass task.")
                if logits.size(1) != len(mapping):
                    raise ValueError(
                        "Multiclass logits size does not match label mapping.")
                mapped = map_multiclass_targets(
                    targets, mapping, logits.size(1))
                l = F.cross_entropy(logits[valid], mapped, reduction='none')

                tmp = torch.zeros(
                    B, device=pred_scalar.device,
                    dtype=pred_scalar.dtype)
                tmp[valid] = l
                loss_mc[:, j] = tmp

            loss = torch.where(is_mc.view(1, -1),
                    loss_mc, loss
                    )

        return loss  

    def forward(
        self, y_pred, y_true, mask,
        task_type=None, mc_label_maps=None):
        losses = self.compute_task_loss(
            y_pred, y_true,
            mask=mask,
            task_type=task_type,
            mc_label_maps=mc_label_maps
            )

        if task_type is None:
            task_type = task_inference(torch.nan_to_num(
                y_true, nan=0.0), mask).to(losses.device)
        else:
            task_type = task_type.to(losses.device)

        is_cls = (task_type != 0)
        alphas = torch.clamp(
            self.log_vars, min=-6.0, max=6.0
            ).view(1, -1).to(losses.device)

        w_cls = torch.exp(-alphas) * losses + 0.5 * alphas
        w_reg = 0.5 * torch.exp(-alphas) * losses + 0.5 * alphas
        weighted = torch.where(is_cls.view(1, -1), w_cls, w_reg)

        task_means, valid = reduce_task_mean(
            weighted, mask.float())
        if not valid.any():
            return torch.tensor(
                0.0, device=losses.device, requires_grad=True)
        return task_means[valid].mean()

    
def MaskedLoss(
    num_tasks=None, task_type=None, mc_label_values=None):
    if num_tasks is None:
        raise ValueError("num_tasks must be provided")

    mc_label_maps = build_mc_label_maps(mc_label_values)

    def masked_loss_function(y_pred, y_true):
        if isinstance(y_pred, dict):
            y_pred_scalar = y_pred.get("scalar")
            y_pred_logits = y_pred.get("mc_logits", {})
        else:
            y_pred_scalar = y_pred
            y_pred_logits = None
        y_true, y_pred, mask = prepare_masked_data(
            y_true, y_pred_scalar)

        tt = task_type
        if tt is None:
            tt = task_inference(
                y_true, mask).to(y_true.device)
        else:
            tt = tt.to(y_true.device)

        if y_pred_scalar is None:
            raise ValueError("y_pred must contain scalar logits")

        loss_cls = F.binary_cross_entropy_with_logits(
            y_pred_scalar, y_true, reduction='none')
        loss_reg = F.mse_loss(
            y_pred_scalar, y_true, reduction='none')

        is_bin = (tt == 1).to(y_pred_scalar.device)
        is_mc = (tt == 2).to(y_pred_scalar.device)

        loss = torch.where(is_bin.view(
            1, -1), loss_cls, loss_reg)

        if is_mc.any():
            if y_pred_logits is None:
                raise ValueError(
                    "Multiclass task detected but logits are missing. "
                    "Model must output per-task multiclass logits."
                )

            B, T = y_pred_scalar.shape
            loss_mc = torch.zeros(
                (B, T), device=y_pred_scalar.device,
                dtype=y_pred_scalar.dtype)

            for j in torch.where(is_mc)[0].tolist():
                valid = mask[:, j]
                if not valid.any():
                    continue

                targets = y_true[valid, j].round().to(torch.long)
                mapping = None
                if mc_label_maps is not None:
                    mapping = mc_label_maps[j]
                if mapping is None:
                    raise ValueError(
                        "Missing label mapping for multiclass task.")

                if isinstance(y_pred_logits, dict):
                    logits = y_pred_logits.get(j)
                else:
                    logits = y_pred_logits[:, j, :]
                if logits is None:
                    raise ValueError(
                        "Missing logits for multiclass task.")
                if logits.size(1) != len(mapping):
                    raise ValueError(
                        "Multiclass logits size does not match label mapping.")
                mapped = map_multiclass_targets(
                    targets, mapping, logits.size(1))
                l = F.cross_entropy(logits[valid], mapped, reduction='none')

                tmp = torch.zeros(
                    B, device=y_pred_scalar.device,
                    dtype=y_pred_scalar.dtype)
                tmp[valid] = l
                loss_mc[:, j] = tmp

            loss = torch.where(is_mc.view(1, -1), loss_mc, loss)
        
        task_means, valid = reduce_task_mean(
            loss, mask.float())
        if not valid.any():
            return torch.tensor(
                0.0, device=loss.device, requires_grad=True)
        return task_means[valid].mean()

    return masked_loss_function
