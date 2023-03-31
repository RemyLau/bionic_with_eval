import torch
import torch.nn as nn
from torch import Tensor

from bionic.utils.common import Device


def masked_scaled_mse(
    output: Tensor,
    target: Tensor,
    weight: Tensor,
    node_ids: Tensor,
    mask: Tensor,
    lambda_: float,
    device=None,
) -> Tensor:
    """Masked and scaled MSE loss.
    """

    if device is None:
        device = Device()

    # Subset `target` to current batch and make dense
    target = target.to(device)
    target = target.adj_t[node_ids, node_ids].to_dense()

    loss = lambda_ * weight * torch.mean(mask.reshape((-1, 1)) * (output - target) ** 2 * mask)

    return loss


cls_criterion = nn.BCEWithLogitsLoss(reduction="none")


def classification_loss(output: Tensor, target: Tensor, mask: Tensor, lambda_: float) -> Tensor:
    """Multi-label classification loss used when labels are provided.
    """

    loss = (1 - lambda_) * (mask.reshape((-1, 1)) * cls_criterion(output, target)).mean()
    return loss
