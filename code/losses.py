import torch
import torch.nn as nn

def calc_loss(batch_dict, cfg):
    total_loss = torch.tensor(0.0, device=cfg.DEVICE)
    if cfg.MSE_WEIGHT > 0:
        for output in batch_dict['outputs']:
            loss = nn.MSELoss()(output, batch_dict['im_orig'])
            total_loss += cfg.MSE_WEIGHT * loss

    return total_loss


