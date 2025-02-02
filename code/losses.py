import torch
import torch.nn as nn

def calc_loss(batch_dict, cfg):
    total_loss = torch.tensor(0.0, device=cfg.DEVICE)
    if cfg.MSE_WEIGHT > 0:
        total_loss += nn.MSELoss()(batch_dict['im_pred'], batch_dict['im_orig'])

    return total_loss


