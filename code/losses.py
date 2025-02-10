import torch
import torch.nn as nn
import wandb

class MSE_loss:
    def __init__(self, cfg):
        self.name = 'MSE'
        self.cfg = cfg
        self.loss = nn.MSELoss()
        self.weight = cfg.MSE_WEIGHT

    def __call__(self, logits, labels):
        return self.loss(logits, labels)

def create_loss_functions_list(cfg):
    loss_functions_list = []
    if cfg.MSE_WEIGHT > 0:
        loss_functions_list.append(MSE_loss(cfg))
    return loss_functions_list

def get_loss_dict(loss_functions_list, batch_dict):
    loss_dict = {
        'Total Loss' : torch.tensor(0.0, device=loss_functions_list[0].cfg.DEVICE),
    }
    for number, output in enumerate(batch_dict['outputs']):
        for loss_function in loss_functions_list:
            loss = loss_function(output, batch_dict['im_orig'])
            loss_dict[f'{loss_function.name} Block {number}'] = loss
            loss_dict['Total Loss'] += loss
            
    return loss_dict


