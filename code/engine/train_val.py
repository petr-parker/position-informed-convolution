from tqdm import tqdm
import wandb
import torch.optim as optim
import torch
import engine.losses as l
import engine.metrics as m

def init_wandb(cfg, loss_functions_list, metrics_list, net):
    wandb.login(key="2e908bb2e1ece8e3954dabad3b089323fb77956a")

    wandb.init(
        project="PIC",
        name=cfg.WANDB_NAME,
        config= {
            "learning_rate": cfg.OPTIMIZER_LR,
            'blocks_num' : cfg.PIC_NUMBER,
        }
    )

    if cfg.LOG_PARAMETERS:
        wandb.watch(net, log="all", log_freq=1)
        for name, _ in net.named_parameters():
            wandb.define_metric(f"grads/{name} grad_norm", step_metric="train step")

    wandb.define_metric(f"Train/Total Loss", step_metric="train step")

    wandb.define_metric(f"Train/learning rate", step_metric="epoch")

    for number in range(cfg.PIC_NUMBER):
        for loss_function in loss_functions_list:
            wandb.define_metric(f"Train/{loss_function.name} Block {number}", step_metric="train step")

    for number in range(cfg.PIC_NUMBER):
        for metric in metrics_list:
            wandb.define_metric(f"Val/{metric.name} Block {number}", step_metric="val step")

def finish_wandb(net):
    wandb.unwatch()
    wandb.finish()

def train_one_epoch(net, optimizer, scheduler, dataloader, loss_functions_list, metrics_list, e):
    net.train()
    print(f'Training epoch [{e + 1}/{net.cfg.EPOCHS}] ...')
    for i, batch_data in enumerate(tqdm(dataloader)):
        step = e * len(dataloader) + i

        batch_data['im_noisy'] = batch_data['im_noisy'].to(torch.float32).to(net.cfg.DEVICE)
        batch_data['im_orig'] = batch_data['im_orig'].to(torch.float32).to(net.cfg.DEVICE)

        optimizer.zero_grad()
        results = net(batch_data)

        loss_dict = l.get_loss_dict(loss_functions_list, results)

        for key, value in loss_dict.items():
            wandb.log(
                {
                    f'Train/{key}' : value.item(),
                    'train step' : step
                }
            )

        loss_dict['Total Loss'].backward()

        if net.cfg.LOG_PARAMETERS:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    norm = param.grad.detach().norm(2).item()
                    wandb.log(
                        {
                            f'grads/{name} grad_norm' : norm,
                            'train step' : step
                        }
                    )

        optimizer.step()
    
    if net.cfg.SCHEDULER_NAME != 'none':
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            f'Train/learning rate' : current_lr,
            'epoch' : e
        })

    print(f'Epoch [{e + 1}/{net.cfg.EPOCHS}] training finshed, loss = ', loss_dict['Total Loss'].item(), '.')

def val_one_epoch(net, dataloader, loss_functions_list, metrics_list, e):
    net.eval()
    print(f'Validating epoch [{e + 1}/{net.cfg.EPOCHS}] ...')
    for i, batch_data in enumerate(tqdm(dataloader)):
        step = e * len(dataloader) + i
        batch_data['im_noisy'] = batch_data['im_noisy'].to(torch.float32).to(net.cfg.DEVICE)
        batch_data['im_orig'] = batch_data['im_orig'].to(torch.float32).to(net.cfg.DEVICE)

        with torch.no_grad():
            results = net(batch_data)

        metrics_dict = m.get_metrics_dict(metrics_list, results)

        for key, value in metrics_dict.items():
            wandb.log(
                {
                    f'Val/{key}' : value.item(),
                    'val step' : step
                }
            )
    print(f'Epoch [{e + 1}/{net.cfg.EPOCHS}] validation finshed.')


def train_val_loop(net, train_dataloader, val_dataloader):
    optimizers = {
        'Adam' : optim.Adam,
        'SGD' : optim.SGD
    }
    optimizer = optimizers[net.cfg.OPTIMZER_NAME](net.parameters(), lr=net.cfg.OPTIMIZER_LR)
    schedulers = {
        'cos' : torch.optim.lr_scheduler.CosineAnnealingLR
    }
    if not net.cfg.SCHEDULER_NAME == 'none':
        scheduler = schedulers[net.cfg.SCHEDULER_NAME](optimizer, T_max=net.cfg.EPOCHS, eta_min=1e-5)
    else:
        scheduler = None
    loss_functions_list = l.create_loss_functions_list(net.cfg)
    metrics_list = m.create_metrcs_list(net.cfg)

    init_wandb(net.cfg, loss_functions_list, metrics_list, net)

    for e in range(net.cfg.EPOCHS):
        train_one_epoch(net, optimizer, scheduler, train_dataloader, loss_functions_list, metrics_list, e)
        val_one_epoch(net, val_dataloader, loss_functions_list, metrics_list, e)

    finish_wandb(net)

