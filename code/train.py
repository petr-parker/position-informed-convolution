from tqdm import tqdm
import wandb
import torch.optim as optim
import torch
import losses as l
import metrics as m

def init_wandb(cfg, loss_functions_list, metrics_list):
    wandb.login(key="2e908bb2e1ece8e3954dabad3b089323fb77956a")

    wandb.init(
        project="PIC",
        name=cfg.WANDB_NAME,
        config= {
            "learning_rate": cfg.OPTIMIZER_LR,
            'blocks_num' : cfg.PIC_NUMBER,
        }
    )

    wandb.define_metric(f"Train/Total Loss", step_metric="train step")

    for number in range(cfg.PIC_NUMBER):
        for loss_function in loss_functions_list:
            wandb.define_metric(f"Train/{loss_function.name} Block {number}", step_metric="train step")

    for number in range(cfg.PIC_NUMBER):
        for metric in metrics_list:
            wandb.define_metric(f"Val/{metric.name} Block {number}", step_metric="val step")


def train_one_epoch(net, optimizer, dataloader, loss_functions_list, metrics_list, e):
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
        optimizer.step()

    print(f'Epoch [{e + 1}/{net.cfg.EPOCHS}] training finshed, loss = ', loss_dict['Total Loss'].item())

def val_one_epoch(net, dataloader, loss_functions_list, metrics_list, e):
    net.eval()
    print(f'Validating epoch [{e + 1}/{net.cfg.EPOCHS}] ...')
    for i, batch_data in enumerate(tqdm(dataloader)):
        step = e * len(dataloader) + i
        batch_data['im_noisy'] = batch_data['im_noisy'].to(torch.float32).to(net.cfg.DEVICE)
        batch_data['im_orig'] = batch_data['im_orig'].to(torch.float32).to(net.cfg.DEVICE)

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


def train_val_loop(net, dataloader):
    optimizer = optim.Adam(net.parameters(), lr=net.cfg.OPTIMIZER_LR)
    loss_functions_list = l.create_loss_functions_list(net.cfg)
    metrics_list = m.create_metrcs_list(net.cfg)

    init_wandb(net.cfg, loss_functions_list, metrics_list)

    for e in range(net.cfg.EPOCHS):
        train_one_epoch(net, optimizer, dataloader, loss_functions_list, metrics_list, e)
        val_one_epoch(net, dataloader, loss_functions_list, metrics_list, e)
    
    wandb.finish()

