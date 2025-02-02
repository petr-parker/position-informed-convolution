from tqdm import tqdm
from losses import calc_loss
import wandb
import torch.optim as optim
import torch

def init_wandb(cfg):
    wandb.login(key="2e908bb2e1ece8e3954dabad3b089323fb77956a")

    wandb.init(
        project="PIC",
        name=cfg.WANDB_NAME,
        config= {
            "learning_rate": cfg.OPTIMIZER_LR,
            'blocks_num' : cfg.PIC_NUMBER,
        }
    )


def train_one_epoch(net, optimizer, dataloader):
    for batch_data in tqdm(dataloader):

        batch_data['im_noisy'] = batch_data['im_noisy'].to(torch.float32).to(net.cfg.DEVICE)
        batch_data['im_orig'] = batch_data['im_orig'].to(torch.float32).to(net.cfg.DEVICE)


        optimizer.zero_grad()
        results = net(batch_data)

        loss = calc_loss(results, net.cfg)
        loss.backward()
        optimizer.step()

        wandb.log({
            "train_loss": loss.item(),
        })
    print(f'Loss = {loss.item()}')


def train(net, dataloader):

    net.train()
    init_wandb(net.cfg)
    optimizer = optim.Adam(net.parameters(), lr=net.cfg.OPTIMIZER_LR)

    for e in range(net.cfg.EPOCHS):
        print(f'Training epoch [{e + 1}/{net.cfg.EPOCHS}] ...')
        train_one_epoch(net, optimizer, dataloader)




