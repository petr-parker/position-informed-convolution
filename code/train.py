from tqdm import tqdm
from losses import calc_loss
import wandb
import torch.optim as optim
import torch
from metrics import calc_metrics
from utils import reconstruct_im

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


def train_one_epoch(net, optimizer, dataloader, e):
    for i, batch_data in enumerate(tqdm(dataloader)):
        step = e * len(dataloader) + i

        batch_data['im_noisy'] = batch_data['im_noisy'].to(torch.float32).to(net.cfg.DEVICE)
        batch_data['im_orig'] = batch_data['im_orig'].to(torch.float32).to(net.cfg.DEVICE)

        optimizer.zero_grad()
        results = net(batch_data)

        total_loss = torch.tensor(0.0, device=net.cfg.DEVICE)
        if net.cfg.MSE_WEIGHT > 0:
            for number, output in enumerate(results['outputs']):
                loss = torch.nn.MSELoss()(output, results['im_orig'])
                wandb.log(
                    {
                        f"Train MSE Loss; Step {number + 1}": loss.item(),
                    },
                    step=step,
                )
                total_loss += net.cfg.MSE_WEIGHT * loss
        total_loss.backward()
        optimizer.step()

    print(f'Loss = {total_loss.item()}')

    im_orig_reconstructed = reconstruct_im(results['im_orig'], net.cfg)
    for number, output in enumerate(results['outputs']):
        output_reconstructed = reconstruct_im(output, net.cfg)
        output_metrics = calc_metrics(output_reconstructed, im_orig_reconstructed)
        for name, value in output_metrics.items():
            wandb.log(
                {
                    f"{name}; Step {number + 1}": value,
                },
                step=step,
            )


def train(net, dataloader):

    net.train()
    init_wandb(net.cfg)
    optimizer = optim.Adam(net.parameters(), lr=net.cfg.OPTIMIZER_LR)

    for e in range(net.cfg.EPOCHS):
        print(f'Training epoch [{e + 1}/{net.cfg.EPOCHS}] ...')
        train_one_epoch(net, optimizer, dataloader, e)




