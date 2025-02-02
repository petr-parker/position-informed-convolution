import tqdm

def train_one_epoch(net, dataloader):

    for


def train(net, dataloader, cfg):

    for e in cfg.EPOCHS:
        train_one_epoch(net, dataloader)
