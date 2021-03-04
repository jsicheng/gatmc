import torch
import yaml

from dataset import MCDataset
from model import GAE
from trainer import Trainer
from utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config

def main(cfg):
    cfg = Config(cfg)

    # device and dataset setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(cfg.root, cfg.dataset_name)
    data = dataset[0].to(device)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_relations = dataset.num_relations
    cfg.num_users = int(data.num_users)

    # set and init model
    model = GAE(cfg, random_init).to(device)
    model.apply(init_xavier)
    
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    
    # train
    trainer = Trainer(
        model, data, calc_rmse, optimizer
    )
    trainer.training(cfg.epochs)


if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)