import os
import tqdm
import wandb
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from dataloader import DataLoader
from models import get_small_covid_net
from train import train_step, val_step
from wandb_config import config
from argparse import ArgumentParser

def train(args):
    loader = DataLoader(args['data_dir'], labels_as_subdir=True, one_hot=True,
            img_size=args['img_size'], train_val_ratio=args['val_ratio'])

    steps_per_epoch = loader.get_train_size()
    val_steps_per_epoch = loader.get_val_size()

    print(args['batch_norm'])
    model = get_small_covid_net(args['img_size'], args['img_size'], 3, batchnorm=args['batch_norm']) 
    optimizer = Adam(lr=args['lr'], beta_1=0.5, beta_2=0.999, amsgrad=True)

    wandb.init(project=config['project'], entity=config['entity'], id=args['run_name'])
    wandb.config.update(args)

    for epoch in range(args['epochs']):
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for batch_idx in range(steps_per_epoch):
                batch = loader.get_train_batch()

                prob, loss, accuracy = train_step(model, optimizer, batch)

                pbar.set_postfix({
                    'train_loss' : loss.numpy(),
                    'train_acc' : accuracy.numpy()
                })

                wandb.log({
                    'train_loss' : f'{loss.numpy():.4f}',
                    'train_acc' : f'{accuracy.numpy():.4f}'
                })

                pbar.update(1)

        with tqdm.tqdm(total=val_steps_per_epoch) as pbar:
            for batchidx in range(val_steps_per_epoch):
                batch = loader.get_val_batch()

                loss, accuracy = val_step(model, batch)

                pbar.set_postfix({
                    'val_loss' : f'{loss.numpy():.4f}',
                    'val_acc' : f'{accuracy.numpy():.4f}'
                })
                
                wandb.log({
                    'val_loss' : loss.numpy(),
                    'val_acc' : accuracy.numpy()
                })
                
                pbar.update(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset folder with sub-folders for each class')
    parser.add_argument('--batch_norm', action='store_false', required=False, help='Whether to apply batch normalization on Pepx modules')
    parser.add_argument('--img_size', type=int, required=False, default=480, help='Default image size of the dataset')
    parser.add_argument('--val_ratio', type=float, required=False, default=0.2, help='Ratio of data for validation')
    parser.add_argument('--epochs', type=int, required=False, default=50, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Number of instances per batch')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the wandb run')

    args = vars(parser.parse_args())

    train(args)
