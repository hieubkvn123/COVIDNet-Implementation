import os
import tqdm
import time
import wandb
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from dataloader import DataLoader
from models import get_small_covid_net
from wandb_config import config
from train import train_step, val_step, is_overfitting
from train import exp_decay_learning_rate, lin_decay_learning_rate
from argparse import ArgumentParser

def train(args):
    # Check if GPU is available
    if(tf.test.is_gpu_available()):
        print('[INFO] GPU-enabled training is running ...')
    else:
        print('[INFO] GPU is not detected ...')

    # Create checkpoint directory
    if(not os.path.exists(args['save_dir'])):
        print('[INFO] Creating checkpoint directory ...')
        os.mkdir(args['save_dir'])

    # Create checkpoint directory for this model
    if(not os.path.exists(os.path.join(args['save_dir'], args['run_name']))):
        print('[INFO] Creating checkpoint directory for model ', args['run_name'])
        os.mkdir(os.path.join(args['save_dir'], args['run_name']))

    if(not os.path.exists(os.path.join(args['save_dir'], args['run_name'], 'models'))):
        print('[INFO] Creating checkpoint models directory ...')
        os.mkdir(os.path.join(args['save_dir'], args['run_name'], 'models'))

    if(not os.path.exists(os.path.join(args['save_dir'], args['run_name'], 'weights'))):
        print('[INFO] Creating checkpoint weights directory ...')
        os.mkdir(os.path.join(args['save_dir'], args['run_name'], 'weights'))
    
    loader = DataLoader(args['data_dir'], labels_as_subdir=True, one_hot=True,
            img_size=args['img_size'], train_val_ratio=args['val_ratio'],
            batch_size=args['batch_size'], random_noise=True) # random_noise = augmentation

    test_loader = DataLoader(args['test_dir'], labels_as_subdir=True, one_hot=True,
            img_size=args['img_size'],test=True,
            batch_size=args['batch_size'], random_noise=False)

    steps_per_epoch = loader.get_train_size()
    val_steps_per_epoch = loader.get_val_size()
    test_steps_per_epoch = test_loader.get_train_size()

    model = get_small_covid_net(args['img_size'], args['img_size'], 3, 
            batchnorm=not args['no_batch_norm'], contrastive=args['contrast_scheme']) 
    optimizer = Adam(lr=args['lr'], beta_1=0.5, beta_2=0.999, amsgrad=True)

    wandb.init(project=config['project'], entity=config['entity'], id=args['run_name'])
    wandb.config.update(args)

    # Some config variables
    lr = args['lr']
    mean_val_losses = []
    mean_test_losses = []
    best_model = model

    # Start training
    try:
        for epoch in range(args['epochs']):
            print(f'\n\nEpoch #[{epoch + 1}/{args["epochs"]}]')
            with tqdm.tqdm(total=steps_per_epoch) as pbar:  
                if(args['lr_sched'] == 'exp'):
                    lr = exp_decay_learning_rate(lr, epoch + 1, decay_steps=args['epochs'])
                else:
                    lr = lin_decay_learning_rate(lr, epoch + 1, decay_steps=args['epochs'], end_lr=0.1*args['lr'])

                optimizer.learning_rate.assign(lr)
                print('[INFO] Learning rate updated to ', optimizer.learning_rate.numpy(), ' ...')

                for batch_idx in range(steps_per_epoch):
                    batch = loader.get_train_batch()

                    prob, loss, accuracy = train_step(model, optimizer, batch, contrastive=args['contrast_scheme'])

                    pbar.set_postfix({
                        'train_loss' : f'{loss.numpy():.4f}',
                        'train_acc' : f'{accuracy.numpy():.4f}'
                    })

                    wandb.log({
                        'train_loss' : loss.numpy(),
                        'train_acc' : accuracy.numpy()
                    })

                    pbar.update(1)

            time.sleep(1.0)


            # ---- Validation ---- #
            print("\nValidating...")
            val_losses = []
            val_accs = []
            with tqdm.tqdm(total=val_steps_per_epoch) as pbar:
                for batchidx in range(val_steps_per_epoch):
                    batch = loader.get_val_batch()

                    loss, accuracy = val_step(model, batch, contrastive=args['contrast_scheme'])

                    pbar.set_postfix({
                        'val_loss' : f'{loss.numpy():.4f}',
                        'val_acc' : f'{accuracy.numpy():.4f}'
                    })
                    
                    val_losses.append(loss.numpy())
                    val_accs.append(accuracy.numpy())
                    
                    pbar.update(1)

            wandb.log({
                'val_loss' : np.array(val_losses).mean(), # loss.numpy(),
                'val_acc' : np.array(val_accs).mean() # accuracy.numpy()
            })
            mean_val_losses.append(np.array(val_losses).mean())
            time.sleep(1.0)

            # ---- Testing ---- #
            print('\nTesting...')
            test_losses = []
            test_accs = []
            with tqdm.tqdm(total=test_steps_per_epoch) as pbar:
                for batchidx in range(test_steps_per_epoch):
                    batch = test_loader.get_train_batch()

                    loss, accuracy = val_step(model, batch, contrastive=args['contrast_scheme'])

                    pbar.set_postfix({
                        'test_loss' : f'{loss.numpy():.4f}',
                        'test_acc' : f'{accuracy.numpy():.4f}'
                    })

                    test_losses.append(loss.numpy())
                    test_accs.append(accuracy.numpy())
                    
                    pbar.update(1)

                    
            wandb.log({
                'test_loss' : np.array(test_losses).mean(), # loss.numpy(),
                'test_acc' : np.array(test_accs).mean() # accuracy.numpy()
            })
            mean_test_losses.append(np.array(test_losses).mean())
            time.sleep(1.0)

            # Detect overfitting
            if(is_overfitting(mean_val_losses) or is_overfitting(mean_test_losses)):
                print('[INFO] Overfitting detected, training halted ...')
                break

            # Check if best_model can be saved based on validation loss
            if(mean_val_losses[-1] == min(mean_val_losses)):
                print('[INFO] Saving best model ...')
                best_model = model

            # Checkpoint model
            if((epoch + 1) % args['saved_every'] == 0):
                print('[INFO] Checkpointing ...')
                model.save(os.path.join(args['save_dir'], args['run_name'], 'models', f'model_step_{epoch+1}.h5'))
                model.save_weights(os.path.join(args['save_dir'], args['run_name'], 'weights', f'model_step_{epoch+1}.weights.h5'))
    except:
        print('[INFO] Training halted unexpectedly, saving best model ...')
        best_model.save(os.path.join(args['save_dir'], args['run_name'], 'models', f'best_model.h5'))
        best_model.save_weights(os.path.join(args['save_dir'], args['run_name'], 'weights', f'best_model.weights.h5'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset folder with sub-folders for each class')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the testing dataset folder with sub-folders for each class')
    parser.add_argument('--contrast_scheme', type=str, required=False, default=None, choices=['arcface', 'cosface', 'sphereface'], help='Select contrastive learning scheme/loss')
    parser.add_argument('--no_batch_norm', action='store_true', required=False, help='Whether to apply batch normalization on Pepx modules')
    parser.add_argument('--img_size', type=int, required=False, default=480, help='Default image size of the dataset')
    parser.add_argument('--val_ratio', type=float, required=False, default=0.2, help='Ratio of data for validation')
    parser.add_argument('--epochs', type=int, required=False, default=50, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Number of instances per batch')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_sched', type=str, required=False, default='exp', choices=['exp', 'lin'], help='Types of learning rate scheduler. "exp" for exponential decay, "lin" for linear decay')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the wandb run')
    parser.add_argument('--patience', type=int, required=False, help='Patience for early stopping')
    parser.add_argument('--saved_every', type=int, required=False, default=5, help='Number of steps to save model weights once every time')
    parser.add_argument('--save_dir', type=str, required=False, default='./checkpoints', help='Name of checkpoint folder')

    args = vars(parser.parse_args())

    train(args)
