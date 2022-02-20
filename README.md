# COVIDNet-Implementation
A replication of COVIDNet for COVID-19 diagnosis.
The original paper can be found [here](https://arxiv.org/abs/2003.09871)

# Introduction
In this work, I proposed some minor modifications to the original "COVID-Net: A Tailored Deep Convolutional Neural
Network Design for Detection of COVID-19 Cases from Chest X-Ray Images" paper. In the original COVID-Net
implementation, the author proposed a lightweight CNN architecture tailored for COVID-19 diagnosis. In order to
increase the robustness of the method, I incorporated several contrastive learning scheme to encourage the network
to learn discriminative features between positive and negative cases. Also, with the employment of contrastive learning,
we can get rid of the pretraining task with ImageNet, which is considerably more time consuming.

# Set-up

- [Data preparation and preprocessing](https://github.com/hieubkvn123/COVIDNet-Implementation/tree/main/data)

## 1. Install requirements
Install the python libraries used in this project with
```
pip3 install -r requirements.txt
```

## 2. Training
To run training, configure wandb options in wandb\_conf.py :
```python
config = {
    "project" : "wandb_project_name",
    "entity" : "wandb_username"
}
```

Then run the training script main.py with the following options :
```python
parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset folder with sub-folders for each class')
parser.add_argument('--test_dir', type=str, required=True, help='Path to the testing dataset folder with sub-folders for each class')
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
```

For example :
```
python3 main.py --data_dir data/covidx/train 
			--test_dir data/covidx/test 
			--run_name test_run_12 
			--no_batch_norm
```

# TODO
- [x] Create the dataloader for COVIDx dataset.
- [x] Replicate the original COVID-Net model.
- [x] Create a training script and push the result on Wandb.ai.
	- [x] Add another loop for testing in the training loop.
	- [x] Add early stopping and overfitting detection.
	- [x] Add regularizers in each conv layer.
	- [x] Create a learning rate scheduler.
	- [x] Add image augmentation in the data loader.
		- So far the augmentation includes : Rotation, Translation, Horizontal Flip.
- [x] Replicate the experiment result in COVID-Net baseline.
- [x] Implement changes and further improvements to COVID-Net.
	- [x] Add regularization (L1 and L2) to the PEPx module and test run.
	- [ ] Clip the logits features using L2 normalization and test run.
	- [x] Add the Additive Angular Margin Penalty head to the logits layer.

# Reference
- Orignal COVID-Net paper : https://arxiv.org/abs/2003.09871
- COVIDx X-ray images dataset for Covid-19 classification : https://www.kaggle.com/andyczhao/covidx-cxr2?select=train.txt
- COVID-Net PyTorch implementation : https://github.com/iliasprc/COVIDNet
