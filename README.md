# COVIDNet-Implementation
A replication of COVIDNet for COVID-19 diagnosis.
The original paper can be found [here](https://arxiv.org/abs/2003.09871)

# TODO
- [x] Create the dataloader for COVIDx dataset.
- [x] Replicate the original COVID-Net model.
- [x] Create a training script and push the result on Wandb.ai.
	- [x] Add another loop for testing in the training loop.
	- [x] Add early stopping and overfitting detection.
	- [x] Add regularizers in each conv layer.
	- [ ] Create a learning rate scheduler.
	- [ ] Add image augmentation in the data loader.
- [ ] Replicate the experiment result in COVID-Net baseline.
- [ ] Implement changes and further improvements to COVID-Net.

# Reference
- Orignal COVID-Net paper : https://arxiv.org/abs/2003.09871
- COVIDx X-ray images dataset for Covid-19 classification : https://www.kaggle.com/andyczhao/covidx-cxr2?select=train.txt
- COVID-Net PyTorch implementation : https://github.com/iliasprc/COVIDNet
