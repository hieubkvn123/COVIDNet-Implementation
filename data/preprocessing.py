import os
import tqdm
import glob

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--train_dir', required=False, default='covidx/train', help='Path to training directory')
parser.add_argument('--train_txt', required=False, default='covidx/train.txt', help='Path to training label file')
parser.add_argument('--test_dir', required=False, default='covidx/test', help='Path to testing directory')
parser.add_argument('--test_txt', required=False, default='covidx/test.txt', help='Path to testing label file')
args = vars(parser.parse_args())

# Open both train and test labels
with open(args['train_txt'], 'r') as f:
    train_idx = f.readlines()

with open(args['test_txt'], 'r') as f:
    test_idx = f.readlines()


# Create sub-directories for each class
if(not os.path.exists(os.path.join(args['train_dir'], 'positive'))):
    os.mkdir(os.path.join(args['train_dir'], 'positive'))

if(not os.path.exists(os.path.join(args['train_dir'], 'negative'))):
    os.mkdir(os.path.join(args['train_dir'], 'negative'))

if(not os.path.exists(os.path.join(args['test_dir'], 'positive'))):
    os.mkdir(os.path.join(args['test_dir'], 'positive'))

if(not os.path.exists(os.path.join(args['test_dir'], 'negative'))):
    os.mkdir(os.path.join(args['test_dir'], 'negative'))

# Re-format training dataset
for line in tqdm.tqdm(train_idx):
    tokens = line.strip().split(' ')
    
    file_name = tokens[1]
    class_name = tokens[2]

    old_file = os.path.join(args['train_dir'], file_name)
    new_file = os.path.join(args['train_dir'], class_name, file_name)

    os.rename(old_file, new_file)


# Re-format testing dataset
for line in tqdm.tqdm(test_idx):
    tokens = line.strip().split(' ')
    
    file_name = tokens[1]
    class_name = tokens[2]

    old_file = os.path.join(args['test_dir'], file_name)
    new_file = os.path.join(args['test_dir'], class_name, file_name)

    os.rename(old_file, new_file)

