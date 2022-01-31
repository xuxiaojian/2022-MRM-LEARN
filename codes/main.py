import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json

import tools.util as util
from method.network import Network
from data_loader.mri_3decho import mri_3decho
from data_loader.mri_2dechoft import mri_2dechoft

# init the env
util.init_env()

# init class
dataset_dict = {
    'mri_3decho': mri_3decho,
    'mri_2dechoft': mri_2dechoft,
}

# define the project folder
directory = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN'

# read configuration
config_file_name = directory + '/codes' + '/configs/config_test_LEARN-BIO.json'
with open(config_file_name) as File:
    config = json.load(File)

# complete the path accordingly
config["setting"]["root_path"] = directory + '/' + config["setting"]["root_path"]
config["setting"]["data_path"] = directory + '/' + config["setting"]["data_path"]
config["train"]["src_path"] = directory + '/' + config["train"]["src_path"]

# load GPUs usage
os.environ["CUDA_VISIBLE_DEVICES"] = config['setting']['gpu_index']

# run method for trainning/testing
method = Network(config=config)
mode = config['setting']['mode']
if mode == 'train':
    train_dataset, valid_dataset = dataset_dict[config['setting']['dataset']](config, 'train')
    test_dataset = dataset_dict[config['setting']['dataset']](config, 'test')
    method.train(train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset)
if mode == 'test':
    test_dataset = dataset_dict[config['setting']['dataset']](config, 'test')
    method.test(test_dataset=test_dataset)
