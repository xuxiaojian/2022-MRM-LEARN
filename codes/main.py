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

# read configuration
config_file_name = 'configs/config_train_LEARN-IMG.json'
with open(config_file_name) as File:
    config = json.load(File)
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
