import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys 
from os.path import dirname, abspath
dir = dirname(dirname(abspath(__file__)))
sys.path.append(dir)

from scipy.io import loadmat
from tools.util import *
from shutil import copyfile

def gen_midecube0_data(folders=None, saveData=False, input_rootpath=None, output_rootpath=None, norm='midcube0'):
    check_and_mkdir(output_rootpath)
    try:
        dict = loadmat('{}/{}.mat'.format(output_rootpath, norm))
    except:
        dict = {}

    for conter in range(len(folders)):
        subjfolder = folders[conter]
        print('[loading data] {}'.format(subjfolder))
        img_truth = loadmat('{}/{}/ima_comb_{}.mat'.format(input_rootpath, subjfolder, subjfolder), verify_compressed_data_integrity=False)['ima_comb']
        nheight, nwidth, nslice, necho = img_truth.shape
        abs_img_truth = abs(img_truth)
        if norm == 'midcube0':
            data = np.mean(abs_img_truth[..., nslice//2, 0])
            print('midecube0 = {:.3e}'.format(data))
            dict['para_' + str(subjfolder)] = data
        elif norm == 'maxecho0':
            data = np.amax(abs_img_truth[..., 0], axis=(0, 1))
            [print('{} = {:.3e}'.format(i, data[i])) for i in range(nslice)]
            dict[str(subjfolder)] = data
        else:
            print('wrong norm method')

    if saveData:
        dirDataName = '{}'.format(output_rootpath)
        check_and_mkdir(dirDataName)
        data_name = '{}/{}.mat'.format(output_rootpath, norm)
        print('[storing data] {} '.format(data_name))
        print(dict)
        sio.savemat(data_name, dict)

# compute the normalization parameters
example_motion_files = ['017_9990']
input_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/truth'
output_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/truth'
norm= 'midcube0'
gen_midecube0_data(folders=example_motion_files, saveData=True, input_rootpath=input_rootpath, output_rootpath=output_rootpath, norm=norm)



