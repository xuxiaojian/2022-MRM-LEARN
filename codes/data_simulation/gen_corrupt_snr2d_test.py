import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy.io import loadmat
import numpy as np
import h5py
import tifffile as tif
from tools.util import *
import numpy.fft as nf
from shutil import copyfile
import copy


# input data: nheight - nwidth - nslice - necho
# output data: nheight - nwidth - nslice - necho
def gen_data_snr3d(folders=None, saveData=False, save_ksp=False, input_rootpath=None, output_rootpath=None):

    output_rootpath = output_rootpath
    check_and_mkdir(output_rootpath)
    for conter in range(len(folders)):
        # load data: nheight - nwidth - nslice - necho
        subfolder = folders[conter]
        np.random.seed(int(folders[conter].replace('C', '').replace('_V', '')))
        file_path = '{}/{}/ima_comb_{}.mat'.format(input_rootpath, subfolder, subfolder)

        print('loading {}'.format(file_path))
        truth_complex_ori = loadmat(file_path, verify_compressed_data_integrity=False)['ima_comb']
        nheight, nwidth, nslice, necho = truth_complex_ori.shape

        # make motion folder
        dirDataName = '{}/{}'.format(output_rootpath, subfolder)
        check_and_mkdir(dirDataName)

        if keeptruth:
            corrupt_data_name = '{}/ima_comb_{}.mat'.format(dirDataName, subfolder)
            copyfile(file_path, corrupt_data_name)
            print('[cpoying data] {} '.format(corrupt_data_name))
            continue
        else:
            # get k-space data
            ksp = img2ksp(truth_complex_ori)
            # generate corruption
            for sample in range(nsample):
                label = 'n{}'.format(nmotion)
                label += '_{}'.format(corrupt_type)
                ksp_corrupt = ksp.copy()
                for motion in range(nmotion):
                    np.random.seed(int(folders[conter].replace('C', '').replace('_V', '')) * (motion+1) * (sample + 1))
                    # shift image
                    shift = [get_rand_int(shift_h_rng[motion]), get_rand_int(shift_w_rng[motion]), 0, 0]
                    rotate = get_rand_int(rotate_rng[motion])

                    w_start = rand_list[motion][get_rand_int([0, len(rand_list[motion]) - 1])]
                    w_end = w_start + get_rand_int(w_rng[motion])
                    # label += '_([{:03d}-{:03d}])_([{:d}_{:d}]_{:d})'.format(w_start, w_end, shift[0], shift[1], rotate)

                    # get k-space data for shifted image
                    ipt_complex_shiftnrotate = shiftnrotate(truth_complex_ori, shift, rotate)
                    ksp_shiftnrotate = img2ksp(ipt_complex_shiftnrotate)

                    # replace the ksp
                    ksp_corrupt[:, w_start:w_end, :, :] = ksp_shiftnrotate[:, w_start:w_end, :, :]

                # convert corrupted ksp to img
                corrupt_complex_ori = ksp2img(ksp_corrupt)
                if saveData or save_ksp:
                    if saveData:
                        corrupt_data_name = '{}/ima_comb_{}_{}.mat'.format(dirDataName, subfolder, label)
                        print('[storing data] {} '.format(corrupt_data_name))
                        sio.savemat(corrupt_data_name, {'ima_comb': corrupt_complex_ori})
                    if save_ksp:
                        corrupt_data_name = '{}/ksp_{}_{}.mat'.format(dirDataName, subfolder, label)
                        print('[storing data] {} '.format(corrupt_data_name))
                        sio.savemat(corrupt_data_name, {'ksp': ksp_corrupt})


# server3
input_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/truth'
output_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/motion'

keeptruth = False
set_random = False
nmotion= 3 # can be 3, 6, 9 for light, moderate and heavy motions
nsample = 1
corrupt_type = 'snr2d_mid'
if set_random:
    # nrand{}-{}
    motion_rng = [1, 10]                          #[1, 5][1, 10][1, 15]
    w_rng = [[1, 10]] *   motion_rng[1]           #[1, 5] [1,10][1, 15]
    shift_h_rng = [[-15, 15]]  *   motion_rng[1]  #[-5, 5] [-15, 15] [-20, 20]
    shift_w_rng = [[-15, 15]]  *   motion_rng[1]  #[-5, 5] [-15, 15] [-20, 20]
    rotate_rng = [[-15, 15]]   *   motion_rng[1]  #[-10, 10] [-15, 15][-30, 30]
    rand_list = [np.concatenate((np.arange(0, 192 // 2 - 15), np.arange(192 // 2 + 15, 191)), axis=0)] * motion_rng[1]
else:
    shift_h_rng = [[5, 5], [-8, -8], [11, 11],
                   [6, 6], [-9, -9], [12, 12],
                   [7, 7], [-10, -10], [13, 13]]
    shift_w_rng = list(-np.array(shift_h_rng))
    rotate_rng = shift_h_rng

    w_rng = [[7, 7], [5, 5], [3, 3]] * 3
    rand_list = [[30, 30], [50, 50], [160, 160],
                 [10, 10], [60, 60], [180, 180],
                 [20, 20], [130, 130], [170, 170]]

np.random.seed(123)
example_file=['C08_V2']
gen_data_snr3d(folders=example_file, saveData=True, save_ksp=False, input_rootpath=input_rootpath, output_rootpath=output_rootpath)

# good_files = ['C01_V2', 'C01_V3', 'C02_V2', 'C02_V3', 'C03_V2', 'C03_V3', 'C04_V2', 'C04_V3', 'C06_V2',
#               'C06_V3','C07_V2', 'C07_V3', 'C08_V2', 'C08_V3', 'C10_V3', 'C11_V1', 'C12_V1', 'C14_V1',
#               'C15_V1', 'C16_V1','C17_V1', 'C18_V1', 'C22_V1', 'C23_V1', 'C24_V1', 'C25_V1', 'C26_V1']
# middel_files = ['C05_V2', 'C05_V3', 'C09_V3', 'C10_V2', 'C19_V1', 'C21_V1']
# bad_files = ['C13_V1', 'C20_V1']
# new_motion_files = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
# new_good_files = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
