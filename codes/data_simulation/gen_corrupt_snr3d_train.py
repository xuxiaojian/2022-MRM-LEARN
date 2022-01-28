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
import tools.transcript as transcript
import copy


# input data: nheight - nwidth - nslice - necho
# output data: nheight - nwidth - nslice - necho
def gen_data_snr3d(folders=None, saveData=False, save_ksp=False, save_tif=False, input_rootpath=None, output_rootpath=None):
    output_rootpath = output_rootpath
    check_and_mkdir(output_rootpath)
    for conter in range(len(folders)):
        # load data: nheight - nwidth - nslice - necho
        subfolder = folders[conter]
        file_path = '{}/{}/ima_comb_{}.mat'.format(input_rootpath, subfolder, subfolder)

        print('loading {}'.format(file_path))
        truth_complex_ori = loadmat(file_path, verify_compressed_data_integrity=False)['ima_comb']
        nheight, nwidth, nslice, necho = truth_complex_ori.shape

        # make motion folder
        dirDataName = '{}/{}'.format(output_rootpath, subfolder)
        check_and_mkdir(dirDataName)
        transcript.start(dirDataName + '/logfile_{}.log'.format(corrupt_type))

        # data_name = '{}/ima_comb_{}.tif'.format(dirDataName, subfolder)
        # tif.imwrite(data_name, abs(truth_complex_ori[:,:,:,0].transpose(2, 0, 1)).astype('float32'), imagej=True, ijmetadata={'Slice': nslice})
        # print('[storing tif] {} '.format(data_name))
        # print(f'{subfolder}: size = {nheight}, {nwidth}, {nslice}, {necho}')
        # continue

        if keeptruth:
            corrupt_data_name = '{}/ima_comb_{}.mat'.format(dirDataName, subfolder)
            copyfile(file_path, corrupt_data_name)
            print('[cpoying data] {} '.format(corrupt_data_name))
            continue
        else:
            # get k-space truth
            ksp = img2ksp(truth_complex_ori)
            # generate mask
            for sample in range(nsample):
                label = 'nrand{}-{}'.format(motion_rng[0], motion_rng[1])
                label += '_{}'.format(corrupt_type)
                ksp_corrupt = ksp.copy()

                # volumwise
                if corrupt_pattern == 'volumwise':
                    img = truth_complex_ori.copy()
                    nmotion = get_rand_int(motion_rng)
                    print('>>>>>> sample = {}/{}, motion = {}'.format(sample, nsample, nmotion))
                    for motion in range(nmotion):
                        shift = [get_rand_int(shift_h_rng[motion]), get_rand_int(shift_w_rng[motion]), 0, 0]
                        # low
                        # rotate = [0, 0, 0]
                        # if get_rand_int([0, 1]) == 1: rotate[0] = get_rand_int(rotate_rng_hw[motion])
                        # if get_rand_int([0, 1]) == 1: rotate[1] = get_rand_int(rotate_rng_hs[motion])
                        # if get_rand_int([0, 1]) == 1: rotate[2] = get_rand_int(rotate_rng_ws[motion])
                        # print(f'\tshift = [shift_h, shift_w] = {shift},  rotate = [hw, hs, ws] = {rotate}')

                        # mid
                        rotate = [get_rand_int(rotate_rng_hw[motion]),  get_rand_int(rotate_rng_hs[motion]), get_rand_int(rotate_rng_ws[motion])]

                        ipt_complex_shiftnrotate = shiftnrotate(img, shift, rotate, snr_type='3d')
                        # ksp for snr-ed image
                        ksp_shiftnrotate = img2ksp(ipt_complex_shiftnrotate)
                        # replace the ksp
                        w_start = rand_list[motion][get_rand_int([0, len(rand_list[motion]) - 1])]
                        w_end = w_start + get_rand_int(w_rng[motion])
                        print('\tmotion, shift, rotate, wrng = ', motion, shift, rotate, [w_start, w_end])
                        ksp_corrupt[:, w_start:w_end, :, :] = ksp_shiftnrotate[:, w_start:w_end, :, :]
                    corrupt_complex_ori = ksp2img(ksp_corrupt)
                # slicewise
                elif corrupt_pattern == 'slicewise':
                    for slice in range(nslice):
                        img = truth_complex_ori.copy()
                        nmotion = get_rand_int(motion_rng)
                        print('>>>>>> sample = {}/{}, slice = {}/{}, motion = {}'.format(sample, nsample, slice, nslice, nmotion))
                        for motion in range(nmotion):
                            shift = [get_rand_int(shift_h_rng[motion]), get_rand_int(shift_w_rng[motion]), 0, 0]
                            rotate = get_rand_int(rotate_rng[motion], size=3)
                            ipt_complex_shiftnrotate = shiftnrotate(img, shift, rotate, snr_type='3d')[:, :, slice, :]
                            # ksp for snr-ed image
                            ksp_shiftnrotate = img2ksp(ipt_complex_shiftnrotate)
                            # replace the ksp
                            w_start = rand_list[motion][get_rand_int([0, len(rand_list[motion]) - 1])]
                            w_end = w_start + get_rand_int(w_rng[motion])
                            print('\tmotion, shift, rotate, wrng = ', motion, shift, rotate, [w_start, w_end])
                            ksp_corrupt[:, w_start:w_end, slice, :] = ksp_shiftnrotate[:, w_start:w_end, :]
                    corrupt_complex_ori = ksp2img(ksp_corrupt)
                # save data
                if saveData or save_ksp:
                    if saveData:
                        corrupt_data_name = '{}/ima_comb_{}_{}.mat'.format(dirDataName, subfolder, label)
                        sio.savemat(corrupt_data_name, {'ima_comb': corrupt_complex_ori})
                        print('[storing data] {} '.format(corrupt_data_name))
                    if save_ksp:
                        corrupt_data_name = '{}/ksp_{}_{}.mat'.format(dirDataName, subfolder, label)
                        sio.savemat(corrupt_data_name, {'ksp': ksp_corrupt})
                        print('[storing ksp] {} '.format(corrupt_data_name))
                    if save_tif:
                        corrupt_data_name = '{}/ima_comb_{}_{}.tif'.format(dirDataName, subfolder, label)
                        tif.imwrite(corrupt_data_name, abs(corrupt_complex_ori[:,:,:,0].transpose(2, 0, 1)).astype('float32'), imagej=True, ijmetadata={'Slice': nslice})
                        print('[storing tif] {} '.format(corrupt_data_name))



# server3
input_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/truth'
output_rootpath = '/export1/project/xiaojianxu/projects/2022-MRM-LEARN/data/motion'

corrupt_type = 'snr3d_midnew'
keeptruth = False
set_random = True
nsample = 1
corrupt_pattern= 'volumwise'

if set_random:
    # nrand{}-{}
    motion_rng = [1, 10]                           #[1, 5][1, 10][1, 15]
    w_rng = [[1, 10]] *   motion_rng[1]           #[1, 5] [1,10][1, 15]
    shift_h_rng = [[-15, 15]]  *   motion_rng[1]  #[-5, 5] [-15, 15] [-20, 20]
    shift_w_rng = [[-15, 15]]  *   motion_rng[1]  #[-5, 5] [-15, 15] [-20, 20]
    rotate_rng = [[-15, 15]]   *   motion_rng[1]  #[-10, 10] [-15, 15][-30, 30]

    # h, w, s, e
    rotate_rng_hw = [[-15, 15]]   *   motion_rng[1]
    rotate_rng_hs = [[-15, 15]]   *   motion_rng[1]
    rotate_rng_ws = [[-15, 15]]   *   motion_rng[1]

    rand_list = [np.concatenate((np.arange(0, 192 // 2 - 15), np.arange(192 // 2 + 15, 191)), axis=0)] * motion_rng[1]
else:
    # n{}_snr3d_midnew
    shift_h_rng = [[5, 5], [-8, -8], [11, 11],
                   [6, 6], [-9, -9], [12, 12],
                   [7, 7], [-10, -10], [13, 13]]
    shift_w_rng = copy.deepcopy(shift_h_rng) #list(-np.array(shift_h_rng))
    rotate_rng = copy.deepcopy(shift_h_rng) #list(-np.array(shift_h_rng))

    w_rng = [[7, 7], [5, 5], [3, 3]] * 3
    rand_list = [[30, 30], [50, 50], [160, 160],
                 [10, 10], [60, 60], [180, 180],
                 [20, 20], [130, 130], [170, 170]]

np.random.seed(123)
example_file=['C08_V2']
gen_data_snr3d(folders=example_file, saveData=True, save_ksp=False, save_tif=True, input_rootpath=input_rootpath, output_rootpath=output_rootpath)


# good_files = ['C01_V2', 'C01_V3', 'C02_V2', 'C02_V3', 'C03_V2', 'C03_V3', 'C04_V2', 'C04_V3', 'C06_V2',
#               'C06_V3','C07_V2', 'C07_V3', 'C08_V2', 'C08_V3', 'C10_V3', 'C11_V1', 'C12_V1', 'C14_V1',
#               'C15_V1', 'C16_V1','C17_V1', 'C18_V1', 'C22_V1', 'C23_V1', 'C24_V1', 'C25_V1', 'C26_V1']

# new_new_motion_files = ['017_9990', '022_10027', '038_12863', 'HV003_Prisma_6']
# middel_files = ['C05_V2', 'C05_V3', 'C09_V3', 'C10_V2', 'C19_V1', 'C21_V1']
# bad_files = ['C13_V1', 'C20_V1']
# new_motion_files = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
# new_good_files = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']