import tensorflow as tf
import os
import shutil
import math
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import scipy.io as sio
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.io import loadmat
import json
import scipy
import numpy.fft as nf

import random



np.set_printoptions(formatter={'float': lambda x: "{:0.2f}".format(x)})



def init_env(seed_value=0):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)


def to_mat_np(data: np.ndarray, file_name: str):
    sio.savemat(file_name=file_name + '.mat', mdict={
        'data': data
    })

def norm_data(data, method=None, subjname='C01_V2', config=None):
    # input: num_slice, num_height, num_width, num_echo
    # or     num_slice, num_height, num_width, num_channel
    if method is None or method == 'None' or method == 'none':
        return data
    elif method == 'maxmin':
        data = data - np.amin(data)
        data = data / np.amax(data)
        ref = None
    elif method == 'midcube0': # mean of the midddle slice of echo 0
        ref = loadmat(config['data_path'] + 'truth/midcube0.mat')['para_' + subjname]
        data = data/ref
    elif method == 'maxecho0':
        ref = loadmat(config['data_path'] + 'truth/maxecho0.mat')['para_' + subjname]
        data = data / np.squeeze(ref)[..., None, None, None]
    elif method == 'max':
        ref = np.amax(abs(data), axis=(1, 2), keepdims=True)
        data = data / ref
    elif method == 'mean':
        ref = np.mean(abs(data), axis=(1, 2), keepdims=True)
        data = data / ref
    elif method == 'midcube':
        num_slice, num_height, num_width, num_channel = data.shape
        ref = np.mean(abs(data[num_slice//2, ...]))
        data = data / ref
    else:
        print("error: wrong normalization in processing data.")
        exit(1)
    print(f'The input mGRE data is scaled up by dividing {ref}')
    return data

def make_3d_grid(img_tensor, num_column=5):
    # input shape = n - echo - height - width -  channel or n - height - width -  channel
    def transform(img_tensor_):
        ret = img_tensor_
        if len(ret.shape)==3:
            # width, height, channel
            ret = tf.expand_dims(ret, 0)
        phase, width, height, channel = ret.shape
        phase   = int(phase)
        width   = int(width)
        height  = int(height)
        channel = int(channel)

        if phase == 10:
            num_column = 5
        else:
            num_column = 1

        ret = tf.contrib.gan.eval.image_grid(ret, (math.ceil(phase / num_column), num_column), (width, height), channel)
        ret = ret[0]
        return ret

    img_channel = int(img_tensor.shape[-1])
    if img_channel != 1:
        img_tensor_unstack = tf.unstack(img_tensor, axis=-1)
        opt = tf.concat(img_tensor_unstack, axis=0)
        opt = tf.expand_dims(opt, axis=-1)
    else:
        opt = img_tensor

    opt = tf.map_fn(transform, opt) # opt : ? * 10 * 256 * 192 * 1(fix)
    print("[make_3d_grid] shape of opt: % s" % str(opt.shape))

    return opt


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def tf_snr(pre, y):
    def tf_log10(x):
        return tf.log(x) / tf.log(tf.constant(10, dtype=y.dtype))
    return 20 * tf_log10(tf.norm(tf.reshape(y, shape=[-1])) / tf.norm(tf.reshape(y, shape=[-1]) - tf.reshape(pre, shape=[-1])))

def evaluateSnr(xtrue, x):
    return 20 * np.log10(np.linalg.norm(xtrue.flatten('F')) / np.linalg.norm(xtrue.flatten('F') - x.flatten('F')))

def copytree_code(save_path, src_path):
    max_code_save = 100
    for i in range(max_code_save):
        code_path = save_path + 'code%d/' % i
        if not os.path.exists(code_path):
            shutil.copytree(src=src_path, dst=code_path)
            break


def dict_to_md_table(config: dict):
    # Convert a python dict to markdown table

    info = str()
    for section in config.keys():

        info += '## ' + section + '\n'
        info += '|  Key  |  Value |\n|:----:|:---:|\n'

        for i in config[section].keys():
            info += '|' + i + '|' + str(config[section][i]) + '|\n'

        info += '\n\n'

    return info

def addtext(image, text):
    draw = ImageDraw.Draw(image)
    x = 0
    y = 0
    htext = 10
    for line in text:
        draw.text((x, y), line)
        y= y + htext
    return image

def make_channels(data, inputType='complex', outputType='complex', new_dim=False):
    if outputType == 'complex':
        if new_dim:
            data = np.concatenate((data.real[..., None], data.imag[..., None]), -1)
        else:
            data = np.concatenate((data.real, data.imag), -1)
    elif outputType == 'magnitude':
        return abs(data)
    elif outputType == 'real':
        return data.real
    elif outputType == 'imag':
        return data.imag
    else:
        print('Cannot add channels!')
        exit(1)
    return data


def addwgn(x, inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10 ** (-inputSnr / 20)
    noise = np.random.normal(loc=0, scale=1, size=np.shape(x))
    noise = noise / np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y

def shiftnrotate(img, shift, rotate, datatype='complex', snr_type = '2d', mode='constant'):
    tmp_img = img
    if datatype == 'complex':
        tmp_img_real = tmp_img.real
        tmp_img_imag = tmp_img.imag
        if snr_type == '2d': # data shape = [h, w, e], snr2d: shift and roate with all constant mode
            # shift
            if not all(v == 0 for v in shift):
                tmp_img_real = scipy.ndimage.shift(tmp_img_real, shift, mode=mode, cval=0.0)
                tmp_img_imag = scipy.ndimage.shift(tmp_img_imag, shift, mode=mode, cval=0.0)
            # rotate
            if rotate != 0:
                tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate, reshape=False, mode=mode, cval=0.0)
                tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate, reshape=False, mode=mode, cval=0.0)
        elif snr_type == '3d': # data shape = [h, w, s, e], snr3d_mid: all with nearst mode, snr3d_low: shift constant but rotate with nearest., snr3d_midnew: all constant
            # shift
            if not all(v == 0 for v in shift):
                tmp_img_real = scipy.ndimage.shift(tmp_img_real, shift, mode=mode, cval=0.0)
                tmp_img_imag = scipy.ndimage.shift(tmp_img_imag, shift, mode=mode, cval=0.0)
            # rotate
            if rotate[0] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[0], mode=mode, cval=0.0, axes=(0, 1), reshape=False)
            if rotate[1] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[1], mode=mode, cval=0.0, axes=(0, 2), reshape=False)
            if rotate[2] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[2], mode=mode, cval=0.0, axes=(1, 2), reshape=False)

            if rotate[0] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[0], mode=mode, cval=0.0, axes=(0, 1), reshape=False)
            if rotate[1] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[1], mode=mode, cval=0.0, axes=(0, 2), reshape=False)
            if rotate[2] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[2], mode=mode, cval=0.0, axes=(1, 2), reshape=False)
        else:
            pass
        # make complex numbers
        tmp_img = tmp_img_real + 1j * tmp_img_imag
    else:
        print('datatype not found!')
        exit(1)
    return tmp_img

def img2ksp(img):
    nheight = img.shape[0]
    nwidth = img.shape[1]
    imatmp2 = nf.fftshift(nf.ifft(nf.fftshift(img, axes=1), n=nwidth, axis=1), axes=1)
    ksp = nf.fftshift(nf.fft(nf.fftshift(imatmp2, axes=0), n=nheight, axis=0), axes=0)
    return ksp

def ksp2img(ksp):
    nheight = ksp.shape[0]
    nwidth = ksp.shape[1]
    imatmp2 = nf.fftshift(nf.fft(nf.fftshift(ksp, axes=1), n=nwidth, axis=1), axes=1)
    ipt = nf.fftshift(nf.ifft(nf.fftshift(imatmp2, axes=0), n=nheight, axis=0), axes=0)
    return ipt


def corrupt_on_run(ipt_complex, skull_mask, corrupt_dict=None):
    corrupt_type = corrupt_dict['corrupt_type']
    motion_rng = corrupt_dict['motion_rng']
    rand_list = corrupt_dict['rand_list']
    w_rng = corrupt_dict['w_rng']
    shift_h_rng = corrupt_dict['shift_h_rng']
    shift_w_rng = corrupt_dict['shift_w_rng']
    rotate_rng = corrupt_dict['rotate_rng']
    inputType = corrupt_dict['inputType']
    noise_level = corrupt_dict['noise_level']

    # we get complex data for ipt =  height - width - echo
    nheight, nwidth, nchannel = ipt_complex.shape
    nmotion = get_rand_int(motion_rng)
    ###################################
    #######      simulation    #######
    ###################################
    # image to ksp
    ksp = img2ksp(ipt_complex)
    #######   drop simulation   #######
    if corrupt_type == 'drop_ksp':
        for m in range(nmotion):
            w_start = rand_list[get_rand_int([0, len(rand_list) - 1])]
            w_end = w_start + get_rand_int(w_rng)
            ksp[:, w_start:w_end, :] = 0
    #######   drop simulation   #######
    if corrupt_type == 'replace_ksp':
        ksp_true = ksp.copy()
        for m in range(nmotion):
            w_start = rand_list[get_rand_int([0, len(rand_list) - 1])]
            w_end = w_start + get_rand_int(w_rng)
            shift = get_rand_int(shift_w_rng)
            ksp[:, w_start:w_end, :] = ksp_true[:, (w_start+shift):(w_end+shift), :]
    #######   shiftNrotate_img simulation   #######
    if corrupt_type == 'shiftNrotate_img':
        for m in range(nmotion):
            # shift image
            shift = [get_rand_int(shift_h_rng), get_rand_int(shift_w_rng), 0]
            rotate = get_rand_int(rotate_rng)
            ipt_complex_shiftnrotate = shiftnrotate(ipt_complex, shift, rotate)
            # ksp for shifted image
            ksp_shiftnrotate = img2ksp(ipt_complex_shiftnrotate)
            # replace the ksp
            w_start = rand_list[get_rand_int([0, len(rand_list) - 1])]
            w_end = w_start + get_rand_int(w_rng)
            ksp[:, w_start:w_end, :] = ksp_shiftnrotate[:, w_start:w_end, :]
    ###################################
    #######   finalize  #######
    ###################################
    # ksp to image
    ipt = ksp2img(ksp)
    # apply mask
    if skull_mask is not None:
        if len(ipt.shape) == len(skull_mask.shape):
            ipt = ipt * skull_mask
        else:
            ipt = np.squeeze(ipt) * np.squeeze(skull_mask)
    # get data
    if noise_level > 0:
        ipt = addwgn(ipt, noise_level)
    return ipt


def get_rand_int(data_range, size=None):
    rand_num = np.random.random_integers(data_range[0], data_range[1], size=size)
    return rand_num