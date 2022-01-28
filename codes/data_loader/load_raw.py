import h5py
import numpy as np
import nibabel as nib
from glob import glob
import scipy.io as sio
from tools.util import norm_data
from scipy.io import loadmat
from shutil import copyfile

####### inputs #######:
# truth/motion/ft : 256 * 192 * 72 * 10
# R2s/t1w/mask: 256 * 192 * 72
####### outputs #######:
# outputs: truth/motion/ft: 2decho: 72 * 256 * 192 * 10 [* channle]
# outputs: R2s/t1w/mask: 256 * 192 * 72 * 1
def __general_precessing(ipt, fileType='truth', subjname='C01_V2', dataType='complex', dataDim='nxye',
                         slice_rng={'60': [19, 50]}, norm='midcube0', mask=None, config=None):

    ####### reshape data (256 * 192 * 72 * 10 /256 * 192 * 72  --> 72 * 256 * 192 * 10/72 * 256 * 192) #######
    print('data before processing:', ipt.shape)
    ipt = np.swapaxes(np.swapaxes(ipt, 0, 2), 1, 2)

    ####### normalize data (72 * 256 * 192 * 10/2) #######
    if fileType == 'truth' or fileType == 'motion' or fileType == 'ft':
        num_slice, num_height, num_width, num_echo = ipt.shape
        ipt = norm_data(ipt, method=norm, subjname=subjname, config=config)
    elif fileType =='t1w' or fileType =='R2s' or fileType =='mask':
        ipt = np.expand_dims(ipt, axis=-1)
        num_slice, num_height, num_width, num_channel = ipt.shape
        ipt = norm_data(ipt, method=norm, subjname=subjname, config=config)

    ####### maskout data (72 * 256 * 192 * 10/2) #######
    if mask is not None:
        mask = np.swapaxes(np.swapaxes(mask, 0, 2), 1, 2)[..., None]
        ipt = ipt * mask
    ###### crop the data (72 * 256 * 192 * ? --> x * 256 * 192 * ?) #######
    slice_rng_subj = slice_rng[str(num_slice)]
    ipt = ipt[slice_rng_subj[0]:slice_rng_subj[1]]

    ###### get the correct dataType (x * 256 * 192 * ?) #######
    if fileType == 'truth' or fileType == 'motion' or fileType == 'ft':
        if dataDim == 'NXYE':
            opt = ipt
        elif dataDim == 'NeXYC':
            opt = ipt.transpose(0, 3, 1, 2) # n * e * 256 * 192
            opt = opt.reshape((-1, num_height, num_width, 1), order='F')
        else:
            print('dataDim not found!')
            exit(1)
    elif fileType == 't1w' or fileType =='R2s': #  or fileType =='mask'
        if dataDim == 'NXYC':
            opt = ipt
        else:
            print('dataDim not found!')
            exit(1)
    elif fileType =='mask':
        if dataDim == 'NXYC': #or dataDim == 'nxye'
            opt = ipt
        elif dataDim == 'NeXYC':
            opt = np.repeat(ipt, 10, axis=-1)
            opt = opt.transpose(0, 3, 1, 2)  # 72 * 256 * 192 * 1
            opt = opt.reshape((-1, num_height, num_width, 1), order='F')
        elif dataDim == 'NXYEC':
            opt = np.repeat(ipt, 10, axis=-1)[..., None]
        else:
            print('dataDim not found!')
            exit(1)
    else:
        print('fileType not found!')
        exit(1)

    if opt.shape[2] == 190:
        opt = np.concatenate((opt, opt[:,:,-2:,:]), axis=2)

    # opt = np.abs(opt) if dataType == 'magnitude' else opt # .astype(np.float32)
    print('data after processing:', opt.shape)
    return opt

###########################################
# Biomedical model loading
###########################################
def mri_complex(basic_dict={},data_config={}):
    # basic_dict
    data_path = basic_dict['data_path']
    subj_indexes = basic_dict['subj_indexes'] #list
    slice_rng = basic_dict['slice_rng'] #list

    # data_config
    fileType = data_config['fileType']
    dataType = data_config['dataType']
    dataDim = data_config['dataDim']
    norm = data_config['norm']
    rm_skull = data_config['rm_skull']
    mask_type = data_config['mask_type']

    subj_files = []
    subj_names = []
    # output = [subj0, ....subjn]


    for subj in subj_indexes:
        if rm_skull:
            if mask_type == 'Braincalmask_New_':
                mask = loadmat(data_path + 'truth/{}/{}{}.mat'.format(subj, mask_type, subj))['Mask']  # 256*192*72
            elif mask_type == 'WholeBrainmask':
                mask = loadmat(data_path + 'truth/{}/{}{}.mat'.format(subj, mask_type, subj))['braincalmsk']  # 256*192*72
            else:
                print('no mask_type found')
                exit(0)
        else:
            mask = None
        if fileType == 'motion':
            rate_index = data_config['rate_index']
            label = data_config['label']
            label_temp = label.format(rate_index)
            files_all = glob(data_path + 'motion/{}/ima_comb_{}{}.mat'.format(subj, subj, label_temp))
            files_all.sort()
            sample_index = data_config['sample_index']
            name = files_all[sample_index]
            data = loadmat(files_all[sample_index])['ima_comb']
            data = __general_precessing(data, fileType='motion', subjname=subj, dataType=dataType, dataDim=dataDim,
                                        slice_rng=slice_rng, norm=norm, mask=mask, config=basic_dict)
        elif fileType == 'truth':
            files_path = data_path + 'truth/{}/ima_comb_{}.mat'.format(subj, subj)
            name = files_path
            data = loadmat(files_path)['ima_comb']
            data = __general_precessing(data, fileType='truth', subjname=subj, dataType=dataType, dataDim=dataDim,
                                        slice_rng=slice_rng, norm=norm, mask=mask, config=basic_dict)
        elif fileType == 'ft':
            files_path = data_path + 'truth/{}/Ffun1st-{}.mat'.format(subj, subj)
            name = files_path
            data = loadmat(files_path)['F_norm']
            data = abs(data)
            data[data > 1.3] = 1.0
            data = __general_precessing(data, fileType='ft',  subjname=subj, dataType=dataType, dataDim=dataDim,
                                        slice_rng=slice_rng, norm=None, mask=mask, config=basic_dict)
        elif fileType == 'S0R2s':
            files_path = data_path + 'truth/{}/R2s1st-{}.mat'.format(subj, subj)
            fileKeys = data_config['fileKeys']
            name = files_path
            data = np.array([])
            for key in fileKeys:
                name += key
                tmp_data = loadmat(files_path)[key]
                tmp_data = tmp_data/1e6 if key=='t1w' else tmp_data
                tmp_data = __general_precessing(tmp_data, fileType=key, subjname=subj, dataType=dataType, dataDim=dataDim,
                                            slice_rng=slice_rng, norm=norm[key], mask=mask, config=basic_dict)
                data = np.concatenate((data, tmp_data), axis=-1) if data.size else tmp_data
        elif fileType == 'mask':
            if mask_type == 'Braincalmask_New_':
                files_path = data_path + 'truth/{}/{}{}.mat'.format(subj, mask_type, subj)
                data = loadmat(files_path)['Mask']
            elif mask_type == 'WholeBrainmask':
                files_path = data_path + 'truth/{}/{}{}.mat'.format(subj, mask_type, subj)
                data = loadmat(files_path)['braincalmsk']
            else:
                print('no mask_type found')
                exit(0)
            name = files_path
            data = __general_precessing(data, fileType='mask', subjname=subj, dataType=dataType, dataDim=dataDim,
                                        slice_rng=slice_rng, norm=None, mask=None, config=basic_dict)
        else:
            print("mri_complex: wrong type of mri_complex data.")
            exit(1)
        print("Loaded MatFile Pointer in Path: {} \n".format(name))
        subj_files.append(data)
        subj_names.append(name)

    return subj_files, subj_names
