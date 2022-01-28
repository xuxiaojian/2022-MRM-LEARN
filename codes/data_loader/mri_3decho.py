from data_loader import TFDatasetBase, compute_nearby_index
from data_loader.load_raw import *
from tools.util import *


class MRI3DEcho(TFDatasetBase):
    def __init__(self, all_config={}, mode='train'):

        self.mode = mode
        self.dataset_name   = 'mri_3decho'
        self.settings_type  = 'offrun_settings'

        self.inputType      = all_config['dataset']['inputType']
        self.outputType     = all_config['dataset']['outputType']
        self.rm_skull       = all_config['dataset']['rm_skull'][self.mode]
        self.norm           = all_config['dataset']['norm']
        self.R2s_norm       = all_config['dataset']['R2s_norm']
        self.mask_type = all_config['dataset']['mask_type'][self.mode]

        self.motion_rng   = all_config['dataset'][self.settings_type]['motion_rng']
        self.noise_level = all_config['dataset'][self.settings_type]['noise_level']

        ipt_label = all_config['dataset'][self.settings_type]['{}_ipt_label'.format(self.mode)]
        sample_batches      = all_config['dataset']['sample_batches']
        sampels_each_rate   = all_config['dataset']['sampels_each_rate'][mode]
        is_shuffle          = True if mode == 'train' else False

        # general parameters
        basic_dict = {}
        basic_dict['data_path']     = all_config['setting']['data_path']
        basic_dict['subj_indexes']  = all_config['dataset']['{}_subj_indexes'.format(mode)]
        basic_dict['slice_rng']     = all_config['dataset']['slice_rng']
        # special parameters
        gt_dict = {'fileType':'truth',  'dataType':self.outputType, 'dataDim':'NXYE', 'norm':self.norm, 'rm_skull': self.rm_skull, 'mask_type':self.mask_type}
        mask_dict = {'fileType':'mask', 'dataType':self.outputType, 'dataDim':'NXYEC',   'norm':None, 'rm_skull': self.rm_skull, 'mask_type':self.mask_type}

        print('=================================================================')
        print("[{}] Reading {} Data : ".format(self.dataset_name, mode))
        print('=================================================================')

        self.__ipt_files = []
        self.__ipt_names = []
        for rate_index in self.motion_rng:
            rate_ipts = []
            rate_name = []
            for sample_index in sampels_each_rate:
                ipt_dict = {'fileType': 'motion', 'dataType':self.outputType, 'dataDim': 'NXYE', 'norm': self.norm,
                            'rm_skull': self.rm_skull, 'rate_index':rate_index, 'sample_index':sample_index,'label': ipt_label, 'mask_type':self.mask_type}
                files, names = mri_complex(basic_dict=basic_dict, data_config=ipt_dict)
                rate_ipts.append(files)
                rate_name.append(names)
            self.__ipt_files.append(rate_ipts)
            self.__ipt_names.append(rate_name)

        self.__gt_files, self.__gt_names = mri_complex(basic_dict=basic_dict, data_config=gt_dict)
        if self.mask_type != 'none':
            self.__skullmask_files, _ = mri_complex(basic_dict=basic_dict, data_config=mask_dict)
        else:
            self.__skullmask_files = []
            for subj in self.__gt_files:
                self.__skullmask_files.append(np.ones(subj.shape + (1, )))

        # [[[c1-s1-subj1, ..., c1-s1-subjn], [c1-s2], [c1-s3]],
        #  [[c2-s1], [c2-s2], [c2-s3]],
        # ]
        self.__indexes_map = []
        for rate_idx in range(len(self.__ipt_files)):
            num_sample = len(self.__ipt_files[rate_idx])
            for sample_idx in range(num_sample):
                num_subj = len(self.__ipt_files[rate_idx][sample_idx])
                for subj_idx in range(num_subj):
                    num_slice = self.__ipt_files[rate_idx][sample_idx][subj_idx].shape[0]
                    for slice_idx in range(num_slice):
                        self.__indexes_map.append([rate_idx, sample_idx, subj_idx, slice_idx])
        super().__init__(is_shuffle=is_shuffle, sample_index=sample_batches)

    def __len__(self):
        return self.__indexes_map.__len__()

    def __getitem__(self, item):
        rate_idx, sample_idx, subj_idx, slice_idx = self.__indexes_map[item]
        ipt= self.__ipt_files[rate_idx][sample_idx][subj_idx][slice_idx] # 256*192*10
        gt = self.__gt_files[subj_idx][slice_idx]  # 256*192*10
        skull_mask = self.__skullmask_files[subj_idx][slice_idx]  # 256*192*10*1

        ipt = make_channels(ipt, inputType=self.inputType, outputType=self.outputType, new_dim=True)
        gt = make_channels(gt, inputType=self.inputType, outputType=self.outputType, new_dim=True)

        ft_palce_holder = np.ones(ipt.shape)
        return ipt.astype(np.float32), gt.astype(np.float32), ft_palce_holder.astype(np.float32), skull_mask.astype(np.float32)

    def getitem_names(self, item):
        rate_idx, sample_idx, subj_idx, slice_idx = self.__indexes_map[item]
        return self.__ipt_names[rate_idx][sample_idx][subj_idx],\
               self.__gt_names[subj_idx]

# get the dataset
def mri_3decho(config, mode: str):
    assert mode == 'train' or mode == 'test', 'mode is not train or test'
    if mode == 'train':
        train_dataset = MRI3DEcho(all_config=config, mode='train')
        valid_dataset = MRI3DEcho(all_config=config, mode='valid')
        return train_dataset, valid_dataset
    if mode == 'test':
        test_dataset = MRI3DEcho(all_config=config, mode='test')
        return test_dataset
