# [Learning-based Motion Artifact Removal Networks (LEARN) for Quantitative  Mapping](https://arxiv.org/pdf/2109.01622.pdf)

[![Watch the video](https://github.com/xuxiaojian/2022-MRM-LEARN/blob/main/examples/gif.gif)](https://youtu.be/2DVl2lS-dbU)

![Alt Text](https://github.com/xuxiaojian/2022-MRM-LEARN/blob/main/examples/gif.gif)







## Abstract

**Purpose**: To introduce two novel learning-based motion artifact removal networks (LEARN) for the estimation of quantitative motion- and B0-inhomogeneity-corrected R2* maps from motion-corrupted multi-Gradient-Recalled Echo (mGRE) MRI data.

**Methods**: We train two convolutional neural networks (CNNs) to correct motion artifacts for high-quality estimation of quantitative B0-inhomogeneity-corrected R2* maps from mGRE sequences. The first CNN, LEARN-IMG, performs motion correction on complex mGRE images, to enable the subsequent computation of high-quality motion-free quantitative R2* (and any other mGRE-enabled) maps using the standard voxel-wise analysis or machine-learning-based analysis. The second CNN, LEARN-BIO, is trained to directly generate motion- and B0-inhomogeneity-corrected quantitative R2* maps from motion-corrupted magnitude-only mGRE images by taking advantage of the biophysical model describing the mGRE signal decay. We show that both CNNs trained on synthetic MR images are capable of suppressing motion artifacts while preserving details in the predicted quantitative R2* maps. Significant reduction of motion artifacts on experimental in vivo motion-corrupted data has also been achieved by using our trained models.

**Conclusion**: Both LEARN-IMG and LEARN-BIO can enable the computation of high-quality motion- and B0-inhomogeneity-corrected R2* maps. LEARN-IMG performs motion correction on mGRE images and relies on the subsequent analysis for the estimation of R2* maps, while LEARN-BIO directly performs motion- and B0-inhomogeneity-corrected R2* estimation. Both LEARN-IMG and LEARN-BIO jointly process all the available gradient echoes, which enables them to exploit spatial patterns available in the data. The high computational speed of LEARN-BIO is an advantage that can lead to a broader clinical application.

**Authored by**: Xiaojian Xu, Satya V.V.N. Kothapalli, Jiaming Liu, Sayan Kahali, Weijie Gan, Dmitriy A. Yablonskiy, Ulugbek S. Kamilov

## How to run the code

### Models

  Four pre-trained models can be downloaded from [Google drive](https://drive.google.com/drive/folders/1qqBGX0-cI-2Jck3tGnT1OjCkIuTTS-To?usp=sharing). Once downloaded, place them into `./results`. The detail information of the provided models are listed below:

  - **LEARN-BIO-finetuned_2021-11-01-12-27-38_mri_2dechoft_bnloss_relu/model/best-snr.h5**
    - This is a LEARN-BIO model trained on 2d motion (in-plane translationa and rotation) data and fintuned on 3d motion (in-plane translatiob and 3d roation).
  
  - **LEARN-BIO-warmup_2021-02-20-00-22-11_mri_2dechoft_bnloss_relu/model/latest.h5** 
    - This is a LEARN-BIO model trained on 2d motion (in-plane translationa and rotation) data.
  
  - **LEARN-IMG-finetuned_2021-10-27-02-35-53_mri_3decho_bnloss/model/best-snr.h5**
    - This is a LEARN-IMG model trained on 2d motion (in-plane translationa and rotation) data and fintuned on 3d motion (in-plane translatiob and 3d roation).
  
  - **LEARN-IMG-warmup_2021-02-24-03-56-55_mri_3decho_bnloss/model/best-snr.h5**
    - This is a LEARN-IMG model trained on 2d motion (in-plane translationa and rotation) data .

 

### Data
  Data of two examplar subjects (017_9990 and C08_V2) can be downloaded from [Google drive](https://drive.google.com/drive/folders/1qqBGX0-cI-2Jck3tGnT1OjCkIuTTS-To?usp=sharing). Once downloaded, place them into `./data`. The detail information of the provided data are listed below:
  - **motion/017_9990/**: This is experimental motion-corrupted data. Folder only contains mGRE data.
  - **motion/C08_V2/**: This is simulated motion-corupted data. Folder only contains mGRE data. The name indiates how each data is simulated. For example, 
    - [snr2d_mid] represents the motion is simulated with 2D inplane shfit and roatation wihle [snr3d_midnew] represent (in-plane shift and 3d roation). 
    - [n3], [n6], and [n9] represents light, moderate and heavy motion, respectively.
  
  - **truth/017_9990/**: This is an experimental motion-corrupted data. Folder contains mGRE data, F(t) function, mask and S0 and R2* (see table below for more details). Note that the mGRE data in this folder is identity to the one in **motion/017_9990/**. 
  - **truth/C08_V2/**: This is an simulated motion-corrupted data. Folder contains mGRE data, F(t) function, mask and S0 and R2* (see table below for more details).

  |Files|Explaination|
  |---|:--:|
  |Braincalmask_New_X.mat| Brain mask ||
  |Ffun1st-X.mat| F(t) function||
  |ima_comb_X.mat| mGRE data||
  |monoexpo_X.mat| S0 (namespace: t1w) and R2* (namespace: ) ccomputed using NLLS from mGRE data||


### Dependencies instllation
- Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- Install the dependencies using the given learn.yml file.
  ```
  $ conda env create -f learn.yml
  ```
- Activate the enviorment
    ```
    $ conda activate learn.yml
    ```

### Test with pretrained models
The configuration for running different subjects data and models are provided in folder [configs]. The current demo tests LEARN-BIO-finedtuned model on experimental data 017_9990, to run such a demo, 
- open config_test_LEARN-BIO.json file, and change the following fields in the config files to your local path: "root_path", "data_path",  "src_path".
-  in your terminal, type
  ```
  $ python main.py
  ```
- The test results will be stored in the ./results/model_name/model_name_test folder.

More generally, to try different models, please change the following instructions. 
  - Open 'config_test_LEARN-X.json' file, where this file  is 'config_test_LEARN-BIO.json' for a LEARN_BIO model and 'config_test_LEARN-IMG.json' for a LEARN-IMG model.
  - In 'config_test_LEARN-X.json' file, set "save_folder" and the corresponding "weight_file" to the pre-trained model you want to test.
- modify [config_file_name] to 'config_test_LEARN-X.json' in the in main.py.

### Train with examplar data
You can also train your own model using our codes. We here illustrate the procedure with our simulated data C08_V2.
  - Open 'config_train_LEARN-X.json' file, where this file is 'config_train_LEARN-BIO.json' for training a LEARN_BIO model and 'config_train_LEARN-IMG.json' for a LEARN-IMG model.
  - Change the data to your data, as we only have one examplar data have set
      - "train_subj_indexes": ["C08_V2"],
      - "valid_subj_indexes": ["C08_V2"],
 - To train your model on 2D motion data, set "train_ipt_label": "_nrand1-10_snr2d_mid". 
 - [Optional] To finetune your model on 3D motion data, set "train_ipt_label": "_nrand1-10_snr3d_midnew", "restore": true and "restore_folder" to its inital models, e.g., "restore_folder":"LEARN-BIO-warmup_2021-02-20-00-22-11_mri_2dechoft_bnloss_relu".
 -  in the in main.py fille, modify [config_file_name] to 'config_train_LEARN-X.json', e.g., change it to 'config_train_LEARN-IMG.json' for testing LEARN-IMG model.
- just type
  ```
  $ python main.py
  ```
- The test results will be stored in the ./results/ folder.
