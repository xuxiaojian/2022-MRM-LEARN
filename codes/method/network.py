import tensorflow as tf
from tools.util import tf_snr, copytree_code, check_and_mkdir, dict_to_md_table, make_3d_grid, to_mat_np
import numpy as np
from tqdm import tqdm
import tensorflow.keras.backend as k
from data_loader import TFDatasetBase
from tensorflow.python.client import timeline
from .model import unet_3d, unet_2d
from .multi_gpu_tools import average_op, average_gradients, assign_to_device
from datetime import datetime
from scipy.io import savemat
import numpy.fft as nf
import csv

class Network:
    def __init__(self, config):
        self.config = config
        self.now = datetime.now()
        self.te = tf.convert_to_tensor(np.array(range(4, 44, 4)) / 1e3, dtype="float32")

    # @staticmethod
    def metrics(self, predict, ground_truth):
        # only compare magnitue
        if self.config['dataset']['outputType'] == 'complex':
            if predict.shape[-1] == 20:
                predict = tf.concat([predict[..., :10][..., None], predict[..., 10:][..., None]], -1)
                ground_truth = tf.concat([ground_truth[..., :10][..., None], ground_truth[..., 10:][..., None]], -1)
            predict = tf.expand_dims(tf.abs(tf.dtypes.complex(predict[..., 0], predict[..., 1])), axis=-1)
            ground_truth = tf.expand_dims(tf.abs(tf.dtypes.complex(ground_truth[..., 0], ground_truth[..., 1])), axis=-1)
        if len(predict.shape)>4 or len(ground_truth.shape)>4:
            predict = tf.squeeze(predict)
            ground_truth = tf.squeeze(ground_truth)
        psnr = tf.reduce_mean(tf.image.psnr(predict, ground_truth, max_val=1))
        ssim = tf.reduce_mean(tf.image.ssim(predict, ground_truth, max_val=1))
        snr  = tf_snr(pre=predict, y=ground_truth)
        return psnr, ssim, snr

    def getbiomed(self, data, ft, mask):
        S0_Pred = data[..., 0, None] #* mask
        R2s_Pred = data[..., 1, None] #* mask ### remember
        physical_data = tf.multiply(tf.multiply(tf.math.exp(tf.multiply(tf.math.negative(R2s_Pred), self.te)), S0_Pred), ft)
        return physical_data

    def changeto5d(self, data, compare=None, dtype=['complex', 'complex']):
        if data.shape[-1] == 20: # make n x y 20 to n x y 10 2
            data = tf.concat([data[..., :10][..., None], data[..., 10:][..., None]], -1)
            compare = tf.concat([compare[..., :10][..., None], compare[..., 10:][..., None]], -1)

        dataset = self.config['setting']['dataset']
        data_mag = tf.cast(tf.expand_dims(tf.abs(tf.dtypes.complex(data[..., 0], data[..., 1])), axis=-1), dtype=tf.float32) \
            if dtype[0] == 'complex' else data
        compare_mag = tf.cast(tf.expand_dims(tf.abs(tf.dtypes.complex(compare[..., 0], compare[..., 1])), axis=-1), dtype=tf.float32) \
            if dtype[1] == 'complex' else compare

        ## let's don't show the complex part
        compare_mag = data_mag if data_mag.shape[1:] != compare_mag.shape[1:] else compare_mag
        err_mag = data_mag - compare_mag
        if len(data_mag.shape) == 4:
            if data_mag.shape[-1] == 10 or data_mag.shape[-1] == 20:
                # mri_2decho: n * height * width * echo
                # mri_2dechorr(ipt): n * height * width * echo
                data = tf.transpose(tf.expand_dims(data, axis=-1), [0, 3, 1, 2, 4])
                data_mag = tf.transpose(tf.expand_dims(data_mag, axis=-1), [0, 3, 1, 2, 4])
                err_mag = tf.transpose(tf.expand_dims(err_mag, axis=-1), [0, 3, 1, 2, 4])
        elif len(data_mag.shape) == 5:
            # mri_3decho: n * height1 * width2 * echo * channel
            data = tf.transpose(data, [0, 3, 1, 2, 4])
            data_mag = tf.transpose(data_mag, [0, 3, 1, 2, 4])
            err_mag = tf.transpose(err_mag, [0, 3, 1, 2, 4])
        if dtype[0] == 'complex':
            data_mag = tf.concat([data, data_mag, err_mag], -1)
        else:
            data_mag = tf.concat([data_mag, err_mag], -1)
        return data_mag

    def test(self, test_dataset: TFDatasetBase):
        print("Test_dataset: Shape= %s Dtype= %s Length= %d" % (
            str(test_dataset.shape),
            str(test_dataset.dtype),
            test_dataset.__len__()))

        batch_size  = self.config['test']['batch_size']
        model_path  = self.config['setting']['root_path'] + self.config['setting']['save_folder'] \
                      + '/model/' + self.config['test']['weight_file']
        description = self.config['test']['description']

        if self.config['method']['model'] == 'unet3d':
            kernel_size = self.config['method']['unet3d']['kernel_size']
            filters_root = self.config['method']['unet3d']['filters_root']
            conv_times = self.config['method']['unet3d']['conv_times']
            up_down_times = self.config['method']['unet3d']['up_down_times']
            if_relu = self.config['method']['unet3d']['if_relu']
            if_residule = self.config['method']['unet3d']['if_residule']
            model = unet_3d(input_shape=test_dataset.shape[0],
                            output_channel=self.config['method']['unet3d']['output_channel'],
                            kernel_size=kernel_size,
                            filters_root=filters_root,
                            conv_times=conv_times,
                            up_down_times=up_down_times,
                            if_relu = if_relu,
                            if_residule = if_residule)

        if self.config['method']['model'] == 'unet2d':
            kernel_size = self.config['method']['unet2d']['kernel_size']
            filters_root = self.config['method']['unet2d']['filters_root']
            conv_times = self.config['method']['unet2d']['conv_times']
            up_down_times = self.config['method']['unet2d']['up_down_times']
            if_relu = self.config['method']['unet2d']['if_relu']
            if_residule = self.config['method']['unet2d']['if_residule']
            model = unet_2d(input_shape=test_dataset.shape[0],
                            output_channel= self.config['method']['unet2d']['output_channel'], #2, #test_dataset.shape[1][-1],  # train_dataset.shape[1][-1], 10-->2, 10-->10
                            kernel_size=kernel_size,
                            filters_root=filters_root,
                            conv_times=conv_times,
                            up_down_times=up_down_times,
                            if_relu=if_relu,
                            if_residule = if_residule)

        # read data
        dataset = self.config['setting']['dataset']
        ipt_op, _, _, _ = test_dataset(batch_size=len(test_dataset))

        # start loading the model
        with tf.Session() as sess:
            ipt = sess.run([ipt_op])
            model.load_weights(model_path)
            pre = model.predict(ipt, batch_size=batch_size, verbose=1)
            if dataset == 'mri_3decho':
                pre = pre[..., 0] + 1j * pre[..., 1]  # N * 256 * 192 * 10
                pre = np.transpose(pre, axes=(1, 2, 0, 3))
            if dataset == 'mri_2dechoft': # n x y 2 -> x, y, n, 2
                pre = np.transpose(pre, axes=(1, 2, 0, 3))

            # For real motion data, input data was padded, we shall remvoed the padded part.
            # For simulated data, input data was NOT padded, therefore comment the following line.
            pre = pre[:,:190,:,:]

        ##########################
        ###### save data #########
        ##########################
        dst_path = self.config['setting']['root_path'] + self.config['setting']['save_folder'] + '/'
        rootpath = dst_path + '{}_test/'.format(self.config['setting']['save_folder'])
        check_and_mkdir(rootpath)
        rootpath = rootpath + '/' + self.config['dataset']['test_subj_indexes'][0] + '/'
        check_and_mkdir(rootpath)

        dst_file_name = self.config['setting']['save_folder'] + '-' + self.config['test']['weight_file'] + '-' + description + '-' + \
                     test_dataset.getitem_names(0)[0].replace(self.config['setting']['data_path'], '').replace('/', '_').replace('.mat', '')
        savemat(rootpath + dst_file_name + '-pre.mat', {'pre': pre})
        print('[Save Mat]: ' + dst_file_name + '-pre.mat')
        return

    def train(self, train_dataset: TFDatasetBase, valid_dataset: TFDatasetBase, test_dataset: TFDatasetBase):
        print("train_dataset: Shape= %s Dtype= %s Length= %d" % (
            str(train_dataset.shape),
            str(train_dataset.dtype),
            train_dataset.__len__())
              )

        print("valid_dataset: Shape= %s Dtype= %s Length= %d" % (
            str(valid_dataset.shape),
            str(valid_dataset.dtype),
            valid_dataset.__len__())
              )

        print("test_dataset: Shape= %s Dtype= %s Length= %d" % (
            str(test_dataset.shape),
            str(test_dataset.dtype),
            test_dataset.__len__())
              )

        ##########################
        # Read Config
        ##########################
        batch_size    = self.config['train']['batch_size']
        learning_rate = self.config['train']['learning_rate']
        runtime_batch = self.config['train']['runtime_batch']
        train_epoch   = self.config['train']['train_epoch']
        save_epoch    = self.config['train']['save_epoch']
        src_path      = self.config['train']['src_path']
        lossType      = self.config['train']['lossType']
        dataset       = self.config['setting']['dataset']
        rm_skull      = self.config['dataset']['rm_skull']

        num_gpu = np.fromstring(self.config['setting']['gpu_index'], dtype=np.int, sep=',').__len__()
        print("Number of GPUs: %d" % num_gpu)

        ##########################
        # Build Path and Copy Code
        ##########################
        if self.config['setting']['restore']:
            restore_path = self.config['setting']['root_path'] + '%s/'%self.config['setting']['restore_folder']
            model_restore_name = self.config['setting']['model_name']
            model_restore_path = restore_path + 'model/' + model_restore_name
            global_epoch = self.config['setting']['global_epoch'] + 1
            global_batch = self.config['setting']['global_batch'] + 1
            save_path = self.config['setting']['root_path'] + '%s_%s/' % (str(self.now.strftime("%Y-%m-%d-%H-%M-%S")), self.config['setting']['save_folder'])
        else:
            save_path = self.config['setting']['root_path'] + '%s_%s/' % (str(self.now.strftime("%Y-%m-%d-%H-%M-%S")), self.config['setting']['save_folder'])
            global_batch = 0
            global_epoch = 0

        check_and_mkdir(save_path)
        model_path = save_path + 'model/'
        check_and_mkdir(model_path)
        timeline_path = save_path + 'timeline/'
        check_and_mkdir(timeline_path)
        valid_path = save_path + 'valid/'
        check_and_mkdir(valid_path)

        copytree_code(save_path, src_path)

        ##########################
        # Get TFTensor Operator
        ##########################
        dy_lr = tf.placeholder(tf.float32)
        with tf.device('/cpu:0'):
            tower_grads = []
            tower_psnr  = []
            tower_ssim  = []
            tower_snr   = []
            tower_loss_tl = []
            if self.config['method']['model'] == 'unet3d':
                kernel_size   = self.config['method']['unet3d']['kernel_size']
                filters_root  = self.config['method']['unet3d']['filters_root']
                conv_times    = self.config['method']['unet3d']['conv_times']
                up_down_times = self.config['method']['unet3d']['up_down_times']
                if_relu = self.config['method']['unet3d']['if_relu']
                if_residule = self.config['method']['unet3d']['if_residule']
                model = unet_3d(input_shape=train_dataset.shape[0],
                                output_channel=self.config['method']['unet3d']['output_channel'],
                                kernel_size=kernel_size,
                                filters_root=filters_root,
                                conv_times=conv_times,
                                up_down_times=up_down_times,
                                if_relu = if_relu,
                                if_residule = if_residule)

            if self.config['method']['model'] == 'unet2d':
                kernel_size = self.config['method']['unet2d']['kernel_size']
                filters_root = self.config['method']['unet2d']['filters_root']
                conv_times = self.config['method']['unet2d']['conv_times']
                up_down_times = self.config['method']['unet2d']['up_down_times']
                if_relu = self.config['method']['unet2d']['if_relu']
                if_residule = self.config['method']['unet2d']['if_residule']
                model = unet_2d(input_shape=train_dataset.shape[0],
                                output_channel= self.config['method']['unet2d']['output_channel'], #train_dataset.shape[1][-1], #train_dataset.shape[1][-1], 10-->2, 10-->10
                                kernel_size=kernel_size,
                                filters_root=filters_root,
                                conv_times=conv_times,
                                up_down_times=up_down_times,
                                if_relu = if_relu,
                                if_residule= if_residule)
            model.summary()
            
            # read data
            tri_x, tri_gt, tri_ft, tri_mask = train_dataset(batch_size * num_gpu)
            val_x, val_gt, val_ft, val_mask = valid_dataset(batch_size)

            tri_x_sample, tri_gt_sample, tri_ft_sample, tri_mask_sample = train_dataset.sample()
            val_x_sample, val_gt_sample, val_ft_sample, val_mask_sample = valid_dataset.sample()
            tst_x_sample, tst_gt_sample, tst_ft_sample, tst_mask_sample = test_dataset.sample()

            for i in range(num_gpu):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    tri_mask_per_gpu = tri_mask[i * batch_size: (i + 1) * batch_size]
                    tri_gt_per_gpu = tri_gt[i * batch_size: (i + 1) * batch_size]
                    tri_x_per_gpu = tri_x[i * batch_size: (i+1) * batch_size]
                    tri_x_pre_per_gpu = model(tri_x_per_gpu)
                    if dataset=='mri_2dechoft':
                        tri_ft_per_gpu = tri_ft[i * batch_size: (i + 1) * batch_size]
                        tri_x_pre_per_gpu = self.getbiomed(tri_x_pre_per_gpu, tri_ft_per_gpu, None)

                    tri_psnr_batch, tri_ssim_batch, tri_snr_batch = self.metrics(predict=tri_x_pre_per_gpu * tri_mask_per_gpu, ground_truth=tri_gt_per_gpu * tri_mask_per_gpu)
                    tower_psnr.append(tri_psnr_batch)
                    tower_ssim.append(tri_ssim_batch)
                    tower_snr.append(tri_snr_batch)

                    if lossType == 'l1':
                        tri_loss_tl = tf.losses.absolute_difference(labels=tri_gt_per_gpu * tri_mask_per_gpu, predictions=tri_x_pre_per_gpu * tri_mask_per_gpu)
                    elif lossType == 'l2':
                        tri_loss_tl = tf.losses.mean_squared_error(labels=tri_gt_per_gpu * tri_mask_per_gpu, predictions=tri_x_pre_per_gpu * tri_mask_per_gpu)

                    tower_loss_tl.append(tri_loss_tl)
                    optimizer = tf.train.AdamOptimizer(learning_rate=dy_lr)
                    grads = optimizer.compute_gradients(tri_loss_tl)
                    tower_grads.append(grads)
                    if i == 0:
                        val_pre = model(val_x)
                        val_to_com = val_gt
                        if dataset=='mri_2dechoft':
                            val_pre = self.getbiomed(val_pre, val_ft, None)

                        val_psnr_batch, val_ssim_batch, val_snr_batch = self.metrics(
                            predict=val_pre * val_mask, ground_truth=val_to_com* val_mask)
                        val_psnr_epoch, val_psnr_epoch_update = tf.metrics.mean(val_psnr_batch)
                        val_ssim_epoch, val_ssim_epoch_update = tf.metrics.mean(val_ssim_batch)
                        val_snr_epoch,  val_snr_epoch_update  = tf.metrics.mean(val_snr_batch)

                        tri_pre_sample = model(tri_x_sample) * tri_mask_sample
                        val_pre_sample = model(val_x_sample) * val_mask_sample
                        tst_pre_sample = model(tst_x_sample) * tst_mask_sample

        tower_psnr = average_op(tower_psnr)
        tower_ssim = average_op(tower_ssim)
        tower_snr  = average_op(tower_snr)

        tri_psnr_epoch, tri_psnr_epoch_update = tf.metrics.mean(tower_psnr)
        tri_ssim_epoch, tri_ssim_epoch_update = tf.metrics.mean(tower_ssim)
        tri_snr_epoch, tri_snr_epoch_update = tf.metrics.mean(tower_snr)
        tri_metrics_epoch_update = [tri_psnr_epoch_update, tri_ssim_epoch_update, tri_snr_epoch_update]

        tower_loss_tl = average_op(tower_loss_tl)
        tri_loss_tl_epoch, tri_loss_tl_epoch_update = tf.metrics.mean(tower_loss_tl)
        tri_loss_epoch_update = [tri_loss_tl_epoch_update]
        tri_epoch_update = tri_metrics_epoch_update + tri_loss_epoch_update
        val_epoch_update = [val_psnr_epoch_update, val_ssim_epoch_update, val_snr_epoch_update]
        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)

        # Model Save Op.
        model_metrics = [val_psnr_epoch, val_ssim_epoch, val_snr_epoch]
        model_tags = ['psnr', 'ssim', 'snr']

        model_best = np.zeros(shape=(len(model_tags),))
        model_epoch = np.zeros(shape=(len(model_tags),))
        model_info = tf.placeholder(dtype=tf.string)

        ##########################
        # Set Summary
        ##########################
        tf.summary.text(name='init/config_pnp.json', tensor=tf.constant(dict_to_md_table(self.config)))
        tf.summary.text(name='model/save_info', tensor=model_info)

        tf.summary.scalar(name='batch/train/psnr', tensor=tower_psnr)
        tf.summary.scalar(name='batch/train/ssim', tensor=tower_ssim)
        tf.summary.scalar(name='batch/train/snr', tensor=tower_snr)

        tf.summary.scalar(name='batch/train/loss_tl', tensor=tower_loss_tl)
        tf.summary.scalar(name='batch/train/lr', tensor=dy_lr)

        tf.summary.scalar(name='epoch/train/psnr', tensor=tri_psnr_epoch)
        tf.summary.scalar(name='epoch/train/ssim', tensor=tri_ssim_epoch)
        tf.summary.scalar(name='epoch/train/snr', tensor=tri_snr_epoch)

        tf.summary.scalar(name='epoch/train/loss_tl', tensor=tri_loss_tl_epoch)

        tf.summary.scalar(name='epoch/valid/psnr', tensor=val_psnr_epoch)
        tf.summary.scalar(name='epoch/valid/ssim', tensor=val_ssim_epoch)
        tf.summary.scalar(name='epoch/valid/snr', tensor=val_snr_epoch)

        tf.summary.image(name='init/train/ipt', tensor=make_3d_grid(img_tensor=self.changeto5d(tri_x_sample, compare=tri_gt_sample, dtype=[self.config['dataset']['inputType'], self.config['dataset']['outputType']])), max_outputs=20)
        tf.summary.image(name='init/train/gt', tensor=make_3d_grid(img_tensor=self.changeto5d(tri_gt_sample, compare=tri_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])), max_outputs=20)

        tf.summary.image(name='init/valid/ipt', tensor=make_3d_grid(img_tensor=self.changeto5d(val_x_sample, compare=val_gt_sample, dtype=[self.config['dataset']['inputType'], self.config['dataset']['outputType']])), max_outputs=20)
        tf.summary.image(name='init/valid/gt', tensor=make_3d_grid(img_tensor=self.changeto5d(val_gt_sample, compare=val_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])), max_outputs=20)

        tf.summary.image(name='init/test/ipt', tensor=make_3d_grid(img_tensor=self.changeto5d(tst_x_sample, compare=tst_gt_sample, dtype=[self.config['dataset']['inputType'], self.config['dataset']['outputType']])), max_outputs=20)
        tf.summary.image(name='init/test/gt', tensor=make_3d_grid(img_tensor=self.changeto5d(tst_gt_sample, compare=tst_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])),max_outputs=20)

        tf.summary.image(name='epoch/train/pre', tensor=make_3d_grid(img_tensor=self.changeto5d(tri_pre_sample, compare=tri_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])), max_outputs=20)
        tf.summary.image(name='epoch/valid/pre', tensor=make_3d_grid(img_tensor=self.changeto5d(val_pre_sample, compare=val_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])), max_outputs=20)
        tf.summary.image(name='epoch/test/pre', tensor=make_3d_grid(img_tensor=self.changeto5d(tst_pre_sample, compare=tst_gt_sample, dtype=[self.config['dataset']['outputType'], self.config['dataset']['outputType']])), max_outputs=20)

        tf.summary.histogram(name='init/train/ipt', values=tri_x_sample)
        tf.summary.histogram(name='init/train/gt', values=tri_gt_sample)
        tf.summary.histogram(name='init/valid/ipt', values=val_x_sample)
        tf.summary.histogram(name='init/valid/gt', values=val_gt_sample)

        tf.summary.histogram(name='epoch/train/pre', values=tri_pre_sample)
        tf.summary.histogram(name='epoch/valid/pre', values=val_pre_sample)

        summary_init  = tf.summary.merge(tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope='init'))
        summary_batch = tf.summary.merge(tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope='batch'))
        summary_epoch = tf.summary.merge(tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope='epoch'))
        summary_model = tf.summary.merge(tf.get_collection(key=tf.GraphKeys.SUMMARIES, scope='model'))

        ##########################
        # Begin Training
        ##########################
        train_steps = len(train_dataset) // (batch_size * num_gpu)
        valid_steps = len(valid_dataset) // batch_size

        config_sess = tf.ConfigProto()
        config_sess.gpu_options.allow_growth = True
        with tf.Session(config=config_sess) as sess:
            tf.global_variables_initializer().run()
            try: #Recover
                model.load_weights(filepath=model_restore_path)
                print("Found Model [{}] and Continue Training" .format(model_restore_path))
            except Exception as e:
                print("No Continue Training and Start New One.")
                print("Exception Info is %s" % e)

            summary_writer = tf.summary.FileWriter(logdir=save_path, graph=sess.graph)
            summary_init_eval = sess.run(summary_init)
            summary_writer.add_summary(summary_init_eval, global_step=global_epoch) #global_epoch

            tri_x_sample_eval, tri_gt_sample_eval, \
                val_x_sample_eval, val_gt_sample_eval \
                = sess.run([tri_x_sample, tri_gt_sample, val_x_sample, val_gt_sample])
            to_mat_np(tri_x_sample_eval, valid_path + 'tri-sample-x')
            to_mat_np(tri_gt_sample_eval, valid_path + 'tri-sample-gt')
            to_mat_np(val_x_sample_eval, valid_path + 'val-sample-x')
            to_mat_np(val_gt_sample_eval, valid_path + 'val-sample-gt')

            # training
            for epoch in range(global_epoch, train_epoch):
                tf.local_variables_initializer().run()
                for _ in tqdm(range(train_steps), desc="Train in Epoch [%d/%d]" % (epoch + 1, train_epoch)):
                    if global_batch % runtime_batch == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary_batch_eval, _, _ = sess.run([summary_batch, train_op, tri_epoch_update],
                                                            feed_dict={k.learning_phase(): 1,
                                                                       dy_lr: learning_rate},
                                                            options=run_options,
                                                            run_metadata=run_metadata)
                        summary_writer.add_run_metadata(run_metadata, 'batch-%d' % global_batch)
                        timeline_eval = timeline.Timeline(run_metadata.step_stats)
                        timeline_eval = timeline_eval.generate_chrome_trace_format()
                        with open(timeline_path + 'timeline-%d.json' % global_batch, 'w') as f:
                            f.write(timeline_eval)
                    else:
                        summary_batch_eval, _, _ = sess.run([summary_batch, train_op, tri_epoch_update],
                                                            feed_dict={k.learning_phase(): 1,
                                                                       dy_lr: learning_rate})
                    summary_writer.add_summary(summary_batch_eval, global_step=global_batch)
                    global_batch += 1

                for _ in tqdm(range(valid_steps), desc='Valid in Epoch [%d/%d]' % (epoch + 1, train_epoch)):
                    sess.run(val_epoch_update, feed_dict={k.learning_phase(): 0}) # xiaojian ??
                summary_epoch_eval, model_metrics_eval = sess.run([summary_epoch, model_metrics],
                                                                  feed_dict={k.learning_phase(): 0,
                                                                             dy_lr: learning_rate})
                summary_writer.add_summary(summary_epoch_eval, global_step=global_epoch)

                # Model Checkpoint
                model.save_weights(model_path + 'latest.h5')
                if (global_epoch + 1) % save_epoch == 0 or ((global_epoch + 1) % 10 == 0 and (global_epoch + 1) <= 100) or (global_epoch + 1) <=10:
                    model.save_weights(model_path + 'epoch-%d.h5' % global_epoch)
                    # Save Valid
                    tri_pre_sample_eval, val_pre_sample_eval = sess.run([tri_pre_sample, val_pre_sample])
                    to_mat_np(tri_pre_sample_eval, valid_path + 'tri-sample-epoch-%d' % global_epoch)
                    to_mat_np(val_pre_sample_eval, valid_path + 'val-sample-epoch-%d' % global_epoch)

                model_info_eval = ""
                for i in range(len(model_tags)):
                    if model_metrics_eval[i] > model_best[i]:
                        model.save_weights(model_path + 'best-%s.h5' % model_tags[i])
                        model_info_eval += "[%s] improve in epoch %d, from %.4f to %.4f \n\n" % (
                            model_tags[i], global_epoch, model_best[i], model_metrics_eval[i])

                        model_best[i] = model_metrics_eval[i]
                        model_epoch[i] = global_epoch
                    else:
                        model_info_eval += "%s doesn't improve, the best is in epoch %d with value as %.4f \n\n" % (
                            model_tags[i], model_epoch[i], model_best[i])

                summary_model_eval = sess.run(summary_model, feed_dict={model_info: model_info_eval})
                summary_writer.add_summary(summary_model_eval, global_step=global_epoch)

                global_epoch += 1
