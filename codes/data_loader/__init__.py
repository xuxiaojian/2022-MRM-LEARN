import tensorflow as tf
import numpy as np

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


# noinspection PyBroadException
class TFDatasetBase:
    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __init__(self,
                 is_shuffle: bool = False,
                 num_parallel_calls: int = -1,
                 sample_index: list or tuple = (0, 1)):

        if num_parallel_calls <= 0:
            num_parallel_calls = None

        self.dtype = tuple((i.dtype for i in self[0]))
        self.shape = tuple((i.shape for i in self[0]))

        self.__dataset = tf.data.Dataset.from_tensor_slices(tf.range(len(self)))
        if is_shuffle:
            self.__dataset = self.__dataset.shuffle(buffer_size=len(self))

        self.__dataset = self.__dataset.map(self.__getitem_tf__,  num_parallel_calls=num_parallel_calls)
        self.__dataset = self.__dataset.repeat()

        self.__sample = tf.data.Dataset.from_tensor_slices(np.array(sample_index))
        self.__sample = self.__sample.map(self.__getitem_tf__)
        self.__sample = self.__sample.repeat()
        self.__sample = self.__sample.batch(sample_index.__len__())

    def __getitem_tf__(self, index):
        output_ = tf.py_function(lambda x: self[x], [index], [i for i in self.dtype])
        for i in range(len(output_)):
            output_[i].set_shape(self.shape[i])
        return output_

    def __call__(self, batch_size):
        return self.__dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE). \
            make_one_shot_iterator().get_next()

    def sample(self):
        return self.__sample.make_one_shot_iterator().get_next()

    def getitem_names(self):
        pass


def compute_nearby_index(num_nearby_indexes, cur_index, total_index) -> list:
    opt = np.array([i for i in range(num_nearby_indexes)]) + cur_index - int(num_nearby_indexes / 2)

    if cur_index >= (total_index - int(num_nearby_indexes / 2)):
        opt -= total_index

    return opt.tolist()

