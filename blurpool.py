import numpy as np
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

'''
class BlurPool2D(Layer):
    """
    https://arxiv.org/abs/1904.11486 https://github.com/adobe/antialiased-cnns
    """
    def __init__(self, kernel_size=5, stride=2, **kwargs):
        
        self.strides = (stride, stride)
        self.kernel_size = kernel_size
        self.padding = ((int(1 * (kernel_size - 1) / 2), int(np.ceil(1 * (kernel_size - 1) / 2))),
                        (int(1 * (kernel_size - 1) / 2), int(np.ceil(1 * (kernel_size - 1) / 2))))
        # x = np.ceil(1.*(kernel_size-1)/2)
        # y = ''.join(x)
        # self.padding =[int(1.*(kernel_size-1)/2), int(y), int(1.*(kernel_size-1)/2), int(y)]

        if self.kernel_size==1:     self.kernel = [1,]
        elif self.kernel_size==2:   self.kernel = [1, 1]
        elif self.kernel_size==3:   self.kernel = [1, 2, 1]
        elif self.kernel_size==4:   self.kernel = [1, 3, 3, 1]
        elif self.kernel_size==5:   self.kernel = [1, 4, 6, 4, 1]
        elif self.kernel_size==6:   self.kernel = [1, 5, 10, 10, 5, 1]
        elif self.kernel_size==7:   self.kernel = [1, 6, 15, 20, 15, 6, 1]
        self.kernel = np.array(self.kernel, dtype=np.float32)

        super(BlurPool2D, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1] 
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)
        

    def call(self, x):
        k = self.kernel
        k = k[:, None] * k[None, :]
        k = k / K.sum(k)
        k = K.tile (k[:, :, None, None], (1, 1, K.int_shape(x)[-1], 1))                
        k = K.constant(k, dtype=K.floatx())
        x = K.spatial_2d_padding(x, padding=self.padding)
        x = K.depthwise_conv2d(x, k, strides=self.strides, padding='valid')
        return x
'''
class BlurPool2D(Layer):
    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]
