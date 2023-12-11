import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.python.ops.numpy_ops import np_config
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# https://neptune.ai/blog/keras-loss-functions

### 0.8 MSE + 0.2 Lap ###
alpha = 0.2

def total_loss_mse_no_reduction(y_true, y_pred): 
    tf.print("gt", y_true.shape)
    tf.print("pred", y_pred.shape)

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true, lapKernel)
    predLap = K.conv2d(y_pred, lapKernel)

    lap = K.square(trueLap - predLap)
    
    blurry_channel = (1-alpha) * mse_channel_0 + alpha * lap[:,:,:,0] # non_bin
    binary_channel = (1-alpha) * mse_channel_1 + alpha * lap[:,:,:,1] # with_bin

    
    return 0.3*binary_channel + 0.7*blurry_channel
    # return 0.8*binary_channel + 0.2*blurry_channel

def total_loss_mse_lap_ssim(y_true, y_pred): 

    if y_true.shape[-1]>1:
        mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0]) # non_bin
        mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1]) # with_bin
        
        lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
         
        trueLap = K.conv2d(y_true, lapKernel)
        predLap = K.conv2d(y_pred, lapKernel)

        lap = K.square(trueLap - predLap)

        ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
        ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
        
        loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
        loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

        loss =0.3*loss_channel_0 + 0.7*loss_channel_1

    elif y_true.shape[-1]==1:
        loss=total_loss(y_true, y_pred)

    return loss

def total_loss(y_true, y_pred): 
    mse = keras.losses.MeanSquaredError()(y_true, y_pred)
    
    lapKernel = K.constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],shape = [3, 3, 1, 1])
    print('#'*20)    
    
    trueLap = K.conv2d(y_true, lapKernel)
    predLap = K.conv2d(y_pred, lapKernel)
    lap = K.sum(K.square(trueLap - predLap), axis=-1)

    ssim = ssim_loss(y_true, y_pred)

    loss = 0.1 * mse + 0.2 * lap+0.7*ssim
    
    return loss

class cal_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(cal_loss, self).__init__()
        self.real_weight = 1.0
        self.enhance_weight = 1 - self.real_weight
        self.alpha, self.beta1, self.beta2, self.gamma = 1., 0., 0.2, 0.5

    def call(self, y_true, y_pred):

        # lapKernel = K.constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],shape = [3, 3, 1, 1])
        lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])


        mse_real = keras.losses.MeanSquaredError()(y_true, y_pred)
        # L1 = tf.reduce_mean(tf.math.abs(y_true - y_pred))

        trueLap = K.conv2d(y_true, lapKernel)
        predLap = K.conv2d(y_pred, lapKernel)
        lap_real = K.sum(K.square(trueLap - predLap), axis=-1)
        loss = self.alpha * lap_real + self.beta1 * mse_real
        return loss

### MSSIM ###

def MSSSIMLossFit(y_true, y_pred):
    # loss= tf.image.ssim_multiscale(y_true, y_pred, 1.0, filter_size=3)

    loss = 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val = 1.0, filter_size=3)) +1e-7
    # print("loss",loss)
    return loss

def MSSSIMLoss(y_true, y_pred):
    loss = 0
    ####^ above are correct
    if len(y_true.shape)==4:
        for batch in range(y_true.shape[0]):
            # print(y_true[batch].numpy().shape)
            # print(y_true[batch])
            # print("*"*50)
            # print(y_pred[batch])
            # print(tf.image.ssim_multiscale(y_true[batch].numpy(), y_pred[batch].numpy(), 1.0,filter_size=3))
            loss += 1-tf.reduce_mean(tf.image.ssim_multiscale(y_true[batch].numpy(), y_pred[batch].numpy(), 1.0,filter_size=3)) 
        # print(loss)
        # exit()
        return loss/y_true.shape[0]
    else:
        tf.print(tf.image.ssim_multiscale(y_true, y_pred, 1.0,filter_size=3))
        loss = 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0,filter_size=3)) 
        # tf.print(loss)
        return loss

    # MSSSIM = tf.image.ssim_multiscale( y_true, y_pred, 1.0, (0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    # return MSSSIM

### SSIM ###
def ssim_loss(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def SSIMMAE(y_true,y_pred):
    #https://arxiv.org/pdf/1511.08861.pdf
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = 0.84 * ssim_loss(y_true,y_pred)+(1-0.84)* mae(y_true,y_pred)
    return loss


### contrastive loss ###
# https://pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)

    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))

    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


def FocalLoss(targets, inputs, alpha=.25, gamma=2):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://stackoverflow.com/questions/63278000/how-to-use-custom-loss-function-for-keras
    """    
    
    targets = K.cast(targets, 'float32')

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def charbonnier_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))
    return loss


if __name__ == '__main__':
    true = tf.random.uniform((32,184,40,2), maxval=1)
    pred = tf.random.uniform((32,184,40,2), maxval=1)
    # loss = binary_focal_loss(alpha=.25, gamma=2)(true, pred)
    aaa = total_loss_mse_no_reduction(true, pred).numpy()
    # aaaa=BinaryFocalLoss(gamma=2)(true, pred)

    # MSSSIMLossFit_loss = SSIMMAE(true, pred)
    # print(loss)
    print(aaa)
    print(aaa.shape)
    # print(MSSSIMLossFit_loss)