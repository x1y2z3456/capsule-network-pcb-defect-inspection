# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras import backend as K
from keras.layers import Layer, Input, Conv2D, AveragePooling2D
from keras.layers import Lambda, Reshape
from keras import activations
from keras.datasets import cifar10
from keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW
import seaborn as sns
from matplotlib import pyplot as plt
import time
from keras.utils.layer_utils import count_params
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.callbacks import Callback


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


def caps_batch_dot(x, y):
    x = K.expand_dims(x, 2)
    if K.int_shape(x)[3] is not None:
        y = K.permute_dimensions(y, (0, 1, 3, 2))
    o = tf.matmul(x, y)
    return K.squeeze(o, 2)


def specificity_score(y_true, y_pred, labels=None, pos_label=1, average='binary'):
    """
    Compute the specificity score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    if average == 'micro':
        specificity = tn / (tn + fp)
    elif average == 'macro':
        specificity = (tn / (tn + fp) + tp / (tp + fn)) / 2
    elif average == 'weighted':
        specificity = (tn / (tn + fp) * (tn + fn) + tp / (tp + fn) * (fp + tp)) / (tn + fp + fn + tp)
    elif average == 'binary':
        specificity = tn / (tn + fp)
    else:
        raise ValueError("Unsupported average type.")
    return specificity

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

def add_commas(num):
    num_str = str(num)
    if len(num_str) <= 3:
        return num_str
    else:
        return add_commas(num_str[:-3]) + ',' + num_str[-3:]


class Capsule(Layer):
    """ A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).
    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )
    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            if input_shape[-2] is None:
                raise ValueError("Input Shape must be defied if weights not shared.")
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.
        This change can improve the feature representation of Capsule.
        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(caps_batch_dot(c, hat_inputs))
            if i < self.routings - 1:
                b = caps_batch_dot(o, hat_inputs)
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Length(Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the
    same shape with y_true in margin_loss. Using this layer as model's output can
    directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    source: https://github.com/XifengGuo/CapsNet-Keras/
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4

    if epoch >= 20:
        lr *= 1e-1
#     elif epoch >= 200:
#         lr *= 1e-2
    print('Learning rate: ', lr)
    return lr


def wd_schedule(epoch):
    """Weight Decay Schedule
    Weight decay is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        wd (float32): weight decay
    """
    wd = 1e-5

    if epoch >= 20:
        wd *= 1e-1
#     elif epoch >= 200:
#         wd *= 1e-2
    print('Weight decay: ', wd)
    return wd

# just copy the implement of LearningRateScheduler, and then change the lr with weight_decay
@keras_export('keras.callbacks.WeightDecayScheduler')
class WeightDecayScheduler(Callback):
    """Weight Decay Scheduler.

    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight decay as output (float).
        verbose: int. 0: quiet, 1: update messages.

    ```python
    # This function keeps the weight decay at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    def scheduler(epoch):
      if epoch < 10:
        return 0.001
      else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = WeightDecayScheduler(scheduler)
    model.fit(data, labels, epochs=100, callbacks=[callback],
              validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self, schedule, verbose=0):
        super(WeightDecayScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'weight_decay'):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        try:  # new API
            weight_decay = float(K.get_value(self.model.optimizer.weight_decay))
            weight_decay = self.schedule(epoch, weight_decay)
        except TypeError:  # Support for old API for backward compatibility
            weight_decay = self.schedule(epoch)
        if not isinstance(weight_decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.weight_decay, weight_decay)
        if self.verbose > 0:
            print('\nEpoch %05d: WeightDecayScheduler reducing weight '
                  'decay to %s.' % (epoch + 1, weight_decay))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['weight_decay'] = K.get_value(self.model.optimizer.weight_decay)

class Lion(tf.keras.optimizers.legacy.Optimizer):
  r"""Optimizer that implements the Lion algorithm."""

  def __init__(self,
               learning_rate=0.0001,
               beta_1=0.9,
               beta_2=0.99,
               wd=0,
               name='lion',
               **kwargs):
    """Construct a new Lion optimizer."""

    super(Lion, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('wd', wd)

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Lion, self)._prepare_local(var_device, var_dtype, apply_state)

    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    wd_t = tf.identity(self._get_hyper('wd', var_dtype))
    lr = apply_state[(var_device, var_dtype)]['lr_t']
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            beta_1_t=beta_1_t,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            one_minus_beta_2_t=1 - beta_2_t,
            wd_t=wd_t))

  @tf.function(jit_compile=True)
  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    var_t = var.assign_sub(
        coefficients['lr_t'] *
        (tf.math.sign(m * coefficients['beta_1_t'] +
                      grad * coefficients['one_minus_beta_1_t']) +
         var * coefficients['wd_t']))
    with tf.control_dependencies([var_t]):
      m.assign(m * coefficients['beta_2_t'] +
               grad * coefficients['one_minus_beta_2_t'])

  @tf.function(jit_compile=True)
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    m_t = m.assign(m * coefficients['beta_1_t'])
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
    var_t = var.assign_sub(coefficients['lr'] *
                           (tf.math.sign(m_t) + var * coefficients['wd_t']))

    with tf.control_dependencies([var_t]):
      m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
      m_t = m_t.assign(m_t * coefficients['beta_2_t'] /
                       coefficients['beta_1_t'])
      m_scaled_g_values = grad * coefficients['one_minus_beta_2_t']
      m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

  def get_config(self):
    config = super(Lion, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'wd': self._serialize_hyperparameter('wd'),
    })
    return config

data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_densenet_capsule_trained_model-r8-r6-r5.h5'

start = time.time()
batch_size = 40
num_classes = 2
epochs = 350
DATASET_PATH = './'
IMAGE_SIZE = (200, 200)
BATCH_SIZE = batch_size
NUM_EPOCHS = epochs

# A common Conv2D model
input_image = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_image)

x = Reshape((-1, 512))(base_model.output)
x = Capsule(32, 16, 3, True)(x)
x = Capsule(32, 16, 3, True)(x)
capsule = Capsule(num_classes, 32, 7, True)(x) 
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=base_model.input, outputs=output)

lr = 1e-4
# adam = Adam(lr=lr)
# adam = AdamW(learning_rate=lr, weight_decay=1e-4)
# adam = AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))
lion = Lion(learning_rate=lr_schedule(0), beta_1=0.9, beta_2=0.99, wd=wd_schedule(0))
# model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=lion, metrics=['accuracy'])
# model.summary()
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
wd_callback = WeightDecayScheduler(wd_schedule)

if not data_augmentation:
    print('Not using data augmentation.')
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
else:
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       channel_shift_range=5,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       rescale=1./255)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

train_set = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                              target_size=IMAGE_SIZE,
                                              interpolation='bicubic',
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=BATCH_SIZE)

valid_set = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                              target_size=IMAGE_SIZE,
                                              interpolation='bicubic',
                                              class_mode='categorical',
                                              shuffle=False,
                                              batch_size=BATCH_SIZE)

for data_batch, labels_batch in train_set:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# callbacks
log = CSVLogger(DATASET_PATH + 'log-capsnet-latest-15-200-full-size-da-densenet121-r8-r6-r5.csv')
checkpoint = ModelCheckpoint(DATASET_PATH + 'weights-capsnet-latest-15-200-full-size-da-densenet121-r8-r6-r5-{epoch:02d}.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
model.fit(train_set,
#           steps_per_epoch=train_set.samples // batch_size,
          epochs=epochs,
#           validation_data=train_generator(test_set, batch_size),
          validation_data=valid_set,
#           validation_steps=test_set.samples // batch_size,
#           batch_size=batch_size,
          callbacks=[log, checkpoint, reduce_lr, lr_callback, wd_callback])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# 評估模型
predict_start = time.time()
y_pred = model.predict(valid_set, steps=len(valid_set), verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_true = valid_set.classes
class_names = list(valid_set.class_indices.keys())
predict_end = time.time()
print('predict time: %s sec' % (predict_end - predict_start))

# 顯示模型訓練參數
print("Trainable Parameters：%s" % add_commas(count_params(model.trainable_weights)))

# 計算模型準確率
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# 顯示分類報告
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
specificity = specificity_score(y_true, y_pred)
print('Specificity: {:.2f}%'.format(specificity * 100))
top1_errors = 1 - np.mean(y_true == y_pred)
print('Top-1 Error: {:.2f}%'.format(top1_errors * 100))

# 顯示混淆矩陣
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# 绘制 confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cm_capsnet_latest15-200-full-size-da-densenet121-r8-r6-r5.png')
# plt.show()
end = time.time()
print('elapse time(s): ', format_time(int(end - start)))
