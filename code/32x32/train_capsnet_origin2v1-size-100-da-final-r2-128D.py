# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, activations, utils, models, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from matplotlib import pyplot as plt
import time
from keras.utils.layer_utils import count_params


def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(
        tf.square(x), axis, keepdims=True) + tf.keras.backend.epsilon()
    scale = tf.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return tf.reduce_sum(y_true * tf.square(tf.keras.backend.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * tf.square(tf.keras.backend.relu(y_pred - margin)), axis=-1)

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


class Capsule(layers.Layer):

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
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):

        if self.share_weights:
            hat_inputs = tf.keras.backend.conv1d(inputs, self.kernel)
        else:
            hat_inputs = tf.keras.backend.local_conv1d(
                inputs, self.kernel, [1], [1])

        batch_size = tf.shape(inputs)[0]
        input_num_capsule = tf.shape(inputs)[1]
        hat_inputs = tf.reshape(hat_inputs,
                                (batch_size, input_num_capsule,
                                 self.num_capsule, self.dim_capsule))
        hat_inputs = tf.transpose(hat_inputs, (0, 2, 1, 3))

        b = tf.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(
                tf.keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if tf.keras.backend.backend() == 'theano':
                    o = tf.reduce_sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Capsule, self).get_config()
        config.update({'num_capsule': self.num_capsule,
                       'dim_capsule': self.dim_capsule,
                       'routings': self.routings,
                       'share_weights': self.share_weights,
                       'activation': activations.serialize(self.activation)})
        return config


def add_commas(num):
    num_str = str(num)
    if len(num_str) <= 3:
        return num_str
    else:
        return add_commas(num_str[:-3]) + ',' + num_str[-3:]


start = time.time()
batch_size = 32
num_classes = 2
epochs = 100
DATASET_PATH = './'
IMAGE_SIZE = (32, 32)
BATCH_SIZE = batch_size
NUM_EPOCHS = epochs

# 一个常规的 Conv2D 模型
input_image = layers.Input(shape=(None, None, 3))
x = layers.Conv2D(filters=128, kernel_size=7, strides=1)(input_image)
x = layers.Conv2D(filters=128, kernel_size=7, strides=1, activation='relu')(x)
x = layers.Conv2D(filters=128, kernel_size=7, strides=2)(x)
x = layers.Conv2D(filters=128, kernel_size=7, strides=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=1)(input_image)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=1)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=1)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=5, strides=2)(x)

x = layers.Reshape((-1, 128))(x)
capsule = Capsule(num_classes, 16, 3, True)(x)
output = layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = models.Model(inputs=input_image, outputs=output)

# 使用 margin loss
lr = 1e-4
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
# model.summary()

# 可以比较有无数据增益对应的性能
data_augmentation = True

if not data_augmentation:
    print('Not using data augmentation.')
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
else:
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       channel_shift_range=5,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                              target_size=IMAGE_SIZE,
                                              interpolation='bicubic',
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=BATCH_SIZE)

valid_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
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
log = callbacks.CSVLogger(DATASET_PATH + 'log-capsnet-origin2v1-128D-2-size-100-da-final-r2.csv')
checkpoint = callbacks.ModelCheckpoint(DATASET_PATH + 'weights-capsnet-origin2v1-128D-2-size-100-da-final-r2-{epoch:02d}.h5', monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=1)

model.fit(train_set,
#           steps_per_epoch=train_set.samples // batch_size,
          epochs=epochs,
#           validation_data=train_generator(test_set, batch_size),
          validation_data=valid_set,
#           validation_steps=test_set.samples // batch_size,
#           batch_size=batch_size,
          callbacks=[log, checkpoint])


# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-capsnet-origin2v1-128D-2-size-100-da-final-r2.h5'
# 儲存訓練好的模型
model.save(WEIGHTS_FINAL)

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
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix-capsnet-origin2v1-128D-2-size-100-da-final-r2.png')
# plt.show()
end = time.time()
print('elapse time(s): ', format_time(int(end - start)))
