# -*- coding: utf-8 -*-
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, activations, utils, models, callbacks
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from matplotlib import pyplot as plt
from keras.utils.layer_utils import count_params
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preinput_resnet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preinput_vgg19
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as preinput_densenet121
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as preinput_inception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preinput_mobilenet


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

def train_generator(generator, batch_size, shift_fraction=0.):

    while True:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])

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

def predict_model(model, test_set):
    
    # 顯示模型訓練參數
    print("Trainable Parameters：%s" % add_commas(count_params(model.trainable_weights)))
    
    # 評估模型
    y_pred_org = model.predict(test_set, steps=len(test_set), verbose=1)
    y_pred = np.argmax(y_pred_org, axis=1)
    y_true = test_set.classes
    class_names = list(test_set.class_indices.keys())

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
    # print(cm)

    # 绘制 confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return y_pred_org


batch_size = 40
num_classes = 2
epochs = 100
DATASET_PATH = './'
IMAGE_SIZE = (200, 200)
BATCH_SIZE = batch_size
NUM_EPOCHS = epochs


# 比較其他CNN

# 定義4個模型的預測結果
print('Inception-V4')
test_datagen = ImageDataGenerator(preprocessing_function=preinput_inception, rescale=1./255)
test_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                            target_size=IMAGE_SIZE,
                                            interpolation='bicubic',
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)

y_true = test_set.classes
model_0 = models.load_model('model-inceptionv4-final-full-size-da.h5')
model_0.load_weights('weights-inceptionv4-full-size-da-28.h5')
y_pred_0_org = predict_model(model_0, test_set)

print('DenseNet-121')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                            target_size=IMAGE_SIZE,
                                            interpolation='bicubic',
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)

y_true = test_set.classes
model_1 = models.load_model('model-densenet121-final-full-size.h5')
# model_1.load_weights('weights-densenet121-full-size-da-98.h5')
# model_1.load_weights('weights-densenet121-full-size-no-da-81.h5')
y_pred_1_org = predict_model(model_1, test_set)

print('ResNet-50')
test_datagen = ImageDataGenerator(preprocessing_function=preinput_densenet121, rescale=1./255)
test_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                            target_size=IMAGE_SIZE,
                                            interpolation='bicubic',
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)

y_true = test_set.classes
model_2 = models.load_model('model-resnet50-final-full-size-da.h5')
y_pred_2_org = predict_model(model_2, test_set)

print('VGGNet-19')
test_datagen = ImageDataGenerator(preprocessing_function=preinput_densenet121, rescale=1./255)
test_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                            target_size=IMAGE_SIZE,
                                            interpolation='bicubic',
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)

y_true = test_set.classes
model_3 = models.load_model('model-vgg19-final-full-size-da.h5')
y_pred_3_org = predict_model(model_3, test_set)

print('MobileNet-V2')
test_datagen = ImageDataGenerator(preprocessing_function=preinput_mobilenet, rescale=1./255)
test_set = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                            target_size=IMAGE_SIZE,
                                            interpolation='bicubic',
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)

y_true = test_set.classes
model_4 = models.load_model('model-mobilenetv2-final-full-size-da.h5')
y_pred_4_org = predict_model(model_4, test_set)

# 計算每個模型的FPR，TPR和閾值
# fpr_0, tpr_0, thresholds_0 = roc_curve(y_true, y_pred_org_200_100[:, 1])
fpr_0, tpr_0, thresholds_0 = roc_curve(y_true, y_pred_0_org[:, 1])
fpr_1, tpr_1, thresholds_1 = roc_curve(y_true, y_pred_1_org[:, 1])
fpr_2, tpr_2, thresholds_2 = roc_curve(y_true, y_pred_2_org[:, 1])
fpr_3, tpr_3, thresholds_3 = roc_curve(y_true, y_pred_3_org[:, 1])
fpr_4, tpr_4, thresholds_4 = roc_curve(y_true, y_pred_4_org[:, 1])

# 計算每個模型的ROC曲線下面積（AUC）
auc_0 = auc(fpr_0, tpr_0)
auc_1 = auc(fpr_1, tpr_1)
auc_2 = auc(fpr_2, tpr_2)
auc_3 = auc(fpr_3, tpr_3)
auc_4 = auc(fpr_4, tpr_4)

# 設置圖形的大小
fig = plt.figure(figsize=(8, 6), dpi=80)

# 繪製所有模型的ROC曲線
plt.plot(fpr_0, tpr_0, color='blue', lw=2, label='Inception-V4 (AUC = %0.4f)' % auc_0)
plt.plot(fpr_1, tpr_1, color='green', lw=2, label='DenseNet-121 (AUC = %0.4f)' % auc_1)
plt.plot(fpr_2, tpr_2, color='red', lw=2, label='ResNet-50 (AUC = %0.4f)' % auc_2)
plt.plot(fpr_3, tpr_3, color='orange', lw=2, label='VGG-19 (AUC = %0.4f)' % auc_3)
plt.plot(fpr_4, tpr_4, color='purple', lw=2, label='MobileNet-V2 (AUC = %0.4f)' % auc_4)

# 設置圖形的標題和軸標籤
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 添加圖例
plt.legend(loc='lower right')

# 顯示圖形
plt.show()
