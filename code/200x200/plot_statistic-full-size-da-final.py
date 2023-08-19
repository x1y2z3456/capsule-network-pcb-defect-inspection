# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.gridspec as gridspec

def plot_fig_with_text(filename, nrows):
    
    print(f'Data Record Number: {nrows}')
    # 读取 CSV 文件
    data = pd.read_csv(filename, nrows=nrows)
    accuracy = np.array(data['accuracy'])
    val_accuracy = np.array(data['val_accuracy'])

    # 获取最后一行数据
    last_row = data.iloc[-1]

    # 獲取最大val acc的那行數據
    max_val_acc_index = np.argmax(val_accuracy)
    max_val_acc = val_accuracy[max_val_acc_index]
    max_row = data.iloc[max_val_acc_index]

#     num_ticks = 8  # 切分數
#     step = round(np.ceil(nrows / num_ticks))  # 刻度間距
    tick_step = {20: 2, 40: 5, 60: 10, 80: 10, 100: 10, 200: 10, 350: 50}
    

    # 创建网格布局
    fig = plt.figure(figsize=(12, 45))
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[1, 1, 2.2, 1], hspace=0.07)

    # 绘制训练准确度图表
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data['epoch'], data['accuracy'], label='Accuracy')
    ax1.plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy')
    ax1.scatter(last_row['epoch'], last_row['accuracy'], label=f'Accuracy: {last_row["accuracy"]:.4f}')
    ax1.scatter(last_row['epoch'], last_row['val_accuracy'], label=f'Val Accuracy: {last_row["val_accuracy"]:.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_step[nrows]))  # 設定X軸刻度
    ax1.legend()

    # 添加训练准确度的散点标记
    ax1.annotate(f"Max Val Accuracy: {max_row['val_accuracy']:.4f}\nCurrent Accuracy: {max_row['accuracy']:.4f}\nCurrent Val Loss: {max_row['val_loss']:.4f}\nCurrent Loss: {max_row['loss']:.4f}\nEpoch: {max_row['epoch']:.0f}", 
                 xy=(max_row['epoch'], max_row['val_accuracy']), 
                 xytext=(0.3, 0.2), textcoords='axes fraction', arrowprops=dict(facecolor='black', arrowstyle="->"))

    # 绘制训练损失图表
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(data['epoch'], data['loss'], label='Loss')
    ax2.plot(data['epoch'], data['val_loss'], label='Validation Loss')
    ax2.scatter(last_row['epoch'], last_row['loss'], label=f'Loss: {last_row["loss"]:.4f}')
    ax2.scatter(last_row['epoch'], last_row['val_loss'], label=f'Val Loss: {last_row["val_loss"]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_step[nrows]))  # 設定X軸刻度
    ax2.legend()
    
    # 添加训练損失的散点标记
#     min_row = data.iloc[35] #第35行的兩者損失最接近，也都比較小
#     ax2.annotate(f"Max Val Accuracy: {min_row['val_accuracy']:.4f}\nCurrent Accuracy: {min_row['accuracy']:.4f}\nCurrent Val Loss: {min_row['val_loss']:.4f}\nCurrent Loss: {min_row['loss']:.4f}\nEpoch: {min_row['epoch']:.0f}", 
#                  xy=(min_row['epoch'], min_row['val_accuracy']), 
#                  xytext=(0.3, 0.2), textcoords='axes fraction', arrowprops=dict(facecolor='black', arrowstyle="->"))
    
    # 取出需要分析的資料列
    y = data['val_loss'].values
    y_= data['loss'].values
    x = data['val_accuracy'].values
    x_= data['accuracy'].values
    
    # 找到极值点
    max_indices = np.where(np.r_[True, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:], True])[0]
    min_indices = np.where(np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], True])[0]

    # 找到相对低点
    valleys = []
    for min_idx in min_indices:
        if min_idx == 0 or min_idx == len(y) - 1:
            continue
        if y[min_idx] < np.max(y[max_indices[max_indices < min_idx]]):
            valleys.append(min_idx)
    ax2.plot(valleys, y[valleys], 'o', markersize=4, color='green')
#     for idx, val in zip(valleys, y[valleys]):
#         if idx >= 20:
#             ax2.annotate(f'{val:.2f}', xy=(idx, val), xytext=(-40, 30),
#                          textcoords='offset points', ha='center', va='bottom',
#                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#     print(valleys)
#     # 資料
#     table_data = [['epoch', 'loss', 'val loss', 'accuracy', 'val accuracy'],]
#     for valley in valleys:
#         table_data.append([valley, f'{y_[valley]:.4f}', f'{y[valley]:.4f}', f'{x_[valley]:.4f}', f'{x[valley]:.4f}'])

#     # 繪製表格
# #     fig, ax = plt.subplots()
#     ax3 = fig.add_subplot(gs[2])
#     ax3.axis('off')
#     ax3.table(cellText=table_data, loc='center')

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(data['epoch'], data['loss'], label='Loss')
    ax4.scatter(last_row['epoch'], last_row['loss'], label=f'Loss: {last_row["loss"]:.4f}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(tick_step[nrows]))  # 設定X軸刻度
    ax4.legend()

    # 显示图表
#     plt.xticks(np.arange(0, nrows+1, 2))
    plt.show()


# plot_fig_with_text('log-200epoch.csv', 200)
# plot_fig_with_text('log-capsnet-new-7.csv', 200)

print('DenseCapsNet 100 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-densenet121-r8.csv', 100)
print('ResCapsNet 100 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-resnet50-r5.csv', 100)
print('VGGCapsNet 100 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-vgg19-r8.csv', 100)


print('New DenseCapsNet 200 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-densenet121-r8-r2.csv', 200)

print('New CapsNet V3 200 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-densenet121-r8-r6.csv', 200)

print('New CapsNet V3 350 epoch')
plot_fig_with_text('log-capsnet-latest-15-200-full-size-da-densenet121-r8-r6-r3.csv', 350)