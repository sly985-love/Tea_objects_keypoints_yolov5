import numpy as np
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

size = 300  # resize图片的大小，运行时如果爆显存的话把这里调小即可


# get_data(Input_path,Label)
# 作用：读取Input_path里的图片，并给每张图打上自定义标签Label
def get_data(Input_path, Label):
    Image_names = os.listdir(Input_path)  # 获取目录下所有图片名称列表
    data = np.zeros((len(Image_names), size * size * 3))  # 初始化一个np.array数组用于存数据,自己图片是n维的就把3改成n即可
    label = np.zeros((len(Image_names), 1))  # 初始化一个np.array数组用于存数据

    # 为当前文件下所有图片分配自定义标签Label
    for k in range(len(Image_names)):
        label[k][0] = Label

    for i in range(len(Image_names)):
        image_path = os.path.join(Input_path, Image_names[i])
        img = cv2.imread(image_path)
        img = cv2.resize(img, (size, size))  # (size,size,3)
        img = img.flatten()  # (3*size*size,)
        data[i] = img
    return data, label


# 重点来了，这里是根据自己想查看的数据来自定义修改代码,得到自己的x_train和y_train
# x_train是待分析的数据
# y_train是待分析的自定义标签
# 比如，我想分析训练集中5个类的分布情况
# 先读取每一个类的数据，然后给他们自定义标签1-5
# 然后把data拼在一起,label拼在一起，前者叫x_train,后者叫y_train
# data1, label1 = get_data(r'D:\yolotea6\all_specials _nt\huang1', 1)
# data2, label2 = get_data(r'D:\yolotea6\all_specials _nt\huang2', 2)
# data3, label3 = get_data(r'D:\yolotea6\all_specials _nt\lv1', 3)
# data4, label4 = get_data(r'D:\yolotea6\all_specials _nt\lv2', 4)
# data5, label5 = get_data(r'D:\yolotea6\all_specials _nt\lv3', 5)
# data6, label6 = get_data(r'D:\yolotea6\all_specials _nt\zi1', 6)
# data7, label7 = get_data(r'D:\yolotea6\all_specials _nt\zi2', 7)

# data1, label1 = get_data(r'D:\yolotea6\all\lv1', 1)
# data2, label2 = get_data(r'D:\yolotea6\all\lv2', 2)
# data3, label3 = get_data(r'D:\yolotea6\all\lv3', 3)
# data4, label4 = get_data(r'D:\yolotea6\all\lv4', 4)
# data5, label5 = get_data(r'D:\yolotea6\all\lv5', 5)
# data6, label6 = get_data(r'D:\yolotea6\all\lv6', 6)
# data7, label7 = get_data(r'D:\yolotea6\all\lv7', 7)

# data1, label1 = get_data(r'D:\yolotea6\all_new\huang1', 1)
# data2, label2 = get_data(r'D:\yolotea6\all_new\zi1', 2)
# data3, label3 = get_data(r'D:\yolotea6\all_new\lv1', 3)
# data4, label4 = get_data(r'D:\yolotea6\all_new\lv2', 4)
# data5, label5 = get_data(r'D:\yolotea6\all_new\lv6', 5)
# data6, label6 = get_data(r'D:\yolotea6\all_new\lv6', 6)
# data7, label7 = get_data(r'D:\yolotea6\all_new\lv7', 7)

data1, label1 = get_data(r'D:\yolotea6\all_new\split1\xi', 1)
data2, label2 = get_data(r'D:\yolotea6\all_new\split1\mao', 2)
# data3, label3 = get_data(r'D:\yolotea6\all_new\split1\mao', 3)
# data4, label4 = get_data(r'D:\yolotea6\all_new\split1\lv3', 4)
# data5, label5 = get_data(r'D:\yolotea6\all_new\lv2', 5)
# data8, label8 = get_data(r'D:\yolotea6\all\huang1', 8) D:\yolotea6\all_new\huang1
# data9, label9 = get_data(r'D:\yolotea6\all\huang2', 9)
# data10, label10 = get_data(r'D:\yolotea6\all\zi1', 10)
# data11, label11 = get_data(r'D:\yolotea6\all\zi2', 11)
# mao xi lv2 lv6

# data1, label1 = get_data(r'D:\yolotea6\all\huang1', 1)
# data2, label2 = get_data(r'D:\yolotea6\all\huang2', 2)
# data3, label3 = get_data(r'D:\yolotea6\all\zi1', 3)
# data4, label4 = get_data(r'D:\yolotea6\all\zi2', 4)

# data1, label1 = get_data(r'D:\yolotea6\all_month _nt\4', 1)
# data2, label2 = get_data(r'D:\yolotea6\all_month _nt\5', 2)
# data1, label1 = get_data(r'D:\yolotea6\all_specials _nt\lv1', 1)
# data2, label2 = get_data(r'D:\yolotea6\all_specials _nt\lv2', 2)
# data3, label3 = get_data(r'D:\yolotea6\all_specials _nt\zi1', 3)
# data4, label4 = get_data(r'D:\yolotea6\all_specials _nt\lv3', 4)
# data1, label1 = get_data(r'../data_set/bs150/', 1)
# data2, label2 = get_data(r'../data_set/b150/', 2)
# data3, label3 = get_data(r'../data_set/bb150/', 3)
# data3, label3 = get_data('../data_set/radar_oldANDyoung/train/3', 3)
# data4, label4 = get_data('../data_set/radar_oldANDyoung/train/4', 4)
# data5, label5 = get_data('../data_set/radar_oldANDyoung/train/5', 5)
# 得出数据后把他们拼起来
# data = np.vstack((data1, data2, data3, data4, data5, data6, data7))
# label = np.vstack((label1, label2, label3, label4, label5, label6, label7))
# data = np.vstack((data1))
# label = np.vstack((label1))

data = np.vstack((data1, data2))
label = np.vstack((label1, label2))

# data = np.vstack((data1, data2, data3))
# label = np.vstack((label1, label2, label3))

# data = np.vstack((data1, data2, data3, data4))
# label = np.vstack((label1, label2, label3, label4))
#
# data = np.vstack((data1, data2, data3, data4, data5))
# label = np.vstack((label1, label2, label3, label4, label5))

# data = np.vstack((data1, data2, data3, data4, data5, data6))
# label = np.vstack((label1, label2, label3, label4, label5, label6))

# data = np.vstack((data1, data2, data3, data4, data5, data6, data7))
# label = np.vstack((label1, label2, label3, label4, label5, label6, label7))

# data = np.vstack((data1, data2, data3, data4, data5, data6, data7,data8, data9, data10, data11))
# label = np.vstack((label1, label2, label3, label4, label5, label6, label7,label8, label9, label10, label11))
(x_train, y_train) = (data, label)
print(y_train.shape)  # (n_samples,1)
print(x_train.shape)  # (n_samples,size*size*3)

# t-SNE，输出结果是(n_samples,2)
# TSNE的参数和sklearn的T-SNE一样，不懂的自行查看即可
# tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=32, device=0)
tsne = TSNE(n_iter=10000, verbose=1, perplexity=5, random_state=0)
tsne_results = tsne.fit_transform(x_train)

print(tsne_results.shape)  # (n_samples,2)

# 画图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, title='TSNE-radar')

# Create the scatter
# ax.scatter()的用法自行百度
scatter = ax.scatter(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    c=y_train,
    # cmap=plt.cm.get_cmap('Paired'),
    # alpha=0.4,
    s=10)

# ax.legend添加类标签
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)

# 显示图片
plt.show()


# # 保存图片
# plt.savefig('./tSNE_radar.jpg')

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        # for i in range(x_train.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(int(label[i])),
                 fontdict={'weight': 'bold', 'size': 9})
        # print(label[i])
        # print("color",plt.cm.Set1(int(label[i])))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_embedding_3D(data, label, title):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)
    ax = plt.figure().add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(label[i]), color=plt.cm.Set1(int(label[i])),
                fontdict={'weight': 'bold', 'size': 9})
    return ax


print('Begining......')  # 时间会较长，所有处理完毕后给出finished提示
tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
result_2D = tsne_2D.fit_transform(data)
tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
result_3D = tsne_3D.fit_transform(data)
print('Finished......')
# 调用上面的两个函数进行可视化
label = np.squeeze(label)
fig1 = plot_embedding_2D(result_2D, label, 't-SNE')
plt.show()
# plt.savefig('./tSNE2.jpg')
label = np.squeeze(label)
fig2 = plot_embedding_3D(result_3D, label, 't-SNE')
plt.show()
plt.savefig('./tSNE3.jpg')
