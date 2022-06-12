import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# 导入工具类
import resnet_util

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 模型测试(模型路径)
def resnet_test(model_src):
    _, test_data, class_names = resnet_util.load_data("E://pythonTest/food-101/images/train2",
                                                      "E://pythonTest/food-101/images/test2", 224, 224)
    model = tf.keras.models.load_model(model_src)
    loss, accuracy = model.evaluate(test_data)
    print('resnet模型的测试集准确率为：',accuracy)
    # 测试集正确标签
    test_real_labels = []
    # 测试集预测标签
    test_pre_labels = []
    # 测试集预测错误信息，保存图片数组，真正标签数组，被预测错误数组
    img_show = []
    img_label_true = []
    img_label_error = []
    # 分别取出图片张量和标签
    for test_batch_images, test_batch_labels in test_data:
        test_batch_labels = test_batch_labels.numpy()
        # 模型预估
        test_batch_pres = model.predict(test_batch_images)
        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # 将推理对应的标签取出
        for i, val in enumerate(test_batch_labels_max):
            # 如果预测错误 取出错误数据
            if test_batch_labels_max[i] != test_batch_pres_max[i]:
                img_label_true.append(test_batch_labels_max[i])
                img_label_error.append(test_batch_pres_max[i])
                img_show.append(test_batch_images[i])
            # 添加到标签
            test_real_labels.append(val)
            test_pre_labels.append(test_batch_pres_max[i])
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    print(heat_maps)
    heat_maps_float = heat_maps / heat_maps_sum
    # 调用画热力图
    # write_heatmap(heat_maps, resnet_util.label)
    # 画错误部分样本
    # error_img(img_show, img_label_true, img_label_error, resnet_util.label)
    # 计算准确率，精确率，和召回率
    calculate_APR(heat_maps, 50)

# 画热力图  (热力图基本信息，标签， 是否展示)
def write_heatmap(array_info, label, show=False):
    fig, ax = plt.subplots()
    im = ax.imshow(array_info, cmap="Greens")
    ax.set_xticks(np.arange(len(label)))
    ax.set_yticks(np.arange(len(label)))
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加具体数值
    for i in range(len(label)):
        for j in range(len(label)):
            text = ax.text(j, i, round(array_info[i, j], 2),
                           ha="center", va="center", color="orange")
    ax.set_xlabel("预测值：")
    ax.set_ylabel("真实值：")
    ax.set_title('resnet模型混淆矩阵')
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig('resnet_heatmap1.png', dpi=300)
    if show:
        plt.show()


# 画错误分类部分样本 (错误集张量数组， 正确的标签数组， 预测错误的标签数组，中文名字数组， 是否展示)
def error_img(img, true_label, error_label, chinese_name, show=False):
    num = 16
    plt.figure()
    for i in range(num):  # 只显示前16个
        # 从张量数组转为numpy数组
        numpy_out = np.array(img[i])
        plt.subplot(4, 4, i + 1, xticks=[], yticks=[])  # 4*4子图显示
        # 转为unit8格式正常显示
        plt.imshow(numpy_out.astype(np.uint8))
        plt.title(f'{chinese_name[true_label[i]]}--> {chinese_name[error_label[i]]}')  # 显示标题
        plt.subplots_adjust(wspace=2, hspace=0.1)
    plt.savefig('resnet_error_img.png', dpi=300)
    if show:
        plt.show()


# https://cloud.tencent.com/developer/article/1510724(准确率，精确率，和召回率介绍1)
# https://blog.csdn.net/qwe1110/article/details/103391632(准确率，精确率，和召回率介绍2)
# 计算准确率，精确率，和召回率(混淆矩阵,每个训练集样本)
def calculate_APR(confusion_matrix,test_data_size):
    confusion_matrix = confusion_matrix.astype(int)
    # 获取分类数量
    data_len = len(confusion_matrix)
    # 准确率 精确率 召回率
    accuracy = 0
    precision = 0
    recall = 0
    # 总训练集数
    total = test_data_size * len(confusion_matrix)
    # 获取对角数据
    diagonal = np.diagonal(confusion_matrix)
    # 计算精确率
    column_arr = np.sum(confusion_matrix, axis=0)
    print(column_arr)
    for i in column_arr:
        precision += ((diagonal / i) * (1 / data_len))
    # 计算召回率
    row_arr = np.sum(confusion_matrix, axis=1)
    print(row_arr)
    for i in row_arr:
        recall += ((diagonal / i) * (1 / data_len))
    # 计算准确率
    accuracy = np.sum(diagonal) / total
    print(accuracy)
    print(precision)
    print(recall)


if __name__ == '__main__':
    resnet_test("../models/resnet_foot_easy.h5")
