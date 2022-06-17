import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

label = ['苹果派', '鸡翅', '巧克力蛋糕', '甜甜圈', '菲力牛排', '年糕', '饺子', '汉堡包', '热狗', '冰淇淋']
img_shape = (224, 224, 3)
class_num = 10

# 加载数据 （训练集路径，测试集路径，宽，高）
def load_data(train_src, test_src, width, height):
    # 训练集数据
    # label_mode 是 categorical, labels是形状为（batch_size, num_classes）的float32张量，表示类索引的one-hot编码。
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_src,
        label_mode='categorical',
        seed=100,
        image_size=(width, height))
    # 测试集数据
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_src,
        label_mode='categorical',
        seed=100,
        image_size=(width, height))
    identity_name = train_data.class_names
    return train_data, test_data, identity_name


