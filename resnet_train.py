import tensorflow as tf
import matplotlib.pyplot as plt
<<<<<<< HEAD
import resnet_util


# resnet模型（图片的维度, 分类种类数量）
def resnet_model1(shape=resnet_util.img_shape,class_num=resnet_util.class_num):
    base_model = tf.keras.applications.resnet_v2.ResNet101V2(input_shape=shape,
                                                             include_top=False,
                                                             weights='imagenet')

    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=shape),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 绘制损失率和精确率折线图(训练历史)
def show_curve(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_mobilenet.png', dpi=300)
    

# 开始训练(训练次数)
def start_train(epochs=30):
    train_data, test_data, class_names = resnet_util.load_data("E://pythonTest/food-101/images/train2",
                                                               "E://pythonTest/food-101/images/test2", 224, 224)
    # 加载resnet模型
    model = resnet_model1()
    history = model.fit(train_data, validation_data=test_data, epochs=epochs)
    model.save("models/resnet_foot_easy.h5")
    # 绘制损失率和精确率折线图
    show_curve(history)


if __name__ == '__main__':
    start_train()
    
=======
import numpy as np


# 加载数据 （训练集路径，测试集路径，宽，高）
def load_data(train_src, test_src, width, height):
    # 训练集数据
    # label_mode 是 categorical, labels是形状为（batch_size, num_classes）的float32张量，表示类索引的one-hot编码。
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_src,
        label_mode='categorical',
        seed=100,
        image_size=(height, width))
    # 测试集数据
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_src,
        label_mode='categorical',
        seed=100,
        image_size=(width, height))
    identity_name = train_data.class_names
    return train_data, test_data, identity_name
>>>>>>> origin/main
