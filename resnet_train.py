import tensorflow as tf
import matplotlib.pyplot as plt
import resnet_util


# resnet模型（图片的维度, 分类种类数量）
def resnet_model1(shape=resnet_util.img_shape,class_num=resnet_util.class_num):
    base_model = tf.keras.applications.resnet_v2.ResNet101V2(input_shape=shape,
                                                             include_top=False,
                                                             weights='imagenet')

    base_model.trainable = True
    for layers in base_model.layers[:-10]:
        layers.trainable = False

    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=shape),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
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
    plt.savefig('models/results_resnet.png', dpi=300)
    

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
    