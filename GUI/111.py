import cv2
import tensorflow as tf
from PIL import Image
import numpy as np


resnet_model = tf.keras.models.load_model("../models/resnet_foot_easy.h5")
class_name = ['苹果派', '鸡翅', '巧克力蛋糕', '甜甜圈', '菲力牛排', '汤圆', '饺子', '汉堡包', '热狗', '冰淇淋']
img = cv2.imread('E:\pythonTest\\ve_test\\vegetables_tf2.3-master\GUI\\val_img\\2.jpg')
img_init = cv2.resize(img, (224, 224))
cv2.imwrite('target.png', img_init)
img = Image.open('target.png')
img = np.asarray(img)  # 将图片转化为numpy的数组
outputs = resnet_model.predict(img.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
result_index = int(np.argmax(outputs))
result = class_name[result_index]
print(result)
