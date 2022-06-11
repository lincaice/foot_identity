import shutil
import os


# 根据txt文件批量创建文件夹
def mk_dir():
    file = "E://pythonTest/food-101/images/test/"
    file_object = open('E://pythonTest/food-101/classes.txt', 'rU')
    for line in file_object:
        folder = file + line
        folder = folder.strip()
        os.makedirs(folder)


# 根据txt文件批量划分测试集和训练集
def split_picture():
    file_object = open('E://pythonTest/food-101/test.txt', 'rU')  # txt文件路径
    try:
        for line in file_object:
            x = line.split("/")[0]
            shutil.move('E://pythonTest/food-101/images/train/' + line.rstrip('\n') + '.jpg',
                        'E://pythonTest/food-101/images/test/' + x)
    finally:
        file_object.close()


# 批量删除文件
def cut_some_picture():
    # 根据txt删除
    file_object = open('E://pythonTest/food-101/test.txt', 'rU')  # txt文件路径
    num = 0
    file_name = ''
    try:
        for line in file_object:
            x = line.split("/")[0]
            if file_name != x:
                file_name = x
                num = 0
            #     删除图片数量
            if num < 200:
                os.remove('E://pythonTest/food-101/images/test/' + line.rstrip('\n') + '.jpg')
            num += 1
    finally:
        file_object.close()


if __name__ == '__main__':
    # mk_dir()
    # split_picture()
    cut_some_picture()
