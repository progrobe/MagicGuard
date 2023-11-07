import os 
import shutil
import os	
import sys

source = '/home/mist/relight/ziyi/data/val/'
destination = './data/imagenet_val/'
dirs = os.listdir(source)


train_class = 0
train_pic = 0
test_class = 0
test_pic = 0

for dir in dirs:
    img_files = os.listdir(source + dir)
    if not os.path.exists(destination + 'train/' + dir):
        os.mkdir(destination + 'train/' + dir)
    if not os.path.exists(destination + 'test/' + dir):
        os.mkdir(destination + 'test/' + dir)
    print(train_class)
    count = 0

    for file in img_files:
        if count < 5:
            # shutil.copyfile(source + dir + '/' + file, destination + 'test/' + dir  + '/' + file)
            print(source + dir + '/' + file)
            print(destination + 'test/' + dir + '/' + file)
            count += 1
            test_pic += 1
    # sys.exit()
        else:
            # shutil.copyfile(source+dir+'/'+file, destination +'train/' + dir + '/' + file)
            count += 1
            train_pic += 1
        if count == 1: #test下已经复制了一张
            test_class += 1
        elif count == 6: #train下已经复制了一张
            train_class += 1

print('train class:', train_class)
print('test class :', test_class)
print('train pic :', train_pic)
print('test pic :', test_pic)