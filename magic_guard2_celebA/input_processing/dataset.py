import os
from tqdm import tqdm
import shutil
# import cv2

dataset_path = 'dataset/'
image_path = 'image/img_align_celeba'
identity_file_path = 'image/identity_CelebA.txt'

def sort_by_id():
    with open(identity_file_path, 'r') as f:
        lines = f.readlines()
        lines = [l.split(' ') for l in lines]
        for pair in lines:
            if os.path.exists(os.path.join(dataset_path, pair[1])):
                shutil.copy(os.path.join(image_path, pair[0]), os.path.join(dataset_path, pair[1]))
            else:
                os.mkdir(os.path.join(dataset_path, pair[1]))
                shutil.copy(os.path.join(image_path, pair[0]), os.path.join(dataset_path, pair[1]))
    f.close()

def truncate():
    nb_face = 0
    directories = os.listdir(dataset_path)
    for di in tqdm(directories):
        if nb_face == 300:
            shutil.rmtree(os.path.join(dataset_path, di))
            continue

        if len(os.listdir(os.path.join(dataset_path, di))) == 30:
            # os.rename(os.path.join(dataset_path, di), os.path.join(dataset_path, str(nb_face)))
            nb_face += 1
        else:
            shutil.rmtree(os.path.join(dataset_path, di))

def prepare_data(nb_class):
    for cls in tqdm(os.listdir('dataset/')[:nb_class]):
        shutil.copytree(os.path.join('dataset/', cls), os.path.join('data/train/', cls))
    for cls in tqdm(os.listdir('data/train')):
        path = os.path.join('data/train/', cls)
        for img in os.listdir(path)[:2]:
            shutil.move(os.path.join(path, img), 'data/trigger')

def trigger_set():
    src_path = '/wangrun/data/VggFace2/vggface2_train/'
    dst_path = '/wangrun/ziyi/data/train/'
    for i in os.listdir(src_path)[:50]:
        if i in os.listdir(dst_path):
            continue
        else:
            shutil.copytree(src_path + i, dst_path + i)

if __name__ == '__main__':
    # for im in tqdm(os.listdir('data/test/wm/')):
    #     face_img = cv2.imread('data/test/wm/' + im)
    #     face_img = cv2.resize(face_img, (128, 128))
    #     cv2.imwrite('data/test/wm/' + im, face_img)

    base_dir = 'data/test/'
    for cls in tqdm(os.listdir(base_dir)):
        if cls == 'wm':
            continue
        else:
            folder_path = base_dir + cls + '/'
            for img in os.listdir(base_dir + cls)[: 1]:
                shutil.copy(folder_path + img, 'data/test/wm/')
