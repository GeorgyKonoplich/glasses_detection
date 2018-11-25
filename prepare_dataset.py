import zipfile
from utils import *
from shutil import copyfile

'''
code for preparing original celebA dataset
'''

save_path = '/home/neuralbee/Downloads/img_align_celeba.zip'
path_to_attr = '/home/neuralbee/Downloads/list_attr_celeba.txt'
celebA_dir = './datasets/celebaA/'

prepare_data_dir(celebA_dir)
with_g, without_g = preprocess_celeba(path_to_attr)

with zipfile.ZipFile(save_path) as zf:
    zf.extractall(celebA_dir)

dir_with_glasses = './dataset/with_glasses/'
dir_without_glasses = './dataset/without_glasses/'

for img_name in with_g:
    img = cv2.imread(celebA_dir + img_name)
    img = get_head(img)
    cv2.imwrite(dir_with_glasses + img_name)

for img_name in without_g:
    img = cv2.imread(celebA_dir + img_name)
    img = get_head(img)
    cv2.imwrite(dir_without_glasses + img_name, img)



