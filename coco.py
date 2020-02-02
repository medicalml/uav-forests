import shutil
import subprocess
import os


dir_with_imgs = 'data_out/'
coco_path = '/home/h/mmdetection/data/coco'
img_dir = 'JPEGImages'
coco_annot_dir = 'annotations'
voc_annot_dir = 'Annotations'
train_json = 'instances_train2017.json'
val_json = 'instances_val2017.json'
train_2017 = 'train2017'
val_2017 = 'val2017'
dir_2007 = 'VOC2007'
voc_path = '/home/h/mmdetection/data/VOCdevkit/'
train_temp_dir = 'train_temp'
val_temp_dir = 'val_temp'
#remove 
if os.path.exists(os.path.join(coco_path, train_2017)):
    shutil.rmtree(os.path.join(coco_path, train_2017))

if os.path.exists(os.path.join(coco_path, val_2017)):
    shutil.rmtree(os.path.join(coco_path, val_2017))

if os.path.exists(os.path.join(coco_path, coco_annot_dir, train_json)):
    os.remove(os.path.join(coco_path, coco_annot_dir, train_json))

if os.path.exists(os.path.join(coco_path, coco_annot_dir, val_json)):
    os.remove(os.path.join(coco_path, coco_annot_dir, val_json))

if os.path.exists(os.path.join(coco_path, 'images')):
    shutil.rmtree(os.path.join(coco_path, 'images'))
os.mkdir(os.path.join(coco_path, 'images'))

subprocess.call(['cp','-r',  os.path.join(dir_2007, img_dir), coco_path])
subprocess.call(['cp','-r',  os.path.join(coco_path, img_dir), os.path.join(coco_path, 'images',train_2017)])
subprocess.call(['cp','-r',  os.path.join(coco_path, img_dir), os.path.join(coco_path, 'images',val_2017)])
shutil.rmtree(os.path.join(coco_path, img_dir))


shutil.rmtree(train_temp_dir)
os.mkdir(train_temp_dir)
shutil.rmtree(val_temp_dir)
os.mkdir(val_temp_dir)

with open('VOC2007/ImageSets/Main/train.txt') as train:
    for line in train:
        shutil.copy('VOC2007/Annotations/'+line[:-1]+'.xml', train_temp_dir)

with open('VOC2007/ImageSets/Main/val.txt') as train:
    for line in train:
        shutil.copy('VOC2007/Annotations/'+line[:-1]+'.xml', val_temp_dir)



subprocess.call(['python3', 'voc2coco.py', train_temp_dir, os.path.join(coco_path, 'annotations/'+train_json)])
subprocess.call(['python3', 'voc2coco.py', val_temp_dir, os.path.join(coco_path, 'annotations/'+val_json)])