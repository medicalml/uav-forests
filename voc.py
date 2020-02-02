from pascal_voc_writer import Writer
import csv
import os
import tqdm
import cv2
import pandas as pd
import re
import shutil
import subprocess

def get_coors_from_polygon(polygon):
    if polygon.startswith('POINT'):
        return None
    else:
        coors = polygon.split('((')[1][:-2] # POLYGON ((192 134, 250 134, 250 173, 192 173, 192 134)) -> 192 134, 250 134, 250 173, 192 173, 192 134
        coors = [str(int(float(s))) for s in re.findall(r'-?\d+\.?\d*', coors)]
        return coors[1], coors[0],coors[5], coors[2]

def embed_annot(img, img_name, output_dir, coors_list):
    label = ''
    for coors in coors_list:
        x1, y1, x2, y2 = [int(coor) for coor in coors]
        # print(x1,y1,x2,y2,conf,label)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),6)
    cv2.imwrite(output_dir + '/' +img_name, img)

dir_with_imgs = 'data_out/'

voc_path = '/home/h/mmdetection/data/VOCdevkit/'
annotation_name = dir_with_imgs + 'annotation.csv'

row_num = 0
with open(annotation_name,'r') as x:
    for line in x:
        row_num += 1

row_num -= 1

dir_2007 = 'VOC2007'
dir_2012 = 'VOC2012'

img_dir = 'JPEGImages'
annot_dir = 'Annotations'
image_sets_dir = 'ImageSets'
train_val_annot_dir = 'Main'
train_key = 'train'
val_key = 'val'
test_key = 'test'

img_with_annot_embedded_path = 'img_annot_embed'

if os.path.exists(img_with_annot_embedded_path):
    shutil.rmtree(img_with_annot_embedded_path)    
os.mkdir(img_with_annot_embedded_path)


for year_dir in [dir_2007, dir_2012]:
    if os.path.exists(year_dir):
        shutil.rmtree(year_dir)
    for in_dir in [annot_dir, img_dir, image_sets_dir]:
        if in_dir == image_sets_dir:
            os.makedirs(os.path.join(year_dir, in_dir, train_val_annot_dir), exist_ok=True)
        else:
            os.makedirs(os.path.join(year_dir, in_dir), exist_ok=True)

train_part = 0.9
val_part = 0.05
test_part = 1-train_part-val_part

print(row_num)
train_image_set = []
reader = pd.read_csv(annotation_name)
coor_column = reader['bbox']
img_name_column = reader['patch_number']

images_names = {'train':[], 'val':[], 'test':[]}

for i, (patch_number, polygon) in tqdm.tqdm(enumerate(zip(img_name_column, coor_column)), total = row_num):
    if i > 0 and img_name_column[i] == img_name_column[i-1]:
        continue
    if i != 0:
        if i < train_part*row_num:
            key = train_key
        elif (val_part+train_part)*row_num > i > train_part*row_num:
            key = val_key
        else:
            key = test_key     
        
        file_to_read = 'patch_' + str(patch_number) + '.png'
        img_full_path  =dir_with_imgs + file_to_read
        img = cv2.imread(img_full_path)
        if img.shape[2] != 3:
            print("problem kanalowy")
            continue
        img_shape = img.shape[:2]
        
        if get_coors_from_polygon(polygon) is not None:
            images_names[key].append(str(patch_number))
            for year_dir in [dir_2007, dir_2012]:
                img_annot_list = []
                cv2.imwrite(os.path.join(year_dir, img_dir) + '/' + str(patch_number)+'.jpg', img)
                writer = Writer(os.path.join(year_dir, img_dir) + '/' + str(patch_number)+'.jpg', img_shape[0], img_shape[1],database=year_dir)
                
                for additional_index in range(i, row_num):
                    if img_name_column[additional_index] == patch_number:
                        coors = get_coors_from_polygon(coor_column[additional_index])
                        if coors is not None:
                            writer.addObject('cow', coors[0], coors[1], coors[2], coors[3])
                            img_annot_list.append(coors)
                    else:
                        break
                writer.save(os.path.join(year_dir, annot_dir) + '/' + str(patch_number)+'.xml')
            embed_annot(img, str(patch_number)+'.png', img_with_annot_embedded_path, img_annot_list)
                

for some_dir in [dir_2007, dir_2012]:
    for annot_type in ['train', 'trainval', 'test', 'val']:
        with open(os.path.join(some_dir, image_sets_dir, train_val_annot_dir)+'/'+annot_type+'.txt', 'w') as img_list_file:    
            if annot_type == 'trainval':
                for line in images_names['train']:
                    img_list_file.write(line+'\n')
                for line in images_names['val']:
                    img_list_file.write(line+'\n')
            else:
                for line in images_names[annot_type]:
                    img_list_file.write(line+'\n')


for created_dir in [dir_2007, dir_2012]:
    if os.path.exists(os.path.join(voc_path, created_dir)):
        shutil.rmtree(os.path.join(voc_path, created_dir))
    subprocess.call(['cp','-r',  created_dir, voc_path ])



subprocess.call(['python', 'coco.py'])