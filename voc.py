from pascal_voc_writer import Writer
import csv
import os
import tqdm
import cv2

dir_with_imgs = 'data_out/'

annotation_name = dir_with_imgs + 'annotation.csv'

row_num = 0
with open(annotation_name,'r') as x:
    for line in x:
        row_num += 1

row_num -= 1

annot_dir_2007 = '2007Annotations'
annot_dir_2012 = '2012Annotations'
img_dir = 'JPEGImages'

train_path = 'train'
val_path = 'val'
test_path = 'test'

for f in [train_path, val_path, test_path]:    
    if not os.path.exists(f):
        os.mkdir(f)
    for p in [annot_dir_2007, annot_dir_2012, img_dir]:
        if not os.path.exists(f + '/' + p):
            os.mkdir(f + '/' + p)

train_part = 0.8
test_part = 0.1
val_part = 0.1

print(row_num)
train_image_set = []
with open(annotation_name, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in tqdm.tqdm(enumerate(reader), total = row_num):
        if i != 0:
            if i < row_num*train_part:
                if 1 ==1:
                    file_to_read = 'patch_' + row[0].split(',')[1] + '.png'
                    #print(dir_with_imgs + file_to_read)
                    img_full_path  =dir_with_imgs + file_to_read
                    img = cv2.imread(img_full_path)
                    img_shape = img.shape[:2]
                    annot = row[-10:-2]
                    annot[0] = annot[0][2:]
                    for i in range(1,8):
                        annot[i] = annot[i][:-1]
                    
                    #print(annot[2], annot[1], annot[0], annot[5])
                    #print(file_to_read)
                    train_image_set.append('patch_' + row[0].split(',')[1])
                    cv2.imwrite(train_path + '/' + img_dir + '/' + file_to_read[:-3]+'jpg', img)
                    for database_name, annot_dir in zip(['VOC2007', 'VOC2012'], [annot_dir_2007, annot_dir_2012]):
                        writer = Writer(train_path + '/' + img_dir + '/' + file_to_read, img_shape[0], img_shape[1],database=database_name)
                        print("ann", annot, "bb", annot[2], annot[1], annot[0], annot[5])
                        for i in [2,1,0,5]:
                            if len(annot[i]):
                                annot[i] = str(int(float(annot[i])))

                        writer.addObject('cow', annot[2], annot[1], annot[0], annot[5])
                        writer.save(train_path + '/' + annot_dir + '/'+ file_to_read[:-3]+'xml')