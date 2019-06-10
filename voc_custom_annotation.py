import xml.etree.ElementTree as ET
from os import getcwd
import os

sets=['elderly']

classes = ["glasses","key","person","phone","shoes","wallet"]
#classes = ["person"]

def convert_annotation(image_set, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(image_set, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    directory_in_str = "VOCdevkit/VOC"+image_set+"/JPEGImages/"
    directory = os.fsencode(directory_in_str)

    list_file = open('%s.txt'%(image_set), 'w')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") : 
            #print(directory)
            #print(filename)
            #print(os.path.splitext(filename)[0])
            image_path = filename
            image_id = os.path.splitext(filename)[0]
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s'%(wd, image_set, image_path))
            convert_annotation(image_set, image_id, list_file)
            list_file.write('\n')
            continue
        else:
            continue

    list_file.close()



