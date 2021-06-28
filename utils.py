import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import os

def read_xml(path):
    cls_idx = {'face':1}
    xml_list = []
    #for xml_file in glob.glob(path + '/*.xml'):
    for xml_file in os.listdir(path):
        xml_file = os.path.join(path,xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_data = [root.find('path').text]
        for member in root.findall('object'):
            value = [
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     cls_idx[member[0].text]
                     ]
            xml_data.append(value)
        xml_list.append(xml_data)
    return xml_list

def decode_xml(xml_list,now_index,
               num_class=2,num_query=100,
               batch_size=1,img_size=224):
    if(now_index+batch_size>len(xml_list)):
        now_index = 0
    location = np.zeros([batch_size,num_query,4],dtype=np.float32)
    image = np.zeros([batch_size,img_size,img_size,3],dtype=np.float32)
    class_label = np.zeros([batch_size,num_query,num_class],dtype=np.float32)
    
    for i in range(batch_size):
        image[now_index] = mpimg.imread(xml_list[now_index][0])
        for j in range(1,len(xml_list[now_index])):
            x,y = xml_list[now_index][j][0],xml_list[now_index][j][1]
            w = xml_list[now_index][j][2]-xml_list[now_index][j][0]
            h = xml_list[now_index][j][3]-xml_list[now_index][j][1]
            location[now_index][j-1] = [x/img_size,y/img_size,w/img_size,h/img_size]
            class_label[now_index][j-1][xml_list[now_index][1][-1]] = 1
        for j in range(len(xml_list[now_index])-1,num_query):
            class_label[now_index][j][0] = 1
        
    return image,class_label,location

def draw_bounding_box(image,location):
    for i in range(len(location)):
        #if(sum(location[i])==0):
        #    break
        x,y,w,h = location
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
    

    

train_data = read_xml('E:\\tensorflow\\DERT\\data')
a,b,c = decode_xml(train_data,0,num_query=10)
#print(b)
#draw_bounding_box(a[0],c[0])


