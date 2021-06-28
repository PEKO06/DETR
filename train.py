from model import DERT
from utils import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET

query = 10
def match_loss(x):
    class_label,bbox_label,class_logit,bbox_logit = x
    print('class_label',class_label)
    print('bbox_label',bbox_label)
    print('class_logit',class_logit)
    print('bbox_logit',bbox_logit)
    class_loss = list()
    box_loss = list()
    for i in range(class_logit.shape[0]):
        cs_ls = list()
        bx_ls = list()
        for j in range(class_logit.shape[0]):
            cs_ls.append(tf.reduce_sum(tf.pow(class_logit[j]-class_label[i],2)))
            bx_ls.append(tf.reduce_sum(tf.abs(bbox_logit[j]-bbox_label[i])))
        class_loss.append(tf.concat(tf.expand_dims(cs_ls,axis=0),axis=-1))
        box_loss.append(tf.concat(tf.expand_dims(bx_ls,axis=0),axis=-1))
    class_loss = tf.concat(class_loss,axis=0)
    box_loss = tf.concat(box_loss,axis=0)
    
    total_loss = list()
    index_save = list()
    for i in range(class_loss.shape[0]):
        a = tf.argmin(class_loss[i])
        index_save.append(tf.expand_dims(tf.cast(a,tf.float32),axis=-1))
        b = tf.constant(np.ones([query,1],dtype=np.float32)*10000)
        total_loss.append(tf.expand_dims(class_loss[i][a]+(box_loss[i][a]*class_label[a][1]),axis=-1))
        class_loss = tf.concat([class_loss[:,:a],b,class_loss[:,a+1:]],axis=-1)
    total_loss = tf.concat(total_loss,axis=-1)
    index_save = tf.concat(index_save,axis=-1)
    total_loss = tf.reduce_sum(total_loss)
    print('total_loss',total_loss)
    
    return [class_label,bbox_label,index_save,total_loss]

def draw_b_box(img,cls_logit_output,b_box_output,num_query=10):
    print(cls_logit_output)
    for i in range(num_query):
        if(cls_logit_output[i][1]>0.95):
            box = np.clip(np.array(b_box_output[i]*224,dtype=np.uint8),0,224)
            draw_bounding_box(np.array(img,dtype=np.uint8),box)
    

if(__name__=='__main__'):
    
    with tf.Graph().as_default(): 
        train_data = read_xml('E:\\tensorflow\\DERT\\data')
        input_img = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32)
        bounding_box_label = tf.placeholder(shape=[None,query,4],dtype=tf.float32)
        class_label = tf.placeholder(shape=[None,query,2],dtype=tf.float32)
        
        dert_model = DERT.dert(input_img,trainable=True,num_query=query)
        dert_class_logit,dert_bounding_box_out = dert_model.get_dert_model()
        dert_class_logit = tf.nn.softmax(dert_class_logit)
        
        #loss = match_loss(dert_class_logit,dert_bounding_box_out,
        #                  class_label,bounding_box_label,num_query=10)
        _,_,index,loss = tf.map_fn(lambda x: match_loss(x),[class_label,bounding_box_label,
                                                  dert_class_logit,dert_bounding_box_out])
        loss = tf.reduce_mean(loss)
        #print(index)
        print(loss)
        
        with tf.name_scope('opt'):
            global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            opt = tf.train.AdamOptimizer(1e-5,name='optimizer')
            with tf.control_dependencies(update_ops):
                grads = opt.compute_gradients(loss)
                train_op = opt.apply_gradients(grads, global_step=global_step)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(10000):
                batch_img,batch_class_label,batch_box_label = decode_xml(train_data,0,num_query=query)
                
                ls,_ = sess.run([loss,train_op],
                                feed_dict={input_img:batch_img,
                                           class_label:batch_class_label,
                                           bounding_box_label:batch_box_label})
                
                if(i%100==0 and i!=0):
                    d_cls_logit,d_b_box = sess.run([dert_class_logit,dert_bounding_box_out],
                                                   feed_dict={input_img:batch_img})
                    for j in range(d_cls_logit.shape[0]):
                        draw_b_box(batch_img[j],d_cls_logit[j],d_b_box[j],num_query=query)
                    print(ls)
                
        
                    
        
    