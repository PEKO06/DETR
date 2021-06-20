import tensorflow as tf
import numpy as np
from PositionEmbedding import *
from resnet import *
from Transformer import *


class dert():
    
    def __init__(self,input_img,trainable,num_class=2,num_query=100,
                 num_encoder_layers=6,num_decoder_layers=6):
        
        self.project_conv_dim = 256
        self.num_class = 2
        self.num_query = num_query
        Resnet = resnet(input_img,trainable)
        self.Resnet_backbone = Resnet.build_net_18()
        masks = tf.cast(tf.zeros_like(self.Resnet_backbone)[:,:,:,0],dtype=tf.bool)
        self.pos_encoding = PositionEmbeddingSine(self.project_conv_dim//2).get_position_embbed(masks)
        self.query_embbed = tf.get_variable('query_embbed',shape=[num_query,self.project_conv_dim],
                                            dtype=tf.float32,initializer=tf.zeros_initializer())
        self.query_embbed = tf.expand_dims(self.query_embbed,axis=0)
        self.query_embbed = tf.tile(self.query_embbed,[tf.shape(self.pos_encoding)[0],1,1])
        self.target = tf.zeros_like(self.query_embbed)
        
        print(self.pos_encoding)
        print(self.query_embbed)
        print(self.target)
    
    def get_dert_model(self):
        project_conv = tf.layers.conv2d(self.Resnet_backbone,self.project_conv_dim,1,1,padding='SAME')
        feature_embbed = tf.reshape(project_conv,(-1,project_conv.shape[1]*project_conv.shape[2],self.project_conv_dim))
        print('feature_embbed',feature_embbed)
        Transformer_encoder = transformer_encoder(feature_embbed,self.pos_encoding,model_dim=256,
                                                  num_head=8,dim_feedforward=1024,
                                                  dropout=0.1,num_encoder_layers=6)
        print('Transformer_encoder',Transformer_encoder)
        
        Transformer_decoder = transformer_decoder(self.target,Transformer_encoder,self.pos_encoding,self.query_embbed,
                                                  model_dim=256,num_head=8,dim_feedforward=1024,
                                                  dropout=0.1,num_decoder_layers=6)
        
        
        print('Transformer_decoder',Transformer_decoder)
        
        class_out = tf.layers.dense(Transformer_decoder,self.num_class)
        
        box_out = tf.layers.dense(Transformer_decoder,self.project_conv_dim//2)
        box_out = tf.nn.relu(box_out)
        box_out = tf.layers.dense(box_out,4)
        box_out = tf.nn.sigmoid(box_out)
        
        return class_out,box_out
    
