import tensorflow as tf
import numpy as np

class resnet:
    def __init__(self,input_x,trainable):
        
        def preprocess(input_x):
            x = input_x/255.0
            x = tf.image.resize_bilinear(x, (224,224))
            return x
        
        self.input_x = preprocess(input_x)
        self.trainable = trainable
        self.ch = 32
        self.expand_dims = 4
    
    def resnet_base(self,x,ch,scope='resnet_base'):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(x,64,7,2,padding='SAME')
            x = tf.layers.batch_normalization(x,trainable=self.trainable)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x,3,2,padding='SAME')
            return x
    
    def build_net_18(self):
        self.layer1 = self.resnet_base(self.input_x,self.ch)
        self.layer2 = self.resnet_block(self.layer1,self.ch,3,stride=1,scope='layer2')
        self.layer3 = self.resnet_block(self.layer2,self.ch*2,3,stride=2,scope='layer3')
        self.layer4 = self.resnet_block(self.layer3,self.ch*4,3,stride=2,scope='layer4')
        self.layer5 = self.resnet_block(self.layer4,self.ch*8,3,stride=2,scope='layer5')
        
        print(self.layer1)
        print(self.layer2)
        print(self.layer3)
        print(self.layer4)
        print(self.layer5)
        
        return self.layer5
        
    
    def resnet_block(self,x,ch,layer_num,stride,scope=''):
        with tf.variable_scope(scope):
            for i in range(layer_num-1):
                if(i==0 and stride==2):
                    x = self.resnet_bottleneck(x,ch,2,scope='bottleneck_{}'.format(i))
                else:
                    x = self.resnet_bottleneck(x,ch,1,scope='bottleneck_{}'.format(i))
        return x
    
    def resnet_bottleneck(self,x,ch,stride,scope=''):
        with tf.variable_scope(scope):
            short_cut = x
            x = tf.layers.conv2d(x,ch,1,1,padding='SAME')
            x = tf.layers.batch_normalization(x,trainable=self.trainable)
            x = tf.nn.relu(x)
            
            x = tf.layers.conv2d(x,ch,3,stride,padding='SAME')
            x = tf.layers.batch_normalization(x,trainable=self.trainable)
            x = tf.nn.relu(x)
            
            x = tf.layers.conv2d(x,ch*self.expand_dims,1,1,padding='SAME')
            x = tf.layers.batch_normalization(x,trainable=self.trainable)
            
            if(stride==2):
                short_cut = tf.layers.average_pooling2d(short_cut,2,2,padding='SAME')
            if(x.shape[-1]!=short_cut.shape[-1]):
                short_cut = tf.layers.conv2d(short_cut,ch*self.expand_dims,1,1,padding='SAME')
                short_cut = tf.layers.batch_normalization(short_cut,trainable=self.trainable)
            
            x = x+short_cut
            
            return tf.nn.relu(x)
