import tensorflow as tf
import numpy as np


def multi_head_attention(q,k,v,dim,num_head):
    
    devide_dim = dim//num_head
    q_hw,k_hw,v_hw = q.shape[1],k.shape[1],v.shape[1]
    #print(q,k,v)
    q = tf.layers.dense(q,dim)
    q = tf.reshape(q,(-1,q_hw*num_head,devide_dim))
    
    k = tf.layers.dense(k,dim)
    k = tf.reshape(k,(-1,k_hw*num_head,devide_dim))
    
    v = tf.layers.dense(v,dim)
    v = tf.reshape(v,(-1,v_hw*num_head,devide_dim))

    qk = tf.matmul(q,tf.transpose(k,[0,2,1]))*(devide_dim**(-0.5))
    qk_attention = tf.nn.softmax(qk,axis=-1)

    qkv = tf.matmul(qk_attention,v)
    qkv = tf.reshape(qkv,(-1,q_hw,dim))

    return qkv

def transformer_encoder(x,pos_embbed,model_dim=256,num_head=8,dim_feedforward=1024,
                        dropout=0.1,num_encoder_layers=6):
    
    for i in range(num_encoder_layers):
        q = x+pos_embbed
        k = x+pos_embbed
        v = x
        mh_attention = multi_head_attention(q,k,v,model_dim,num_head)
        mh_attention = tf.contrib.layers.layer_norm(mh_attention+x,begin_norm_axis=-1,begin_params_axis=-1)
        
        ffn = tf.layers.dense(mh_attention,dim_feedforward)
        ffn = tf.nn.relu(ffn)
        ffn = tf.layers.dense(ffn,model_dim)
        x = tf.contrib.layers.layer_norm(mh_attention+x,begin_norm_axis=-1,begin_params_axis=-1)
    
    return x

def transformer_decoder(target,encoder_memory,pos_embbed,query_embbed,model_dim=256,num_head=8,dim_feedforward=1024,
                        dropout=0.1,num_decoder_layers=6):
    
    for i in range(num_decoder_layers):
        query_q = target+query_embbed
        query_k = target+query_embbed
        query_v = target

        query_attention = multi_head_attention(query_q,query_k,query_v,model_dim,num_head)
        query_attention = tf.contrib.layers.layer_norm(query_attention+target,begin_norm_axis=-1,begin_params_axis=-1)
        
        q = target+query_attention
        k = encoder_memory+pos_embbed
        v = encoder_memory
        mh_attention = multi_head_attention(q,k,v,model_dim,num_head)
        mh_attention = tf.contrib.layers.layer_norm(mh_attention+target,begin_norm_axis=-1,begin_params_axis=-1)
        
        
        ffn = tf.layers.dense(mh_attention,dim_feedforward)
        ffn = tf.nn.relu(ffn)
        ffn = tf.layers.dense(ffn,model_dim)
        target = tf.contrib.layers.layer_norm(mh_attention+ffn,begin_norm_axis=-1,begin_params_axis=-1)
    return target