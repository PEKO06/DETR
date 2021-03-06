import numpy as np
import tensorflow as tf


class PositionEmbeddingSine():

    def __init__(self, num_pos_features=64, temperature=10000,
                 normalize=False, scale=None, eps=1e-6):

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps

    

    def get_position_embbed(self,mask):
        not_mask = tf.cast(~mask, tf.float32)
        y_embed = tf.math.cumsum(not_mask, axis=1)
        x_embed = tf.math.cumsum(not_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tf.range(self.num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t
        
        pos_x = tf.stack([tf.math.sin(pos_x[..., 0::2]),
                          tf.math.cos(pos_x[..., 1::2])], axis=4)

        pos_y = tf.stack([tf.math.sin(pos_y[..., 0::2]),
                          tf.math.cos(pos_y[..., 1::2])], axis=4)
        #print(pos_x)
        pos_x = tf.reshape(pos_x,(-1,pos_x.shape[1]*pos_x.shape[2],pos_x.shape[3]*2))
        pos_y = tf.reshape(pos_y,(-1,pos_y.shape[1]*pos_y.shape[2],pos_y.shape[3]*2))

        pos_emb = tf.concat([pos_y, pos_x], axis=2)
        return pos_emb
    