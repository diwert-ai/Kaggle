import mish

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.layers import Activation

def build_backbone(x_in, img_size=128):
    backbone_net = DenseNet121(include_top=False, weights=None, input_shape=(img_size, img_size, 1))
    x = backbone_net (x_in)
    x_avg = layers.GlobalAveragePooling2D()(x)
    x_max = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([x_max, x_avg])
    x = layers.Activation('Mish', name='mish_backbone') (x)
    return x
  
def build_head(x_in, n, name=None, drop_out=0.5,wd = 1e-2):
    x = layers.BatchNormalization()(x_in)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(wd), 
                        bias_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = layers.Activation('Mish', name='mish_act2_'+name) (x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(drop_out)(x)
    x = layers.Dense(n, name=name, activation='softmax')(x)
    return x
  
def build_md121_v2_model(img_size=128, drop_out=0.5, wd = 1e-2):
    x_in = layers.Input(shape=(img_size, img_size, 1))
    x = build_backbone(x_in, img_size)
    out_root = build_head(x, 168,'root',drop_out=drop_out,wd=wd)
    out_vowel = build_head(x, 11,'vowel',drop_out=drop_out,wd=wd)
    out_consonant = build_head(x,7,'consonant',drop_out=drop_out,wd=wd)
    
    model = tf.keras.Model(inputs=x_in, outputs=[out_root, out_vowel, out_consonant])
    
    return model