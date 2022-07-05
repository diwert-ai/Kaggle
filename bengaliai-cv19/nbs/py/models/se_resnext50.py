import tensorflow as tf
from tf2cv.model_provider import get_model as tf2cv_get_model
from tensorflow.keras import layers


def build_srnext_head(x_in, n, name=None, drop_out=None):
    #x = layers.Activation('Mish', name='mish_act2_'+name) (x_in)
    if drop_out is not None:
        x = layers.Dropout(drop_out)(x_in)
    else: 
        x = x_in
    x = layers.Dense(n, name=name, activation='softmax')(x)
    return x

def build_srnext_head_wod(x_in, n, name=None):
    #x = layers.Activation('Mish', name='mish_act2_'+name) (x_in)
    #x = layers.Dropout(drop_out)(x_in)
    x = layers.Dense(n, name=name, activation='softmax')(x_in)
    return x

def build_se_resnext50_model(img_size=128, drop_out=0.5):
    x_in = layers.Input(shape=(img_size, img_size, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(img_size,img_size))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
    x = layers.GlobalAveragePooling2D()(x)
    #x_max = layers.GlobalMaxPooling2D()(x)
    #x = layers.Concatenate()([x_max, x_avg])
    out_root = build_srnext_head(x, 168,'root',drop_out=drop_out)
    out_vowel = build_srnext_head(x, 11,'vowel',drop_out=drop_out)
    out_consonant = build_srnext_head(x,7,'consonant',drop_out=drop_out)
    
    model = tf.keras.Model(inputs=x_in, outputs=[out_root, out_vowel, out_consonant])
    
    return model



def build_se_resnext50_root_model(img_size=128, drop_out=0.5):
    x_in = layers.Input(shape=(img_size, img_size, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(img_size,img_size))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
    x = layers.GlobalAveragePooling2D()(x)
    #x_max = layers.GlobalMaxPooling2D()(x)
    #x = layers.Concatenate()([x_max, x_avg])
    out_root = build_srnext_head(x, 168,'root',drop_out=drop_out)
    model = tf.keras.Model(inputs=x_in, outputs=out_root)
    
    return model

def build_se_resnext50_fs_root_model(height=137,width=236, drop_out=None):
    x_in = layers.Input(shape=(height, width, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(height,width))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
    x = layers.GlobalAveragePooling2D()(x)
    #x_max = layers.GlobalMaxPooling2D()(x)
    #x = layers.Concatenate()([x_max, x_avg])
    out_root = build_srnext_head(x, 168,'root',drop_out=drop_out)
    model = tf.keras.Model(inputs=x_in, outputs=out_root)
    
    return model
    
def build_se_resnext50_model_fs(height=137,width=236, drop_out=0.5):
    x_in = layers.Input(shape=(height, width, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(height,width))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
    x = layers.GlobalAveragePooling2D()(x)
    #x_max = layers.GlobalMaxPooling2D()(x)
    #x = layers.Concatenate()([x_max, x_avg])
    out_root = build_srnext_head(x, 168,'root',drop_out=drop_out)
    out_vowel = build_srnext_head(x, 11,'vowel',drop_out=drop_out)
    out_consonant = build_srnext_head(x,7,'consonant',drop_out=drop_out)
    
    model = tf.keras.Model(inputs=x_in, outputs=[out_root, out_vowel, out_consonant])
    
    return model

def build_se_resnext50_model_fs_wod(height=137,width=236):
    x_in = layers.Input(shape=(height, width, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(height,width))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
    x = layers.GlobalAveragePooling2D()(x)
    #x_max = layers.GlobalMaxPooling2D()(x)
    #x = layers.Concatenate()([x_max, x_avg])
    out_root = build_srnext_head_wod(x, 168,'root')
    out_vowel = build_srnext_head_wod(x, 11,'vowel')
    out_consonant = build_srnext_head_wod(x,7,'consonant')
    
    model = tf.keras.Model(inputs=x_in, outputs=[out_root, out_vowel, out_consonant])
    
    return model

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                        axis = [1, 2], 
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool
    
def build_se_resnext50_model_GeM_fs(height=137,width=236, drop_out=None):
    x_in = layers.Input(shape=(height, width, 1))
    se_model =  tf2cv_get_model("seresnext50_32x4d", pretrained=False, data_format="channels_last", in_channels=1, in_size=(height,width))
    se_backbone  = se_model.layers[0]
    x = se_backbone.layers[0](x_in)
    for i in range(4):
        x = se_backbone.layers[i+1](x)
        
    lambda_layer = layers.Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x)

    out_root = build_srnext_head(x, 168,'root',drop_out=drop_out)
    out_vowel = build_srnext_head(x, 11,'vowel',drop_out=drop_out)
    out_consonant = build_srnext_head(x,7,'consonant',drop_out=drop_out)
    
    model = tf.keras.Model(inputs=x_in, outputs=[out_root, out_vowel, out_consonant])
    
    return model
