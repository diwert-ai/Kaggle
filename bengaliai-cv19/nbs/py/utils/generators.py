import tensorflow as tf
import numpy as np
import cv2

_STATS = (0.0692, 0.2051)
_STATS_FS = (0.055029564364430086, 0.17228061284674265)

def split_into_3_outputs(y_batch):
    
    y_root =tf.keras.utils.to_categorical(y_batch[0],168)
    y_vowel=tf.keras.utils.to_categorical(y_batch[1],11)
    y_cons =tf.keras.utils.to_categorical(y_batch[2],7)
    
    return [y_root,y_vowel,y_cons]

def aux_data_gen(generator):
    while True:
        batch = next(generator)
        batch_x = (batch[0].astype(np.float32)/255.0 - _STATS[0])/_STATS[1]
        yield batch_x, split_into_3_outputs(batch[1])

def mixup_data_gen(generator1, generator2, alpha=0.4):
    while True:
      x1,y1 = next(generator1)
      x2,y2 = next(generator2)
      bs = x1.shape[0]
      l = np.random.beta(alpha, alpha, bs)

      y1_root =  y1[0]
      y1_vowel = y1[1]
      y1_cons =  y1[2]

      y2_root =  y2[0]
      y2_vowel = y2[1]
      y2_cons =  y2[2]

      x_l = l.reshape(bs, 1, 1, 1)
      y_l = l.reshape(bs, 1)

      x = x1 * x_l + x2 * (1 - x_l)

      y_root =  y1_root  *  y_l + y2_root  * (1 - y_l)
      y_vowel = y1_vowel *  y_l + y2_vowel * (1 - y_l)
      y_cons =  y1_cons  *  y_l + y2_cons  * (1 - y_l)

      yield x,[y_root,y_vowel,y_cons]

def hybrid_data_gen (mixup_gen,cutout_gen):
    while True:
        p = np.random.rand()
        if p > 0.5:
            batch_x,y = next(mixup_gen)
        else:
            batch_x,y = next(cutout_gen)
        yield batch_x,y

def prep_batch (batch):
    batch_x = (batch[0].astype(np.float32)/255.0 - _STATS[0])/_STATS[1]
    y_root =tf.keras.utils.to_categorical(batch[1][0],168)
    y_vowel=tf.keras.utils.to_categorical(batch[1][1],11)
    y_cons =tf.keras.utils.to_categorical(batch[1][2],7)
    return batch_x,[y_root,y_vowel,y_cons]
    
def prep_root_batch (batch):
    batch_x = (batch[0].astype(np.float32)/255.0 - _STATS[0])/_STATS[1]
    y_root=tf.keras.utils.to_categorical(batch[1][0],168)
    return batch_x,[y_root]
    
def prep_batch_fs_shuffled(batch, st=_STATS_FS):
    batch_x = (batch[0].astype(np.float32)/255.0 - st[0])/st[1]
    y_root =tf.keras.utils.to_categorical(batch[1][0],168)
    y_vowel=tf.keras.utils.to_categorical(batch[1][1],11)
    y_cons =tf.keras.utils.to_categorical(batch[1][2],7)
    
    bs = batch_x.shape[0]
    
    shuffled_idx = np.random.permutation(bs)
    batch_x_shuffled = np.zeros((bs,batch_x.shape[1],batch_x.shape[2],1))
    y_root_shuffled = np.zeros((bs,168))
    y_vowel_shuffled = np.zeros((bs,11))
    y_cons_shuffled = np.zeros((bs,7))
    for i in range(bs):
        j = shuffled_idx[i]
        batch_x_shuffled[i]=batch_x[j]
        y_root_shuffled[i]=y_root[j]
        y_vowel_shuffled[i]=y_vowel[j]
        y_cons_shuffled[i]=y_cons[j]
    
    return batch_x_shuffled,[y_root_shuffled,y_vowel_shuffled,y_cons_shuffled]
    

def prep_batch_fs (batch, st=_STATS_FS):
    batch_x = (batch[0].astype(np.float32)/255.0 - st[0])/st[1]
    y_root =tf.keras.utils.to_categorical(batch[1][0],168)
    y_vowel=tf.keras.utils.to_categorical(batch[1][1],11)
    y_cons =tf.keras.utils.to_categorical(batch[1][2],7)
    return batch_x,[y_root,y_vowel,y_cons]

def mix_batches(batch1,batch2,alpha=0.4):
    x1,y1 = prep_batch(batch1)
    x2,y2 = prep_batch(batch2)
    bs = x1.shape[0]
    assert(bs == x2.shape[0])
  
    l = np.random.beta(alpha, alpha, bs)

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

    x_l = l.reshape(bs, 1, 1, 1)
    y_l = l.reshape(bs, 1)

    x = x1 * x_l + x2 * (1 - x_l)

    y_root =  y1_root  *  y_l + y2_root  * (1 - y_l)
    y_vowel = y1_vowel *  y_l + y2_vowel * (1 - y_l)
    y_cons =  y1_cons  *  y_l + y2_cons  * (1 - y_l)

    return x,[y_root,y_vowel,y_cons]

def mix_root_batches(batch1,batch2,alpha=0.4):
    x1,y1 = prep_root_batch(batch1)
    x2,y2 = prep_root_batch(batch2)
    bs = x1.shape[0]
    assert(bs == x2.shape[0])
  
    l = np.random.beta(alpha, alpha, bs)

    y1_root =  y1[0]

    y2_root =  y2[0]


    x_l = l.reshape(bs, 1, 1, 1)
    y_l = l.reshape(bs, 1)

    x = x1 * x_l + x2 * (1 - x_l)

    y_root =  y1_root  *  y_l + y2_root  * (1 - y_l)


    return x,[y_root]

def mix_batches_fs(batch1,batch2,alpha=0.4, st=_STATS_FS):
    x1,y1 = prep_batch_fs(batch1,st=st)
    x2,y2 = prep_batch_fs(batch2,st=st)
    bs = x1.shape[0]
    assert(bs == x2.shape[0])
    

  
    l = np.random.beta(alpha, alpha, bs)

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

    x_l = l.reshape(bs, 1, 1, 1)
    y_l = l.reshape(bs, 1)

    x = x1 * x_l + x2 * (1 - x_l)

    y_root =  y1_root  *  y_l + y2_root  * (1 - y_l)
    y_vowel = y1_vowel *  y_l + y2_vowel * (1 - y_l)
    y_cons =  y1_cons  *  y_l + y2_cons  * (1 - y_l)

    return x,[y_root,y_vowel,y_cons]

def mix_batches_fs_shuffled(batch1,batch2,alpha=0.4, st=_STATS_FS):
    x1,y1 = prep_batch_fs_shuffled(batch1,st=st)
    x2,y2 = prep_batch_fs_shuffled(batch2,st=st)
    bs = x1.shape[0]
    assert(bs == x2.shape[0])
    

  
    l = np.random.beta(alpha, alpha, bs)

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

    x_l = l.reshape(bs, 1, 1, 1)
    y_l = l.reshape(bs, 1)

    x = x1 * x_l + x2 * (1 - x_l)

    y_root =  y1_root  *  y_l + y2_root  * (1 - y_l)
    y_vowel = y1_vowel *  y_l + y2_vowel * (1 - y_l)
    y_cons =  y1_cons  *  y_l + y2_cons  * (1 - y_l)

    return x,[y_root,y_vowel,y_cons]
    

def cutmix_root_batches(batch1,batch2,alpha=0.4):
    x1,y1 = prep_root_batch(batch1)
    x2,y2 = prep_root_batch(batch2)

    bs = x1.shape[0]
    assert(bs == x2.shape[0])
    
    _IMG_SIZE=x1.shape[1]

    y1_root =  y1[0]
    y2_root =  y2[0]


  
    cut_ratio = np.random.beta(alpha, alpha, bs)
    cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
    label_ratio = cut_ratio.reshape(bs, 1)
    cut_img = x2

    x = x1
    for i in range(bs):
        cut_size = int((_IMG_SIZE-1) * cut_ratio[i])
        by1 = np.random.randint(0, (_IMG_SIZE-1) - cut_size)
        bx1 = np.random.randint(0, (_IMG_SIZE-1) - cut_size)
        by2 = by1 + cut_size
        bx2 = bx1 + cut_size
        cut_arr = cut_img[i][by1:by2, bx1:bx2]
        cutmix_img = x1[i]
        cutmix_img[by1:by2, bx1:bx2] = cut_arr
        x[i] = cutmix_img
            
    y_root =  y1_root *  (1 - (label_ratio ** 2)) + y2_root *  (label_ratio ** 2)

    return x,[y_root]

def cutmix_batches_fs(batch1,batch2,alpha=0.4,h=137,w=236,st=_STATS_FS):
    x1,y1 = prep_batch_fs(batch1,st=st)
    x2,y2 = prep_batch_fs(batch2,st=st)

    bs = x1.shape[0]
    assert(bs == x2.shape[0])
    

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

  
    cut_ratio = np.random.beta(alpha, alpha, bs)
    cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
    label_ratio = cut_ratio.reshape(bs, 1)
    cut_img = x2

    x = x1
    for i in range(bs):
        #cut_size = int((_IMG_SIZE-1) * cut_ratio[i])
        cut_size_w = int((w-1)*cut_ratio[i])
        cut_size_h = int((h-1)*cut_ratio[i])
        by1 = np.random.randint(0, (h-1) - cut_size_h)
        bx1 = np.random.randint(0, (w-1) - cut_size_w)
        by2 = by1 + cut_size_h
        bx2 = bx1 + cut_size_w
        cut_arr = cut_img[i][by1:by2, bx1:bx2]
        cutmix_img = x1[i]
        cutmix_img[by1:by2, bx1:bx2] = cut_arr
        x[i] = cutmix_img
            
    y_root =  y1_root *  (1 - (label_ratio ** 2)) + y2_root *  (label_ratio ** 2)
    y_vowel = y1_vowel * (1 - (label_ratio ** 2)) + y2_vowel * (label_ratio ** 2)
    y_cons =  y1_cons *  (1 - (label_ratio ** 2)) + y2_cons *  (label_ratio ** 2)


    return x,[y_root,y_vowel,y_cons]


    
def h_generator(m_gen1,m_gen2,c_gen):
    while True:
        p = np.random.rand()
        if p>0.5:
            batch1 = next(m_gen1)
            batch2 = next(m_gen2)
            x_res,y_res = mix_batches(batch1,batch2)
        else:
            batch = next(c_gen)
            x_res,y_res = prep_batch(batch)

        yield x_res,y_res

def cutmix_batches(batch1,batch2,alpha=0.4):
    x1,y1 = prep_batch(batch1)
    x2,y2 = prep_batch(batch2)

    bs = x1.shape[0]
    assert(bs == x2.shape[0])
    
    _IMG_SIZE=x1.shape[1]

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

  
    cut_ratio = np.random.beta(alpha, alpha, bs)
    cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
    label_ratio = cut_ratio.reshape(bs, 1)
    cut_img = x2

    x = x1
    for i in range(bs):
        cut_size = int((_IMG_SIZE-1) * cut_ratio[i])
        by1 = np.random.randint(0, (_IMG_SIZE-1) - cut_size)
        bx1 = np.random.randint(0, (_IMG_SIZE-1) - cut_size)
        by2 = by1 + cut_size
        bx2 = bx1 + cut_size
        cut_arr = cut_img[i][by1:by2, bx1:bx2]
        cutmix_img = x1[i]
        cutmix_img[by1:by2, bx1:bx2] = cut_arr
        x[i] = cutmix_img
            
    y_root =  y1_root *  (1 - (label_ratio ** 2)) + y2_root *  (label_ratio ** 2)
    y_vowel = y1_vowel * (1 - (label_ratio ** 2)) + y2_vowel * (label_ratio ** 2)
    y_cons =  y1_cons *  (1 - (label_ratio ** 2)) + y2_cons *  (label_ratio ** 2)


    return x,[y_root,y_vowel,y_cons]

def hmcm_generator(gen1,gen2):
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        p = np.random.rand()
        if p>0.5:
            x_res,y_res = mix_batches(batch1,batch2)
        else:
            x_res,y_res = cutmix_batches(batch1,batch2)

        yield x_res,y_res

def g3_generator(gen1,gen2, gen3):
    while True:
        p = np.random.rand()
        if p>0.5:
            batch = next(gen3)
            x_res,y_res = prep_batch(batch)
        else:
            batch1 = next(gen1)
            batch2 = next(gen2)
            q = np.random.rand()
            if q>0.5:
                x_res,y_res = mix_batches(batch1,batch2)
            else:
                x_res,y_res = cutmix_batches(batch1,batch2)
        yield x_res,y_res

def g3_generator_fs(gen1,gen2, gen3,h=137,w=236,st=_STATS_FS):
    while True:
        p = np.random.rand()
        if p>0.5:
            batch = next(gen3)
            x_res,y_res = prep_batch_fs(batch,st=st)
        else:
            batch1 = next(gen1)
            batch2 = next(gen2)
            q = np.random.rand()
            if q>0.5:
                x_res,y_res = mix_batches_fs(batch1,batch2,st=st)
            else:
                x_res,y_res = cutmix_batches_fs(batch1,batch2,h=h,w=w,st=st)
        yield x_res,y_res

def g3_root_generator(gen1,gen2, gen3):
    while True:
        p = np.random.rand()
        if p>0.5:
            batch = next(gen3)
            x_res,y_res = prep_root_batch(batch)
        else:
            batch1 = next(gen1)
            batch2 = next(gen2)
            q = np.random.rand()
            if q>0.5:
                x_res,y_res = mix_root_batches(batch1,batch2)
            else:
                x_res,y_res = cutmix_root_batches(batch1,batch2)
        yield x_res,y_res

def cutmix_generator(gen1,gen2):
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        x_res,y_res = cutmix_batches(batch1,batch2)

        yield x_res,y_res

def cutout_batch_fs(batch, p=0.5, s_l=0.05,s_h=0.3,r_1=0.2,r_2=5.0, st=_STATS_FS):
        x,y = prep_batch_fs(batch,st=st)
        p_1 = np.random.rand()

        if p_1 > p:
            return x,y

        bs,img_h, img_w, img_c = x.shape

        for i in range(bs):
          while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

          x[i,top:top + h, left:left + w, :] = 0

        return x,y

def ps_batch_fs(batch, n=4, p=0.5, s_l=0.05,s_h=0.15,r_1=0.8,r_2=1.2, st=_STATS_FS):
        x,y = prep_batch_fs(batch,st=st)
        p_1 = np.random.rand()

        if p_1 > p:
            return x,y

        bs,img_h, img_w, img_c = x.shape

        for i in range(bs):
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / n*r))
            h = int(np.sqrt(s * r/n))
            left = np.random.randint(0, img_w-w-1,size=n)
            top =  np.random.randint(0, img_h-h-1,size=n)
            for j in range(n):
              x[i,top[j]:top[j] + h, left[j]:left[j] + w, :] = 0

        return x,y

#cutout+progressive+mixup
def cpsm_generator_fs(gen1,gen2, gen3,st=_STATS_FS):
    while True:
        p = np.random.rand()
        #print(f'p={p}')
        if p>0.5:
            batch = next(gen3)
            q  = np.random.rand()
            #print(f'q={q}')
            if q>0.5:
              #print('cutout')
              x_res,y_res = cutout_batch_fs(batch,st=st)
            else:
              #print('ps')
              ps_n = np.random.randint(4,16)
              x_res,y_res = ps_batch_fs(batch,n=ps_n)
        else:
            #print('mixup')
            batch1 = next(gen1)
            batch2 = next(gen2)
            mx_alpha = np.random.uniform(0.1,0.4)
            x_res,y_res = mix_batches_fs(batch1,batch2,alpha=mx_alpha,st=st)

        yield x_res,y_res

def test_batch_generator(frame, train_dir,batch_size=64, img_size=128):    
    
    num_imgs = len(frame)
    
    for batch_start in range(0, num_imgs,batch_size):   
            cur_batch_size = min(num_imgs,batch_start+batch_size)-batch_start

            idx = np.arange(batch_start,batch_start+cur_batch_size)
            names_batch = frame.iloc[idx,0].values
            imgs_batch = np.zeros((cur_batch_size,img_size,img_size,1))
            
            for j in range(cur_batch_size):
                img = cv2.imread(train_dir+'/'+names_batch[j],0)
                img = (img.astype(np.float32)/255.0 - _STATS[0])/_STATS[1]
                imgs_batch[j,:,:,0] = img

            yield imgs_batch

def test_batch_generator_fs(frame, train_dir,batch_size=64, height=137, width=236,st=_STATS_FS):    
    
    num_imgs = len(frame)
    
    for batch_start in range(0, num_imgs,batch_size):   
            cur_batch_size = min(num_imgs,batch_start+batch_size)-batch_start

            idx = np.arange(batch_start,batch_start+cur_batch_size)
            names_batch = frame.iloc[idx,0].values
            imgs_batch = np.zeros((cur_batch_size,height,width,1))
            
            for j in range(cur_batch_size):
                img = cv2.imread(train_dir+'/'+names_batch[j],0)
                img = (img.astype(np.float32)/255.0 - st[0])/st[1]
                imgs_batch[j,:,:,0] = img

            yield imgs_batch

#reference https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
    
    
def rand_bbox(size, lam):
    H = size[1]
    W = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_batches_fs2(batch1,batch2,alpha=1.0,h=137,w=236,st=_STATS_FS):
    x1,y1 = prep_batch_fs(batch1,st=st)
    x2,y2 = prep_batch_fs(batch2,st=st)

    bs = x1.shape[0]
    assert(bs == x2.shape[0])

    shape = x1.shape
    

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

  
    cut_ratio = np.random.beta(alpha, alpha, bs)
    label_ratio = cut_ratio.reshape(bs, 1)

    x = x1
    for i in range(bs):
        bx1,by1,bx2,by2 = rand_bbox(shape,cut_ratio[i])
        x[i,by1:by2,bx1:bx2,:] = x2[i,by1:by2, bx1:bx2,:]
            
    y_root =  y1_root *  label_ratio  + y2_root *  (1-label_ratio)
    y_vowel = y1_vowel * label_ratio  + y2_vowel * (1-label_ratio)
    y_cons =  y1_cons *  label_ratio  + y2_cons *  (1-label_ratio)


    return x,[y_root,y_vowel,y_cons]

def cutmix_batches_fs2_shuffled(batch1,batch2,alpha=1.0,h=137,w=236,st=_STATS_FS):
    x1,y1 = prep_batch_fs_shuffled(batch1,st=st)
    x2,y2 = prep_batch_fs_shuffled(batch2,st=st)

    bs = x1.shape[0]
    assert(bs == x2.shape[0])

    shape = x1.shape
    

    y1_root =  y1[0]
    y1_vowel = y1[1]
    y1_cons =  y1[2]

    y2_root =  y2[0]
    y2_vowel = y2[1]
    y2_cons =  y2[2]

  
    cut_ratio = np.random.beta(alpha, alpha, bs)
    label_ratio = cut_ratio.reshape(bs, 1)

    x = x1
    for i in range(bs):
        bx1,by1,bx2,by2 = rand_bbox(shape,cut_ratio[i])
        x[i,by1:by2,bx1:bx2,:] = x2[i,by1:by2, bx1:bx2,:]
            
    y_root =  y1_root *  label_ratio  + y2_root *  (1-label_ratio)
    y_vowel = y1_vowel * label_ratio  + y2_vowel * (1-label_ratio)
    y_cons =  y1_cons *  label_ratio  + y2_cons *  (1-label_ratio)


    return x,[y_root,y_vowel,y_cons]
