#dataset.py

from common import *
from configure import *


from augmentation import *

data_dir='/content'

def make_fold(mode='train-0'):
    
    if 'train' in mode:
        df = pd.read_csv(data_dir+'/train_extended.csv')
        fold = int(mode[-1])
        df.id_dir = df.id_dir.apply(lambda x: str(x).zfill(5))
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    if 'test' in mode:
        df_valid = pd.read_csv(data_dir+'/test_extended.csv')
        df_valid.id_dir = df_valid.id_dir.apply(lambda x: str(x).zfill(5))
        return df_valid

def null_augment(r):
    image = r['image']
    return r


class BTDataset(Dataset):
    def __init__(self, df,df_meta_ext,mri_type='FLAIR', augment=null_augment):
        super().__init__()
        self.df = df
        self.df_meta_ext = df_meta_ext
        self.mri_type = mri_type
        self.augment = augment
        self.length = len(df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        return string


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        df_imgs = self.df_meta_ext[(self.df_meta_ext['BraTS21ID']==d.BraTS21ID) & (self.df_meta_ext['mri_id']==self.mri_type)].reset_index()
        #print(f'ds {d[self.mri_type]-1} {d.BraTS21ID} {self.mri_type}')

        image = np.zeros((512,512))
        for i in range(64):
          dm = df_imgs.iloc[random.randint(0, d[self.mri_type]-1)]
          image_file = data_dir + '/%s/%s/%s/%s.jpg' % (d.set, d.id_dir,self.mri_type,dm.image_id)
          image_slice = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
          x = (i // 8)*64
          y = (i % 8)*64
          image[x:x+64,y:y+64]=image_slice
        
        mgmt = d.MGMT_value

        r = {
            'index' : index,
            'd' : d,
            'image' : image,
            'mgmt' : mgmt,
        }
        if self.augment is not None: r = self.augment(r)
        return r


def null_collate(batch):
    collate = defaultdict(list)

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)

    # ---
    batch_size = len(batch)
    mgmt = np.ascontiguousarray(np.stack(collate['mgmt'])).astype(np.float32)
    collate['mgmt'] = torch.from_numpy(mgmt)

    image = np.stack(collate['image'])
    image = image.reshape(batch_size, 1, image_size,image_size).repeat(3,1)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image)

    return collate

def draw_batch(imbatch):
  bs = imbatch.shape[0]
  fig, axs = plt.subplots(1, bs, figsize=(64, 64))
  for i in range(bs):
    axs[i].imshow(imbatch[i,0,:,:],cmap='gray')
  plt.show()

#===============================================================

def run_check_dataset():
    df_train, df_valid = make_fold(mode='train-1')
    #df_valid = make_fold(mode='test')

    dataset = BTDataset(df_valid,df_meta_ext) #null_augment
    print(dataset)

    for i in range(5):
        i = np.random.choice(len(dataset))
        r = dataset[i]

        print('index ' , i)
        print(r['d'])
        print(r['mgmt'])
        print('')
        #image_show('image', r['image'], resize=1)
        #cv2.waitKey(0)

    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate,
    )
    for t,batch in enumerate(loader):
        if t>30: break

        print(t, '-----------')
        print('index : ', batch['index'])
        print('image : ')
        print('\t', batch['image'].shape, batch['image'].is_contiguous())
        print('mgmt : ')
        print('\t', batch['mgmt'])
        print('\t', batch['mgmt'].shape, batch['mgmt'].is_contiguous())
        print('')

        draw_batch(batch['image'])

     

def run_check_augment():
    def augment(r):
        image = r['image']
        for fn in np.random.choice([
            lambda image : do_random_scale(image, mag=0.20),
            lambda image : do_random_stretch_y(image, mag=0.20),
            lambda image : do_random_stretch_x(image, mag=0.20),
            lambda image : do_random_shift(image, mag=int(0.20*image_size)),
            lambda image : (image)],1):
          image= fn(image)

        for fn in np.random.choice([
            lambda image : do_random_rotate(image, mag=15),
            lambda image : do_random_hflip(image),
            lambda image : (image)],1):
          image= fn(image)

        # ------------------------
        for fn in np.random.choice([
            lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
            lambda image : do_random_noise(image, mag=0.05),
            lambda image : do_random_guassian_blur(image),
            lambda image : do_random_blurout(image, size=0.25, num_cut=2),
            lambda image : do_random_clahe(image),
            lambda image : do_histogram_norm(image),
            lambda image : image,],1):
          image = fn(image)

        r['image']=image
        return r

    #---
    df_train, df_valid = make_fold('train-1')
    dataset = BTDataset(df_train,df_meta_ext,augment=augment)
    print(dataset)

    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate,
    )
    for t,batch in enumerate(loader):
        if t>4: break
        draw_batch(batch['image'])