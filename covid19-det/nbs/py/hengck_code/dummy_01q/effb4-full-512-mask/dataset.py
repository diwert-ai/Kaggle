from common import *
from configure import *

from siim import *
from augmentation2 import *


# def do_resize_image():
#     dump_dir = '/root/share1/kaggle/2021/siim-covid-19/data/siim-covid19-detection/test_full_512'
#     os.makedirs(dump_dir,exist_ok=True)
#
#     file = glob.glob('/root/share1/kaggle/2021/siim-covid-19/data/siim-covid19-detection/test_full/*.png')
#     print(len(file))
#     for i,f in enumerate(file):
#         print(i)
#         image = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
#         image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
#         cv2.imwrite( dump_dir + '/' + f.split('/')[-1], image)
#     exit(0)
# do_resize_image()


def make_fold(mode='train-1'):
    if 'train' in mode:
        df_study = pd.read_csv(data_dir+'/train_study_level.csv')
        df_fold  = pd.read_csv(data_dir+'/df_fold_rand830.csv')
        df_meta  = pd.read_csv(data_dir+'/df_meta.csv')

        df_study.loc[:, 'id'] = df_study.id.str.replace('_study', '')
        df_study = df_study.rename(columns={'id': 'study_id'})

        #---
        df = df_study.copy()
        df = df.merge(df_fold, on='study_id')
        df = df.merge(df_meta, left_on='study_id', right_on='study')

        duplicate = read_list_from_file(data_dir + '/duplicate.txt')
        df = df[~df['image'].isin(duplicate)]

        #---
        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    if 'test' in mode:
        df_meta  = pd.read_csv(data_dir+'/df_meta.csv')
        df_valid = df_meta[df_meta['set']=='test'].copy()

        for l in study_name_to_label.keys():
            df_valid.loc[:,l]=0
        df_valid = df_valid.reset_index(drop=True)
        return df_valid

def null_augment(r):
    image = r['image']
    # if image[:2].shape != (image_size, image_size):
    #     r['image'] = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
    return r


class SiimDataset(Dataset):
    def __init__(self, df, augment=null_augment):
        super().__init__()
        self.df = df
        self.augment = augment
        self.length = len(df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        string += '\tlabel distribution\n'
        for i in range(num_study_label):
            n = self.df[study_label_to_name[i]].sum()
            string += '\t\t %d %26s: %5d (%0.4f)\n'%(i, study_label_to_name[i], n, n/len(self.df) )
        return string


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        #image_file = data_dir + '/%s_640/%s/%s/%s.png' % (d.set, d.study, d.series, d.image)
        image_file = data_dir + '/%s/%s.jpg' % (d.set, d.image)
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        onehot = d[study_name_to_label.keys()].values

        if d.set == 'train':
            mask_file = data_dir + '/%s_mask/%s.jpg' % (d.set, d.image)
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros_like(image)


        r = {
            'index' : index,
            'd' : d,
            'image' : image,
            'mask' : mask,
            'onehot' : onehot,
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
    onehot = np.ascontiguousarray(np.stack(collate['onehot'])).astype(np.float32)
    collate['onehot'] = torch.from_numpy(onehot)

    image = np.stack(collate['image'])
    image = image.reshape(batch_size, 1, image_size,image_size).repeat(3,1)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image)


    mask = np.stack(collate['mask'])
    mask = mask.reshape(batch_size, 1, image_size,image_size)
    mask = np.ascontiguousarray(mask)
    mask = mask.astype(np.float32) / 255
    collate['mask'] = torch.from_numpy(mask)

    return collate



#===============================================================

def run_check_dataset():
    df_train, df_valid = make_fold(mode='train-1')
    #df_valid = make_fold(mode='test')

    dataset = SiimDataset(df_valid) #null_augment
    print(dataset)

    for i in range(50):
        i = np.random.choice(len(dataset))
        r = dataset[i]

        print('index ' , i)
        print(r['d'])
        print(r['onehot'])
        print('')
        image_show('image', r['image'], resize=1)
        image_show('mask', r['mask'], resize=1)
        cv2.waitKey(0)

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
        print('mask : ')
        print('\t', batch['mask'].shape, batch['mask'].is_contiguous())
        print('onehot : ')
        print('\t', batch['onehot'])
        print('\t', batch['onehot'].shape, batch['onehot'].is_contiguous())
        print('')


def run_check_augment():
    def augment(image):
        #image = do_random_hflip(image)

        #image = do_random_rotate(image, mag=20)
        #image = do_random_scale(image, mag=0.2)
        #image = do_random_stretch_x(image, mag=0.2)
        #image = do_random_stretch_y(image, mag=0.2)
        #image = do_random_shift( image, mag=64 )


        #image = do_random_blurout(image, size=0.10, num_cut=16)
        #image = do_random_noise(image)
        #image = do_random_guassian_blur(image)
        #image = do_random_intensity_shift_contast(image)

        image = do_random_clahe(image)
        #image = do_histogram_norm(image)


        return image

    #---
    df_train, df_valid = make_fold('train-1')
    dataset = SiimDataset(df_train)
    print(dataset)

    for i in range(500):
        r = dataset[i]
        image = r['image']


        print('%2d --------------------------- '%(i))
        image_show('image', image)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1 =  augment(image.copy())
                image_show('image1', image1)
                cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
    #run_check_augment()