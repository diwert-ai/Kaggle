from common import *
from siim import *
from configure import *

from augmentation3 import *



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










df_fold_file = 'df_fold_rand830.csv'
def make_fold(mode='train-1'):
    if 'train' in mode:
        df_annotate = pd.read_csv(data_dir + '/df_annotate.csv')
        df_annotate.loc[:, 'ncx'] = df_annotate.nx.values + df_annotate.nw.values / 2
        df_annotate.loc[:, 'ncy'] = df_annotate.ny.values + df_annotate.nh.values / 2

        #---
        df_study = pd.read_csv(data_dir+'/train_study_level.csv')
        df_fold  = pd.read_csv(data_dir+'/%s'%df_fold_file)
        df_meta  = pd.read_csv(data_dir+'/df_meta_hw.csv')

        df_study.loc[:, 'id'] = df_study.id.str.replace('_study', '')
        df_study = df_study.rename(columns={'id': 'study_id'})

        #---
        df = df_study.copy()
        df = df.merge(df_fold, on='study_id')
        df = df.merge(df_meta, left_on='study_id', right_on='study')

        #---
        duplicate = read_list_from_file(data_dir + '/duplicate.txt')
        df = df[~df['image'].isin(duplicate)]
        df_annotate = df_annotate[~df_annotate['image_id'].isin(duplicate)]

        #---
        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_annotate, df_train, df_valid

    if 'test' in mode:
        df_meta  = pd.read_csv(data_dir+'/df_meta_hw.csv')
        df_valid = df_meta[df_meta['set']=='test'].copy()

        df_annotate = pd.DataFrame()
        df_annotate.loc[:,'image_id'] = df_valid.image.values
        df_annotate = df_annotate.merge(df_meta[['image','width','height']],left_on='image_id',right_on='image')
        df_annotate.loc[:,'class_id'] = 0
        df_annotate.loc[:, 'ncx'] = 0
        df_annotate.loc[:, 'ncy'] = 0
        df_annotate.loc[:, 'nw'] = 0
        df_annotate.loc[:, 'nh'] = 0


        for l in study_name_to_label.keys():
            df_valid.loc[:,l]=0
        df_valid = df_valid.reset_index(drop=True)
        return df_annotate, df_valid







def null_augment(r):
    image = r['image']
    # if image[:2].shape != (image_size, image_size):
    #     r['image'] = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
    return r



def make_annotate(df):

    cls_pos = []
    cls_neg = []
    for n in range(num_head):
        s = feature_size[n]
        num_anchor = len(anchor_size[n])

        pos = np.zeros((num_anchor, num_class, s, s), np.float32)
        neg = np.ones ((num_anchor, num_class, s, s), np.float32)

        cls_pos.append(pos)
        cls_neg.append(neg)

    #---

    non_zero = df['class_id'].values!=0
    norm_box = df[['nx0','ny0','nx1','ny1']].values
    norm_box = norm_box[non_zero]

    if len(norm_box)==0:
        box_index = []
        box_coord = []

    else:
        num_truth  = len(norm_box)
        num_anchor = len(norm_anchor)
        giou   = np_generalized_box_iou(norm_box, norm_anchor)
        argmax = giou.argmax(-1)
        giou[np.arange(num_truth), argmax] = 1  #always include best match

        ma, mt = np.meshgrid(range(num_anchor), range(num_truth))

        #for regression
        pos = (giou > 0.75)
        box_index = ma[pos]
        box_coord = norm_box[mt[pos]]*[[image_size, image_size, image_size, image_size,]]

        #for classification
        non_neg = (giou > 0.5)
        for a,t in zip(ma[non_neg],mt[non_neg]):
            n, k, y, x = anchor_index[a]
            cls_neg[n][k, 0, y, x] = 0

        pos = (giou > 0.5)
        for a,t in zip(ma[pos],mt[pos]):
            n, k, y, x = anchor_index[a]
            cls_pos[n][k, 0, y, x] = 1

    annotate = {
        'box_index': box_index,
        'box_coord': box_coord,
        'cls_pos' : cls_pos,
        'cls_neg' : cls_neg,
    }
    return annotate


class SiimDataset(Dataset):
    def __init__(self, df_annotate, df, augment=null_augment):
        super().__init__()
        self.gb = df_annotate.groupby('image_id') #9eb725cdb713
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
        df = self.gb.get_group(d.image).copy()

        g = self.gb.get_group(d.image)
        annotate = g[['class_id', 'ncx', 'ncy', 'nw', 'nh', ]].values
        annotate = annotate[annotate[:, 0] == 1]

        #image_file = data_dir + '/%s_640/%s/%s/%s.png' % (d.set, d.study, d.series, d.image)
        #image_file = data_dir + '/%s_full_%d/%s.png' % (d.set, image_size, d.image)
        image_file = data_dir + '/%s/%s.jpg' % (d.set, d.image)
        image  = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        onehot = d[study_name_to_label.keys()].values

        if d.set == 'train':
            #mask_file = data_dir + '/%s_mask_full_512/%s.png' % (d.set, d.image)
            mask_file = data_dir + '/%s_mask/%s.jpg' % (d.set, d.image)
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            #if mask[:2].shape != (image_size, image_size):
            #    mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
        else:
            mask = np.zeros_like(image)


        r = {
            'index' : index,
            'd'  : d,
            'df' : df,
            'annotate' : annotate,
            'image'  : image,
            'mask'   : mask,
            'onehot' : onehot,
        }
        if self.augment is not None: r = self.augment(r)
        return r


def null_collate(batch):
    collate = defaultdict(list)
    for b,r in enumerate(batch):
        for k, v in r.items():
            if k=='annotate':
                collate[k].append(np.hstack(
                    [np.full((len(v),1),b),v]
                ))
            else:
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

    annotate = np.concatenate(collate['annotate'])
    collate['annotate'] = torch.from_numpy(annotate).float()
    return collate



#===============================================================

def run_check_dataset():
    #df_annotate, df_train, df_valid = make_fold(mode='train-1')
    df_annotate, df_valid = make_fold(mode='test')

    dataset = SiimDataset(df_annotate, df_valid) #null_augment
    print(dataset)

    for i in range(20):
        #i = 272 #np.random.choice(len(dataset))
        r = dataset[i]

        print('index ' , i)
        print(r['d'])
        print(r['onehot'])
        print(r['annotate'])
        print('')
        image_show('image', r['image'], resize=1)
        image_show('mask', r['mask'], resize=1)

        #---
        #draw annotate
        # image = cv2.cvtColor(r['image'],cv2.COLOR_GRAY2BGR)
        # image_show('annotate box',overlay2, resize=1)
        cv2.waitKey(1)

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
        print('\t', batch['annotate'])
        print('\t', batch['annotate'].shape, batch['annotate'].is_contiguous())
        print('')


def run_check_augment():
    def draw_annotate(image, annotate):
        overlay = image.copy()
        h,w = image.shape
        for c, ncx, ncy, nw, nh in annotate:
            x0 = int((ncx-nw/2)*w)
            x1 = int((ncx+nw/2)*w)
            y0 = int((ncy-nh/2)*h)
            y1 = int((ncy+nh/2)*h)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), 255, 3)
        return overlay

    def augment(image,mask,annotate):
        #image,mask,annotate = do_random_hflip(image,mask,annotate)

        #image,mask,annotate = do_random_rotate(image,mask,annotate, mag=20)
        #image,mask,annotate = do_random_scale(image,mask,annotate, mag=0.2)
        #image,mask,annotate = do_random_stretch_x(image,mask,annotate, mag=0.2)
        #image,mask,annotate = do_random_stretch_y(image,mask,annotate, mag=0.2)
        image,mask,annotate = do_random_shift( image,mask,annotate, mag=64 )


        #image = do_random_blurout(image, size=0.10, num_cut=16)
        #image = do_random_noise(image)
        #image = do_random_guassian_blur(image)
        #image = do_random_intensity_shift_contast(image)

        #image = do_random_clahe(image)
        #image = do_histogram_norm(image)


        return image,mask,annotate

    #---
    df_annotate, df_train, df_valid = make_fold('train-1')
    dataset = SiimDataset(df_annotate, df_train)
    print(dataset)

    for i in range(2,500):
        r = dataset[i]
        image = r['image']
        mask = r['mask']
        annotate = r['annotate']


        print('%2d --------------------------- '%(i))
        overlay = draw_annotate(image, annotate)
        image_show('image', image)
        image_show('mask', mask)
        image_show('overlay', overlay)

        cv2.waitKey(1)
        #continue

        if 1:
            for i in range(100):
                image1,mask1,annotate1 =  augment(image.copy(),mask.copy(),annotate.copy())
                overlay1 = draw_annotate(image1, annotate1)

                image_show('image1', image1)
                image_show('mask1', mask1)
                image_show('overlay1', overlay1)
                cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    #run_check_dataset()
    run_check_augment()