import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
matplotlib.use('TkAgg')

#from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans


from common import *
from siim import *

from lib.net.lookahead import *
from lib.net.radam import *
#from madgrad import MADGRAD

from model import *
from dataset import *
from utils.general import wh_iou

#-----------------------------------------



def run_kmean_anchors(
    num_anchor=9,
    anchor_match_ratio_thresh=4.0,
    image_size=640,  gen=1000):



    df_annotate, df_train, df_valid = make_fold('train-0')
    df_train = pd.concat([df_train, df_valid])
    train_dataset = SiimDataset(df_annotate, df_train)

    norm_box_size = []
    num_train = len(train_dataset)
    for i in range(num_train):
        print('\r',i,end='',flush=True)
        r = train_dataset[i]
        annotate = r['annotate']
        norm_box_size.append(annotate[:,[3,4]])

    print('')
    norm_box_size = np.concatenate(norm_box_size)
    num_box = len(norm_box_size)
    print('norm_box_size :', norm_box_size.shape)
    #

    # kmeans calculation
    wh = norm_box_size
    #center, dist = kmeans(wh, num_anchor, iter=30)  # points, mean distance
    kmeans = KMeans(n_clusters=num_anchor)
    assign = kmeans.fit_predict(wh)
    center = kmeans.cluster_centers_

    s = (center[:,0]*center[:,1])**0.5


    #plot results
    print(center)
    for i in range(num_anchor):
        v = assign==i
        plt.scatter(wh[v][:,0], wh[v][:,1], alpha=0.3)
    plt.show()

    # compute metrics
    r = wh[:, None] / center[None]
    r = np.minimum(r, 1. / r).min(2)  # ratio metric
    iou = wh_iou(torch.tensor(wh), torch.tensor(center)).data.cpu().numpy()   # iou metric
    return x, x.max(1)[0]  # x, best_x


    zz=0

 

'''
center*640
array([[     175.17,      235.17],
       [     162.95,      148.85],
       [     210.62,      434.81],
       [     98.285,      149.48],
       [     106.39,      217.16],
       [     124.45,      288.62],
       [     150.78,       367.9],
       [       81.6,      87.335],
       [     195.04,      322.88]])

'''
# main #################################################################
if __name__ == '__main__':
    run_kmean_anchors()

