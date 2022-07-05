import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from siim import *

from lib.net.lookahead import *
from lib.net.radam import *
#from madgrad import MADGRAD

from model import *
from dataset import *


matplotlib.use('TkAgg')
#----------------
import torch.cuda.amp as amp

class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)

is_mixed_precision = True  #True #False
#----------------



def train_augment(r):
    image = r['image']
    mask = r['mask']
    annotate = r['annotate']

    if 1:
        for fn in np.random.choice([
            lambda image, mask, annotate : do_random_scale(image, mask, annotate, mag=0.35),
            lambda image, mask, annotate : do_random_stretch_y(image, mask, annotate, mag=0.35),
            lambda image, mask, annotate : do_random_stretch_x(image, mask, annotate, mag=0.35),
            lambda image, mask, annotate : do_random_shift(image, mask, annotate, mag=int(0.35*image_size)),
            lambda image, mask, annotate : (image, mask, annotate)
        ],1):
            image, mask, annotate = fn(image, mask, annotate)

        for fn in np.random.choice([
            lambda image, mask, annotate : do_random_rotate(image, mask, annotate, mag=15),
            lambda image, mask, annotate : do_random_hflip(image, mask, annotate),
            lambda image, mask, annotate : (image, mask, annotate)
        ],1):
            image, mask, annotate = fn(image, mask, annotate)

        # ------------------------
        for fn in np.random.choice([
            lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
            lambda image : do_random_noise(image, mag=0.05),
            lambda image : do_random_guassian_blur(image),
            lambda image : do_random_blurout(image, size=0.3, num_cut=3),
            #lambda image : do_random_clahe(image),
            #lambda image : do_histogram_norm(image),
            lambda image : image,
        ],1):
            image = fn(image)

    r['image'] = image
    r['mask'] = mask
    r['annotate'] = annotate
    return r



def do_valid(net, valid_loader):

    valid_loss = [0,0]
    valid_num  = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        annotate = batch['annotate'].cuda()
        image = batch['image'].cuda()
        onehot = batch['onehot']
        label = onehot.argmax(-1)

        with torch.no_grad():
            #with amp.autocast():
                predict = data_parallel(net, image)

                truth = make_truth(annotate)
                loss_cls, loss_box, loss_obj = modified_yolo_loss(predict, truth)

        #----------
        valid_num += batch_size
        valid_loss[0] += batch_size*loss_box.item()
        valid_loss[1] += batch_size*loss_obj.item()
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))
    #print('')
    #----------------------
    valid_loss[0] = valid_loss[0]/valid_num
    valid_loss[1] = valid_loss[1]/valid_num
    return [0, 0, valid_loss[0], valid_loss[1]]



# start here ! ###################################################################################


def run_train():
    fold = 1
    for fold in [0,]:
        out_dir = out_dir = \
            '/root/share1/kaggle/2021/siim-covid-19/result/try50-opacity/yolov5x-full-640-v4-1/fold%d'%fold
        initial_checkpoint = \
            out_dir + '/checkpoint/00012500_model.pth'  #
            #out_dir + '/checkpoint/00008500_model.pth' #

        start_lr   = 0.00001#1
        batch_size = 16 #14 #22


        ## setup  ----------------------------------------
        for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir + '/' + f, exist_ok=True)
        # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

        log = Logger()
        log.open(out_dir + '/log.train.txt', mode='a')
        log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        log.write('\t%s\n' % COMMON_STRING)
        log.write('\t__file__ = %s\n' % __file__)
        log.write('\tout_dir  = %s\n' % out_dir)
        log.write('\n')

        ## dataset ------------------------------------
        df_annotate, df_train, df_valid = make_fold('train-%d'%fold)
        train_dataset = SiimDataset(df_annotate, df_train, train_augment)#train_augment
        valid_dataset = SiimDataset(df_annotate, df_valid, )

        train_loader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size,
            drop_last   = True,
            num_workers = 0,
            pin_memory  = True,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn  = null_collate,
        )
        valid_loader  = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size  = 16,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate,
        )

        log.write('df_fold_file  : %s\n'%(df_fold_file))
        log.write('train_dataset : \n%s\n'%(train_dataset))
        log.write('valid_dataset : \n%s\n'%(valid_dataset))
        log.write('\n')

        #check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

        ## net ----------------------------------------
        log.write('** net setting **\n')
        if is_mixed_precision:
            scaler = amp.GradScaler()
            net = AmpNet().cuda()
        else:
            net = Net().cuda()


        if initial_checkpoint is not None:
            f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
            start_iteration = f['iteration']
            start_epoch = f['epoch']
            state_dict  = f['state_dict']
            net.load_state_dict(state_dict,strict=True)  #True
        else:
            start_iteration = 0
            start_epoch = 0


        log.write('net=%s\n'%(type(net)))
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        log.write('\n')

        # -----------------------------------------------
        if 0: ##freeze
            for p in net.block0.backbone.parameters(): p.requires_grad = False


        optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
        #optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
        #optimizer = MADGRAD( filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum= 0.9, weight_decay= 0, eps= 1e-06)
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr, momentum=0.9)

        num_iteration = 80000
        iter_log    = 500
        iter_valid  = 500
        iter_save   = list(range(0, num_iteration, 500))#1*1000

        log.write('optimizer\n  %s\n'%(optimizer))
        log.write('\n')


        ## start training here! ##############################################
        log.write('** start training here! **\n')
        log.write('   fold = %d\n'%(fold))
        log.write('   is_mixed_precision = %s \n'%str(is_mixed_precision))
        log.write('   batch_size = %d\n'%(batch_size))
        log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
        log.write('                      |----- VALID ---|---- TRAIN/BATCH --------------\n')
        log.write('rate     iter   epoch | loss    map   | loss0  loss1  | time          \n')
        log.write('----------------------------------------------------------------------\n')
                  #0.00000   0.00* 0.00  | 0.000  0.000  | 0.000  0.000  |  0 hr 00 min

        def message(mode='print'):
            if mode==('print'):
                asterisk = ' '
                loss = batch_loss
            if mode==('log'):
                asterisk = '*' if iteration in iter_save else ' '
                loss = train_loss

            text = \
                '%0.5f  %5.3f%s %4.2f  | '%(rate, iteration/10000, asterisk, epoch,) +\
                '%4.3f  %4.3f  %4.3f  %4.3f  | '%(*valid_loss,) +\
                '%4.3f  %4.3f  %4.3f  | '%(*loss,) +\
                '%s' % (time_to_str(timer() - start_timer,'min'))

            return text

        #----
        valid_loss = np.zeros(4,np.float32)
        train_loss = np.zeros(3,np.float32)
        batch_loss = np.zeros_like(train_loss)
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = 0
        loss0 = torch.FloatTensor([0]).cuda().sum()
        loss1 = torch.FloatTensor([0]).cuda().sum()
        loss2 = torch.FloatTensor([0]).cuda().sum()


        start_timer = timer()
        iteration = start_iteration
        epoch = start_epoch
        rate = 0
        while  iteration < num_iteration:

            for t, batch in enumerate(train_loader):

                if iteration in iter_save:
                    if iteration != start_iteration:
                        torch.save({
                            'state_dict': net.state_dict(),
                            'iteration': iteration,
                            'epoch': epoch,
                        }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                        pass

                if (iteration % iter_valid == 0):
                    if iteration!=start_iteration:
                        valid_loss = do_valid(net, valid_loader)  #
                        pass

                if (iteration % iter_log == 0):
                    print('\r', end='', flush=True)
                    log.write(message(mode='log') + '\n')


                # learning rate schduler ------------
                rate = get_learning_rate(optimizer)

                # one iteration update  -------------
                batch_size = len(batch['index'])
                image    = batch['image'].cuda()
                annotate = batch['annotate'].cuda()
                mask     = batch['mask'].cuda()
                onehot   = batch['onehot'].cuda()
                label    = onehot.argmax(-1)

                #----
                net.train()
                optimizer.zero_grad()

                if is_mixed_precision:
                    with amp.autocast():
                        predict = data_parallel(net, image)

                        truth = make_truth(annotate)
                        loss_cls, loss_box, loss_obj = modified_yolo_loss(predict, truth)

                        loss0 =  loss_cls #not used
                        loss1 =  loss_box
                        loss2 =  loss_obj

                    scaler.scale(loss1+loss2).backward()
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                    scaler.step(optimizer)
                    scaler.update()


                else :
                    assert(False)
                    print('fp32')
                    predict = data_parallel(net, image)
                    #<todo> loss = ...

                    #(loss0+loss1).backward()
                    (loss1).backward()
                    optimizer.step()




                # print statistics  --------
                epoch += 1 / len(train_loader)
                iteration += 1

                batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
                sum_train_loss += batch_loss
                sum_train += 1
                if iteration % 100 == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train = 0

                print('\r', end='', flush=True)
                print(message(mode='print'), end='', flush=True)


                #debug--------------------------
                if 0:
                    def draw_yolo_true_objectness(objectness, truth, b, mode='sum', image_size=None):
                        if image_size is None:
                            image_size=feature_size[0]
                        num_anchor = 3
                        overlay = []
                        for n in range(num_head):
                            s = feature_size[n]
                            o = np.zeros((num_anchor,s,s), np.float32)

                            i = np.where(truth['index'][n][0]==b)[0]
                            num_box = len(i)
                            if num_box>0:
                                a  = truth['index'][n][2][i]
                                gj = truth['index'][n][3][i]
                                gi = truth['index'][n][4][i]
                                o[a, gj, gi] = 1

                            if mode == 'max':  o = o.max(0)
                            if mode == 'min':  o = o.min(0)
                            if mode == 'mean': o = o.mean(0)
                            if mode == 'sum':  o = o.sum(0)

                            o = cv2.resize(o, dsize = (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                            cv2.rectangle(o, (0, 0),  (image_size, image_size), 1, 1)
                            overlay.append(o)

                        overlay = np.hstack(overlay)
                        return overlay

                    def draw_yolo_objectness(objectness, mode='sum',  image_size=None):
                        if image_size is None:
                            num_anchor, image_size, image_size = objectness[0].shape

                        overlay = []
                        for p in objectness:
                            p = p.copy()
                            if mode == 'max':  p = p.max(0)
                            if mode == 'min':  p = p.min(0)
                            if mode == 'mean': p = p.mean(0)
                            if mode == 'sum':  p = p.sum(0)

                            p = cv2.resize(p, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
                            cv2.rectangle(p, (0, 0), (image_size, image_size), 1, 1)
                            overlay.append(p)

                        overlay = np.hstack(overlay)
                        return overlay

                    #----
                    def draw_yolo_true_box(overlay, truth, b):
                        for n in range(num_head):
                            i = np.where(truth['index'][n][0]==b)[0]
                            num_box = len(i)
                            if num_box==0: continue

                            s = feature_stride[n]
                            true_box = truth['box'][n][i]

                            a  = truth['index'][n][2][i]
                            gj = truth['index'][n][3][i]
                            gi = truth['index'][n][4][i]
                            for k in range(num_box):
                                dx, dy, w, h = true_box[k] #true box at that scale
                                cx=int(s*(gi[k]+dx))
                                cy=int(s*(gj[k]+dy))
                                w = int(s*w)
                                h = int(s*h)
                                x0=cx-w//2
                                y0=cy-h//2
                                x1=cx+w//2
                                y1=cy+h//2
                                cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,0,255),3)

                                #---
                                cx = int((gi[k]+0.5)*s)
                                cy = int((gj[k]+0.5)*s)
                                w = int(anchor_size[n][a[k]][0])
                                h = int(anchor_size[n][a[k]][1])
                                x0 = cx - w // 2
                                y0 = cy - h // 2
                                x1 = cx + w // 2
                                y1 = cy + h // 2
                                cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)
                                cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)


                        return overlay

                    def draw_yolo_box(overlay, box, truth, b):
                        for n in range(num_head):
                            i = np.where(truth['index'][n][0]==b)[0]
                            num_box = len(i)
                            if num_box==0: continue

                            a  = truth['index'][n][2][i]
                            gj = truth['index'][n][3][i]
                            gi = truth['index'][n][4][i]
                            for k in range(num_box):
                                cx, cy, w, h = box[n][a[k],gj[k],gi[k]] #
                                cx = int(cx)
                                cy = int(cy)
                                w = int(w)
                                h = int(h)
                                x0=cx-w//2
                                y0=cy-h//2
                                x1=cx+w//2
                                y1=cy+h//2
                                cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,255,0),2)
                        return overlay


                    #----
                    predict = infer_prediction(predict)
                    predict = [p.data.float().cpu().numpy() for p in predict]
                    objectness =[p[...,4] for p in predict]
                    box = [p[...,:4] for p in predict]

                    truth = {
                        'index': [(
                            m[0].data.cpu().numpy(),
                            m[1].data.cpu().numpy(),
                            m[2].data.cpu().numpy(),
                            m[3].data.cpu().numpy(),
                            m[4].data.cpu().numpy(),
                        ) for m in truth['index']],
                        'label': [m.data.cpu().numpy() for m in truth['label']],
                        'box'  : [m.data.cpu().numpy() for m in truth['box']],
                    }

                    image = image.permute(0,2,3,1).contiguous()
                    image = image.data.cpu().numpy()
                    image = (image*255).astype(np.uint8)

                    mask = mask.permute(0,2,3,1).repeat(1,1,1,3)
                    mask = mask.data.cpu().numpy()
                    mask = (mask*255).astype(np.uint8)

                    for b in range(batch_size):

                        overlay1 = draw_yolo_true_objectness(objectness, truth, b, mode='sum')
                        overlay2 = draw_yolo_objectness([p[b] for p in objectness])

                        overlay3 = mask[b].copy()
                        overlay3 = draw_yolo_true_box(overlay3, truth, b)
                        overlay4 = mask[b].copy()
                        overlay4 = draw_yolo_box(overlay4, [p[b] for p in box], truth, b)



                        # ---
                        image_show('image', image[b], resize=0.5)
                        image_show('mask', mask[b], resize=0.5)
                        #image_show('truth', overlay0, resize=0.5)

                        image_show_norm('true objectness', overlay1, min=0, max=1, resize=2)
                        image_show_norm('objectness', overlay2, min=0, max=1, resize=2)
                        image_show('true box', overlay3, resize=0.5)
                        image_show('box', overlay4, resize=0.5)
                        cv2.waitKey(0)
        log.write('\n')



def run_check_anchor_goodness():
    df_annotate, df_train, df_valid = make_fold('train-0')
    df_train = pd.concat([df_train,df_valid])
    train_dataset = SiimDataset(df_annotate, df_train, )

    result = defaultdict(list)

    num_image = len(train_dataset)
    for i in range(num_image):
        if i >300: break
        r = train_dataset[i]
        print(i,r['d'].image)

        annotate = r['annotate']
        num_box = len(annotate)
        if num_box==0 : continue

        annotate = np.hstack([np.full((num_box, 1), 0), annotate])
        annotate = torch.from_numpy(annotate).float()

        truth = make_truth(annotate,anchor_match_ratio_threshold=100)
        true_index = truth['index']
        true_box   = truth['box']

        norm_anchor_size = torch.FloatTensor(make_norm_anchor_size())
        for n in range(num_head):
            b, u, a, gj, gi = true_index[n]
            p = torch.zeros((len(a),6))

            pxy = (p[:, [0, 1]].sigmoid() * 2) - 0.5
            pwh = (p[:, [2, 3]].sigmoid() * 2) ** 2 * norm_anchor_size[n][a]
            box = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(box.T, true_box[n], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            iou = iou.clamp(0)
            iou = iou.data.cpu().numpy()
            #print(iou)

            for j in range(num_box):
                result['%s_%02d'%(r['d'].image,j)].extend(iou[u==j])

    #--------
    num_target = len(result)
    hist_max = np.zeros(11)
    hist_all = np.zeros(11)
    for k,v in result.items():
        v = np.array(v)
        v1 = np.round(v * 10).astype(np.int32)
        hist_all += np.bincount(v1, minlength=11)

        m = np.round(v * 10).max().astype(np.int32)
        hist_max[m]+=1
        zz=0

    plt.plot(np.arange(11), hist_max)
    #plt.plot(np.arange(11), np.log(hist_all+0.000001))
    #plt.plot(np.arange(11), hist_all)
    plt.show()

    zz=0


# main #################################################################
if __name__ == '__main__':
    run_train()
    #run_check_anchor_goodness()

