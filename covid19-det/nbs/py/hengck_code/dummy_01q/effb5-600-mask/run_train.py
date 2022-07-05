import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from siim import *

# from lib.net.lookahead import *
# from lib.net.radam import *
from madgrad import MADGRAD

from model import *
from dataset import *


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
    # if image[:2].shape != (image_size, image_size):
    #     image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

    if 1:
        for fn in np.random.choice([
            lambda image, mask : do_random_scale(image, mask, mag=0.20),
            lambda image, mask : do_random_stretch_y(image, mask, mag=0.20),
            lambda image, mask : do_random_stretch_x(image, mask, mag=0.20),
            lambda image, mask : do_random_shift(image, mask, mag=int(0.20*image_size)),
            lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        for fn in np.random.choice([
            lambda image, mask : do_random_rotate(image, mask, mag=15),
            lambda image, mask : do_random_hflip(image, mask),
            lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        # ------------------------
        for fn in np.random.choice([
            lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
            lambda image : do_random_noise(image, mag=0.05),
            lambda image : do_random_guassian_blur(image),
            lambda image : do_random_blurout(image, size=0.25, num_cut=2),
            #lambda image : do_random_clahe(image),
            #lambda image : do_histogram_norm(image),
            lambda image : image,
        ],1):
            image = fn(image)

    r['image'] = image
    r['mask'] = mask
    return r


def do_valid(net, valid_loader):

    valid_probability = []
    valid_truth = []
    valid_num = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image = batch['image'].cuda()
        onehot = batch['onehot']
        label = onehot.argmax(-1)

        with torch.no_grad():
            #with amp.autocast():
                logit, mask = data_parallel(net,image)
                probability = F.softmax(logit,-1)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(label.data.cpu().numpy())
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))
    #print('')
    #----------------------
    truth = np.concatenate(valid_truth)
    probability = np.concatenate(valid_probability)
    predict = probability.argsort(-1)[::-1]

    loss = np_loss_cross_entropy(probability,truth)
    topk = (predict==truth.reshape(-1,1))
    acc  = topk[:, 0]
    topk = topk.mean(0).cumsum()
    acc = [acc[truth==i].mean() for i in range(num_study_label)]

    #---
    map  = np_metric_map_curve_by_class(probability, truth)*(4/6)

    return [loss, map.mean(), topk[0], topk[1]]



# start here ! ###################################################################################


def run_train():

    #for fold in [0,1,2,3,4]:
    for fold in [1,]:
        out_dir = out_dir = \
            '/root/share1/kaggle/2021/siim-covid-19/result/try-delivery/effb3-full-512-mask/fold%d-fine'%fold
        initial_checkpoint = \
            '/root/share1/kaggle/2021/siim-covid-19/result/try-delivery/effb3-full-512-mask/fold%d/checkpoint/00007400_model.pth'%fold
            #out_dir + '/checkpoint/00007700_model.pth' #None#

        start_lr   = 0.0001#1
        batch_size = 8 #14 #22

        num_iteration = 12000
        iter_log    = 200
        iter_valid  = 200
        iter_save   = list(range(0, num_iteration+1, 200))

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
        df_train, df_valid = make_fold('train-%d'%fold)
        train_dataset = SiimDataset(df_train, train_augment)
        valid_dataset = SiimDataset(df_valid, )

        train_loader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size,
            drop_last   = True,
            num_workers = 4,
            pin_memory  = True,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn  = null_collate,
        )
        valid_loader  = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size  = 16,
            drop_last   = False,
            num_workers = 4,
            pin_memory  = True,
            collate_fn  = null_collate,
        )

        log.write('train_dataset : \n%s\n'%(train_dataset))
        log.write('valid_dataset : \n%s\n'%(valid_dataset))
        log.write('\n')


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


        #optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
        #optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
        optimizer = MADGRAD( filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum= 0.9, weight_decay= 0, eps= 1e-06)


        # num_iteration = 8000
        # iter_log    = 100
        # iter_valid  = 100
        # iter_save   = list(range(0, num_iteration, 100))#1*1000

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
                    #if iteration!=start_iteration:
                        valid_loss = do_valid(net, valid_loader)  #
                        pass

                if (iteration % iter_log == 0):
                    print('\r', end='', flush=True)
                    log.write(message(mode='log') + '\n')


                # learning rate schduler ------------
                rate = get_learning_rate(optimizer)

                # one iteration update  -------------
                batch_size = len(batch['index'])
                image = batch['image'].cuda()
                truth_mask = batch['mask'].cuda()
                truth_mask = F.interpolate(truth_mask, size=(32,32), mode='bilinear', align_corners=False)
                onehot = batch['onehot'].cuda()
                label = onehot.argmax(-1)

                #----
                net.train()
                optimizer.zero_grad()

                if is_mixed_precision:
                    with amp.autocast():
                        logit, mask = data_parallel(net, image)
                        loss0 = F.cross_entropy(logit, label)
                        loss1 = F.binary_cross_entropy_with_logits(mask, truth_mask)

                    #scaler.scale(loss0).backward()
                    #scaler.scale(loss1).backward()
                    scaler.scale(loss0+loss1).backward()
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                    scaler.step(optimizer)
                    scaler.update()


                else :
                    assert(False)
                    print('fp32')
                    logit, mask = data_parallel(net, image)
                    loss0 = F.cross_entropy(logit, label)
                    loss1 = F.binary_cross_entropy_with_logits(mask, truth_mask)

                    (loss0+loss1).backward()
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


        log.write('\n')


# main #################################################################
if __name__ == '__main__':
    run_train()

