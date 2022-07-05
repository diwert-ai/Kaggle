import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from siim import *

from dataset import *
from model import *


# start here ! ###################################################################################
def probability_to_df_study(df_valid, probability):
    df_study = pd.DataFrame()
    df_study.loc[:,'id'] = df_valid.study + '_study'
    for i in range(num_study_label):
        df_study.loc[:,study_name_to_predict_string[study_label_to_name[i]]]=probability[:,i]

    df_study = df_study.groupby('id', as_index=False).mean()
    df_study.loc[:, 'PredictionString'] = \
           'negative '      + df_study.negative.apply(lambda x: '%0.6f'%x)      + ' 0 0 1 1' \
        + ' typical '       + df_study.typical.apply(lambda x: '%0.6f'%x)       + ' 0 0 1 1' \
        + ' indeterminate ' + df_study.indeterminate.apply(lambda x: '%0.6f'%x) + ' 0 0 1 1' \
        + ' atypical '      + df_study.atypical.apply(lambda x: '%0.6f'%x)      + ' 0 0 1 1'

    df_study = df_study[['id','PredictionString']]
    return df_study

def probability_to_df_image(df_valid, probability, box):
    df_image = pd.DataFrame({'id':[],'PredictionString':[]})
    return df_image



def do_predict(net, valid_loader, tta=['flip','scale']): #flip

    valid_probability = []
    valid_num = 0

    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image'].cuda()
        onehot = batch['onehot']
        label  = onehot.argmax(-1)

        #<todo> TTA
        net.eval()
        with torch.no_grad():
            probability = []
            logit, mask = net(image)
            probability.append(F.softmax(logit,-1))

            if 'flip' in tta:
                logit, mask = net(torch.flip(image,dims=(3,)))
                probability.append(F.softmax(logit,-1))

            if 'scale' in tta:
                # size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
                logit, mask = net(F.interpolate(image, scale_factor=1.33, mode='bilinear', align_corners=False))
                probability.append(F.softmax(logit,-1))

            #--------------
            probability = torch.stack(probability,0).mean(0)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')

    probability = np.concatenate(valid_probability)
    return probability




def run_submit():
    for fold in [0]:
        out_dir = out_dir = \
            '/root/share1/kaggle/2021/siim-covid-19/result/try-delivery/effb3-full-512-mask/fold%d-fine'%fold
        initial_checkpoint = \
            out_dir + '/checkpoint/00008600_model.pth' # None #

        if 1:

            ## setup  ----------------------------------------
            #mode = 'local'
            mode = 'remote'

            submit_dir = out_dir + '/valid/%s-%s'%(mode, initial_checkpoint[-18:-4])
            os.makedirs(submit_dir, exist_ok=True)

            log = Logger()
            log.open(out_dir + '/log.submit.txt', mode='a')
            log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
            log.write('\t%s\n' % COMMON_STRING)
            log.write('\n')

            #
            ## dataset ------------------------------------

            if 'remote' in mode: #1263
                df_valid = make_fold('test')

            if 'local' in mode: #1276 #1256
                df_train, df_valid = make_fold('train-%d' % fold)


            valid_dataset = SiimDataset(df_valid)
            valid_loader  = DataLoader(
                valid_dataset,
                sampler = SequentialSampler(valid_dataset),
                batch_size  = 32,#128, #
                drop_last   = False,
                num_workers = 8,
                pin_memory  = True,
                collate_fn  = null_collate,
            )
            log.write('mode : %s\n'%(mode))
            log.write('valid_dataset : \n%s\n'%(valid_dataset))

            ## net ----------------------------------------
            if 1:
                net = Net().cuda()
                net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)

                #---
                start_timer = timer()
                probability = do_predict(net, valid_loader)
                log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))
                log.write('probability %s \n' % str(probability.shape))

                np.save(submit_dir + '/probability.npy',probability)
                df_valid.to_csv(submit_dir + '/df_valid.csv', index=False)

            else:
                probability = np.load(submit_dir + '/probability.npy')

            #----
            df_study = probability_to_df_study(df_valid, probability)
            df_image = probability_to_df_image(df_valid, None, None)
            df_submit = pd.concat([df_study,df_image])
            df_submit.to_csv(submit_dir + '/submit.csv', index=False)

            log.write('submit_dir : %s\n' % (submit_dir))
            log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
            log.write('df_submit : %s\n' % str(df_submit.shape))
            log.write('%s\n' % str(df_submit))
            log.write('\n')

            if 'local' in mode:
                onehot = df_valid[study_name_to_label.keys()].values
                truth = onehot.argmax(-1)
                predict = probability.argsort(-1)[::-1]

                loss = np_loss_cross_entropy(probability, truth)
                topk = (predict == truth.reshape(-1, 1))
                acc  = topk[:, 0]
                topk = topk.mean(0).cumsum()
                acc  = [acc[truth == i].mean() for i in range(num_study_label)]

                # ---
                map = np_metric_map_curve_by_class(probability, truth)

                # ---
                log.write('loss : %f\n' % (loss))
                log.write('topk : %s\n' % str(topk))
                log.write('map(mean) : %f\n' % map.mean())
                log.write('          : %f\n' % ((4 / 6)* map.mean()))
                for i in range(num_study_label):
                    l = study_label_to_name[i]
                    log.write('%d %30s : %0.5f\n' % (i,l,map[i]))
                log.write('\n\n')
        #exit(0)

def run_remote_ensemble():
    out_dir = '/root/share1/kaggle/2021/siim-covid-19/result/try-delivery/effb3-full-512-mask'
    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')


    submit_dir=[
        out_dir+'/fold0-fine/valid/remote-00008600_model',
        out_dir+'/fold1-fine/valid/remote-00008400_model',
    ]

    probability=0
    for d in submit_dir:
        p = np.load(d + '/probability.npy')
        probability += p**0.5
    probability = probability/len(submit_dir)


    #----
    df_valid = pd.read_csv(submit_dir[1] + '/df_valid.csv')

    df_study  = probability_to_df_study(df_valid, probability)
    df_image  = probability_to_df_image(df_valid, None, None)
    df_submit = pd.concat([df_study, df_image])
    df_submit.to_csv(out_dir + '/effb3-full-512-mask-submit-ensemble2.csv', index=False)

    log.write('submit_dir : %s\n' % (submit_dir))
    log.write('df_submit : %s\n' % str(df_submit.shape))
    log.write('%s\n' % str(df_submit))
    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    #run_submit()
    run_remote_ensemble()

