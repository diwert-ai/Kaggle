import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from siim import *

from dataset import *
from model import *


# start here ! ###################################################################################
def make_df_image(df_valid, detection):
    df_image = pd.DataFrame()
    df_image.loc[:,'id'] = df_valid.image + '_image'
    #df_image.loc[:, 'PredictionString']=''

    predict_string = []
    for i,det in enumerate(detection):
        d = df_valid.iloc[i]

        s = ''
        for x0, y0, x1, y1, c in det:
            x0 = int(x0/image_size*d.width )
            y0 = int(y0/image_size*d.height)
            x1 = int(x1/image_size*d.width )
            y1 = int(y1/image_size*d.height)
            s += ' opacity %0.5f %4d %4d %4d %4d'%(c,x0,y0,x1,y1)
        predict_string.append(s)

    df_image.loc[:, 'PredictionString'] = predict_string
    #df_image = df_image[['id','PredictionString']]
    return df_image


#---------------------------------------------------------


def do_predict(net, valid_loader, tta=[]): #flip

    valid_detection = []
    valid_num = 0

    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image'].cuda()

        onehot = batch['onehot']
        label  = onehot.argmax(-1)
        mask   = batch['mask']

        #<todo> TTA
        net.eval()
        with torch.no_grad():
            predict = net(image)
            predict = infer_prediction(predict)
            predict_flat = pyramid_to_flat(predict)

            detection = do_non_max_suppression(
                    predict_flat,
                    nms_objectness_threshold=0.01,
                    nms_iou_threshold=0.5,
                    nms_pre_max_num=500,
                    nms_post_max_num=25,
            )

            #debug ------------------------------------------------------------------
            if 0:
                image = image.permute(0, 2, 3, 1).contiguous()
                image = image.data.cpu().numpy()
                image = (image * 255).astype(np.uint8)

                mask = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
                mask = mask.data.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                for b in range(batch_size):
                    image_show('image', image[b], resize=1)
                    image_show('mask', mask[b], resize=1)

                    overlay = mask[b].copy()
                    d = detection[b]
                    for x0,y0,x1,y1,s in d:
                        x0 = int(x0)
                        y0 = int(y0)
                        x1 = int(x1)
                        y1 = int(y1)
                        if s>0.5:
                            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
                            draw_shadow_text(overlay, '%0.4f'%s, (x0,y0+15), 0.8, (255,255,255),1)
                        else:
                            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 1)

                    image_show('overlay', overlay, resize=1)
                    cv2.waitKey(0)


        valid_num += batch_size
        valid_detection.extend(detection)
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')

    detection = valid_detection
    return detection





def run_submit():

    for fold in [0]:
        out_dir = out_dir = \
            '/root/share1/kaggle/2021/siim-covid-19/result/try50-opacity/yolov5x-full-640-v4-1/fold%d'%fold
        initial_checkpoint = \
            out_dir + '/checkpoint/00011000_model.pth'  #
            #out_dir + '/checkpoint/00010500_model.pth'  #

        if 1:

            ## setup  ----------------------------------------
            mode = 'local'
            #mode = 'remote'

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
                df_annotate, df_valid = make_fold('test')

            if 'local' in mode: #1276 #1256
                df_annotate, df_train, df_valid = make_fold('train-%d' % fold)
                #df_valid = df_train

            valid_dataset = SiimDataset(df_annotate, df_valid)
            valid_loader  = DataLoader(
                valid_dataset,
                sampler = SequentialSampler(valid_dataset),
                batch_size  = 4,#128, #
                drop_last   = False,
                num_workers = 0,
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
                detection = do_predict(net, valid_loader)
                log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))
                log.write('detection %d \n' % len(detection))

                write_pickle_to_file(submit_dir + '/detection.pickle',detection)
                #df_valid['study'].to_csv(submit_dir + '/study.csv', index=False)
                #df_valid.to_csv(submit_dir + '/df_valid.csv', index=False)

                #write_pickle_to_file(submit_dir + '/study.pickle', df_valid.study.values)
                #exit(0)
            else:
                detection = read_pickle_from_file(submit_dir + '/detection.pickle')
                pass


            #----
            df_image  = make_df_image(df_valid, detection)
            df_submit = df_image
            df_submit.to_csv(submit_dir + '/submit.csv', index=False)

            log.write('submit_dir : %s\n' % (submit_dir))
            log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
            log.write('df_submit : %s\n' % str(df_submit.shape))
            log.write('%s\n' % str(df_submit))
            log.write('\n')

            if 'local' in mode:
                #exit(0)

                #['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']
                df_predict = {
                    'ImageID':[],
                    'LabelName':[],
                    'Conf':[],
                    'XMin':[],
                    'XMax':[],
                    'YMin':[],
                    'YMax':[],
                }
                for i,det in enumerate(detection):
                    d = df_valid.iloc[i]

                    for x0, y0, x1, y1, c in det:
                        x0 = int(x0 / image_size * d.width)
                        y0 = int(y0 / image_size * d.height)
                        x1 = int(x1 / image_size * d.width)
                        y1 = int(y1 / image_size * d.height)

                        df_predict['ImageID'].append(d.image)
                        df_predict['LabelName'].append(0)
                        df_predict['Conf'].append(c)
                        df_predict['XMin'].append(x0)
                        df_predict['XMax'].append(x1)
                        df_predict['YMin'].append(y0)
                        df_predict['YMax'].append(y1)

                df_predict = pd.DataFrame(df_predict)
                log.write('df_predict.shape : %s\n' % str(df_predict.shape))


                #-------------------------------------------------------------------------

                #['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
                df_truth = {
                    'ImageID':[],
                    'LabelName':[],
                    'XMin':[],
                    'XMax':[],
                    'YMin':[],
                    'YMax':[],
                }

                gb = df_annotate.groupby('image_id')
                for i,d in df_valid.iterrows():
                    g = gb.get_group(d.image)

                    for j,f in g.iterrows():
                        if f.class_id==0: continue
                        x0 = f.x
                        y0 = f.y
                        x1 = f.x+f.w
                        y1 = f.y+f.h
                        df_truth['ImageID'].append(d.image)
                        df_truth['LabelName'].append(0)
                        df_truth['XMin'].append(x0)
                        df_truth['XMax'].append(x1)
                        df_truth['YMin'].append(y0)
                        df_truth['YMax'].append(y1)

                df_truth = pd.DataFrame(df_truth)
                log.write('df_truth.shape : %s\n' % str(df_truth.shape))

                map, _ = mean_average_precision_for_boxes(
                    df_truth, df_predict, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=True)

                log.write('map(opacity) : %f\n' % map)
                log.write('map*0.16     : %f\n' % (map/6))
                log.write('\n\n')
        #exit(0)


 
# main #################################################################
if __name__ == '__main__':
	run_submit()
 
