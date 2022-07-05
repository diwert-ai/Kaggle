from common import *
from configure import *

from yolo5.models.yolo import *
from yolo5.utils.torch_utils import *
from yolo5.utils.general import bbox_iou, xywh2xyxy

#######################################################################
import torchvision

#single class only


def do_non_max_suppression(
    predict_flat,
    nms_objectness_threshold = nms_objectness_threshold,
    nms_iou_threshold = nms_iou_threshold,
    nms_pre_max_num   = nms_pre_max_num,
    nms_post_max_num  = nms_post_max_num,
):
    batch_size = len(predict_flat)
    detection = []
    for b in range(batch_size):
        p = predict_flat[b]

        i = p[..., 4] > nms_objectness_threshold
        p = p[i]
        num = len(p)
        if num==0:
            det = np.zeros((0,5),np.float32) #None
            detection.append(det)
            continue

        box = xywh2xyxy(p[:, :4])
        score = p[:,4]
        #<todo> x[:, 5:] *= x[:, 4:5]  #conf = obj_conf * cls_conf

        if num > nms_pre_max_num:  # excess boxes
            i = score.argsort(descending=True)[:nms_pre_max_num]  # sort by confidence
            box = box[i]
            score = score[i]

        i = torchvision.ops.nms(box, score, nms_iou_threshold)
        if len(i) > nms_post_max_num:
            i = i[:nms_post_max_num]

        #<todo> merge NMS (boxes merged using weighted mean)  # sort by confidence
        box = box[i]
        score = score[i]
        det = torch.cat([box,score[:,None]],-1)
        det = det.data.cpu().numpy()
        detection.append(det)

    return detection


def pyramid_to_flat(data):
    flat = []
    for n in range(num_head):
        p = data[n]
        batch_size, num_anchor, w, h, dim = p.shape
        flat.append(p.reshape(batch_size, -1, dim))
    flat = torch.cat(flat, 1)
    return flat

def infer_prediction(predict):
    device = predict[0].device

    z = []  # inference output
    for n in range(num_head):
        p = predict[n]
        batch_size, num_anchor, w, h, dim = p.shape

        # inference
        aa = torch.FloatTensor(anchor_size[n]).to(device)
        yy, xx = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((xx, yy), 2).reshape((1, 1, h, w, 2)).float().to(device)

        d = p.sigmoid()  #objectivesness and class also use sigmoid ...
        d[..., 0:2] = (d[..., 0:2] * 2 - 0.5 + grid) * feature_stride[n]  # xy
        d[..., 2:4] = (d[..., 2:4] * 2) ** 2 * aa.reshape(1, num_anchor, 1, 1, 2)  # wh

        z.append(d)
    return z



#######################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #print("Class Net. Current working dir : %s" % os.getcwd())
        model_cfg_file = '/content/drive/My Drive/kaggle/covid19-det/nbs/py/hengck_code/dummy_01a/custom-yolo5-640-v4/yolo5/models/yolov5m.yaml'
        pretrain_file  = '/content/yolov5m.pt'
        e = Model(
            cfg=model_cfg_file,
            ch=3,
            nc=1,
            anchors=np.array(anchor_size).reshape(num_head,-1).tolist(),
        )
        state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, e.state_dict(), exclude=['anchor'])  # intersect
        e.load_state_dict(state_dict, strict=False)

        #---
        # remove detect layer
        assert (e.save[-3:] == e.model[-1].f)
        removed = list(e.model.children())[:-1]
        self.backbone = torch.nn.Sequential(*removed)
        self.index = e.save
        #print(self.index)

        #---
        self.head = nn.ModuleList([
            nn.Conv2d(192, num_anchor*6, kernel_size=1),
            nn.Conv2d(384, num_anchor*6, kernel_size=1),
            nn.Conv2d(768, num_anchor*6, kernel_size=1),
        ])

        #---
        #<todo> add image classification head
        # self.index.append( ... add feature layer no to use ...)
        #self.logit=nn.Sequential(
        #    nn.Conv2d(768, 1, kernel_size=1),
        #)


    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1

        # yolov5 backbone ----------------------
        # predict = self.e(x)
        z = []
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int):
                    x = z[m.f]
                else:
                    x = [x if i == -1 else z[i] for i in m.f]
            x = m(x)  # run
            z.append(x if m.i in self.index else None)  # cache output
        z = [z[i] for i in self.index[-3:]]
        #--------------------------------------
        predict = []
        for n in range(num_head):
            p = self.head[n](z[n])
            batch_size, num_anchor_dim, h, w = p.shape
            dim = num_anchor_dim//num_anchor
            p = p.reshape(batch_size, num_anchor, dim, h, w).permute(0, 1, 3, 4, 2).contiguous()
            predict.append(p)

        return predict



############################################################################

def make_truth(
    annotate,
    anchor_match_ratio_threshold = anchor_match_ratio_threshold,
):
    device = annotate.device
    label, box, index, anchor_index = [], [], [], []

    norm_anchor_size = torch.FloatTensor(make_norm_anchor_size()).to(device)
    num_target = len(annotate)
    arange1 = torch.arange(num_anchor, device=device).float().reshape(num_anchor, 1).repeat(1, num_target)
    arange2 = torch.arange(num_target, device=device).float().reshape(1, num_target).repeat(num_anchor, 1)
    target = torch.cat([
        annotate.repeat(num_anchor, 1, 1),
        arange1[:, :, None],
        arange2[:, :, None],
    ], 2) #shape: num_anchor, num_target, 7
    # append anchor index
    # target:    # batch_i, class_i,  cx_cy_w_h, anchor_i
    # annotate:  # batch_i, class_i,  cx_cy_w_h

    g = 0.5
    neighbour = torch.tensor([
        [0, 0],
        [1, 0], [0, 1], [-1, 0], [0, -1],      # j,k,l,m
        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    ], device=device).float() * g    # offset

    for i in range(num_head):
        a = norm_anchor_size[i]
        s = feature_size[i]
        gain = torch.tensor([1.,1.,s,s,s,s,1.,1.], device=device)

        # match targets to anchors
        t = target * gain
        if num_target>0:
            # wh ratio
            r = t[:, :, 4:6] / a[:, None]
            valid = torch.max(r, 1. / r).max(2)[0] < anchor_match_ratio_threshold # compare
            # valid = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[valid]  # filter

            # get neighbour
            #   https://www.kaggle.com/c/global-wheat-detection/discussion/172436
            #   https://github.com/ultralytics/yolov5/issues/802
            #   https://zhuanlan.zhihu.com/p/159371985
            #   https://blog.csdn.net/Q1u1NG/article/details/108799441
            #   https://zhuanlan.zhihu.com/p/172121380
            gxy = t[:, 2:4]  # grid xy
            ixy = gain[2:4] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((ixy % 1. < g) & (ixy > 1.)).T
            select = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[select]
            offset = (torch.zeros_like(gxy)[None] + neighbour[:, None])[select]
        else:
            t = target[0]
            offset = 0

        # define
        b, c = t[:, :2].long().T  # image, class
        gxy  = t[:, 2:4]  # grid xy
        gwh  = t[:, 4:6]  # grid wh
        gij  = (gxy - offset).long()
        gi, gj = gij.T  # grid xy indices

        a = t[:, 6].long()  # anchor indices
        u = t[:, 7].long()  # truth  indices
        index.append((b, u, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid_j , grid_i
        box.append(torch.cat((gxy - gij, gwh), 1))  # box
        label.append(c)  # class

    truth = {
        'index': index,
        'label': label,
        'box': box,
    }
    return truth

####################################################################################
def focal_loss_for_binary_cross_entropy_with_logit(logit, truth, gamma=1.5, alpha=0.25):
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    prob = torch.sigmoid(logit)  # prob from logits
    p = truth * prob + (1 - truth) * (1 - prob)
    g = (1.0 - p) ** gamma
    a = truth * (1 - alpha) + (1 - truth) * alpha
    loss = loss * a * g

    loss = loss.mean()
    return loss



def modified_yolo_loss(predict, truth):
    true_index = truth['index']
    true_label = truth['label']
    true_box   = truth['box']
    device = predict[0].device

    loss_cls, loss_box, loss_obj = \
        torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    norm_anchor_size = torch.FloatTensor(make_norm_anchor_size()).to(device)

    # losses
    for i, pred in enumerate(predict):
        b, u, a, gj, gi = true_index[i]  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets

        true_objectness = torch.zeros_like(pred[..., 0], device=device)  # target obj
        if n:
            p = pred[b, a, gj, gi]  # prediction subset corresponding to targets

            # regression
            pxy = (p[:, [0,1]].sigmoid() * 2) - 0.5
            pwh = (p[:, [2,3]].sigmoid() * 2) ** 2 * norm_anchor_size[i][a]
            box = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(box.T, true_box[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            loss_box += (1.0 - iou).mean()  # iou loss


            # classification
            assert(num_class==1)
            if num_class>1:  # cls loss (only if multiple classes)
                pass
                #<todo> ?
                # t = torch.full_like(p[:, 5:], self.cn, device=device)
                # t[range(n), tcls[i]] = self.cp
                # lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            # objectness
            o = iou.detach().clamp(0)
            true_objectness[b, a, gj, gi] = o.type(true_objectness.dtype)    # iou ratio
        loss_obj += loss_level_balance[i] * \
                    focal_loss_for_binary_cross_entropy_with_logit(pred[..., 4], true_objectness)
                    #F.binary_cross_entropy_with_logits(pred[..., 4], true_objectness)

    # batch_size = len(predict[0])
    loss_cls *= loss_cls_balance
    loss_box *= loss_box_balance
    loss_obj *= loss_obj_balance
    return loss_cls, loss_box, loss_obj




############################################################################

def run_check_net():
    batch_size = 2
    #C, H, W = 3, 512, 512
    C, H, W = 3, 640, 640
    image = torch.randn(batch_size, C, H, W)#.cuda()

    net = Net()#.cuda()
    #print(net)
    #print(net.e.model[-1].anchors)

    predict = net(image)

    print(image.shape)
    for n in range(num_head):
        print(n, predict[n].shape)

def run_check_loss():
    norm_anchor_size = torch.FloatTensor(
       [[[1.25000, 1.62500],
         [2.00000, 3.75000],
         [4.12500, 2.87500]],
        [[1.87500, 3.81250],
         [3.87500, 2.81250],
         [3.68750, 7.43750]],
        [[3.62500, 2.81250],
         [4.87500, 6.18750],
         [11.65625, 10.18750]]]
    )
    annotate = torch.FloatTensor(
           # batch_i, class_i,  cx_cy_w_h
           [[0.00000, 0.00000, 0.85330, 0.49471, 0.20789, 0.32667],
            [2.00000, 0.00000, 0.34049, 0.53764, 0.26069, 0.43307],
            [2.00000, 0.00000, 0.78786, 0.46187, 0.33799, 0.33433],
            [3.00000, 0.00000, 0.70699, 0.56112, 0.29921, 0.25383]]
    )


    batch_size=len(annotate)
    p0 = torch.randn(batch_size, 3, 80, 80, 6)
    p1 = torch.randn(batch_size, 3, 40, 40, 6)
    p2 = torch.randn(batch_size, 3, 20, 20, 6)
    predict = [p0,p1,p2]

    truth = make_truth(annotate)
    for n in range(num_head):
        print(n, 'label :', truth['label'][n].shape)
        print(n, 'box   :', truth['box'][n].shape)
        print(n, 'index :', '%dx'%len(truth['index'][n]), truth['index'][n][0].shape)
        print('')

    #---
    loss_cls, loss_box, loss_obj = modified_yolo_loss(predict, truth)
    print(loss_cls, loss_box, loss_obj)


# main #################################################################
if __name__ == '__main__':
    run_check_net()
    #run_check_loss()
