from common import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
data_dir = '/content'
# "negative", "typical", "indeterminate", "atypical"
study_name_to_predict_string = {
    'Negative for Pneumonia'  :'negative',
    'Typical Appearance'      :'typical',
    'Indeterminate Appearance':'indeterminate',
    'Atypical Appearance'     :'atypical',
}

study_name_to_label = {
    'Negative for Pneumonia'  :0,
    'Typical Appearance'      :1,
    'Indeterminate Appearance':2,
    'Atypical Appearance'     :3,
}
study_label_to_name = { v:k for k,v in study_name_to_label.items()}
num_study_label = len(study_name_to_label)

#---
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
from sklearn.metrics import average_precision_score

def np_metric_roc_auc_by_class(probability, truth):
    num_sample, num_label = probability.shape
    score = []
    for i in range(num_label):
        s = roc_auc_score(truth==i, probability[:,i])
        score.append(s)
    score = np.array(score)
    return score



def np_metric_map_curve_by_class(probability, truth):
    num_sample, num_label = probability.shape
    score = []
    for i in range(num_label):
        s = average_precision_score(truth==i, probability[:,i])
        score.append(s)
    score = np.array(score)
    return score


def df_submit_to_predict(df):
    negative = []
    typical = []
    indeterminate = []
    atypical = []
    id = []

    for i,d in df.iterrows():
        if '_image' in d.id: continue
        p = d.PredictionString
        p = p.replace('0 0 1 1','')
        p = p.replace('negative','{"negative":')
        p = p.replace('typical',',"typical":')
        p = p.replace('indeterminate',',"indeterminate":')
        p = p.replace('a,"typical"',',"atypical"')
        p = p+'}'
        p = eval(p)

        negative.append(p['negative'])
        typical.append(p['typical'])
        indeterminate.append(p['indeterminate'])
        atypical.append(p['atypical'])
        id.append(d.id)

    df = pd.DataFrame({
        'id':id,
        'Negative for Pneumonia':negative,
        'Typical Appearance':typical,
        'Indeterminate Appearance':indeterminate,
        'Atypical Appearance':atypical,
    })
    #df = df.set_index('id')
    return df


#################################################################################

def draw_box(box, image, color=(255,255,255), thickness=2):
    num_box = len(box)
    for i in range(num_box):
        x0 = int(round(box[i,0]))
        y0 = int(round(box[i,1]))
        x1 = int(round(box[i,2]))
        y1 = int(round(box[i,3]))
        cv2.rectangle(image, (x0,y0), (x1,y1), color, thickness, cv2.LINE_AA)
    return image


def draw_annotate_box(annotate, image, anchor, thickness=1):

    box_coord = annotate['box_coord']
    box_index = annotate['box_index']

    height,width = image.shape[:2]

    if len(box_index) != 0:
        for t, i in enumerate(box_index):
            ax, ay, aw, ah = anchor[i]
            x0 = ax - aw // 2
            x1 = ax + aw // 2
            y0 = ay - ah // 2
            y1 = ay + ah // 2
            cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), thickness)

        for t, b in enumerate(box_coord):
            x0,y0,x1,y1 = b
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0,0,255), thickness)



    return image


def draw_pyramid_heatmap(heatmap, image_size=None, mode='max'):
    if image_size is None:
        _,_,image_size,image_size = heatmap[0].shape

    overlay = []
    for m in heatmap:
        m = m.copy()
        if mode=='max': m = m.max(0).max(0)   #(num_anchor, num_class, s, s)
        if mode=='min': m = m.min(0).min(0)   #(num_anchor, num_class, s, s)
        if mode=='mean':m = m.mean(0).mean(0) #(num_anchor, num_class, s, s)

        m = cv2.resize(m, dsize=(image_size,image_size),interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(m, (0, 0), (image_size, image_size), 1, 1)
        overlay.append(m)

    overlay = np.hstack(overlay)
    return overlay