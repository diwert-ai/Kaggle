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