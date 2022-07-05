import numpy as np
from sklearn.metrics import  recall_score, confusion_matrix
from tqdm.auto import tqdm

def compute_cm(frame,root_predicts,vowel_predicts,consonant_predicts):
    
    p_root=np.array([np.argmax(x) for x in root_predicts[:]]).reshape(-1)
    p_vowel = np.array([np.argmax(x) for x in vowel_predicts[:]]).reshape(-1)
    p_consonant = np.array([np.argmax(x) for x in consonant_predicts[:]]).reshape(-1)
    l = len(p_root)
    t_root=np.array(frame.iloc[:l,1].values, dtype=int)
    t_vowel=np.array(frame.iloc[:l,2].values, dtype=int)
    t_consonant=np.array(frame.iloc[:l,3].values, dtype=int)
  
    vowel_cm = confusion_matrix(t_vowel, p_vowel)
    vowel_recalls = np.diag(vowel_cm / np.sum(vowel_cm, axis = 1))

    cons_cm = confusion_matrix(t_consonant,p_consonant)
    cons_recalls = np.diag(cons_cm / np.sum(cons_cm, axis = 1))

    root_cm = confusion_matrix(t_root, p_root)
    root_recalls = np.diag(root_cm / np.sum(root_cm, axis = 1))

    return root_recalls,vowel_recalls,cons_recalls

def compute_recall(frame,root_predicts,vowel_predicts,consonant_predicts):
    
    p_root=np.array([np.argmax(x) for x in root_predicts[:]]).reshape(-1)
    p_vowel = np.array([np.argmax(x) for x in vowel_predicts[:]]).reshape(-1)
    p_consonant = np.array([np.argmax(x) for x in consonant_predicts[:]]).reshape(-1)
    l = len(p_root)
    t_root=np.array(frame.iloc[:l,1].values, dtype=int)
    t_vowel=np.array(frame.iloc[:l,2].values, dtype=int)
    t_consonant=np.array(frame.iloc[:l,3].values, dtype=int)
    root_recall = recall_score(t_root, p_root, average='macro')
    vowel_recall = recall_score(t_vowel, p_vowel, average='macro')
    cons_recall = recall_score(t_consonant,p_consonant,average='macro')

    return root_recall,vowel_recall, cons_recall

def compute_root_recall(frame,root_predicts):
    
    p_root=np.array([np.argmax(x) for x in root_predicts[:]]).reshape(-1)
  
    l = len(p_root)
    t_root=np.array(frame.iloc[:l,1].values, dtype=int)
  
    root_recall = recall_score(t_root, p_root, average='macro')
  

    return root_recall

def get_p_dicts(model,generator):
    root_predicts,vowel_predicts, consonant_predicts = [],[],[]
    for batch_x in tqdm(generator):
        batch_predict = model.predict(batch_x)
        for j in range(batch_predict[0].shape[0]):
            root_predicts += [batch_predict[0][j]]
            vowel_predicts += [batch_predict[1][j]]
            consonant_predicts += [batch_predict[2][j]]
    return root_predicts,vowel_predicts,consonant_predicts
    
def get_p_root_dict(model,generator):
    root_predicts= []
    for batch_x in tqdm(generator):
        batch_predict = model.predict(batch_x)
        for j in range(batch_predict.shape[0]):
            root_predicts += [batch_predict[j]]
            
    return root_predicts

def get_p_dicts_fromnp(model,v_np, batch_size=128):
    root_predicts,vowel_predicts, consonant_predicts = [],[],[]
    num_imgs = v_np.shape[0]
    for batch_start in tqdm(range(0, num_imgs,batch_size)):
        cur_batch_size = min(num_imgs,batch_start+batch_size)-batch_start
        idx = np.arange(batch_start,batch_start+cur_batch_size)
        predict = model.predict(v_np[idx])
        for j in range(cur_batch_size):
            root_predicts += [predict[0][j]]
            vowel_predicts += [predict[1][j]]
            consonant_predicts += [predict[2][j]]
    return root_predicts,vowel_predicts,consonant_predicts