from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import gc
import time
import datetime
import math
import pandas as pd

from generators import test_batch_generator, test_batch_generator_fs
from kaggle_metric import get_p_dicts, compute_recall, get_p_root_dict, compute_root_recall

class KaggleValidation(Callback):
    
    def __init__(self, valid_df,train_steps, vals_per_epoch=1,batch_size = 32, logfile=None, initial_epoch=0,suffix=None,
                 train_img_dir='./',
                 output_dir='./',
                 thresholds=[0,0,0],
                 chkp_manager=None):
        super().__init__()
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.current_epoch=initial_epoch
        self.num_batches = train_steps
        self.val_steps = train_steps//vals_per_epoch
        self.logfile = logfile
        self.best_r_kr = thresholds[0]
        self.best_v_kr = thresholds[1]
        self.best_c_kr = thresholds[2]
        self.best_kr=0.5*self.best_r_kr+0.25*(self.best_v_kr+self.best_c_kr)
        self.best_comb_kr = self.best_kr
        self.suffix = suffix
        self.train_img_dir=train_img_dir
        self.output_dir = output_dir
        self.chkp_manager = chkp_manager
       
    def do_savelog(self):
        log_df = pd.DataFrame()
        log_df['epoch']=self.iters
        log_df['lr']=self.lr_values
        log_df['loss']=self.losses
        log_df['root_loss']=self.root_losses
        log_df['vowel_loss']=self.vowel_losses
        log_df['consonant_loss']=self.cons_losses
        log_df['kaggle']=self.val_kaggle_recalls
        log_df['root']=self.val_root_recalls
        log_df['vowel']=self.val_vowel_recalls
        log_df['cons'] =self.val_consonant_recalls
        log_df['time']=self.time_stamp
        log_df.to_csv(self.logfile, index=False, float_format='%.6f')
      
    def do_validation(self,num_iter=0, logs={}):
        valid_gen = test_batch_generator(self.valid_df,self.train_img_dir, batch_size=self.batch_size)
        val_root_preds,val_vowel_preds,val_consonant_preds = get_p_dicts(self.model,valid_gen)
        val_root_recall,val_vowel_recall, val_cons_recall = compute_recall(self.valid_df,
                                                                           val_root_preds,
                                                                           val_vowel_preds,
                                                                           val_consonant_preds)
        val_kaggle_recall = 0.5*val_root_recall+0.25*(val_vowel_recall+val_cons_recall)
        loc_time = math.floor(time.time()-self.time_start)
        print(f'[{str(datetime.timedelta(seconds=loc_time))}] - kaggle:{val_kaggle_recall:.4f} - root:{val_root_recall:.4f} -vowel:{val_vowel_recall:.4f} - cons: {val_cons_recall:.4f}')
        self.val_root_recalls.append(val_root_recall)
        self.val_vowel_recalls.append(val_vowel_recall)
        self.val_consonant_recalls.append(val_cons_recall)
        self.val_kaggle_recalls.append(val_kaggle_recall)
        self.time_stamp.append(str(datetime.timedelta(seconds=loc_time)))
        self.iters.append(num_iter)
        rl = logs.get('root_loss')
        vl = logs.get('vowel_loss')
        cl = logs.get('consonant_loss')
        self.losses.append(2*rl+vl+cl)
        self.root_losses.append(rl)
        self.vowel_losses.append(vl)
        self.cons_losses.append(cl)
        self.lr_values.append(K.get_value(self.model.optimizer.lr))

        if self.logfile != None: self.do_savelog()

        if val_kaggle_recall>self.best_kr:
            self.best_kr = val_kaggle_recall
            print(f'saving weights with kr {self.best_kr}...')
            self.model.save_weights(self.output_dir+f'w_kr_{self.best_kr:.4f}_'+self.suffix+'.h5')
        
        preffix = 'w_best_'
        save_weights = False

        if val_root_recall>self.best_r_kr:
            self.best_r_kr = val_root_recall
            preffix = preffix + f'[r {self.best_r_kr:.5f}]'
            save_weights=True
        
        if val_vowel_recall>self.best_v_kr:
            self.best_v_kr = val_vowel_recall
            preffix = preffix + f'[v {self.best_v_kr:.5f}]'
            save_weights=True
        
        if val_cons_recall>self.best_c_kr:
            self.best_c_kr = val_cons_recall
            preffix = preffix + f'[c {self.best_c_kr:.5f}]'
            save_weights=True
        
        if save_weights==True:
            self.best_comb_kr = 0.5*self.best_r_kr+0.25*(self.best_v_kr+self.best_c_kr)
            print(f'saving weights {preffix} best combined kaggle recall: {self.best_comb_kr}...')
            self.model.save_weights(self.output_dir+preffix+'_'+self.suffix+'.h5')

    def on_train_begin(self, logs={}):
        self.val_root_recalls = []
        self.val_vowel_recalls =[]
        self.val_consonant_recalls = []
        self.val_kaggle_recalls = []
        self.time_stamp = []
        self.iters = []
        self.time_start = time.time()
        self.losses = []
        self.root_losses = []
        self.vowel_losses = []
        self.cons_losses = []
        self.lr_values = []
        
    def on_batch_end(self,batch,logs={}):
        if batch%self.val_steps == self.val_steps-1: 
            print(f'\nbatch {batch}: validation...')
            num_iter = self.current_epoch + batch/self.num_batches
            self.do_validation(num_iter=num_iter,logs=logs)
            gc.collect()
    
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch +=1
        if self.chkp_manager !=None:
            self.chkp_manager._checkpoint.latest_epoch.assign_add(1)
            save_path=self.chkp_manager.save()
            print("\nsaved checkpoint for epoch {}: {}\n".format(int(self.chkp_manager._checkpoint.latest_epoch), save_path))
            
            
class KaggleRootValidation(Callback):
    
    def __init__(self, valid_df,train_steps, vals_per_epoch=1,batch_size = 32, logfile=None, initial_epoch=0,
                    suffix=None,
                    train_img_dir='./',
                    output_dir='./',
                    threshold=0,
                    chkp_manager=None):
        super().__init__()
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.current_epoch=initial_epoch
        self.num_batches = train_steps
        self.val_steps = train_steps//vals_per_epoch
        self.logfile = logfile
        self.best_r_kr = threshold
        self.suffix = suffix
        self.train_img_dir=train_img_dir
        self.output_dir = output_dir
        self.chkp_manager = chkp_manager
       
    def do_savelog(self):
        log_df = pd.DataFrame()
        log_df['epoch']=self.iters
        log_df['lr']=self.lr_values
        log_df['loss']=self.losses
        log_df['root']=self.val_root_recalls
        log_df['time']=self.time_stamp
        log_df.to_csv(self.logfile, index=False, float_format='%.6f')
      
    def do_validation(self,num_iter=0, logs={}):
        valid_gen = test_batch_generator(self.valid_df,self.train_img_dir, batch_size=self.batch_size)
        val_root_preds = get_p_root_dict(self.model,valid_gen)
        val_root_recall = compute_root_recall(self.valid_df,val_root_preds)
        loc_time = math.floor(time.time()-self.time_start)
        print(f'[{str(datetime.timedelta(seconds=loc_time))}] - root:{val_root_recall:.4f}')
        self.val_root_recalls.append(val_root_recall)
        self.time_stamp.append(str(datetime.timedelta(seconds=loc_time)))
        self.iters.append(num_iter)
        self.losses.append(logs.get('loss'))
        self.lr_values.append(K.get_value(self.model.optimizer.lr))

        if self.logfile != None: self.do_savelog()
        
        preffix = 'w_best_'

        if val_root_recall>self.best_r_kr:
            self.best_r_kr = val_root_recall
            preffix = preffix + f'[r {self.best_r_kr:.5f}]'
            print(f'saving weights {preffix}')
            self.model.save_weights(self.output_dir+preffix+'_'+self.suffix+'.h5')

    def on_train_begin(self, logs={}):
        self.val_root_recalls = []
        self.time_stamp = []
        self.iters = []
        self.time_start = time.time()
        self.losses = []
        self.lr_values = []
        
    def on_batch_end(self,batch,logs={}):
        if batch%self.val_steps == self.val_steps-1: 
            print(f'\nbatch {batch}: validation...')
            num_iter = self.current_epoch + batch/self.num_batches
            self.do_validation(num_iter=num_iter,logs=logs)
            gc.collect()
    
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch +=1
        if self.chkp_manager !=None:
            self.chkp_manager._checkpoint.latest_epoch.assign_add(1)
            save_path=self.chkp_manager.save()
            print("\nsaved checkpoint for epoch {}: {}\n".format(int(self.chkp_manager._checkpoint.latest_epoch), save_path))

class KaggleValidationFS(Callback):
    
    def __init__(self, valid_df,train_steps, vals_per_epoch=1,batch_size = 32, logfile=None, initial_epoch=0,suffix=None,
                 train_img_dir='./',
                 output_dir='./',
                 thresholds=[0,0,0],
                 chkp_manager=None,
                 in_shape=[137,236],
                 st=(0,1)):
        super().__init__()
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.current_epoch=initial_epoch
        self.num_batches = train_steps
        self.val_steps = train_steps//vals_per_epoch
        self.logfile = logfile
        self.best_r_kr = thresholds[0]
        self.best_v_kr = thresholds[1]
        self.best_c_kr = thresholds[2]
        self.best_kr=0.5*self.best_r_kr+0.25*(self.best_v_kr+self.best_c_kr)
        self.best_comb_kr = self.best_kr
        self.suffix = suffix
        self.train_img_dir=train_img_dir
        self.output_dir = output_dir
        self.chkp_manager = chkp_manager
        self.in_height = in_shape[0]
        self.in_width = in_shape[1]
        self.st = st
       
    def do_savelog(self):
        log_df = pd.DataFrame()
        log_df['epoch']=self.iters
        log_df['lr']=self.lr_values
        log_df['loss']=self.losses
        log_df['root_loss']=self.root_losses
        log_df['vowel_loss']=self.vowel_losses
        log_df['consonant_loss']=self.cons_losses
        log_df['kaggle']=self.val_kaggle_recalls
        log_df['root']=self.val_root_recalls
        log_df['vowel']=self.val_vowel_recalls
        log_df['cons'] =self.val_consonant_recalls
        log_df['time']=self.time_stamp
        log_df.to_csv(self.logfile, index=False, float_format='%.6f')
      
    def do_validation(self,num_iter=0, logs={}):
        valid_gen = test_batch_generator_fs(self.valid_df,self.train_img_dir, batch_size=self.batch_size,height=self.in_height, width=self.in_width,st=self.st)
        val_root_preds,val_vowel_preds,val_consonant_preds = get_p_dicts(self.model,valid_gen)
        val_root_recall,val_vowel_recall, val_cons_recall = compute_recall(self.valid_df,
                                                                           val_root_preds,
                                                                           val_vowel_preds,
                                                                           val_consonant_preds)
        val_kaggle_recall = 0.5*val_root_recall+0.25*(val_vowel_recall+val_cons_recall)
        loc_time = math.floor(time.time()-self.time_start)
        print(f'[{str(datetime.timedelta(seconds=loc_time))}] - kaggle:{val_kaggle_recall:.4f} - root:{val_root_recall:.4f} -vowel:{val_vowel_recall:.4f} - cons: {val_cons_recall:.4f}')
        self.val_root_recalls.append(val_root_recall)
        self.val_vowel_recalls.append(val_vowel_recall)
        self.val_consonant_recalls.append(val_cons_recall)
        self.val_kaggle_recalls.append(val_kaggle_recall)
        self.time_stamp.append(str(datetime.timedelta(seconds=loc_time)))
        self.iters.append(num_iter)
        rl = logs.get('root_loss')
        vl = logs.get('vowel_loss')
        cl = logs.get('consonant_loss')
        self.losses.append(2*rl+vl+cl)
        self.root_losses.append(rl)
        self.vowel_losses.append(vl)
        self.cons_losses.append(cl)
        self.lr_values.append(K.get_value(self.model.optimizer.lr))

        if self.logfile != None: self.do_savelog()

        if val_kaggle_recall>self.best_kr:
            self.best_kr = val_kaggle_recall
            print(f'saving weights with kr {self.best_kr}...')
            self.model.save_weights(self.output_dir+f'w_kr_{self.best_kr:.4f}_'+self.suffix+'.h5')
        
        preffix = 'w_best_'
        save_weights = False

        if val_root_recall>self.best_r_kr:
            self.best_r_kr = val_root_recall
            preffix = preffix + f'[r {self.best_r_kr:.5f}]'
            save_weights=True
        
        if val_vowel_recall>self.best_v_kr:
            self.best_v_kr = val_vowel_recall
            preffix = preffix + f'[v {self.best_v_kr:.5f}]'
            save_weights=True
        
        if val_cons_recall>self.best_c_kr:
            self.best_c_kr = val_cons_recall
            preffix = preffix + f'[c {self.best_c_kr:.5f}]'
            save_weights=True
        
        if save_weights==True:
            self.best_comb_kr = 0.5*self.best_r_kr+0.25*(self.best_v_kr+self.best_c_kr)
            print(f'saving weights {preffix} best combined kaggle recall: {self.best_comb_kr}...')
            self.model.save_weights(self.output_dir+preffix+'_'+self.suffix+'.h5')

    def on_train_begin(self, logs={}):
        self.val_root_recalls = []
        self.val_vowel_recalls =[]
        self.val_consonant_recalls = []
        self.val_kaggle_recalls = []
        self.time_stamp = []
        self.iters = []
        self.time_start = time.time()
        self.losses = []
        self.root_losses = []
        self.vowel_losses = []
        self.cons_losses = []
        self.lr_values = []
        
    def on_batch_end(self,batch,logs={}):
        if batch%self.val_steps == self.val_steps-1: 
            print(f'\nbatch {batch}: validation...')
            num_iter = self.current_epoch + batch/self.num_batches
            self.do_validation(num_iter=num_iter,logs=logs)
            gc.collect()
    
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch +=1
        logs['kaggle']=self.val_kaggle_recalls[len(self.val_kaggle_recalls)-1]
        if self.chkp_manager !=None:
            self.chkp_manager._checkpoint.latest_epoch.assign_add(1)
            save_path=self.chkp_manager.save()
            print("\nsaved checkpoint for epoch {}: {}\n".format(int(self.chkp_manager._checkpoint.latest_epoch), save_path))

class KaggleValidationRootFS(Callback):
    
    def __init__(self, valid_df,train_steps, vals_per_epoch=1,batch_size = 32, logfile=None, initial_epoch=0,suffix=None,
                 train_img_dir='./',
                 output_dir='./',
                 threshold=0,
                 chkp_manager=None,
                 in_shape=[137,236],
                 st=(0,1)):
        super().__init__()
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.current_epoch=initial_epoch
        self.num_batches = train_steps
        self.val_steps = train_steps//vals_per_epoch
        self.logfile = logfile
        self.best_r_kr = threshold
        self.suffix = suffix
        self.train_img_dir=train_img_dir
        self.output_dir = output_dir
        self.chkp_manager = chkp_manager
        self.in_height = in_shape[0]
        self.in_width = in_shape[1]
        self.st = st
       
    def do_savelog(self):
        log_df = pd.DataFrame()
        log_df['epoch']=self.iters
        log_df['lr']=self.lr_values
        log_df['loss']=self.losses
        log_df['root']=self.val_root_recalls
        log_df['time']=self.time_stamp
        log_df.to_csv(self.logfile, index=False, float_format='%.6f')
      
    def do_validation(self,num_iter=0, logs={}):
        valid_gen = test_batch_generator_fs(self.valid_df,self.train_img_dir, batch_size=self.batch_size,height=self.in_height, width=self.in_width,st=self.st)
        val_root_preds = get_p_root_dict(self.model,valid_gen)
        val_root_recall = compute_root_recall(self.valid_df,val_root_preds)
        loc_time = math.floor(time.time()-self.time_start)
        print(f'[{str(datetime.timedelta(seconds=loc_time))}] - root:{val_root_recall:.4f}')
        self.val_root_recalls.append(val_root_recall)
        self.time_stamp.append(str(datetime.timedelta(seconds=loc_time)))
        self.iters.append(num_iter)
        self.losses.append(logs.get('loss'))
        self.lr_values.append(K.get_value(self.model.optimizer.lr))

        if self.logfile != None: self.do_savelog()
        
        preffix = 'w_best_'

        if val_root_recall>self.best_r_kr:
            self.best_r_kr = val_root_recall
            preffix = preffix + f'[r {self.best_r_kr:.5f}]'
            print(f'saving weights {preffix}')
            self.model.save_weights(self.output_dir+preffix+'_'+self.suffix+'.h5')

    def on_train_begin(self, logs={}):
        self.val_root_recalls = []
        self.time_stamp = []
        self.iters = []
        self.time_start = time.time()
        self.losses = []
        self.lr_values = []
        
    def on_batch_end(self,batch,logs={}):
        if batch%self.val_steps == self.val_steps-1: 
            print(f'\nbatch {batch}: validation...')
            num_iter = self.current_epoch + batch/self.num_batches
            self.do_validation(num_iter=num_iter,logs=logs)
            gc.collect()
    
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch +=1
        logs['root_only']=self.val_root_recalls[len(self.val_root_recalls)-1]
        if self.chkp_manager !=None:
            self.chkp_manager._checkpoint.latest_epoch.assign_add(1)
            save_path=self.chkp_manager.save()
            print("\nsaved checkpoint for epoch {}: {}\n".format(int(self.chkp_manager._checkpoint.latest_epoch), save_path))