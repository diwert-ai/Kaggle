import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
       reference: https://github.com/4uiiurz1/keras-cosine-annealing
    """
    
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0, initial_epoch=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.initial_epoch = initial_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * (epoch + self.initial_epoch) / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)