import tensorflow as tf
import numpy as np

class cosine_with_warmup():
    def __init__(self,max_lr,warmup_epochs,decay_epochs):
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.current_epoch = 0

    def step(self):

        lr_ = None
        
        if self.current_epoch < self.warmup_epochs:  # Linear warmup
            
            lr_ = self.max_lr * (self.current_epoch) / self.warmup_epochs + 1e-6
            self.current_epoch += 1
            
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.decay_epochs)
            lr_ = float(self.max_lr * 0.5 * (1 + tf.cos(np.pi * progress)))
            self.current_epoch += 1
        return lr_

class cosine_annealing_warm_restart_with_warmup():
    def __init__(self,first_cycle_steps,cycle_mult,max_lr,warmup_epochs,gamma):
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.current_epoch = 0
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult

        self.current_cycle_steps = first_cycle_steps
        self.current_cycle_progress = 0
    def step(self):
        
        if self.current_cycle_progress < self.warmup_epochs:
            lr_ = self.max_lr * (self.current_cycle_progress) / self.warmup_epochs + 1e-6
            self.current_cycle_progress += 1
        else:
            progress = (self.current_cycle_progress - self.warmup_epochs) / (self.current_cycle_steps - self.warmup_epochs)
            lr_ =  float(self.max_lr * 0.5 * (1 + tf.cos(np.pi * progress)))
            self.current_cycle_progress += 1
            
            if self.current_cycle_progress == self.current_cycle_steps:
                self.current_cycle_steps = self.current_cycle_steps*self.cycle_mult
                self.max_lr = self.max_lr * self.gamma
                self.current_cycle_progress = 0
        return lr_