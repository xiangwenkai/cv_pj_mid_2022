from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import math

df = pd.read_csv('cutmix_loss.csv')
writer = SummaryWriter()
print(writer.log_dir)

train_loss = list(df['train_loss'])
val_loss = list(df['val_loss'])
err1 = list(df['err1'])
err5 = list(df['err5'])
for epoch in range(200):
    writer.add_scalars('cutmix_loss', {'train_loss':train_loss[epoch], 'val_loss':val_loss[epoch]}, epoch)
    writer.add_scalars('cutmix_err', {'err1':err1[epoch], 'err5':err5[epoch]}, epoch)


df = pd.read_csv('cutout_loss.csv')
writer = SummaryWriter()
print(writer.log_dir)

train_loss = list(df['train_loss'])
val_loss = list(df['val_loss'])
err1 = list(df['err1'])
err5 = list(df['err5'])
for epoch in range(200):
    writer.add_scalars('cutout_loss', {'train_loss':train_loss[epoch], 'val_loss':val_loss[epoch]}, epoch)
    writer.add_scalars('cutout_err', {'err1':err1[epoch], 'err5':err5[epoch]}, epoch)


df = pd.read_csv('mixup_loss.csv')
writer = SummaryWriter()
print(writer.log_dir)

train_loss = list(df['train_loss'])
val_loss = list(df['val_loss'])
err1 = list(df['err1'])
err5 = list(df['err5'])
for epoch in range(200):
    writer.add_scalars('mixup_loss', {'train_loss':train_loss[epoch], 'val_loss':val_loss[epoch]}, epoch)
    writer.add_scalars('mixup_err', {'err1':err1[epoch], 'err5':err5[epoch]}, epoch)


df = pd.read_csv('None_loss.csv')
writer = SummaryWriter()
print(writer.log_dir)

train_loss = list(df['train_loss'])
val_loss = list(df['val_loss'])
err1 = list(df['err1'])
err5 = list(df['err5'])
for epoch in range(200):
    writer.add_scalars('base_loss', {'train_loss':train_loss[epoch], 'val_loss':val_loss[epoch]}, epoch)
    writer.add_scalars('base_err', {'err1':err1[epoch], 'err5':err5[epoch]}, epoch)


