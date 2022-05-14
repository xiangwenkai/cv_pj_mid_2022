import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.datasets as datasets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

data = datasets.CIFAR100('../data', train=True, download=True)


method = 'cutmix'
r = np.random.rand(1)
cutmix_prob = 0.5
beta = 1

def to_pil(m):
    r = Image.fromarray(m[:,:,0])
    g = Image.fromarray(m[:,:,1])
    b = Image.fromarray(m[:,:,2])
    pil_img = Image.merge('RGB', (r,g,b))
    return pil_img


if method == 'cutmix':
    input = data.data[500:600]
    lam = 0.4
    rand_index = torch.randperm(input.shape[0])
    # rand_index = torch.tensor([1,2,0])
    # target_a = target
    # target_b = target[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox((100,3,32,32), lam)
    # bbx1, bby1, bbx2, bby2 = 1, 5, 16, 20
    input1 = input[:]
    input1[:, bbx1:bbx2, bby1:bby2, :] = input[rand_index, bbx1:bbx2, bby1:bby2, :]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.shape[-2] * input.shape[-3]))
    # compute output

    for i in range(3):
        m = input1[i]
        img1 = to_pil(m)
        plt.imshow(img1)
        plt.savefig('graph/cutmix_{}.png'.format(i),dpi=200)
        plt.close()
elif method=='cutout':
    input = data.data[400:403]
    # lam = np.random.beta(beta, beta)
    lam=0.4
    # rand_index = torch.randperm(input.shape[0])
    rand_index = torch.tensor([1, 2, 0])
    # target_a = target
    # target_b = target[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox((3,3,32,32), lam)
    # bbx1, bby1, bbx2, bby2 = 1, 5, 16, 20
    input1 = input[:]
    input1[:, bbx1:bbx2, bby1:bby2, :] = 0
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.shape[-2] * input.shape[-3]))
    # compute output

    for i in range(3):
        m = input1[i]
        img1 = to_pil(m)
        plt.imshow(img1)
        plt.savefig('graph/cutout_{}.png'.format(i),dpi=200)
        plt.close()

elif method=='mixup':
    input = data.data[1500:1600]
    rand_index = torch.randperm(input.shape[0])

    input1 = input[rand_index]

    input2 = input*0.5+input1*0.5
    input2=input2.astype(input.dtype)
    for i in range(3):
        m = input2[i]
        img1 = to_pil(m)
        plt.imshow(img1)
        plt.savefig('graph/mixup_{}.png'.format(i), dpi=200)
        plt.close()