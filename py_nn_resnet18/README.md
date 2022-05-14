
### Train Examples
- CIFAR-100: We used 2 GPUs to train CIFAR-100.
```
python train.py \
--net_type pyramidnet \
--dataset cifar100 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \
--no-verbose
```
- ImageNet: We used 4 GPUs to train ImageNet. 
```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--cutmix_prob 1.0 \
--no-verbose
```

### Test Examples using Pretrained model
- Download [CutMix-pretrained PyramidNet200 (top-1 error: 14.23)](https://www.dropbox.com/sh/o68qbvayptt2rz5/AACy3o779BxoRqw6_GQf_QFQa?dl=0) 
```
python test.py \
--net_type pyramidnet \
--dataset cifar100 \
--batch_size 64 \
--depth 200 \
--alpha 240 \
--pretrained /set/your/model/path/model_best.pth.tar
```
- Download [CutMix-pretrained ResNet50 (top-1 error: 21.40)](https://www.dropbox.com/sh/w8dvfgdc3eirivf/AABnGcTO9wao9xVGWwqsXRala?dl=0)
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 50 \
--pretrained /set/your/model/path/model_best.pth.tar
```

