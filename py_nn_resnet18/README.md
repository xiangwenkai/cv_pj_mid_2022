
### Train Examples
- CIFAR-100: 
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
--augment_prob 0.5 \
--augment cutmix \
--no-verbose
```
