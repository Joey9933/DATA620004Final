export CUDA_VISIBLE_DEVICES=7
epoch=150
# 在imagenet200上 有监督训练 150个epoch
python mytrain.py --model-name ResNet -dataset-name imagenet200 -b 512 --lr 1e-3 --n-views 1 --trainer Supervised --gpu-index 0 --out_dim 200 --epoch $epoch
python mytrain.py --model-name SimCLR -dataset-name imagenet200 -b 512 --lr 1e-3 --n-views 2 --trainer SelfSupervised --gpu-index 0 --out_dim 200 --epoch $epoch
sleep 1