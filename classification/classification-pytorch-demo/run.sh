#!/bin/sh
checkpoint_dir='/workspace/pytorch_demo/checkpoint/'

if [ ! -d $checkpoint_dir ];then
	mkdir -p $checkpoint_dir
else
	rm -rf $checkpoint_dir
	mkdir -p $checkpoint_dir
fi

nohup  python train.py \
	--data_type=pic_folder \
        --label_path='/workspace/dataset/train_imgs/label.txt' \
        --train_folder='/workspace/pytorch_demo/dataset/cifar10/train' \
        --class_num=10 \
        --resume=false \
        --pretrained_model='/workspace/pytorch_demo/pretrained/model.pth' \
        --epoch=100 \
        --batch_size=64 \
        --lr=0.01 \
        --log_step=100 \
        --save_step=1000 \
	--GPU=0 \
        --checkpoint_dir=${checkpoint_dir}>train.log &

python  evaluate.py \
        --test_folder='/workspace/pytorch_demo/dataset/cifar10/test' \
        --log_step=10 \
        --checkpoint_dir=${checkpoint_dir} \
        --GPU=0


