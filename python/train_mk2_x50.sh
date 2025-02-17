#! /bin/bash

var=x50

pfx1=mk2/s1/${var}/0
pfx2=mk2/s2/${var}/0
pfx3=mk2/s3/${var}/0
pfx4=mk2/s4/${var}/0
pfx5=mk2/s5/${var}/0

net="--net x50"

if [[ ! -e $pfx1/final_0_8.pth ]];then
mkdir -p $pfx1

python aqc_training.py ${net} --workers 8 \
    --data /fast/vfonov/registration \
    --db qc_db_20250206_slices.sqlite3 $pfx1 \
    --folds 8 --fold 0 --batch 256 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices \
    --noise 0.1 --lut 0.5 \
    --warmup_lr 1e-10 \
    --n_epochs 100 
fi

if [[ ! -e $pfx2/final_0_8.pth ]];then
mkdir -p $pfx2

python aqc_training.py ${net} --workers 8 \
    --data /fast/vfonov/registration \
    --db qc_db_20250206_slices.sqlite3 $pfx2 \
    --folds 8 --fold 0 --batch 256 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices \
    --noise 0.1 --lut 0.9 \
    --warmup_lr 1e-10 \
    --n_epochs 20 \
    --load $pfx1/best_tnr_0_8.pth
fi


if [[ ! -e $pfx3/final_0_8.pth ]];then
mkdir -p $pfx3

python aqc_training.py \
    ${net} \
    --workers 2 \
    --data /fast/vfonov/registration --db qc_db_20250206_faster.sqlite3  $pfx3 \
    --noise 0.05 --lut 0.0 \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 64 --adamw --warmup_iter 10 --lr 0.00001 \
    --save_final --save_best \
    --load $pfx2/best_tnr_0_8.pth
fi

if [[ ! -e $pfx4/final_0_8.pth ]];then
mkdir -p $pfx4

python aqc_training.py \
    ${net} \
    --workers 2 \
    --data /fast/vfonov/registration --db qc_db_20250206_faster.sqlite3  $pfx4 \
    --noise 0.2 --lut 0.2 \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 64 --adamw --warmup_iter 10 --lr 0.00001 \
    --save_final --save_best \
    --load $pfx3/best_tnr_0_8.pth
fi


if [[ ! -e $pfx5/final_0_8.pth ]];then
mkdir -p $pfx5

python aqc_training.py \
    ${net} \
    --workers 2 \
    --data /fast/vfonov/registration --db qc_db_20250206_faster.sqlite3  $pfx5 \
    --noise 0.1 --lut 0.5 \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 64 --adamw --warmup_iter 10 --lr 0.00001 \
    --nu 4.0 --nl 1.0 --th 5 \
    --save_final --save_best \
    --load $pfx4/best_tnr_0_8.pth
fi
