#! /bin/bash

pfx1=mk2/s1_dist/r18/0

if [[ ! -e $pfx1/final_0_8.pth ]];then
mkdir -p $pfx1

python aqc_training.py --net r18 --workers 8 \
    --data /fast/vfonov/registration \
    --db qc_db_20250206_slices.sqlite3 $pfx1 \
    --folds 8 --fold 0 --batch 1024 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices \
    --noise 0.2 --lut 0.2 \
    --warmup_lr 1e-10 \
    --n_epochs 100 --dist
fi


pfx2=mk2/s2_dist/r18/0

if [[ ! -e $pfx2/final_0_8.pth ]];then
mkdir -p $pfx2

python aqc_training.py --net r18 --workers 8 \
    --data /fast/vfonov/registration \
    --db qc_db_20250206_slices.sqlite3 $pfx2 \
    --folds 8 --fold 0 --batch 1024 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices \
    --noise 0.2 --lut 0.9 \
    --warmup_lr 1e-10 \
    --n_epochs 100 --dist \
    --load $pfx1/final_0_8.pth
fi
