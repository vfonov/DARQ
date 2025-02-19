#! /bin/bash

pfx1=mk2/s1_small_dist/r18/0

if [[ ! -e $pfx1/final_0_8.pth ]];then
mkdir -p $pfx1

python aqc_training.py --net r18 --workers 8 \
    --data /fast/vfonov/registration/pp_npy_smaller_slices \
    --db qc_db_20250217_smaller_slices.sqlite3 $pfx1 \
    --folds 8 --fold 0 --batch 1024 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices --patch 192 --dist \
    --noise 0.1 --lut 0.5 \
    --warmup_lr 1e-10 \
    --n_epochs 100 
fi


