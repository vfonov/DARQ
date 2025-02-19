#! /bin/bash
#pfx2=mk2/s2/r18/0

pfx1=mk2/s1_small_vol_dist/r18/0

if [[ ! -e $pfx1/final_0_8.pth ]];then
mkdir -p $pfx1

python aqc_training.py \
    --net r18 \
    --workers 2 \
    --data /fast/vfonov/registration/pp_npy_ref \
    --db qc_db_20250217_ref.sqlite3  $pfx1 \
    --distcalc \
    --dist \
    --patch 192 \
    --noise 0.1 --lut 0.0  \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 128 --adamw --warmup_iter 10 --lr 0.00001 \
    --save_final --save_best 
fi

