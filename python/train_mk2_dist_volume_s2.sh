#! /bin/bash
#pfx2=mk2/s2/r18/0

prev=mk2/s1_small_dist/r18/0/best_loss_0_8.pth

pfx1=mk2/s1_small_vol_dist/r18/0
pfx2=mk2/s2_small_vol_dist/r18/0

if [[ ! -e $pfx2/final_0_8.pth ]];then
mkdir -p $pfx2

python aqc_training.py \
    --net r18 \
    --workers 2 \
    --data /fast/vfonov/registration/pp_npy_ref \
    --load $pfx1/final_0_8.pth \
    --db qc_db_20250217_ref.sqlite3  $pfx2 \
    --distcalc \
    --dist \
    --patch 192 \
    --noise 0.1 --lut 0.1  \
    --n_epochs 100 \
    --folds 8 --fold 0 --batch 160 \
    --adamw --warmup_iter 4 --lr 0.01 --warmup_lr 1e-10 \
    --save_final --save_best 
fi
