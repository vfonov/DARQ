#! /bin/bash
#pfx2=mk2/s2/r18/0

prev=mk2/s1_small_dist/r18/0/best_loss_0_8.pth

pfx1=mk2/s1_small_vol_dist/r18/0
pfx2=mk2/s2_small_vol_dist/r18/0
pfx3=mk2/s3_small_vol_dist/r18/0
pfx4=mk2/s4_small_vol_dist/r18/0
pfx5=mk2/s5_small_vol_dist/r18/0

if [[ ! -e $pfx5/final_0_8.pth ]];then
mkdir -p $pfx5


## finetune!
python aqc_training.py \
    --net r18 \
    --workers 2 \
    --data /fast/vfonov/registration/pp_npy_ref \
    --load $pfx4/best_loss_0_8.pth \
    --db qc_db_20250217_ref.sqlite3  \
    $pfx5 \
    --distcalc \
    --dist \
    --patch 192 \
    --noise 0.1 --lut 0.1  --nu 0.5 \
    --lin 0.1 \
    --n_epochs 100 \
    --folds 8 --fold 0 --batch 160 \
    --warmup_iter 4 --lr 1.0 --clip 4.0 --warmup_lr 1e-10 \
    --save_final --save_best
fi
