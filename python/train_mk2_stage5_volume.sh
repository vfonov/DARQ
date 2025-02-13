#! /bin/bash
pfx2=mk2/s2/r18/0

pfx3=mk2/s3/r18/0
pfx4=mk2/s4/r18/0
pfx5=mk2/s5/r18/0

if [[ ! -e $pfx5/final_0_8.pth ]];then
mkdir -p $pfx5

python aqc_training.py \
    --net r18 \
    --workers 2 \
    --data /fast/vfonov/registration --db qc_db_20250206_faster.sqlite3  $pfx5 \
    --noise 0.1 --lut 0.5 \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 128 --adamw --warmup_iter 10 --lr 0.00001 \
    --nu 4.0 --nl 1.0 --th 5 \
    --save_final --save_best \
    --load $pfx4/final_0_8.pth
fi

