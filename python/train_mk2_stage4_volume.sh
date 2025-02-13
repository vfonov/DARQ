#! /bin/bash
pfx2=mk2/s2/r18/0

pfx3=mk2/s3/r18/0
pfx4=mk2/s4/r18/0

if [[ ! -e $pfx4/final_0_8.pth ]];then
mkdir -p $pfx4

python aqc_training.py \
    --net r18 \
    --workers 2 \
    --data /fast/vfonov/registration --db qc_db_20250206_faster.sqlite3  $pfx4 \
    --noise 0.2 --lut 0.2 \
    --n_epochs 4 \
    --folds 8 --fold 0 --batch 128 --adamw --warmup_iter 10 --lr 0.00001 \
    --save_final --save_best \
    --load $pfx3/final_0_8.pth
fi

