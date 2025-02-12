#! /bin/bash


mkdir -p mk2/s1/r18/0

python aqc_training.py --net r18 --workers 8 \
    --data /fast/vfonov/registration \
    --db qc_db_20250206_slices.sqlite3 mk2/s1/r18/0 \
    --folds 8 --fold 0 --batch 1024 \
    --adamw --warmup_iter 10 --lr 0.001 \
    --save_final --save_best --slices \
    --noise 0.1 --lut 0.5 \
    --warmup_lr 1e-10 \
    --n_epochs 100 
