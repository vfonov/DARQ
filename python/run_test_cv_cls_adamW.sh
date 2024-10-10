#! /bin/bash

FOLDS=8
lr=0.0001
wd=0.01
pfx=cls_aug/adamw/lr_${lr}_wd_${wd}_pre
mkdir -p $pfx

for m in r18,20,500 ;do # r34,10,128 r50,10,80 r101,10,48 r152,10,32 
 i=( ${m//,/ } )
 for ref in Y ;do # N
   if [[ $ref == Y ]];then 
      suff='_ref'
      param='--ref'
   else
      suff=''
      param=''
   fi
   for f in $(seq 0 $((${FOLDS}-1)) );do
    if [[ ! -e $pfx/model_${i[0]}${suff}/log_${f}_${FOLDS}.json ]];then
        python aqc_training.py \
            --db qc_db_20230820.sqlite3 \
            --png \
            --lr $lr \
            --weight_decay $wd \
            --warmup_iter 100 \
            --clip 1.0 \
            --l2 0.0 \
            --balance \
            --adamw \
            --augment 0.01 \
            $param --fold $f --pretrained \
            --folds $FOLDS --net ${i[0]} \
            --n_epochs ${i[1]} --batch_size ${i[2]}  \
            $pfx/model_${i[0]}${suff} 2>&1 |tee $pfx/log${suff}_${i[0]}_${f}_${FOLDS}.txt
    fi
   done
 done
done
