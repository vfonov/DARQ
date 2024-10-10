#! /bin/bash

pfx=cls_pp2
lr=0.0001
wd=0.01

mkdir -p $pfx
#set -e

# train with reference
for m in r18,20,500 \
         r34,10,300 \
         r50,10,200 \
         r101,10,100 \
         r152,10,70 \
;do
 i=( ${m//,/ } )
 for ref in Y N;do # N
    if [[ $ref == Y ]];then
      suff='_ref'
      param='--ref'
    else
      suff=''
      param=''
    fi

    out=$pfx/model_${i[0]}${suff}
    if [ ! -e $out/final.pth ];then
    mkdir -p $out
    python aqc_training.py \
        --data ../data \
        --db qc_db_20230820.sqlite3 \
        --png \
        --lr $lr \
        --weight_decay $wd \
        --warmup_iter 100 \
        --clip 1.0 \
        --l2 0.0 \
        --adamw \
        --augment 0.01 \
        $param --pretrained \
        --balance \
        --adamw \
        --fold 0 --folds 0 --net ${i[0]} \
        --n_epochs ${i[1]} --batch_size ${i[2]}  \
        --save_final --save_best --save_cpu \
        $out 2>&1 |tee $out/log_${suff}_${i[0]}.txt
    fi
  done
done
