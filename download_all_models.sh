#! /bin/bash
set -e 
prefix="https://github.com/vfonov/DARQ/releases/download/v0.1/"

echo "Downloading all pretrained models"

function download {
    set -e 
    for f in $@;do
        if [ ! -e ${f} ];then
        curl --location "${prefix}/${f}" -o ${f}
        fi
    done
}

download DARQ_models_all.tar.xz

echo "Unpacking..."

tar xJf DARQ_models_all.tar.xz
