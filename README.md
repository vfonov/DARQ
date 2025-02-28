# DEEP QC

Code for the paper Vladimir S. Fonov, Mahsa Dadar, The PREVENT-AD Research Group, D. Louis Collins **"DARQ: Deep learning of quality control for stereotaxic registration of human brain MRI"** https://doi.org/10.1016/j.neuroimage.2022.119266

*Updated version of the previosly available ["Deep learning of quality control for stereotaxic registration of human brain MRI"](https://doi.org/10.1101/303487)*

## Installation (Python version) using *conda* for inference

* CPU version

    ```{shell}
    conda install pytorch-cpu==1.7.1 torchvision==0.8.2 cpuonly -c pytorch 
    conda install scikit-image
    ```

* GPU version

    ```{shell}
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=<your cuda version>  -c pytorch 
    conda install scikit-image
    ```

* (optional) minc toolkit and minc2-simple

   ```{shell}
   conda install -c vfonov minc-toolkit-v2 minc2-simple
   ```

* (optional) nibabel (for analyzing Freesurfer output)

   ```{shell}
   pip install nibabel
   ```

## Running

### Inference

* Inference using jpeg files generated by `minc_aqc.pl` script: `python3 python/aqc_apply.py --net <network> --image <image base> `
* Inference using minc files in stereotaxic space: `python3 python/aqc_apply.py --net <network> --volume <input.mnc> `
* Inference using Freesurfer output: `python3 python/aqc_apply.py  --net <network> --freesurfer <freesurfer subject directory> `

### Training

* Training  in `python` directory `run_all_experiments.sh` - will try to train all networks

## Dependencies

* trainig dependencies: `scikit-image tensorboard`,
* for inference directly on minc files `minc2-simple`
* for inference on freeserfer files: `nibabel`
* minc2-simple (optional): https://github.com/vfonov/minc2-simple

## Files

* Shell scripts:
    * `download_minimal_models.sh`  - download QCResNET-18 with reference pretrained model to run automatic qc (43mb)
    * `download_all_models.sh`  - download all pretrained models to run automatic qc 
* Directory `python`:
    * `run_all_experiments.sh` - run experiments with different versions of ResNet and SquezeNet
    * `aqc_apply.py` - apply pre-trained network
    * `aqc_convert_to_cpu.py`- helper script to convert network from GPU to CPU
    * `aqc_data.py` - module to load QC data
    * `aqc_training.py` - deep nearal net training script
    * `model/resnet_qc.py` - module with ResNET implementation, based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    * `model/util.py` - various helper functions
    * `*.R` - R scripts to generete figures for the paper
* Image files:
    * `mni_icbm152_t1_tal_nlin_sym_09c_0.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_1.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_2.jpg` - reference slices, needed for both training and running pretrained model
* `results` - figures for the paper
* `data` - reference images

## Validating correct operation (requires minc-toolkit and minc2_simple python module)

```{shell}
# create a file with 30 degree rotation transform
param2xfm -rotations 30 0 0  rot_30.xfm
# apply to a template:
itk_resample /opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc --transform rot_30.xfm bad.mnc

# run QC script on good scan
# should print "Pass"
python python/aqc_apply.py --net r18 --volume /opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc

# now on "bad"
# should print "Fail"
python python/aqc_apply.py --net r18 --volume bad.mnc
```

## Validating correct operation on Freesurfer, requires nibabel

```{shell}
# run recon-all on ernie subject from freesurfer installation
recon-all -s ernie -i $SUBJECTS_DIR/sample-001.mgz -i $SUBJECTS_DIR/sample-002.mgz -autorecon1

# run automated QC on freesurfer output
# should pring "Pass"
python python/aqc_apply.py --net r18 --freesurfer $FREESURFER_HOME/subjects/ernie 

```
