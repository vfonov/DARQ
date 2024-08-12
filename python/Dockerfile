FROM mambaorg/micromamba:1.4.9
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER  aqc_apply.py  aqc_convert_to_cpu.py  aqc_data.py  aqc_pic.py /DARQ/python/
COPY --chown=$MAMBA_USER:$MAMBA_USER  model/__init__.py  model/resnet_qc.py  model/squezenet_qc.py  model/util.py /DARQ/python/model/
COPY --chown=$MAMBA_USER:$MAMBA_USER  cls/model_r18_ref/best_tnr.pth /DARQ/python/cls/model_r18_ref/best_tnr.pth
COPY --chown=$MAMBA_USER:$MAMBA_USER  data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg /DARQ/data/

# RUN /opt/conda/bin/curl https://raw.githubusercontent.com/vfonov/DARQ/main/data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg --output /DARQ/data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg && \
#     /opt/conda/bin/curl https://raw.githubusercontent.com/vfonov/DARQ/main/data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg --output /DARQ/data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg && \
#     /opt/conda/bin/curl https://raw.githubusercontent.com/vfonov/DARQ/main/data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg --output /DARQ/data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg
