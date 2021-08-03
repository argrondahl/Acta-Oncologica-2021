Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu
Stage: build

%post
    apt update -y
    apt upgrade -y
    pip install ipython
    pip install tensorflow-addons
    pip install https://github.com/huynhngoc/deoxys-image/archive/master.zip
    pip install https://github.com/huynhngoc/deoxys/archive/master.zip
    pip install opencv-python-headless
    pip install comet-ml
    pip install scikit-image
    pip install scikit-learn
    pip install mypy
    pip install nptyping

%environment
    export KERAS_MODE=TENSORFLOW