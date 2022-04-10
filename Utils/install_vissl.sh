#!/bin/bash   
cd $HOME
git clone --recursive https://github.com/facebookresearch/vissl.git

cd vissl/

git checkout v0.1.6
git checkout -b v0.1.6

# install vissl dependencies
pip install --progress-bar off -r requirements.txt
pip install opencv-python

# update classy vision install to commit compatible with v0.1.6
pip uninstall -y classy_vision
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d

# install vissl dev mode (e stands for editable)
pip install -e .[dev]
