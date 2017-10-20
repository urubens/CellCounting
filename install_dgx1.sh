#!/bin/bash

cd /cellcounting
apt-get update
apt-get install -y libgeos-dev expect
apt-get install -y language-pack-en
pip install Pillow
mkdir Cytomine/
cd Cytomine/
git clone https://github.com/cytomine/Cytomine-python-client.git
cd Cytomine-python-client/
cd client/
python setup.py build
python setup.py install
cd ../../Cytomine
git clone https://github.com/waliens/sldc.git
cd sldc
python setup.py build
python setup.py install
cd /cellcounting
pip install .
