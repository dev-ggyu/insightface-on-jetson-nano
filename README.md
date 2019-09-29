# insightface-on-jetson-nano
insightface on jetson nano

# prepare environment
## 1 apt
sudo apt update
sudo apt install -y  git cmake python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev
sudo apt install clang lldb lld
sudo apt install -y build-essential libatlas-base-dev
sudo apt install gfortran
sudo apt install libopencv-dev graphviz
sudo apt install python3-pip libhdf5-serial-dev hdf5-tools

## 2 tvm
git clone --recursive https://github.com/dmlc/tvm.git
cd tvm
mkdir build
cp cmake/config.cmake build
vi build/config.cmake
  ->enable cuda
  ->enable cudnn
  ->enable llvm

cd build
cmake ..
make -j4

cd ..
cd python
python setup.py install
cd ..

cd nnvm
python setup.py install
cd ..

## 3. coral
> **note:** This step is not required.
You need to install it to run some models on Edge TPU.

**official:** https://coral.withgoogle.com/docs/accelerator/get-started/
**summary:**
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std


mkdir coral && cd coral
git clone https://github.com/google-coral/tflite.git

cd tflite/python/examples
bash download.sh


## 3. framework 
**tensorflow**
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3

**mxnet**
wget https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.0/mxnet-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install mxnet-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install cython

## 4. environment
**set swap file**
git clone https://github.com/JetsonHacksNano/installSwapfile
cd installSwapfile
#usage: installSwapFile [[[-d directory ] [-s size] -a] | [-h]]
 ./installSwapfile   #default 6GB


**install virtual enviroment**
pip install virtualenv virtualenvwrapper
echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.projectenv
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.projectenv
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.projectenv
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.projectenv
source ~/.projectenv


**make virtual enviroment**
mkvirtualenv venv_insightface



# how to run

**compile and deploy model with TVM**
> **note:** you have to run this step.
> 
https://docs.tvm.ai/tutorials/relay_quick_start.html
https://docs.tvm.ai/tutorials/index.html

**make face feature**
> **note:** you have to run this step.

mkdir -p images/facename1
mkdir -p images/facename2
cp face1.jpg images/facename1
cp face2.jpg images/facename2
soure setenv.sh
python prerun/test_learn_known_face_mtcnn.py

**run face recognition**
soure setenv.sh
python apps/test_mtcnn_mp.py
