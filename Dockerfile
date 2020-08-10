FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# install python 3.7
# https://tecadmin.net/install-python-3-7-on-ubuntu-linuxmint/
RUN apt-get update && apt-get install -y build-essential checkinstall
RUN apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev wget libffi-dev
RUN cd /usr/src && wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && tar xzf Python-3.7.0.tgz && cd Python-3.7.0 && ./configure --enable-optimizations && make install

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext-dev # for opencv

# install python packages
RUN pip3 install matplotlib>=3.0.0 tqdm>=4.26.0 tensorboardX>=1.4 torchsummary>=1.5.1
RUN pip3 install numpy>=1.15.3 opencv-python==3.4.3.18 Pillow==5.3.0
RUN pip3 install http://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install torchvision==0.2.1 torchfile==0.1.0 torchnet
