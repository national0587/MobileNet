FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
  && apt-get install -y  python3-dev python3-pip \
  && pip3 install setuptools && pip3 install --upgrade pip \
  && apt-get install -y git m4 autoconf automake libtool flex \
  && apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev \
  && apt-get install -y ssh \
  && apt-get clean
RUN git clone --depth=1 -b v3.0.0 https://github.com/open-mpi/ompi.git /tmp/ompi \
  && cd /tmp/ompi \
  && ./autogen.pl \
  && ./configure --with-cuda \
  && make -j4 \
  && make install \
  && rm -rf /tmp/ompi
RUN pip3 install matplotlib
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV CPATH=$CUDA_PATH/include:$CPATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH

RUN pip3 install cupy==3.0.0a1 chainer==4.0.0a1
RUN pip3 install cython && pip3 install chainermn
RUN pip3 install  pillow
