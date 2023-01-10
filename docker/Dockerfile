FROM ubuntu:20.04 as system

# Set working directory
RUN mkdir /vsss_ws
WORKDIR /vsss_ws

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential \
    g++ \
    cmake \
    git \
    qt5-default \
    libqt5opengl5-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libode-dev \ 
    libboost-dev \
    sudo && \
    apt-get clean && \
    cd /tmp && \
    git clone https://github.com/jpfeltracco/vartypes.git &&\
    cd vartypes && \
    mkdir build && cd build && \
    cmake .. && make && sudo make install

FROM system as simulator

# Install FIRASim
RUN cd /vsss_ws && \
    git clone https://github.com/VSSSLeague/FIRASim.git && \
    cd FIRASim && \
    git checkout tags/v3.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

FROM simulator as display

# Set enviroment variables
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

RUN mkdir -m 700 /tmp/runtime-root
ENV XDG_RUNTIME_DIR=/tmp/runtime-root


FROM display as python

RUN apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt install python3.10 -y && \
    apt install python3-pip -y && \
    pip3 install --upgrade pip


FROM python as files

COPY . /vsss_ws/unballia

# RUN pip3 install -r /vsss_ws/unballia/requirements.txt


# CMD /vsss_ws/FIRASim/bin/FIRASim -H & python3.10 /vsss_ws/unballia/run_training.py
CMD /vsss_ws/FIRASim/bin/FIRASim