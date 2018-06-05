# VERSION 0.3
# AUTHOR:         DSSTNE Docker <dsstne-docker@amazon.com>
# DESCRIPTION:    Docker image for Amazon DSSTNE

FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# Suppress interactive prompts while installing base packages
ENV DEBIAN_FRONTEND=noninteractive

# Add repositories and install base packages
RUN apt-get update && \
    apt-get install -y build-essential libcppunit-dev libatlas-base-dev pkg-config python \
        software-properties-common unzip wget && \
    apt-get update && \
    apt-get clean

# Install a recent CMake
RUN wget https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.sh -O /tmp/cmake-installer.sh && \
    sh /tmp/cmake-installer.sh --prefix=/usr/local --exclude-subdir

# Install OpenMPI
RUN apt-get install -y libopenmpi-dev

# Install JSONCPP
RUN apt-get install -y libjsoncpp-dev

# Install hdf5
RUN apt-get install -y libhdf5-dev

# Install zlib
RUN apt-get install -y zlib1g-dev

# Install netcdf
RUN apt-get install -y libnetcdf-dev

# Install netcdf-c++
RUN apt-get install -y libnetcdf-c++4-dev

# Installing CUBG
RUN cd /tmp && \
    wget https://github.com/NVlabs/cub/archive/1.5.2.zip && \
    unzip 1.5.2.zip && \
    cp -rf cub-1.5.2/cub/ /usr/local/include/ && \
    rm -rf /tmp/*

# Ensure OpenMPI is available on path
ENV PATH=/usr/local/openmpi/bin/:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/openmpi/lib/:${LD_LIBRARY_PATH}

# Build latest version of DSSTNE from source
COPY . /opt/amazon/dsstne
RUN cd /opt/amazon/dsstne && mkdir build && cd build && cmake .. -DCMAKE_CXX_COMPILER=mpiCC && cmake --build . --target install

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
