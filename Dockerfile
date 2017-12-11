# VERSION 0.3
# AUTHOR:         DSSTNE Docker <dsstne-docker@amazon.com>
# DESCRIPTION:    Docker image for Amazon DSSTNE

FROM nvidia/cuda:9.0-cudnn6-devel-ubuntu14.04

# Suppress interactive prompts while installing base packages
ENV DEBIAN_FRONTEND=noninteractive

# Add repositories and install base packages
RUN apt-get update && \
    apt-get install -y build-essential libcppunit-dev libatlas-base-dev pkg-config python \
        software-properties-common unzip wget && \
    add-apt-repository ppa:george-edison55/cmake-3.x && \
    apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install OpenMPI
RUN apt-get install libopenmpi-dev

# Install JSONCPP
RUN apt-get install libjsoncpp-dev

# Install hdf5
RUN cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.12.tar.gz && \
    tar xvfz hdf5-1.8.12.tar.gz && \
    cd hdf5-1.8.12 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

# Install zlib
RUN apt-get install zlib

# Install netcdf
RUN apt-get install libnetcdf-dev

# Install netcdf-c++
RUN apt-get install libnetcdfc++4-dev

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
RUN cd /opt/amazon/dsstne/src/amazon/dsstne && \
    make install

# Add DSSTNE binaries to PATH
ENV PATH=/opt/amazon/dsstne/src/amazon/dsstne/bin/:${PATH}
