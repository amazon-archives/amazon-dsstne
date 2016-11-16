# VERSION 0.2
# AUTHOR:         DSSTNE Docker <dsstne-docker@amazon.com>
# DESCRIPTION:    Docker image for Amazon DSSTNE

FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

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
RUN cd /tmp  &&  \
    wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.2.tar.gz && \
    tar xvfz openmpi-1.8.2.tar.gz && \
    cd openmpi-1.8.2 && \
    ./configure --prefix=/usr/local/openmpi && \
    make && \
    sudo make install && rm -rf /tmp/*

# Install JSONCPP
RUN cd /tmp  && \
    wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz && \
    tar xvfz svn-import.tar.gz && \
    cd jsoncpp-svn-import && \
    mkdir -p build/release && \
    cd build/release && \
    cmake -DCMAKE_BUILD_TYPE=release -DJSONCPP_LIB_BUILD_SHARED=OFF -G "Unix Makefiles" ../.. && \
    make && \
    make install && rm -rf /tmp/*

# Install hdf5
RUN cd /tmp && \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz && \
    tar xvfz hdf5-1.8.12.tar.gz && \
    cd hdf5-1.8.12 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && rm -rf /tmp/*

# Install zlib
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4/zlib-1.2.8.tar.gz && \
    tar xvf zlib-1.2.8.tar.gz && \
    cd zlib-1.2.8 && \
    ./configure && \
    make && \
    make install && rm -rf /tmp/*

# Install netcdf
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.3.tar.gz && \
    tar xvf netcdf-4.1.3.tar.gz && \
    cd netcdf-4.1.3 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && rm -rf /tmp/*

# Install netcdf-cxx
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz && \
    tar xvf netcdf-cxx4-4.2.tar.gz && \
    cd netcdf-cxx4-4.2 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && rm -rf /tmp/*

# Installing CUBG
RUN cd /tmp && \
    wget https://github.com/NVlabs/cub/archive/1.5.2.zip && \
    unzip 1.5.2.zip && \
    cp -rf cub-1.5.2/cub/ /usr/local/include/ && \
    rm -rf /tmp/*

# Ensure OpenMPI is avaiable on path
ENV PATH=/usr/local/openmpi/bin/:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}

# Build latest version of DSSTNE from source
COPY . /opt/amazon/dsstne
RUN cd /opt/amazon/dsstne/src/amazon/dsstne && \
    make install

# Add DSSTNE binaries to PATH
ENV PATH=/opt/amazon/dsstne/src/amazon/dsstne/bin/:${PATH}

