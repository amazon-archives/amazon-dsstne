# VERSION 0.2
# AUTHOR:         DSSTNE Docker <dsstne-docker@amazon.com>
# DESCRIPTION:    Docker image for Amazon DSSTNE

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

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
    wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz && \
    tar xvfz openmpi-2.1.1.tar.gz && \
    cd openmpi-2.1.1 && \
    ./configure CC=gcc CXX=g++ --enable-mpi-cxx --prefix=/usr/local/openmpi && \
    make -j 8 && \
    sudo make install && rm -rf /tmp/*

# Install JSONCPP
RUN cd /tmp  && \
    wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz && \
    tar xvfz svn-import.tar.gz && \
    cd jsoncpp-svn-import && \
    mkdir -p build/release && \
    cd build/release && \
    cmake -DCMAKE_BUILD_TYPE=release -DJSONCPP_LIB_BUILD_SHARED=OFF -G "Unix Makefiles" ../.. && \
    make -j 8 && \
    make install && rm -rf /tmp/*

# Install hdf5
RUN cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.12.tar.gz && \
    tar xvfz hdf5-1.8.12.tar.gz && \
    cd hdf5-1.8.12 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

# Install zlib
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4/zlib-1.2.8.tar.gz && \
    tar xvf zlib-1.2.8.tar.gz && \
    cd zlib-1.2.8 && \
    ./configure && \
    make -j 8 && \
    make install && rm -rf /tmp/*

# Install netcdf
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.3.tar.gz && \
    tar xvf netcdf-4.1.3.tar.gz && \
    cd netcdf-4.1.3 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

# Install netcdf-cxx
RUN cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz && \
    tar xvf netcdf-cxx4-4.2.tar.gz && \
    cd netcdf-cxx4-4.2 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

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
