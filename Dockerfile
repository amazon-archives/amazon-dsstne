# VERSION 0.1
# AUTHOR:         DSSTNE Docker <dsstne-docker@amazon.com>
# DESCRIPTION:    Docker image for Amazon DSSTNE
FROM ubuntu:14.04
RUN apt-get update && apt-get install -y make \
   build-essential \
   cmake \
   g++  \
   gcc \
   libatlas-base-dev \
   wget

ENV CUDA_MAJOR=7.0 \
  CUDA_VERSION=7.0.28 \
  CUDA_MAJOR_U=7_0 \
  GPU_DRIVER_VERSION=346.72

# Change to the /tmp directory
RUN cd /tmp && \
# Download run file
  wget http://developer.download.nvidia.com/compute/cuda/${CUDA_MAJOR_U}/Prod/local_installers/cuda_${CUDA_VERSION}_linux.run && \
# Make the run file executable and extract
  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd`

# Install the specific driver.  (7.0.28 ships with 346.46)
RUN cd /tmp && \
  wget http://us.download.nvidia.com/XFree86/Linux-x86_64/$GPU_DRIVER_VERSION/NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run && \
  chmod +x ./NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run && \
# Install CUDA drivers (silent, no kernel)
  ./NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run -s --no-kernel-module && \
# Install toolkit (silent)
  ./cuda-linux64-rel-*.run -noprompt && \
  rm -rf /tmp/*

# Add to path
ENV PATH=/usr/local/cuda/bin:${PATH}
# Configure dynamic link
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig

# Install OpenMPI
RUN cd /tmp  &&  \
   wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.2.tar.gz && \
   tar xvfz openmpi-1.8.2.tar.gz && \
   cd openmpi-1.8.2 && \
   ./configure --prefix=/usr/local/openmpi  && \
   make && \
   sudo make install && rm -rf /tmp/*

# "Installing JSONCPP"
RUN cd /tmp  && \
    apt-get install -y python && \
    wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz && \
    tar xvfz svn-import.tar.gz &&\
    cd jsoncpp-svn-import &&\
    mkdir -p build/release &&\
    cd build/release &&\
    cmake -DCMAKE_BUILD_TYPE=release -DJSONCPP_LIB_BUILD_SHARED=OFF -G "Unix Makefiles" ../.. && \
    make &&\
    make install && rm -rf /tmp/*

#Installing hdf5
RUN cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.12.tar.gz && \
    tar xvfz hdf5-1.8.12.tar.gz && \
    cd hdf5-1.8.12 && \
    ./configure --prefix=/usr/local &&\
    make && \
    make install && rm -rf /tmp/*

# Installing zlib
RUN cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz &&\
    tar xvf zlib-1.2.8.tar.gz &&\
    cd zlib-1.2.8 &&\
    ./configure && \
    make && \
    make install && rm -rf /tmp/*

#"Installing netcdf"
RUN cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.1.3.tar.gz &&\
    tar xvf netcdf-4.1.3.tar.gz &&\
    cd netcdf-4.1.3 &&\
    ./configure --prefix=/usr/local &&\
    make &&\
    make install && rm -rf /tmp/*

# "Installing netcdf-cxx"
RUN cd /tmp && \
    wget http://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz &&\
    tar xvf netcdf-cxx4-4.2.tar.gz &&\
    cd netcdf-cxx4-4.2 &&\
    ./configure --prefix=/usr/local &&\
    make && \
    make install && rm -rf /tmp/*

#Installing CUBG
RUN cd /tmp && \
    wget https://github.com/NVlabs/cub/archive/1.5.2.zip &&\
    apt-get install -y  unzip &&\
    unzip 1.5.2.zip &&\
    cp -rf cub-1.5.2/cub/ /usr/local/include/ &&\
    rm -rf /tmp/*

ENV PATH=/usr/local/openmpi/bin/:/usr/local/cuda-7.0/bin/:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}

COPY src /opt/amazon/dsstne/src

RUN cd /opt/amazon/dsstne/src/amazon/dsstne && \
    make

ENV PATH /opt/amazon/dsstne/src/amazon/dsstne/bin:${PATH}
