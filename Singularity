Bootstrap: docker
From: nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

%environment
    DEBIAN_FRONTEND=noninteractive
    LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/openmpi/lib/
    PATH=/amazon-dsstne/src/amazon/dsstne/bin/:/usr/local/openmpi/bin/:${PATH}
    export DEBIAN_FRONTEND LD_LIBRARY_PATH PATH

%setup
    echo $SINGULARITY_ROOTFS
    echo "cp -r . $SINGULARITY_ROOTFS/"
    cp Singularity $SINGULARITY_ROOTFS/
    cp -r ./amazon-dsstne/ $SINGULARITY_ROOTFS/

%post
    apt-get update && \
    apt-get install -y build-essential libcppunit-dev libatlas-base-dev pkg-config python \
        software-properties-common unzip wget && \
    add-apt-repository ppa:george-edison55/cmake-3.x && \
    apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

    cd /tmp  &&  \
    wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz && \
    tar xvfz openmpi-2.1.1.tar.gz && \
    cd openmpi-2.1.1 && \
    ./configure CC=gcc CXX=g++ --enable-mpi-cxx --prefix=/usr/local/openmpi && \
    make -j 8 && \
    sudo make install && rm -rf /tmp/*

    cd /tmp  && \
    wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz && \
    tar xvfz svn-import.tar.gz && \
    cd jsoncpp-svn-import && \
    mkdir -p build/release && \
    cd build/release && \
    cmake -DCMAKE_BUILD_TYPE=release -DJSONCPP_LIB_BUILD_SHARED=OFF -G "Unix Makefiles" ../.. && \
    make -j 8 && \
    make install && rm -rf /tmp/*

    cd /tmp && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.9.tar.gz && \
    tar xvfz hdf5-1.8.9.tar.gz && \
    cd hdf5-1.8.9 && \
    ./configure --prefix=/usr/local &&\
    make -j 8 && \
    make install && rm -rf /tmp/*

    cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4/zlib-1.2.8.tar.gz && \
    tar xvf zlib-1.2.8.tar.gz && \
    cd zlib-1.2.8 && \
    ./configure && \
    make -j 8 && \
    make install && rm -rf /tmp/*

    cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.3.tar.gz && \
    tar xvf netcdf-4.1.3.tar.gz && \
    cd netcdf-4.1.3 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

    cd /tmp && \
    wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz && \
    tar xvf netcdf-cxx4-4.2.tar.gz && \
    cd netcdf-cxx4-4.2 && \
    ./configure --prefix=/usr/local && \
    make -j 8 && \
    make install && rm -rf /tmp/*

    cd /tmp && \
    wget https://github.com/NVlabs/cub/archive/1.5.2.zip && \
    unzip 1.5.2.zip && \
    cp -rf cub-1.5.2/cub/ /usr/local/include/ && \
    rm -rf /tmp/*

    export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/openmpi/lib/:${LD_LIBRARY_PATH}
    export PATH=/amazon-dsstne/src/amazon/dsstne/bin:/usr/local/openmpi/bin:/usr/local/bin:/usr/local/include/bin:/usr/local/cuda-7.5/bin:${PATH}
    cd /amazon-dsstne/src/amazon/dsstne && \
    make install
