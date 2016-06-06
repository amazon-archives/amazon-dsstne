# Setup
DSSTNE requires a GPU to run. You can setup and run DSSTNE in several different ways:

* [Setting up with Docker](#setup-on-docker)
* [Setting up on AWS](#setup-on-aws)
* [Setting up on a dev machine](#setup-on-a-dev-machine)

## Setup on Docker
[Docker](http://docker.com/) helps to containerize your installation without affecting any builds locally. The [Dockerfile](../../Dockerfile) is provided for DSSTNE and you can build a docker out of that.

### Matching GPU driver versions
**The NVIDIA driver version in your Docker image must exactly match the version installed on the host**
The `Dockerfile` uses CUDA 7.0 with driver version 346.72. You can verify the installed version on your
machine by running the `nvidia-smi` tool.

```bash
nvidia-smi | grep Version
| NVIDIA-SMI 346.72     Driver Version: 346.72
```

If your machine does not have 346.72 installed, then you must update the `Dockerfile` to install the
same driver that your host machine has.
Note that we have only fully tested installation with CUDA 7.0.28 and driver 346.72 on the host system.

### Loading the nVidia driver on the host 
We have to make sure the nVidia kernel driver is loaded in the host before running the docker for the first time. In the AWS AMI run the following command 
```bash
cd NVIDIA_CUDA-7.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

### Creating the Docker image
 Download the code.
```bash
git clone https://github.com/amznlabs/amazon-dsstne.git
```
Then build the image locally. Make sure there are at least 10GB free on the root of your [docker runtime](https://docs.docker.com/engine/reference/commandline/daemon/).
```bash
cd amazon-dsstne/
docker build -t amazon/dsstne .
```

Once the Docker Image has been created ensure that you start the docker with *privileged* mode so that you can access the GPU drivers
```bash
docker run -it --privileged amazon/dsstne /bin/bash
```

## Setup on AWS
You can also launch a [GPU based AWS instance](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html) from *ami-d6f2e6bc* which has all the prerequisites build in properly. The image is currently available in us-east-1 region (N.Virginia in AWS Console). Launch a GPU based instance (*g2.2xlarge*,*g2.8xlarge*) from the AMI. [Download the code and build](#download-the-code-and-build)

## Setup on a dev machine
Instructions are provided for installation on Ubuntu Linux machines.

### Prerequisites
* [Setup GCC](#gcc-setup) : GCC compiler with C++11 is required.
* [Setup CuBLAS](#cublas-setup) : Blas Libraries
* Cuda Toolkit >= 7.0 is required
* [Setup OpenMPI](#openmpi-setup) : CUDA aware OpenMPI is required.
* [Setup NetCDF](#netcdf-setup) : NetCDF is the native format which DSSTNE engine supports
* [Setup JsonCPP](#jsoncpp-setup) : Configurations are parsed through Json
* [Setup CUB](#cub-setup) : Dependent CUDA libraries which is required by DSSTNE

#### GCC Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get -y update
sudo apt-get install gcc
sudo apt-get install g++

#Fedora 64-bit
sudo dnf check-update
sudo dnf install gcc
sudo dnf install gcc-c++
```
#### Cublas Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get install -y libatlas-base-dev

#Fedora 64-bit
sudo dnf install atlas-devel
```

#### OpenMPI Setup
MPI is used across in DSTTNE to allow multi GPU modeling. OpenMPI package is used as the MPI Platform.

```bash
# Ubuntu/Linux 64-bit
sudo apt-get -y install make
wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.2.tar.gz
tar xvfz openmpi-1.8.2.tar.gz
cd openmpi-1.8.2
./configure --prefix=/usr/local/openmpi
make
sudo make install
```

#### NetCDF Setup
NetCDF is the format which is supported inherently from DSSTNE engine. It is required to install:
* [Setup Hdf5](#hdf5-setup)
* [Setup Zlib](#zlib-setup)
* [Setup NetCDF](#netcdf-setup-1)
* [Setup NetCDFC++](#netcdfc-setup)

#### HDF5 Setup
```bash
# Ubuntu/Linux 64-bit
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.9.tar.gz
tar xvfz hdf5-1.8.9.tar.gz
cd hdf5-1.8.9
./configure --prefix=/usr/local
make
sudo make install
```

#### Zlib Setup
```bash
# Ubuntu/Linux 64-bit
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz
tar xvf zlib-1.2.8.tar.gz
cd zlib-1.2.8
./configure
make
sudo make install
```
#### Netcdf Setup
```bash
# Ubuntu/Linux 64-bit
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.1.3.tar.gz
tar xvf netcdf-4.1.3.tar.gz
cd netcdf-4.1.3
./configure --prefix=/usr/local
make
sudo make install
```
#### Netcdfc++ Setup
```bash
# Ubuntu/Linux 64-bit
wget http://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz
tar xvf netcdf-cxx4-4.2.tar.gz
cd netcdf-cxx4-4.2
./configure --prefix=/usr/local
make
sudo make install
```

#### JsonCPP Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get -y install cmake
wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz
tar xvfz svn-import.tar.gz
cd jsoncpp-svn-import
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=release -DJSONCPP_LIB_BUILD_SHARED=OFF -G "Unix Makefiles" ../..
make
sudo make install
```

#### CUB Setup
```bash
# Ubuntu/Linux 64-bit
wget https://github.com/NVlabs/cub/archive/1.5.2.zip
sudo apt-get install -y unzip
unzip 1.5.2.zip
sudo cp -rf cub-1.5.2/cub/ /usr/local/include/
```

### Download the code and build.
```bash
# Ubuntu/Linux 64-bit
git clone https://github.com/amznlabs/amazon-dsstne.git
cd amazon-dsstne/src/amazon/dsstne
#Add the mpiCC and nvcc compiler in the path
export PATH=/usr/local/openmpi/bin:/usr/local/cuda/bin:$PATH
make
export PATH=`pwd`/bin:$PATH
```

Try running some [examples](examples.md).
