# Setup Guide

DSSTNE requires a GPU to run. This guide covers two different ways that you can setup and run DSSTNE:

* [Setup on AWS using Docker](#setup-on-aws-using-docker)
* [Setup up on your own development machine](#setup-on-your-own-development-machine)

## Setup on AWS using Docker

The first option is to run DSSTNE on AWS, taking advantage of [Docker](http://docker.com/) to containerize your installation. We have provided an AMI and Dockerfile that can be used to build and run DSSTNE on a GPU-based EC2 instance.

### Dockerfile

A [Dockerfile](../../Dockerfile) has been provided as part of the DSSTNE source code, and you can build a Docker image out of that. Our Dockerfile is designed to be used with a Docker plugin called [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), which facilitates communication between a Docker container and the GPU on the host.

When you build this Docker image, all of the dependencies for building DSSTNE will be fetched from third-party sources. By building this image, you are responsible for reading and accepting the relevant software licenses.

### AMI with nvidia-docker

You can launch a [GPU-based EC2 instance](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html) from *ami-fe173884*, which has Docker and nvidia-docker installed. This AMI also includes NVIDIA drivers, and is currently available in the us-east-1 region (N.Virginia in AWS Console).

You can create a new EC2 instance by following the instructions in the [EC2 Launch Instance Wizard](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-25c0eb32) (this link will take you directly to the wizard).

### Login to EC2 instance via SSH

Once you have launched your new GPU-based EC2 instance, you will need to log in as the 'ubuntu' user. Once you have successfully logged in, ensure that you can start a Docker container that can access the GPU drivers on the host. You can do this by running `nvidia-smi` in a new container:

```bash
nvidia-docker run --rm nvidia/cuda nvidia-smi
```

The output should be similar to the following - with no errors or running processes, and indicating that version 367.57 of the NVIDIA GPU Drivers are active:

```
Wed Nov 16 02:11:42 2016
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.57                 Driver Version: 367.57                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |
| N/A   26C    P8    17W / 125W |      0MiB /  4036MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

If that all works, you can proceed to building a DSSTNE Docker image.

### Build the Docker image

To build the latest Docker image, you will need to clone the DSSTNE source code:

```bash
git clone https://github.com/amznlabs/amazon-dsstne.git
```

Then build the image locally:

```bash
cd amazon-dsstne/
docker build -t amazon-dsstne .
```

Note that this will take a while, as it will build DSSTNE's dependencies from source. Feel free to get a cup of coffee while you wait.

### Test the Docker image

```bash
nvidia-docker run --rm -it amazon-dsstne predict
```

The output should look like the following (the missing argument error is expected):

```
Error: Missing required argument: -d: dataset_name is not specified.
Predict: Generates predictions from a trained neural network given a signals/input dataset.
Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_recs>] [-l layer] [-s input_signals_index] [-p score_precision]
    -b batch_size: (default = 1024) the number records/input rows to process in a batch.
    -d dataset_name: (required) name for the dataset within the netcdf file.
    -f samples filterFileName .
    -i input_feature_index: (required) path to the feature index file, used to tranform input signals to correct input feature vector.
    -k num_recs: (default = 100) The number of predictions (sorted by score to generate). Ignored if -l flag is used.
    -l layer: (default = Output) the network layer to use for predictions. If specified, the raw scores for each node in the layer is output in order.
    -n network_file: (required) the trained neural network in NetCDF file.
    -o output_feature_index: (required) path to the feature index file, used to tranform the network output feature vector to appropriate features.
    -p score_precision: (default = 4.3f) precision of the scores in output
    -r input_text_file: (required) path to the file with input signal to use to generate predictions (i.e. recommendations).
    -s filename (required) . to put the output recs to.
```

If this is what you see, you're ready to move on to the [examples]. Note that before running the examples, you should start a shell on a fresh Docker container:

```bash
nvidia-docker run -it amazon-dsstne /bin/bash
```

## Setup on your own development machine
Instructions are provided for installation on Ubuntu 16.04 Linux machines. For prior versions of Ubuntu, usage of Docker is recommended.


### Prerequisites
* [Setup GCC](#gcc-setup) : GCC compiler with C++11 is required.
* [Setup CMake](#cmake-setup) : CMake >= 3.10 is required.
* [Setup CuBLAS](#cublas-setup) : Blas Libraries
* Cuda Toolkit >= 7.0 is required
* [Setup OpenMPI](#openmpi-setup) : CUDA aware OpenMPI is required.
* [Setup NetCDF](#netcdf-setup) : NetCDF is the native format which DSSTNE engine supports
* [Setup JsonCPP](#jsoncpp-setup) : Configurations are parsed through Json
* [Setup CppUnit](#cppunit-setup) : Unit testing framework module for C++ 
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
#### CMake Setup
```bash
# Ubuntu/Linux 64-bit
sudo wget https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.sh -O /tmp/cmake-installer.sh && \
    sh /tmp/cmake-installer.sh --prefix=/usr/local --exclude-subdir
```
#### Bazel Setup (required to build Tensorflow from scratch)
```bash
# Ubuntu/Linux 64-bit
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get install bazel
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
sudo apt-get install libopenmpi-dev
```

#### Build Tensorflow from scratch
Tensorflow is needed because we use some of their APIs to generate Tensorboard-compatible log files.

```bash
git clone https://github.com/FloopCZ/tensorflow_cc.git
mkdir -p tensorflow_cc/build
(cd tensorflow_cc/build && cmake ../tensorflow_cc -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON && sudo make install)
```

#### NetCDF Setup
NetCDF is the format which is supported inherently from DSSTNE engine. It is required to install:
* [Setup Hdf5](#hdf5-setup)
* [Setup Zlib](#zlib-setup)
* [Setup NetCDF](#netcdf-setup-1)
* [Setup NetCDFC++](#netcdfc-setup)

#### HDF5 Setup
```bash
sudo apt-get install libhdf5-dev
```

#### Zlib Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get install zlib1g-dev
```
#### Netcdf Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get install libnetcdf-dev
```
#### Netcdfc++ Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get install libnetcdf-c++4-dev
```

#### JsonCPP Setup
```bash
# Ubuntu/Linux 64-bit
sudo apt-get install libjsoncpp-dev
```

#### CppUnit Setup
```bash
# Ubuntu
sudo apt-get install libcppunit-dev
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
cd amazon-dsstne

# Add the mpiCC and nvcc compiler in the path (if they are not already there)
export PATH=/usr/local/openmpi/bin:/usr/local/cuda/bin:$PATH

# Decide where you want to install Dsstne (defaults to /usr/local if unspecified)
INSTALL_DIR="$(pwd)/install"

# Generate the Makefiles with CMake (you can use another generator like Ninja with -GNinja)
(mkdir build && cd build && cmake .. -DCMAKE_CXX_COMPILER=mpiCC -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}")

# Build and install the library
(cd build && make install)

# Run the tests if you want
(cd build && ctest --output-on-failure)

# Everything will be installed to the given CMAKE_INSTALL_PREFIX; consider adding this to your path
export PATH="${INSTALL_DIR}/bin:${PATH}"
```

Try running some [examples](examples.md).
