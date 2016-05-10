# Amazon DSSTNE: Deep Scalable Sparse Tensor Network Engine

DSSTNE is an open source software library for training and deploying deep neural 
networks using GPUs.  Amazon engineers built DSSTNE to solve deep learning 
problems at Amazon's scale.  DSSTNE is built for production deployment of real-world 
deep learning applications, emphasizing speed and scale over experimental flexibility.

DSSTNE was built with a number of features for production workloads:

* **Multi-GPU Scale**: Training and prediction
both scale out to use multiple GPUs, spreading out computation
and storage in a model-parallel fashion for each layer.  
* **Large Layers**: Model-parallel scaling enables larger networks than 
are possible with a single GPU.  
* **Sparse Data**: DSSTNE is optimized for fast performance on sparse datasets.  Custom GPU kernels perform sparse computation on the GPU, without filling in lots of zeroes.

##License
[License](LICENSE)

##Setup 
* Follow [Setup](docs/getting_started/setup.md) for step by step instructions on installing and setting up DSSTNE

## User Guide 
* Check [User Guide](docs/getting_started/userguide.md) for detailed information about the features in DSSTNE

##Examples
* Check [Examples](docs/getting_started/examples.md) to start trying your first Neural Network Modeling using DSSTNE

##Q&A
[FAQ](FAQ.md)
