

# Amazon DSSTNE: Deep Scalable Sparse Tensor Network Engine

DSSTNE (pronounced "Destiny") is an open source software library for training and deploying recommendation
models with sparse inputs, fully connected hidden layers, and sparse outputs. Models with weight matrices
that are too large for a single GPU can still be trained on a single host. DSSTNE has been used at Amazon
to generate personalized product recommendations for our customers at Amazon's scale. It is designed for
production deployment of real-world applications which need to emphasize speed and scale over experimental 
flexibility.

DSSTNE was built with a number of features for production recommendation workloads:

* **Multi-GPU Scale**: Training and prediction
both scale out to use multiple GPUs, spreading out computation
and storage in a model-parallel fashion for each layer.
* **Large Layers**: Model-parallel scaling enables larger networks than
are possible with a single GPU.
* **Sparse Data**: DSSTNE is optimized for fast performance on sparse datasets, common in recommendation 
problems. Custom GPU kernels perform sparse computation on the GPU, without filling in lots of zeroes.

## Benchmarks
* scottlegrand@ reported [near-linear scaling with multiple GPUs] on the MovieLens recommendation problem 
(https://medium.com/@scottlegrand/first-dsstne-benchmarks-tldr-almost-15x-faster-than-tensorflow-393dbeb80c0f#.ghe74fu1q)
* Directions on how to run a benchmark can be found in [here](benchmarks/Benchmark.md)

## Scaling up
* [Using Spark in AWS EMR and Dockers in AWS ECS ](http://blogs.aws.amazon.com/bigdata/post/TxGEL8IJ0CAXTK/Generating-Recommendations-at-Amazon-Scale-with-Apache-Spark-and-Amazon-DSSTNE)
    

## License
[License](LICENSE)


 
 

## Setup
* Follow [Setup](docs/getting_started/setup.md) for step by step instructions on installing and setting up DSSTNE

## User Guide
* Check [User Guide](docs/getting_started/userguide.md) for detailed information about the features in DSSTNE

## Examples
* Check [Examples](docs/getting_started/examples.md) to start trying your first Neural Network Modeling using DSSTNE

## Q&A
[FAQ](FAQ.md)
