# Sample Neural Network Modeling using DSSTNE

After you have followed [setup](setup.md) we should be able to start neural network modeling using DSSTNE. As with any modeling we will go through the 3 basics steps and walk you though the wrappers which will be used to interact with the DSSTNE Engine.
* [Convert Data](#convert-data)
* [Train](#train)
* [Predict](#predict)

You can check [Summary](#summary) for a quick script to run all the steps.

**MultiGPU Modeling COMING SOON**



## Convert Data
For this example we will use the movie lens data which was transformed into the [format](userguide.md) which DSSTNE recognizes.
```bash
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/data/ml20m-all
```
Once you have downloaded the data we will convert to the NetCDF format. We will try to run an Auto Encoder with the given data set.

### Generate the input data set
```bash
generateNetCDF -d gl_input -i ml20m-all -o gl_input.nc -f features_input -s samples_input -c
```
 This will create the following files:
* gl_input.nc : NetCDF file in the format which DSSTNE Engine understand
* features_input : A index file with the indexes of each neuron
* samples_input : A index file with the indexes of all samples

You can always run **generateNetCDF -h** for more help.

### Generate the output data
Please ensure that the number of examples and the examples are the same between the input and the output. We will generate the output data.
```bash
generateNetCDF -d gl_output -i ml20m-all -o gl_output.nc -f features_output -s samples_input -c
```
Please ensure that **samples_input** is used as we have to ensure that the example index is the same in both input and the output.


## Train
We will train a 3 layer Neural Network with one 128 node hidden layer with Sigmoid as an activation function.
```bash
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/configs/config.json.
cat config.json
{
    "Version" : 0.7,
    "Name" : "AE",
    "Kind" : "FeedForward",  
    "SparsenessPenalty" : {
        "p" : 0.5,
        "beta" : 2.0
    },

    "ShuffleIndices" : false,

    "Denoising" : {
        "p" : 0.2
    },

    "ScaledMarginalCrossEntropy" : {
        "oneTarget" : 1.0,
        "zeroTarget" : 0.0,
        "oneScale" : 1.0,
        "zeroScale" : 1.0
    },
    "Layers" : [
        { "Name" : "Input", "Kind" : "Input", "N" : "auto", "DataSet" : "gl_input", "Sparse" : true },
        { "Name" : "Hidden", "Kind" : "Hidden", "Type" : "FullyConnected", "N" : 128, "Activation" : "Sigmoid", "Sparse" : true },
        { "Name" : "Output", "Kind" : "Output", "Type" : "FullyConnected", "DataSet" : "gl_output", "N" : "auto", "Activation" : "Sigmoid", "Sparse" : true }
    ],

    "ErrorFunction" : "ScaledMarginalCrossEntropy"
}

```
We will train the model for 256 batch side for 10 epochs and the model file will be gl.nc

```bash
train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10
```

We can also train the model across multiple gpus in the same machine and it will model parallelism automatically for you. If you are using g2.8xlarge instance which has 4 GPUs you can run as follows

```bash
mpirun -np 4 train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10
```
## Predict

Once you have finished training now you can start predicting. Since DSSTNE is mostly used for recommendations we also added support for post filtering. The filter follows the same format as the standard DSSTNE format but for each example you can decide which features to remove when we predict. In the following example we will remove all the features which were triggered in the input layer for predicting.
```bash
predict -b 1024 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml20m-all -s recs -r ml20m-all
```

This will result in the top 10 recommendation for each sample in the **recs** file.

## Summary
You can run the full pipeline with the following commands or check at [run_movie_lens_sample.sh](../../src/amazon/dsstne/samples/run_movie_lens_sample.sh).
```bash
#!/bin/bash
# Get the Movie lens Dataset
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/data/ml20m-all
# Generate Input layer and Output Layer
generateNetCDF -d gl_input -i ml20m-all -o gl_input.nc -f features_input -s samples_input -c
generateNetCDF -d gl_output -i ml20m-all -o gl_output.nc -f features_output -s samples_input -c

# Train
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/configs/config.json.
train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

# Predict
predict -b 1024 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml20m-all -s recs -r ml20m-all
```
