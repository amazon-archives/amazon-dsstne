# Examples #

## MovieLens Recommendations ##

After following the [Setup Guide](setup.md), you should be able to start neural network modeling using DSSTNE. For this example we will generate movie recommendations based on the [MovieLens dataset](http://grouplens.org/datasets/movielens/).

As with any modeling we will go through the 3 basics steps and walk you through the wrappers which will be used to interact with the DSSTNE Engine.

* [Converting Data](#converting-data)
* [Training](#training)
* [Prediction](#prediction)

You can check [Summary](#summary) for a quick script to run all the steps.

## Converting Data ##

You can use **wget** to fetch the dataset from [grouplens.org](http://grouplens.org/datasets/movielens/):

    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

Once you have downloaded the dataset, we will convert it to the NetCDF format used by DSSTNE. We will then try to train an Auto Encoder on the data.

### Generate the input data set ###

Generating the input data consist of several steps. First we, need to extract the ratings data from the zip file. There are several other CSV files in the MovieLens dataset, but **ratings.csv** is the only one we need for this example:

    unzip -p ml-20m.zip ml-20m/ratings.csv > ml-20m_ratings.csv

Next, we need to convert **ml-20m_ratings.csv** into the format recognised by **generateNetCDF**. This can be done using **awk**. Assuming your current directory is the **samples/movielens**, this should be as simple as:

    awk -f convert_ratings.awk ml-20m_ratings.csv > ml-20m_ratings

Finally, we can generate the NetCDF input files that will be used by the DSSTNE engine to train a neural network:

    generateNetCDF -d gl_input -i ml-20m_ratings -o gl_input.nc -f features_input -s samples_input -c

This will create the following files:

* gl_input.nc : NetCDF file in the format which DSSTNE Engine understand
* features_input : An index file with the indexes of each neuron
* samples_input : An index file with the indexes of all samples

You can always run **generateNetCDF -h** for more help.

### Generate the output data ###

Now we can generate the output data:

    generateNetCDF -d gl_output -i ml-20m_ratings -o gl_output.nc -f features_output -s samples_input -c

Please ensure that **samples_input** is used as we have to ensure that the index assigned to each example is the same in both input and the output.

## Training ##

We will now train a 3-layer Neural Network with one 128 node hidden layer with Sigmoid as an activation function.

This is the network definition contained in the sample configuration file, **config.json**:

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

We will train the model for 256 batch side for 10 epochs and the model file will be **gl.nc**:

    train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

### Multi-GPU Training ###

We can use OpenMPI to perform model parallel training across the GPUs in the same host. For example, you could run the following on a **g2.8xlarge** which has 4 GPUs:

    mpirun -np 4 train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

## Prediction ##

Once you have finished training now you can start predicting. Since DSSTNE is mostly used for recommendations we also added support for post filtering. The filter follows the same format as the standard DSSTNE format but for each example you can decide which features to remove when we predict. In the following example we will remove all the features which were triggered in the input layer for predicting.

    predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml-20m_ratings -s recs -r ml-20m_ratings

This will result in the top 10 recommendation for each sample in the **recs** file.

## Summary ##

You can run the full pipeline with the following commands, or use [run_movielens_sample.sh](../../samples/movielens/run_movielens_sample.sh) to run the complete example:

    # Fetch Movielens dataset
    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

    # Extract ratings from dataset
    unzip -p ml-20m.zip ml-20m/ratings.csv > ml-20m_ratings.csv

    # Convert ml-20m_ratings.csv to format supported by generateNetCDF
    awk -f convert_ratings.awk ml-20m_ratings.csv > ml-20m_ratings

    # Generate NetCDF files for input and output layers
    generateNetCDF -d gl_input  -i ml-20m_ratings -o gl_input.nc  -f features_input  -s samples_input -c
    generateNetCDF -d gl_output -i ml-20m_ratings -o gl_output.nc -f features_output -s samples_input -c

    # Train the network
    train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

    # Generate predictions
    predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml-20m_ratings -s recs -r ml-20m_ratings

