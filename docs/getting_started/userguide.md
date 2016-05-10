#
 DSSTNE currently only supports Fully Connected layers and the network structure for training is defined though a  config json representation. 

#Data Formats
DSSTNE Engine takes in  Data only in NetCDF Format. There are some wrappers which converts the following format  to NetCDF. The separator between the example and features is TAB
```bash
Example1	Feature1:Feature2:Feature3
Example2	Feature5:Feature2:Feature4
Example3	Feature6:Feature7:Feature8
```

#Neural Network Layer Definition Language
The definitions for the Neural Network fed into DSSTNE is represented in a Json Format. All the Supported feature can be found at [LDL.txt](LDL.txt). Sample one is given below
```bash
{
    "Version" : 0.8,
    "Name" : "2 Hidden Layer",
    "Kind" : "FeedForward",  

    "ShuffleIndices" : false,


    "ScaledMarginalCrossEntropy" : {
        "oneTarget" : 1.0,
        "zeroTarget" : 0.0,
        "oneScale" : 1.0,
        "zeroScale" : 1.0
    },
    "Layers" : [
        { "Name" : "Input", "Kind" : "Input", "N" : "auto", "DataSet" : "input", "Sparse" : true }, 
        { "Name" : "Hidden1", "Kind" : "Hidden", "Type" : "FullyConnected", "Source" : "Input", "N" : 1024, "Activation" : "Relu", "Sparse" : false, "pDropout" : 0.5, "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01 } },
        { "Name" : "Hidden2", "Kind" : "Hidden", "Type" : "FullyConnected", "Source" : ["Hidden1"], "N" : 1024, "Activation" : "Relu", "Sparse" : false, "pDropout" : 0.5, "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01 } },  
        { "Name" : "Output", "Kind" : "Output", "Type" : "FullyConnected", "DataSet" : "output", "N" : "auto", "Activation" : "Sigmoid", "Sparse" : true , "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01, "Bias" : -10.2 }}
    ],
        
    "ErrorFunction" : "ScaledMarginalCrossEntropy"
}

```
#Layers
```bash
"Layers" : [
        { "Name" : "Input", "Kind" : "Input", "N" : "auto", "DataSet" : "gl_input", "Sparse" : true }, 
        { "Name" : "Hidden1", "Kind" : "Hidden", "Type" : "FullyConnected", "Source" : "Input", "N" : 1024, "Activation" : "Relu", "Sparse" : false, "pDropout" : 0.5, "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01 } },
        { "Name" : "Hidden2", "Kind" : "Hidden", "Type" : "FullyConnected", "Source" : ["Hidden1"], "N" : 1024, "Activation" : "Relu", "Sparse" : false, "pDropout" : 0.5, "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01 } },  
        { "Name" : "Output", "Kind" : "Output", "Type" : "FullyConnected", "DataSet" : "gl_output", "N" : "auto", "Activation" : "Sigmoid", "Sparse" : true , "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01, "Bias" : -10.2 }}
    ],
```
Neural Network is represented in Layers in a configuraton json. We only support Fully Connected layers and the layer can be of 3 different kinds

1. Input
   This is the input layer for the Neural Network  and atleast one Input layer is *required* for training. There should be a  *DataSet* required for the input layer. 

2. Hidden
   Hidden Layers are Layers which connect between layers. It Does require a DataSet but rather a *Source*. If *Source* is not mentioned then the previous Layer is taken as Source

3. Output
   Output Layer is the layer where the truths are compared against. Atleast one Output Layer is required and there should be a DataSet also for the Output Layer. 

#Activation
```bash
{ "Name" : "Hidden2", "Kind" : "Hidden", ..............., "Activation" : "Relu", ................ }

```
  Activation function for each layer is passed on as a parameter to the Layer Definition in the field *Activation* . The following Activation functions are supported

*    Sigmoid
*    Tanh
*    RectifiedLinear 
*    Linear
*    ParametricRectifiedLinear
*    SoftPlus
*    SoftSign
*    SoftMax
*    ReluMax
*    LinearMax

#Size 
```bash
{ "Name" : "Input", "Kind" : "Input", "N" : "auto", "DataSet" : "gl_input", "Sparse" : true }

{ "Name" : "Hidden1", "Kind" : "Hidden", "Type" : "FullyConnected", "Source" : "Input", "N" : 1024, "Activation" : ...}

```
Size of the Layer is added to the *N* field in Layer Definition and is represented as an Integer. For Input and Output layers *auto* is also supported so that the size is automtically figured from the dataset

#Initialization
```bash
{ "Name" : "Hidden1", ....... "WeightInit" : { "Scheme" : "Gaussian", "Scale" : 0.01 }............................ }
```
Weight Initialization between the Layers are defined  by *WeightInit* filed in the Layer definition. The supported weight initializers are

* Xavier
* CaffeXavier
* Gaussian
* Uniform
* UnitBall
* Constant


#Optimization
```bash
/ Set to default training mode Nesterov.
    TrainingMode mode=Nesterov;
    pNetwork->SetTrainingMode(mode)
```

  Optimization for the Network is currently passed through the code . Currently Supported optimizers are 
* SGD
* Momentum
* AdaGrad
* Nesterov
* RMSProp
* AdaDelta

