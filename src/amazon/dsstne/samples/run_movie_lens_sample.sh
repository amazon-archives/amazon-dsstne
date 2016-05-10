#!/bin/bash
# Downloads the Move lens Data Set and trains a small model for 10 Epochs and predictions are
# written to 'recs' file
#Get the Movie lens Dataset
#Ensure that the executable location has been added to the PATH
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/data/ml20m-all
#Generate Input layer and Output Layer
generateNetCDF -d gl_input -i ml20m-all -o gl_input.nc -f features_input -s samples_input -c
generateNetCDF -d gl_output -i ml20m-all -o gl_output.nc -f features_output -s samples_input -c

#Train
wget https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/configs/config.json
train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

# Predict
predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml20m-all -s recs -r ml20m-all
