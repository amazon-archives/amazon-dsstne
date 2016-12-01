#!/bin/bash
#
# Downloads the MovieLens dataset and trains a small model for 10 epochs before
# writing predictions to a file called 'recs'.
#

# Fetch Movielens dataset if necessary
if [ -f ml-20m.zip ]; then
    echo "Already downloaded MovieLens dataset."
else
    echo "Downloading MovieLens dataset..."
    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
fi

# Extract ratings from dataset
echo "Extracting ml-20/ratings.csv from ml-20m.zip to ml-20m_ratings.csv"
unzip -p ml-20m.zip ml-20m/ratings.csv > ml-20m_ratings.csv

# Convert ml-20m_ratings.csv to format supported by generateNetCDF
echo "Converting ml-20m_ratings.csv to DSSTNE format"
awk -f convert_ratings.awk ml-20m_ratings.csv > ml-20m_ratings

# Generate NetCDF files for input and output layers
generateNetCDF -d gl_input  -i ml-20m_ratings -o gl_input.nc  -f features_input  -s samples_input -c
generateNetCDF -d gl_output -i ml-20m_ratings -o gl_output.nc -f features_output -s samples_input -c

# Train the network
train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

# Generate predictions
predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml-20m_ratings -s recs -r ml-20m_ratings
