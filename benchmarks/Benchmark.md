#Benchmarks for DSSTNE
We ran the benchmark of DSSTNE using the Movielens data set. Training Parameter Specific
* 27278 input/output dimensions
* 3 Hidden Sigmoid layers with 1024 each
* RMSProp learner
* 256 Batchsize

Time taken to run one epoch is considered for performance comparison


#DSSTNE
Use the Config at [config.json](dsstne/config.json). Follow the [example](../docs/getting_started/examples.md) and change the traning command
```bash
train -i gl_input.nc -o gl_output.nc -d gl -c config.json -b 256 -e 20 -n gl_network.nc
```

#TensorFlow
[autoencoder.py](tf/autoencoder.py) -u 1024 -b 256 -i 1082 -v54 --vocab_size 27278 -l 3 -f /input/data/ml20m-all.remotcc


