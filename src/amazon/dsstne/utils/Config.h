#pragma once

#include <string>
using std::string;

struct Config {
  struct TrainingParameters {
    struct CheckPoint {
      string _sName; // Location to write checkpoint information
      int _Interval; // Number of minutes between writing checkpoint data (default 30) (TODO make it number of epochs?)
    };

    int _Epochs; // Number of training epochs
    int _MiniBatch; // Mini-batch size
    float _Alpha; // Learning rate
    float _Lambda; // Regularization/Weight Decay weight
    float _mu; // Momentum update parameter (TODO make it part of optimization method)
    int _AlphaInterval; // Interval between learning rate updates
    float _AlphaMultiplier; // Amount by which to multiply learning rate per above interval
    int _sOptimizer; // Optimization method, either "SGD", "Momentum", "RMSPROP", or "Nesterov" (default "SGD")
    CheckPoint _CheckPoint;
    bool _ShuffleIndices; // Shuffle training examples once per epoch? (default true)    
  };
  struct PredictionParameters {
    int _MiniBatch; // Mini-batch size (default 500, use 0 for entire dataset)
  };

  Config() {
    _sCommand = -1;
    _RandomSeed = -1; // default -1, sets from time of day
    _TrainingParameters._Epochs = 20;
    _TrainingParameters._MiniBatch = 256;
    _TrainingParameters._Alpha = 0.1;
    _TrainingParameters._Lambda = 0.001;
    _TrainingParameters._mu = 0.9;
    _TrainingParameters._AlphaInterval = 0;
    _TrainingParameters._AlphaMultiplier = 0.9;
    _TrainingParameters._sOptimizer = -1;
    _TrainingParameters._CheckPoint._Interval = 30;
    _TrainingParameters._ShuffleIndices = true;
    _PredictionParameters._MiniBatch = 128;
  }
  ;

  string _sNetwork; // NetCDF, JSON Object, or service object containing network
  int _sCommand; // Command to execute "Train", "Predict"
  int64_t _RandomSeed; // Initializes RNG for reproducible runs (default -1, sets from time of day)
  TrainingParameters _TrainingParameters;
  PredictionParameters _PredictionParameters;
  string _sData; // List of data sources
  string _sResults; // Location to write results (File or S3 Object)
};

bool LoadConfig(const string& fname, Config& config);
