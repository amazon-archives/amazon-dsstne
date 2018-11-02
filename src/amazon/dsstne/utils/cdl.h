#ifndef __CDL_H__
#define __CDL_H__

#include "GpuTypes.h"
#include "NNTypes.h"

struct CDL
{
    CDL();
    int Load_JSON(const string& fname);


    std::string     _networkFileName;    // NetCDF or JSON Object file name (required)
    int             _randomSeed;         // Initializes RNG for reproducible runs (default: sets from time of day)
    Mode            _mode;
    std::string     _dataFileName;

    // training params
    int             _epochs; // total epochs
    int             _batch;  // used by inference as well:  Mini-batch size (default 500, use 0 for entire dataset)
    float           _alpha;
    float           _lambda;
    float           _mu;
    int             _alphaInterval;      // number of epochs per update to alpha - so this is the number of epochs per DSSTNE call
    float           _alphaMultiplier;    // amount to scale alpha every alphaInterval number of epochs
    TrainingMode    _optimizer;
    std::string     _checkpointFileName;
    int             _checkpointInterval;
    bool            _shuffleIndexes;
    std::string     _resultsFileName;
};

#endif
