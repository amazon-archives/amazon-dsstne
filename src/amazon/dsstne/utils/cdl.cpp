#include "GpuTypes.h"
#include "NNTypes.h"
#include "cdl.h"

static std::map<string, TrainingMode> sOptimizationMap = {
    {"sgd",         TrainingMode::SGD},
    {"nesterov",    TrainingMode::Nesterov}
};


CDL::CDL()
{
    _randomSeed = time(NULL);
    _alphaInterval = 0;
    _alphaMultiplier = 0.5f;
    _batch = 1024;
    _checkpointInterval = 1;
    _checkpointFileName = "check";
    _shuffleIndexes = false;
    _resultsFileName = "network.nc";
    _alpha = 0.1f;
    _lambda = 0.001f;
    _mu = 0.9f;
    _optimizer = TrainingMode::SGD;
}


int CDL::Load_JSON(const string& fname)
{
    // flags for mandatory values - check to make sure they have been set during load
    // overall flags
    bool networkSet=false, commandSet=false, dataSet=false;
    // flags for when in training mode
    bool epochsSet=false;

    Json::Reader reader;
    Json::Value index;

    std::ifstream stream(fname, std::ifstream::binary);
    bool parsedSuccess = reader.parse(stream, index, false);
    if (!parsedSuccess)
    {
        printf("CDL::Load_JSON: Failed to parse JSON file: %s, error: %s\n", fname.c_str(), reader.getFormattedErrorMessages().c_str());
        return -1;
    }

    for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
    {
        // Extract JSON object key/value pair
        string name                         = itr.name();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        Json::Value key                     = itr.key();
        Json::Value value                   = *itr;
        string vstring                      = value.isString() ? value.asString() : "";
        std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

        if (name.compare("version") == 0)
        {
            float version = value.asFloat();
            // we only have this first version, but we will have future versions, then we will
            // need to do something, until then noop
        }
        else if (name.compare("network") == 0)
        {
            _networkFileName = value.asString();
            networkSet = true;
        }
        else if (name.compare("data") == 0) {
            _dataFileName = value.asString();
            dataSet = true;
        } else if (name.compare("randomseed") == 0)
            _randomSeed = value.asInt();
        else if (name.compare("command") == 0)
        {
            if (vstring.compare("train") == 0)
                _mode = Mode::Training;
            else if (vstring.compare("predict") == 0)
                _mode = Mode::Prediction;
            else if (vstring.compare("validate") == 0)
                _mode = Mode::Validation;
            else
            {
                printf("*** CDL::Load_JSON: Command unknown:  %s\n", vstring.c_str());
                return -1;
            }
            commandSet = true;
        }
        else if (name.compare("trainingparameters") == 0)
        {
            for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
            {
                string pname                = pitr.name();
                std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                Json::Value pkey            = pitr.key();
                Json::Value pvalue          = *pitr;
                if (pname.compare("epochs") == 0) {
                    _epochs = pvalue.asInt();
                    epochsSet = true;
                } else if (pname.compare("alpha") == 0)
                    _alpha = pvalue.asFloat();
                else if (pname.compare("alphainterval") == 0)
                    _alphaInterval = pvalue.asFloat();
                else if (pname.compare("alphamultiplier") == 0)
                    _alphaMultiplier = pvalue.asFloat();
                else if (pname.compare("mu") == 0)
                    _mu = pvalue.asFloat();
                else if (pname.compare("lambda") == 0)
                    _lambda = pvalue.asFloat();
                else if (pname.compare("checkpointinterval") == 0)
                    _checkpointInterval = pvalue.asFloat();
                else if (pname.compare("checkpointname") == 0)
                    _checkpointFileName = pvalue.asString();
                else if (pname.compare("optimizer") ==0)
                {
                    string pstring = pvalue.isString() ? pvalue.asString() : "";
                    std::transform(pstring.begin(), pstring.end(), pstring.begin(), ::tolower);
                    auto it = sOptimizationMap.find(pstring);
                    if (it != sOptimizationMap.end())
                        _optimizer = it->second;
                    else
                    {
                        printf("CDL::Load_JSON: Invalid TrainingParameter Optimizer: %s\n", pstring.c_str());
                        return -1;
                    }
                }
                else if (pname.compare("results") == 0) {
                    _resultsFileName = pvalue.asString();
                } else {
                    name = pitr.name();
                    printf("CDL::Load_JSON: Invalid TrainingParameter: %s\n", name.c_str());
                    return -1;
                }
            }
        }
        else
        {
            printf("*** CDL::Load_JSON: Unknown keyword:  %s\n", name.c_str());
            return -1;
        }
    }

    // alphaInterval of zero is specified to mean a constant alpha, so
    // if they want a constant alpha, then set up the other variables to make it so
    if (_alphaInterval == 0)
    {
        _alphaInterval = 20;
        _alphaMultiplier = 1;
    }

    if (!networkSet)
    {
        printf("CDL::Load_JSON: Network is required to be set, none found\n");
        return -1;
    }
    if (!commandSet)
    {
        printf("CDL::Load_JSON: Command is required, none found\n");
        return -1;
    }
    if (!dataSet)
    {
        printf("CDL::Load_JSON: Data source file is required to be set\n");
        return -1;
    }
    if (_mode == Mode::Training && !epochsSet)
    {
        printf("CDL::Load_JSON: Mode set to Training, requires number of epochs to be set\n");
        return -1;
    }

    printf("CDL::Load_JSON: %s successfully parsed\n", fname.c_str());

    return 0;
}


