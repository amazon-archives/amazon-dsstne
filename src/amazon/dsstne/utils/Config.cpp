#include "Config.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <json/json.h>

using std::cout;
using std::endl;

bool LoadConfig(const string& fname, Config& config) {
  Json::Value index;
  Json::Reader reader;
  bool bValid = true;

  std::ifstream stream(fname.c_str(), std::ifstream::binary);
  bool parsedSuccess = reader.parse(stream, index, false);

  if (!parsedSuccess) {
    cout << "Failed to parse JSON file" << fname.c_str() << " error " << reader.getFormatedErrorMessages().c_str() << endl;
    bValid = false;
  } else {
    for (Json::ValueIterator itr = index.begin(); itr != index.end(); itr++) {
      // Extract JSON object key/value pair
      string name = itr.memberName();
      std::transform(name.begin(), name.end(), name.begin(), ::tolower);
      Json::Value key = itr.key();
      Json::Value value = *itr;
      string vstring = value.isString() ? value.asString() : "";
      std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

      if (name.compare("network") == 0) {
        config._sNetwork = value.asString();
      } else if (name.compare("command") == 0) {
        if (vstring.compare("train") == 0) {
          config._sCommand = 1; //NNNetwork::Mode::Training; // TODO use enum
        } else if (vstring.compare("predict") == 0) {
          config._sCommand = 0; //NNNetwork::Mode::Prediction;
        } else if (vstring.compare("validate") == 0) {
          config._sCommand = 2; //NNNetwork::Mode::Validation;
        } else {
          cout << "unsupported item " << vstring;
          bValid = false;
        }
      } else if (name.compare("randomseed") == 0) {
        config._RandomSeed = value.asInt();
      } else if (name.compare("trainingparameters") == 0) {
        for (Json::ValueIterator pitr = value.begin(); pitr != value.end(); pitr++) {
          string pname = pitr.memberName();
          std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
          Json::Value pkey = pitr.key();
          Json::Value pvalue = *pitr;
          if (pname.compare("epochs") == 0) {
            config._TrainingParameters._Epochs = pvalue.asInt();
          } else if (pname.compare("minibatch") == 0) {
            config._TrainingParameters._MiniBatch = pvalue.asInt();
          } else if (pname.compare("alpha") == 0) {
            config._TrainingParameters._Alpha = pvalue.asFloat();
          } else if (pname.compare("lambda") == 0) {
            config._TrainingParameters._Lambda = pvalue.asFloat();
          } else if (pname.compare("mu") == 0) {
            config._TrainingParameters._mu = pvalue.asFloat();
          } else if (pname.compare("alphainterval") == 0) {
            config._TrainingParameters._AlphaInterval = pvalue.asInt();
          } else if (pname.compare("alphamultiplier") == 0) {
            config._TrainingParameters._AlphaMultiplier = pvalue.asFloat();
          } else if (pname.compare("optimizer") == 0) {
            string vstring = pvalue.isString() ? pvalue.asString() : "";
            if (vstring.compare("SGD") == 0) {
              config._TrainingParameters._sOptimizer = 0; //TrainingMode::SGD; // TODO: use enum
            } else if (vstring.compare("Momentum") == 0) {
              config._TrainingParameters._sOptimizer = 1; //TrainingMode::Momentum;
            } else if (vstring.compare("AdaGrad") == 0) {
              config._TrainingParameters._sOptimizer = 2; //TrainingMode::AdaGrad;
            } else if (vstring.compare("Nesterov") == 0) {
              config._TrainingParameters._sOptimizer = 3; //TrainingMode::Nesterov;
            } else if (vstring.compare("RMSProp") == 0) {
              config._TrainingParameters._sOptimizer = 4; //TrainingMode::RMSProp;
            } else if (vstring.compare("AdaDelta") == 0) {
              config._TrainingParameters._sOptimizer = 5; //TrainingMode::AdaDelta;
            } else {
              cout << "unsupported item " << vstring;
              bValid = false;
            }
          } else if (pname.compare("checkpoint") == 0) {
            for (Json::ValueIterator pitr2 = pvalue.begin(); pitr2 != pvalue.end(); pitr2++) {
              string pname2 = pitr2.memberName();
              std::transform(pname2.begin(), pname2.end(), pname2.begin(), ::tolower);
              Json::Value pkey2 = pitr2.key();
              Json::Value pvalue2 = *pitr2;
              if (pname2.compare("name") == 0) {
                config._TrainingParameters._CheckPoint._sName = pvalue2.asString();
              } else if (pname2.compare("interval") == 0) {
                config._TrainingParameters._CheckPoint._Interval = pvalue2.asInt();
              } else {
                cout << "unsupported item " << pname2;
                bValid = false;
              }
            }
          } else if (pname.compare("shuffleindices") == 0) {
            config._TrainingParameters._ShuffleIndices = pvalue.asBool();
          } else {
            cout << "unsupported item " << pname;
            bValid = false;
          }
        }
      } else if (name.compare("predictionparameters") == 0) {
        for (Json::ValueIterator pitr = value.begin(); pitr != value.end(); pitr++) {
          string pname = pitr.memberName();
          std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
          Json::Value pkey = pitr.key();
          Json::Value pvalue = *pitr;
          if (pname.compare("minibatch") == 0) {
            config._PredictionParameters._MiniBatch = pvalue.asInt();
          } else {
            cout << "unsupported item " << pname;
            bValid = false;
          }
        }
      } else if (name.compare("data") == 0) {
        config._sData = value.asString();
      } else if (name.compare("results") == 0) {
        config._sResults = value.asString();
      } else {
        cout << "unsupported item " << name;
        bValid = false;
      }
    }
  }

  return bValid;
}
