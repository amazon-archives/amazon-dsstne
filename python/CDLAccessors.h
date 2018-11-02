/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __CDLACCESSORS_H__
#define __CDLACCESSORS_H__

class CDLAccessors {
    public:
        static PyObject* GetRandomSeed(PyObject* self, PyObject* args);
        static PyObject* SetRandomSeed(PyObject* self, PyObject* args);
        static PyObject* GetEpochs(PyObject* self, PyObject* args);
        static PyObject* SetEpochs(PyObject* self, PyObject* args);
        static PyObject* GetBatch(PyObject* self, PyObject* args);
        static PyObject* SetBatch(PyObject* self, PyObject* args);
        static PyObject* GetCheckpointInterval(PyObject* self, PyObject* args);
        static PyObject* SetCheckpointInterval(PyObject* self, PyObject* args);
        static PyObject* GetAlphaInterval(PyObject* self, PyObject* args);
        static PyObject* SetAlphaInterval(PyObject* self, PyObject* args);
        static PyObject* GetShuffleIndexes(PyObject* self, PyObject* args);
        static PyObject* SetShuffleIndexes(PyObject* self, PyObject* args);
        static PyObject* GetAlpha(PyObject* self, PyObject* args);
        static PyObject* SetAlpha(PyObject* self, PyObject* args);
        static PyObject* GetAlphaMultiplier(PyObject* self, PyObject* args);
        static PyObject* SetAlphaMultiplier(PyObject* self, PyObject* args);
        static PyObject* GetLambda(PyObject* self, PyObject* args);
        static PyObject* SetLambda(PyObject* self, PyObject* args);
        static PyObject* GetMu(PyObject* self, PyObject* args);
        static PyObject* SetMu(PyObject* self, PyObject* args);
        static PyObject* GetMode(PyObject* self, PyObject* args);
        static PyObject* SetMode(PyObject* self, PyObject* args);
        static PyObject* GetOptimizer(PyObject* self, PyObject* args);
        static PyObject* SetOptimizer(PyObject* self, PyObject* args);
        static PyObject* GetNetworkFileName(PyObject* self, PyObject* args);
        static PyObject* SetNetworkFileName(PyObject* self, PyObject* args);
        static PyObject* GetDataFileName(PyObject* self, PyObject* args);
        static PyObject* SetDataFileName(PyObject* self, PyObject* args);
        static PyObject* GetCheckpointFileName(PyObject* self, PyObject* args);
        static PyObject* SetCheckpointFileName(PyObject* self, PyObject* args);
        static PyObject* GetResultsFileName(PyObject* self, PyObject* args);
        static PyObject* SetResultsFileName(PyObject* self, PyObject* args);
};

/**
 * Get the random seed from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the random seed integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetRandomSeed(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_randomSeed);
}

/**
 * Set the random seed in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param randomSeed - the random seed integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetRandomSeed(PyObject* self, PyObject* args) {
    int randomSeed = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, randomSeed, "cdl", "Oi");
    pCDL->_randomSeed = randomSeed;
    Py_RETURN_NONE;
}

/**
 * Get the epochs from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the epochs integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetEpochs(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_epochs);
}

/**
 * Set the epochs in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param epochs - the epochs integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetEpochs(PyObject* self, PyObject* args) {
    int epochs = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, epochs, "cdl", "Oi");
    pCDL->_epochs = epochs;
    Py_RETURN_NONE;
}

/**
 * Get the batch from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the batch integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetBatch(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_batch);
}

/**
 * Set the batch in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param batch - the batch integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetBatch(PyObject* self, PyObject* args) {
    int batch = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, batch, "cdl", "Oi");
    pCDL->_batch = batch;
    Py_RETURN_NONE;
}

/**
 * Get the checkpoint interval from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the checkpoint interval integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetCheckpointInterval(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_checkpointInterval);
}

/**
 * Set the checkpoint interval in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param checkpointInterval - the checkpoint interval integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetCheckpointInterval(PyObject* self, PyObject* args) {
    int checkpointInterval = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, checkpointInterval, "cdl", "Oi");
    pCDL->_checkpointInterval = checkpointInterval;
    Py_RETURN_NONE;
}

/**
 * Get the alpha interval from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the alpha interval integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetAlphaInterval(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_alphaInterval);
}

/**
 * Set the alpha interval in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param alphaInterval - the alpha interval integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetAlphaInterval(PyObject* self, PyObject* args) {
    int alphaInterval = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, alphaInterval, "cdl", "Oi");
    pCDL->_alphaInterval = alphaInterval;
    Py_RETURN_NONE;
}

/**
 * Get the shuffle indices from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the shuffle indices integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetShuffleIndexes(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("i", pCDL->_shuffleIndexes); // bool to int conversion is implicit in C++
}

/**
 * Set the shuffle indices in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param shuffleIndices - the shuffle indices integer
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetShuffleIndexes(PyObject* self, PyObject* args) {
    int shuffleIndices = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, int>(args, shuffleIndices, "cdl", "Oi");
    pCDL->_shuffleIndexes = shuffleIndices; // int to bool conversion is implicit in C++
    Py_RETURN_NONE;
}

/**
 * Get the alpha from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the alpha float
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetAlpha(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("f", pCDL->_alpha);
}

/**
 * Set the alpha in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param alpha - the alpha float
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetAlpha(PyObject* self, PyObject* args) {
    float alpha = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, float>(args, alpha, "cdl", "Of");
    pCDL->_alpha = alpha;
    Py_RETURN_NONE;
}

/**
 * Get the alpha multiplier from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the alpha multiplier float
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetAlphaMultiplier(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("f", pCDL->_alphaMultiplier);
}

/**
 * Set the alpha multiplier in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param alphaMultiplier - the alpha multiplier float
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetAlphaMultiplier(PyObject* self, PyObject* args) {
    float alphaMultiplier = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, float>(args, alphaMultiplier, "cdl", "Of");
    pCDL->_alphaMultiplier = alphaMultiplier;
    Py_RETURN_NONE;
}

/**
 * Get the lambda from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the lambda float
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetLambda(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("f", pCDL->_lambda);
}

/**
 * Set the lambda in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param lambda - the lambda float
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetLambda(PyObject* self, PyObject* args) {
    float lambda = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, float>(args, lambda, "cdl", "Of");
    pCDL->_lambda = lambda;
    Py_RETURN_NONE;
}

/**
 * Get the mu from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the mu float
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetMu(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("f", pCDL->_mu);
}

/**
 * Set the mu in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param mu - the mu float
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetMu(PyObject* self, PyObject* args) {
    float mu = 0;
    CDL* pCDL = parsePtrAndOneValue<CDL*, float>(args, mu, "cdl", "Of");
    pCDL->_mu = mu;
    Py_RETURN_NONE;
}

/**
 * Get the mode enumerator from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the mode enumerator string
 * @throws exception upon failure to parse arguments or to build the return value or if the mode enumerator is unsupported
 */
PyObject* CDLAccessors::GetMode(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    map<Mode, string>::iterator it = intToStringModeMap.find(pCDL->_mode);
    if (it == intToStringModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::GetMode received unsupported mode enumerator");
        return NULL;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Set the mode enumerator in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param mode - the mode enumerator string
 * @throws exception upon failure to parse args or if the mode enumerator string is unsupported
 */
PyObject* CDLAccessors::SetMode(PyObject* self, PyObject* args) {
    char* mode = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, mode, "cdl", "Os");
    map<string, Mode>::iterator it = stringToIntModeMap.find(string(mode));
    if (it == stringToIntModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::SetMode received unsupported mode enumerator string");
        return NULL;
    }
    pCDL->_mode = it->second;
    Py_RETURN_NONE;
}

/**
 * Get the training mode enumerator from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the training mode enumerator string
 * @throws exception upon failure to parse arguments or to build the return value or if the mode enumerator is unsupported
 */
PyObject* CDLAccessors::GetOptimizer(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    map<TrainingMode, string>::iterator it = intToStringTrainingModeMap.find(pCDL->_optimizer);
    if (it == intToStringTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::GetOptimizer received unsupported training mode enumerator");
        return NULL;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Set the training mode enumerator in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param trainingMode - the training mode enumerator string
 * @throws exception upon failure to parse arguments or if the training mode enumerator string is unsupported
 */
PyObject* CDLAccessors::SetOptimizer(PyObject* self, PyObject* args) {
    char* trainingMode = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, trainingMode, "cdl", "Os");
    map<string, TrainingMode>::iterator it = stringToIntTrainingModeMap.find(string(trainingMode));
    if (it == stringToIntTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "CDLAccessors::SetOptimizer received unsupported training mode enumerator string");
        return NULL;
    }
    pCDL->_optimizer = it->second;
    Py_RETURN_NONE;
}

/**
 * Get the network filename from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the network filename string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetNetworkFileName(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("s", pCDL->_networkFileName.c_str());
}

/**
 * Set the network filename in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param networkFilename - the network filename string
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetNetworkFileName(PyObject* self, PyObject* args) {
    char* networkFilename = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, networkFilename, "cdl", "Os");
    pCDL->_networkFileName = string(networkFilename);
    Py_RETURN_NONE;
}

/**
 * Get the data filename from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the data filename string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetDataFileName(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("s", pCDL->_dataFileName.c_str());
}

/**
 * Set the data filename in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param dataFilename - the data filename string
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetDataFileName(PyObject* self, PyObject* args) {
    char* dataFilename = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, dataFilename, "cdl", "Os");
    pCDL->_dataFileName = string(dataFilename);
    Py_RETURN_NONE;
}

/**
 * Get the checkpoint filename from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the checkpoint filename string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetCheckpointFileName(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("s", pCDL->_checkpointFileName.c_str());
}

/**
 * Set the checkpoint filename in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param checkpointFilename - the checkpoint filename string
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetCheckpointFileName(PyObject* self, PyObject* args) {
    char* checkpointFilename = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, checkpointFilename, "cdl", "Os");
    pCDL->_checkpointFileName = string(checkpointFilename);
    Py_RETURN_NONE;
}

/**
 * Get the results filename from the source CDL.
 * @param pCDL - the encapsulated source CDL*
 * @return a PyObject* that references the results filename string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* CDLAccessors::GetResultsFileName(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    return Py_BuildValue("s", pCDL->_resultsFileName.c_str());
}

/**
 * Set the results filename in the destination CDL.
 * @param pCDL - the encapsulated destination CDL*
 * @param resultsFilename - the results filename string
 * @throws exception upon failure to parse arguments
 */
PyObject* CDLAccessors::SetResultsFileName(PyObject* self, PyObject* args) {
    char* resultsFilename = NULL;
    CDL* const pCDL = parsePtrAndOneValue<CDL*, char*>(args, resultsFilename, "cdl", "Os");
    pCDL->_resultsFileName = string(resultsFilename);
    Py_RETURN_NONE;
}

#endif
