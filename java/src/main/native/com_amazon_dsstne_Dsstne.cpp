/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */
#include <dlfcn.h>
#include <sstream>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"

#include "jni_util.h"
#include "com_amazon_dsstne_Dsstne.h"

using namespace std;
using namespace dsstne::jni;

using NNDataSetEnums::Attributes;
using NNDataSetEnums::DataType;

namespace
{
const unsigned long SEED = 12134ull;

void *LIB_MPI = NULL;
const char* LIB_MPI_SO = "libmpi.so";

const int ARGC = 1;
char *ARGV = "jni-faux-process";

References REFS;

const std::string _NNLayer = "com/amazon/dsstne/NNLayer";
const std::string _NNDataSet = "com/amazon/dsstne/NNDataSet";
const std::string _OutputNNDataSet = "com/amazon/dsstne/data/OutputNNDataSet";

jmethodID java_ArrayList_;
jmethodID java_ArrayList_add;

jmethodID NNLayer_;

jmethodID NNDataSet_getName;
jmethodID NNDataSet_getAttribute;
jmethodID NNDataSet_getDataTypeOrdinal;

jmethodID NNDataSet_getDimensions;
jmethodID NNDataSet_getDimX;
jmethodID NNDataSet_getDimY;
jmethodID NNDataSet_getDimZ;
jmethodID NNDataSet_getExamples;

jmethodID NNDataSet_getSparseStart;
jmethodID NNDataSet_getSparseEnd;
jmethodID NNDataSet_getSparseIndex;
jmethodID NNDataSet_getData;

jmethodID OutputNNDataSet_getName;
jmethodID OutputNNDataSet_getIndexes;
jmethodID OutputNNDataSet_getScores;

GpuContext* checkPtr(JNIEnv *env, jlong ptr)
{
    GpuContext *gpuContext = (GpuContext*) ptr;
    if (gpuContext == NULL)
    {
        throwJavaException(env, RuntimeException,
                           "GpuContext pointer is null, call init() prior to any other functions");
    }
    return gpuContext;
}
}  //namespace

jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    /*
     * JVM loads dynamic libs into a local namespace,
     * MPI requires to be loaded into a global namespace
     * so we manually load it into a global namespace here
     */
    LIB_MPI = dlopen(LIB_MPI_SO, RTLD_NOW | RTLD_GLOBAL);

    if (LIB_MPI == NULL)
    {
        std::cerr << "Failed to load libmpi.so" << std::endl;
        exit(1);
    }

    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
    {
        return JNI_ERR;
    } else
    {
        java_ArrayList_ = findConstructorId(env, REFS, ArrayList, NO_ARGS_CONSTRUCTOR);
        java_ArrayList_add = findMethodId(env, REFS, ArrayList, "add", "(Ljava/lang/Object;)Z");

        NNLayer_ = findConstructorId(env, REFS, _NNLayer, "(Ljava/lang/String;Ljava/lang/String;IIIIII)V");

        NNDataSet_getName = findMethodId(env, REFS, _NNDataSet, "getName", "()Ljava/lang/String;");
        NNDataSet_getAttribute = findMethodId(env, REFS, _NNDataSet, "getAttribute", "()I");
        NNDataSet_getDataTypeOrdinal = findMethodId(env, REFS, _NNDataSet, "getDataTypeOrdinal", "()I");
        NNDataSet_getDimensions = findMethodId(env, REFS, _NNDataSet, "getDimensions", "()I");
        NNDataSet_getDimX = findMethodId(env, REFS, _NNDataSet, "getDimX", "()I");
        NNDataSet_getDimY = findMethodId(env, REFS, _NNDataSet, "getDimY", "()I");
        NNDataSet_getDimZ = findMethodId(env, REFS, _NNDataSet, "getDimZ", "()I");
        NNDataSet_getExamples = findMethodId(env, REFS, _NNDataSet, "getExamples", "()I");
        NNDataSet_getSparseStart = findMethodId(env, REFS, _NNDataSet, "getSparseStart", "()[J");
        NNDataSet_getSparseEnd = findMethodId(env, REFS, _NNDataSet, "getSparseEnd", "()[J");
        NNDataSet_getSparseIndex = findMethodId(env, REFS, _NNDataSet, "getSparseIndex", "()[J");
        NNDataSet_getData = findMethodId(env, REFS, _NNDataSet, "getData", "()Ljava/nio/ByteBuffer;");

        OutputNNDataSet_getName = findMethodId(env, REFS, _OutputNNDataSet, "getName", "()Ljava/lang/String;");
        OutputNNDataSet_getIndexes = findMethodId(env, REFS, _OutputNNDataSet, "getIndexes", "()[J");
        OutputNNDataSet_getScores = findMethodId(env, REFS, _OutputNNDataSet, "getScores", "()[F");

        return JNI_VERSION_1_6;
    }
}

void JNI_OnUnload(JavaVM *vm, void *reserved)
{
    using namespace dsstne;

    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
    {
        return;
    } else
    {
        deleteReferences(env, REFS);
    }
}

JNIEXPORT jlong JNICALL Java_com_amazon_dsstne_Dsstne_load(JNIEnv *env, jclass clazz, jstring jNetworkFileName,
                                                           jint batchSize)
{
    const char *networkFileName = env->GetStringUTFChars(jNetworkFileName, 0);

    getGpu().Startup(ARGC, &ARGV);
    getGpu().SetRandomSeed(SEED);
    NNNetwork *network = LoadNeuralNetworkNetCDF(networkFileName, batchSize);
    getGpu().SetNeuralNetwork(network);

    env->ReleaseStringUTFChars(jNetworkFileName, networkFileName);
    return (jlong) &getGpu();
}

bool isSupported(uint32_t attributes)
{
    // only support vanilla sparse and dense datasets for now
    static const vector<Attributes> SUPPORTED_ATTRIBUTES(Attributes::Sparse);
    for (auto mask : SUPPORTED_ATTRIBUTES)
    {
        if (attributes & mask)
        {
            attributes -= mask;
        }
    }
    return attributes == 0;
}

tuple<NNDataSetDimensions, uint32_t> getDataDimensions(JNIEnv *env, jobject jDataset)
{
    NNDataSetDimensions dim;
    dim._width = env->CallIntMethod(jDataset, NNDataSet_getDimX);
    dim._length = env->CallIntMethod(jDataset, NNDataSet_getDimY);
    dim._height = env->CallIntMethod(jDataset, NNDataSet_getDimZ);
    dim._dimensions = env->CallIntMethod(jDataset, NNDataSet_getDimensions);
    uint32_t examples = env->CallIntMethod(jDataset, NNDataSet_getExamples);
    return make_tuple(dim, examples);
}

template<typename T> NNDataSetBase* newNNDataSet(JNIEnv *env, jobject jDataset)
{
    jstring jName = (jstring) env->CallObjectMethod(jDataset, NNDataSet_getName);
    const char *name = env->GetStringUTFChars(jName, NULL);
    jint attributes = env->CallIntMethod(jDataset, NNDataSet_getAttribute);

    if (!isSupported(attributes))
    {
        throwJavaException(env, IllegalArgumentException, "Unsupported attributes %u for dataset %s", attributes, name);
    }

    tuple<NNDataSetDimensions, uint32_t> dim = getDataDimensions(env, jDataset);
    NNDataSetBase *dataset;
    if (attributes & Attributes::Sparse)
    {
        /* sparse data */
        dataset = new NNDataSet<T>(get<1>(dim), get<0>(dim), name);
    } else
    {
        /* dense data */
        dataset = new NNDataSet<T>(get<1>(dim), get<0>(dim), name);
    }

    env->ReleaseStringUTFChars(jName, name);
    return dataset;
}

NNDataSetBase* newNNDataSet(JNIEnv *env, jobject jDataset)
{
    DataType dataType = static_cast<DataType>(env->CallIntMethod(jDataset, NNDataSet_getDataTypeOrdinal));
    NNDataSetBase *dataset;
    switch (dataType) {
        case DataType::UInt:
            dataset = newNNDataSet<uint32_t>(env, jDataset);
            break;
        case DataType::Int:
            dataset = newNNDataSet<int>(env, jDataset);
            break;
        case DataType::Float:
            dataset = newNNDataSet<float>(env, jDataset);
            break;
        case DataType::Double:
            dataset = newNNDataSet<double>(env, jDataset);
            break;
        case DataType::Char:
            dataset = newNNDataSet<char>(env, jDataset);
            break;
        case DataType::UChar:
        case DataType::RGB8:
            dataset = newNNDataSet<uint8_t>(env, jDataset);
            break;
        default:
            throwJavaException(
                env, IllegalArgumentException,
                "Unsupported data type: %u. DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8",
                dataType);
    }
    return dataset;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_loadDatasets(JNIEnv *env, jclass clazz, jlong ptr,
                                                                  jobjectArray jDatasets)
{
    using NNDataSetEnums::DataType;

    jsize len = env->GetArrayLength(jDatasets);
    vector<NNDataSetBase*> datasets;

    for (jsize i = 0; i < len; ++i)
    {
        jobject jDataset = env->GetObjectArrayElement(jDatasets, i);
        NNDataSetBase *dataset = newNNDataSet(env, jDataset);
        if (dataset == NULL)
        {

        }
        datasets.push_back(dataset);
    }

    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;
    network->LoadDataSets(datasets);
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    gpuContext->Shutdown();
}

JNIEXPORT jobject JNICALL Java_com_amazon_dsstne_Dsstne_get_1layers(JNIEnv *env, jclass clazz, jlong ptr,
                                                                    jint kindOrdinal)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;
    NNLayer::Kind kind = static_cast<NNLayer::Kind>(kindOrdinal);

    std::vector<const NNLayer*> layers;
    std::vector<const NNLayer*>::iterator it = network->GetLayers(kind, layers);
    if (it == layers.end())
    {
        throwJavaException(env, RuntimeException, "No layers of type %s found in network %s", NNLayer::_sKindMap[kind],
                           network->GetName());
    }

    jobject jLayers = newObject(env, REFS, ArrayList, java_ArrayList_);

    for (; it != layers.end(); ++it)
    {
        const NNLayer* layer = *it;
        const std::string &name = layer->GetName();
        const std::string &datasetName = layer->GetDataSetName();
        jstring jName = env->NewStringUTF(name.c_str());
        jstring jDatasetName = env->NewStringUTF(datasetName.c_str());
        int kind = static_cast<int>(layer->GetKind());
        uint32_t attributes = layer->GetAttributes();

        uint32_t numDim = layer->GetNumDimensions();

        uint32_t lx, ly, lz, lw;
        std::tie(lx, ly, lz, lw) = layer->GetDimensions();

        jobject jInputLayer = newObject(env, REFS, _NNLayer, NNLayer_, jName, jDatasetName, kind, attributes, numDim, lx,
                                        ly, lz);

        env->CallBooleanMethod(jLayers, java_ArrayList_add, jInputLayer);
    }
    return jLayers;
}

bool checkDataset(NNDataSetBase *dstDataset, uint32_t attribute, DataType dataType, const NNDataSetDimensions &dim,
                  uint32_t examples)
{
    return dstDataset->_attributes == attribute
        && dstDataset->_dataType == dataType
        && dstDataset->_dimensions == dim._dimensions
        && dstDataset->_width == dim._width
        && dstDataset->_width == dim._length
        && dstDataset->_width == dim._height
        && dstDataset->_examples == examples;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_predict(JNIEnv *env, jclass clazz, jlong ptr, jint k,
                                                             jobjectArray jInputDatasets, jobjectArray jOutputDatasets)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;

    vector<const NNLayer*> inputLayers;
    network->GetLayers(NNLayer::Kind::Input, inputLayers);
    uint32_t batchSize = network->GetBatch();

    // expect jInputs.size == network.inputLayers.size and jOutputs.size == network.outputLayers.size
    jsize inputLen = env->GetArrayLength(jInputDatasets);

    using NNDataSetEnums::DataType;

    for (jsize i = 0; i < inputLen; ++i)
    {
        jobject jInputDataset = env->GetObjectArrayElement(jInputDatasets, i);
        jstring jDatasetName = (jstring) env->CallObjectMethod(jInputDataset, NNDataSet_getName);
        const char *datasetName = env->GetStringUTFChars(jDatasetName, NULL);
        tuple<NNDataSetDimensions, uint32_t> dim = getDataDimensions(env, jInputDataset);
        uint32_t attribute = env->CallIntMethod(jInputDataset, NNDataSet_getAttribute);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jInputDataset, NNDataSet_getDataTypeOrdinal));

        const NNLayer *layer = network->GetLayerForDataSet(datasetName, NNLayer::Kind::Input);
        if (!layer)
        {
            throwJavaException(env, IllegalArgumentException, "No matching layer found in network %s for dataset: %s",
                               network->GetName(), datasetName);
        }

        NNDataSetBase *dstDataset = layer->GetDataSet();
        if(!checkDataset(dstDataset, attribute, dataType, get<0>(dim), get<1>(dim))) {
            throwJavaException(env, IllegalArgumentException, "Input dataset %s does not match the layer %s dataset",
                               datasetName, layer->GetName());
        }

        // java data direct buffer may contain sparse or dense data depending on the type of the dataset
        jobject srcByteBuffer = env->CallObjectMethod(jInputDataset, NNDataSet_getData);
        const void *srcDataNative = env->GetDirectBufferAddress(srcByteBuffer);

        if (dstDataset->_attributes == Attributes::Sparse)
        {
            /* copy sparse data */
            jlongArray jSparseStart = (jlongArray) env->CallObjectMethod(jInputDataset, NNDataSet_getSparseStart);
            jlongArray jSparseEnd = (jlongArray) env->CallObjectMethod(jInputDataset, NNDataSet_getSparseEnd);
            // TODO measure performance diff if we use GetPrimitiveArrayCritical
            jlongArray jSparseIndex = (jlongArray) env->CallObjectMethod(jInputDataset, NNDataSet_getSparseIndex);
            jlong *sparseStart = env->GetLongArrayElements(jSparseStart, NULL);
            jlong *sparseEnd = env->GetLongArrayElements(jSparseEnd, NULL);
            jlong *sparseIndex = env->GetLongArrayElements(jSparseIndex, NULL);

            dstDataset->SetSparseData((uint64_t*) sparseStart, (uint64_t*)sparseEnd, srcDataNative, (uint32_t*)sparseIndex);
            env->ReleaseLongArrayElements(jSparseStart, sparseStart, JNI_ABORT);
            env->ReleaseLongArrayElements(jSparseEnd, sparseEnd, JNI_ABORT);
            env->ReleaseLongArrayElements(jSparseIndex, sparseIndex, JNI_ABORT);
        } else
        {
            /* copy dense data */
            dstDataset->SetData(srcDataNative);
        }

        env->ReleaseStringUTFChars(jDatasetName, datasetName);
    }

    // there is only one batch in the dataset; always start from position 0
    network->SetPosition(0);
    network->PredictBatch();

    jsize outputLen = env->GetArrayLength(jOutputDatasets);
    for (jsize i = 0; i < outputLen; ++i)
    {
        jobject jOutputDataset = env->GetObjectArrayElement(jOutputDatasets, i);

        // output dataset name is set to output layer name
        jstring jDatasetName = (jstring) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getName);
        const char *datasetName = env->GetStringUTFChars(jDatasetName, NULL);

        jfloatArray jScores = (jfloatArray) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getScores);
        jlongArray jIndexes = (jlongArray) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getIndexes);

        float *scores = (float*) env->GetPrimitiveArrayCritical(jScores, NULL);
        uint32_t *indexes = (uint32_t*) env->GetPrimitiveArrayCritical(jIndexes, NULL);

        NNFloat *outputUnitBuffer = network->GetUnitBuffer(datasetName);
        uint32_t stride = network->GetBufferSize(datasetName) / batchSize;

        kCalculateTopK(outputUnitBuffer, scores, indexes, batchSize, stride, k);

        env->ReleasePrimitiveArrayCritical(jScores, scores, 0);
        env->ReleasePrimitiveArrayCritical(jIndexes, indexes, 0);
        env->ReleaseStringUTFChars(jDatasetName, datasetName);
    }
}

