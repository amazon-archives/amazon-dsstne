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

#include "amazon/dsstne/engine/DsstneContext.h"

#include "jni_util.h"
#include "com_amazon_dsstne_Dsstne.h"

using namespace std;
using namespace dsstne;
using namespace dsstne::jni;

using NNDataSetEnums::Attributes;
using NNDataSetEnums::DataType;

namespace
{
void *LIB_MPI = NULL;
const char* LIB_MPI_SO = "libmpi.so";

References REFS;

const std::string _NNLayer = "com/amazon/dsstne/NNLayer";
const std::string _NNDataSet = "com/amazon/dsstne/NNDataSet";
const std::string _OutputNNDataSet = "com/amazon/dsstne/data/OutputNNDataSet";

jmethodID java_ArrayList_;
jmethodID java_ArrayList_add;

jmethodID NNLayer_;

jmethodID NNDataSet_getName;
jmethodID NNDataSet_getLayerName;
jmethodID NNDataSet_getAttribute;
jmethodID NNDataSet_getDataTypeOrdinal;

jmethodID NNDataSet_getDimensions;
jmethodID NNDataSet_getDimX;
jmethodID NNDataSet_getDimY;
jmethodID NNDataSet_getDimZ;
jmethodID NNDataSet_getExamples;
jmethodID NNDataSet_getStride;

jmethodID NNDataSet_getSparseStart;
jmethodID NNDataSet_getSparseEnd;
jmethodID NNDataSet_getSparseIndex;
jmethodID NNDataSet_getData;

jmethodID OutputNNDataSet_getName;
jmethodID OutputNNDataSet_getLayerName;
jmethodID OutputNNDataSet_getIndexes;
jmethodID OutputNNDataSet_getScores;

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
        NNDataSet_getLayerName = findMethodId(env, REFS, _NNDataSet, "getLayerName", "()Ljava/lang/String;");
        NNDataSet_getAttribute = findMethodId(env, REFS, _NNDataSet, "getAttribute", "()I");
        NNDataSet_getDataTypeOrdinal = findMethodId(env, REFS, _NNDataSet, "getDataTypeOrdinal", "()I");
        NNDataSet_getDimensions = findMethodId(env, REFS, _NNDataSet, "getDimensions", "()I");
        NNDataSet_getDimX = findMethodId(env, REFS, _NNDataSet, "getDimX", "()I");
        NNDataSet_getDimY = findMethodId(env, REFS, _NNDataSet, "getDimY", "()I");
        NNDataSet_getDimZ = findMethodId(env, REFS, _NNDataSet, "getDimZ", "()I");
        NNDataSet_getExamples = findMethodId(env, REFS, _NNDataSet, "getExamples", "()I");
        NNDataSet_getStride = findMethodId(env, REFS, _NNDataSet, "getStride", "()I");
        NNDataSet_getSparseStart = findMethodId(env, REFS, _NNDataSet, "getSparseStart", "()[J");
        NNDataSet_getSparseEnd = findMethodId(env, REFS, _NNDataSet, "getSparseEnd", "()[J");
        NNDataSet_getSparseIndex = findMethodId(env, REFS, _NNDataSet, "getSparseIndex", "()[J");
        NNDataSet_getData = findMethodId(env, REFS, _NNDataSet, "getData", "()Ljava/nio/ByteBuffer;");

        OutputNNDataSet_getName = findMethodId(env, REFS, _OutputNNDataSet, "getName", "()Ljava/lang/String;");
        OutputNNDataSet_getLayerName = findMethodId(env, REFS, _OutputNNDataSet, "getLayerName", "()Ljava/lang/String;");
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
    DsstneContext *dc = new DsstneContext(networkFileName, batchSize);
    env->ReleaseStringUTFChars(jNetworkFileName, networkFileName);
    return (jlong) dc;
}

NNDataSetDimensions getDataDimensions(JNIEnv *env, jobject jDataset)
{
    NNDataSetDimensions dim;
    dim._width = env->CallIntMethod(jDataset, NNDataSet_getDimX);
    dim._length = env->CallIntMethod(jDataset, NNDataSet_getDimY);
    dim._height = env->CallIntMethod(jDataset, NNDataSet_getDimZ);
    dim._dimensions = env->CallIntMethod(jDataset, NNDataSet_getDimensions);
    return dim;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_loadDatasets(JNIEnv *env, jclass clazz, jlong ptr,
                                                                  jobjectArray jDatasets)
{
    using NNDataSetEnums::DataType;

    jsize len = env->GetArrayLength(jDatasets);
    vector<NNDataSetDescriptor> datasetDescriptors;

    for (jsize i = 0; i < len; ++i)
    {
        jobject jDataset = env->GetObjectArrayElement(jDatasets, i);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jDataset, NNDataSet_getDataTypeOrdinal));
        jstring jName = (jstring) env->CallObjectMethod(jDataset, NNDataSet_getName);
        const char *name = env->GetStringUTFChars(jName, NULL);
        jint attributes = env->CallIntMethod(jDataset, NNDataSet_getAttribute);
        int examples = env->CallIntMethod(jDataset, NNDataSet_getExamples);
        int stride = env->CallIntMethod(jDataset, NNDataSet_getStride);
        NNDataSetDimensions dim = getDataDimensions(env, jDataset);

        // for dense data x*y*z == stride, for sparse data x*y*z*sparseDensity == stride
        float sparseDensity = ((double) (dim._width * dim._length * dim._height)) / (double) stride;

        NNDataSetDescriptor descriptor;
        descriptor._name = name;
        descriptor._attributes = attributes;
        descriptor._dataType = dataType;
        descriptor._dim = dim;
        descriptor._examples = examples;
        descriptor._sparseDensity = sparseDensity;

        datasetDescriptors.push_back(descriptor);
        env->ReleaseStringUTFChars(jName, name);
    }

    DsstneContext *dc = DsstneContext::fromPtr(ptr);
    dc->initInputLayerDataSets(datasetDescriptors);
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
    DsstneContext *dc = DsstneContext::fromPtr(ptr);
    delete dc;
}

JNIEXPORT jobject JNICALL Java_com_amazon_dsstne_Dsstne_get_1layers(JNIEnv *env, jclass clazz, jlong ptr,
                                                                    jint kindOrdinal)
{
    DsstneContext *dc = DsstneContext::fromPtr(ptr);
    NNNetwork *network = dc->getNetwork();
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

        jobject jInputLayer = newObject(env, REFS, _NNLayer, NNLayer_, jName, jDatasetName, kind, attributes, numDim,
                                        lx, ly, lz);

        env->CallBooleanMethod(jLayers, java_ArrayList_add, jInputLayer);
    }
    return jLayers;
}

void checkDataset(JNIEnv *env, NNDataSetBase *dstDataset, uint32_t attribute, DataType dataType,
                  const NNDataSetDimensions &dim, uint32_t examples)
{
    if(dstDataset->_attributes != attribute)
    {
        throwJavaException(env, IllegalArgumentException, "Attribute mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if(dstDataset->_dataType != dataType)
    {
        throwJavaException(env, IllegalArgumentException, "Data type mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if(dstDataset->_dimensions != dim._dimensions)
    {
        throwJavaException(env, IllegalArgumentException, "Dimension mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if (dstDataset->_width != dim._width)
    {
        throwJavaException(env, IllegalArgumentException, "Width mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if (dstDataset->_length != dim._length)
    {
        throwJavaException(env, IllegalArgumentException, "Length mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if (dstDataset->_height != dim._height)
    {
        throwJavaException(env, IllegalArgumentException, "Height mismatch in dataset %s", dstDataset->_name.c_str());
    }
    if(dstDataset->_examples != examples)
    {
        throwJavaException(env, IllegalArgumentException, "Examples mismatch in dataset %s", dstDataset->_name.c_str());
    }
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_predict(JNIEnv *env, jclass clazz, jlong ptr, jint k,
                                                             jobjectArray jInputDatasets, jobjectArray jOutputDatasets)
{
    DsstneContext *dc = DsstneContext::fromPtr(ptr);
    NNNetwork *network = dc->getNetwork();

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
        jstring jLayerName = (jstring) env->CallObjectMethod(jInputDataset, NNDataSet_getLayerName);
        const char *datasetName = env->GetStringUTFChars(jDatasetName, NULL);
        const char *layerName = env->GetStringUTFChars(jLayerName, NULL);
        uint32_t examples = env->CallIntMethod(jInputDataset, NNDataSet_getExamples);
        NNDataSetDimensions dim = getDataDimensions(env, jInputDataset);
        uint32_t attribute = env->CallIntMethod(jInputDataset, NNDataSet_getAttribute);
        DataType dataType = static_cast<DataType>(env->CallIntMethod(jInputDataset, NNDataSet_getDataTypeOrdinal));

        const NNLayer *layer = network->GetLayer(layerName);
        if (!layer)
        {
            throwJavaException(env, IllegalArgumentException, "No matching layer found in network %s for dataset: %s",
                               network->GetName(), datasetName);
        }

        NNDataSetBase *dstDataset = layer->GetDataSet();
        checkDataset(env, dstDataset, attribute, dataType, dim, examples);

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

            dstDataset->SetSparseData((uint64_t*) sparseStart, (uint64_t*) sparseEnd, srcDataNative,
                                      (uint32_t*) sparseIndex);
            env->ReleaseLongArrayElements(jSparseStart, sparseStart, JNI_ABORT);
            env->ReleaseLongArrayElements(jSparseEnd, sparseEnd, JNI_ABORT);
            env->ReleaseLongArrayElements(jSparseIndex, sparseIndex, JNI_ABORT);
        } else
        {
            /* copy dense data */
            dstDataset->SetData(srcDataNative);
        }

        env->ReleaseStringUTFChars(jDatasetName, datasetName);
        env->ReleaseStringUTFChars(jLayerName, layerName);
    }

    // there is only one batch in the dataset; always start from position 0
    network->SetPosition(0);
    network->PredictBatch();

    jsize outputLen = env->GetArrayLength(jOutputDatasets);
    for (jsize i = 0; i < outputLen; ++i)
    {
        jobject jOutputDataset = env->GetObjectArrayElement(jOutputDatasets, i);

        // output dataset name is set to output layer name
        jstring jLayerName = (jstring) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getLayerName);
        const char *layerName = env->GetStringUTFChars(jLayerName, NULL);

        jfloatArray jScores = (jfloatArray) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getScores);
        jlongArray jIndexes = (jlongArray) env->CallObjectMethod(jOutputDataset, OutputNNDataSet_getIndexes);

        NNLayer *outputLayer = network->GetLayer(layerName);

        uint32_t x, y, z, w;
        tie(x, y, z, w) = outputLayer->GetDimensions();
        uint32_t stride = x * y * z;

        float *scores = (float*) env->GetPrimitiveArrayCritical(jScores, NULL);

        if (k > 0)
        {
            NNFloat *outputUnitBuffer = network->GetUnitBuffer(layerName);
            long *indexes = (long*) env->GetPrimitiveArrayCritical(jIndexes, NULL);
            NNFloat *dScores;
            uint32_t *dIndexes;
            uint32_t *hIndexes = (uint32_t*) calloc(k * batchSize, sizeof(uint32_t));
            cudaMalloc(&dScores, k * batchSize * sizeof(NNFloat));
            cudaMalloc(&dIndexes, k * batchSize * sizeof(uint32_t));

            kCalculateTopK(outputUnitBuffer, dScores, dIndexes, batchSize, stride, k);

            cudaMemcpy(scores, dScores, k * batchSize * sizeof(NNFloat), cudaMemcpyDeviceToHost);
            cudaMemcpy(hIndexes, dIndexes, k * batchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaFree(dScores);
            cudaFree(dIndexes);

            for (size_t i = 0; i < k * batchSize; ++i)
            {
                indexes[i] = (long) hIndexes[i];
            }
            free(hIndexes);

            env->ReleasePrimitiveArrayCritical(jIndexes, indexes, 0);
        } else
        {
            // return the entire output layer, no need to set indexes since we are returning the whole thing
            outputLayer->GetUnits((NNFloat*) scores);
        }
        env->ReleasePrimitiveArrayCritical(jScores, scores, 0);
        env->ReleaseStringUTFChars(jLayerName, layerName);
    }
}

