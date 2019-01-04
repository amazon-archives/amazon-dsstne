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

#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include "amazon/dsstne/knn/KnnData.h"
#include "amazon/dsstne/knn/DataReader.h"
#include "amazon/dsstne/knn/KnnExactGpu.h"
#include "com_amazon_dsstne_knn_KNearestNeighborsCuda.h"

static const char KNN_RESULT[] = "com/amazon/dsstne/knn/KnnResult";
static const char STRING[] = "java/lang/String";

static const char ILLEGAL_ARGUMENT_EXCEPTION[] = "java/lang/IllegalArgumentException";
static const char CLASS_NOT_FOUND_EXCEPTION[] = "java/lang/ClassNotFoundException";
static const char NO_SUCH_METHOD_EXCEPTION[] = "java/lang/NoSuchMethodException";
static const char FILE_NOT_FOUND_EXCEPTION[] = "java/io/FileNotFoundException";
static const char RUNTIME_EXCEPTION[] = "java/lang/RuntimeException";
static const char NULL_POINTER_EXCEPTION[] = "java/lang/NullPointerException";

static jclass JCLASS_KNN_RESULT;
static jmethodID JMETHODID_KNN_RESULT_CONSTRUCTOR;

static jclass JCLASS_STRING;

void throw_java_exception(JNIEnv *env, const char *exceptionType, const char *msg)
{
  jclass exc = env->FindClass(exceptionType);
  if (exc == 0)
  {
    // Default to a RuntimeException if the requested exception doesn't exist.
    exc = env->FindClass(RUNTIME_EXCEPTION);
  }
  env->ThrowNew(exc, msg);
}

void throw_java_exception(JNIEnv *env, const char *exceptionType, const std::string &msg)
{
  throw_java_exception(env, exceptionType, msg.c_str());
}

jclass find_class(JNIEnv *env, const char *className)
{
  jclass clazz = env->FindClass(className);
  if (clazz == NULL)
  {
    throw_java_exception(env, CLASS_NOT_FOUND_EXCEPTION, className);
  }
  return clazz;
}

jmethodID find_method_id(JNIEnv *env, const char* className, const char *methodName, const char *methodDescriptor)
{
  jclass clazz = find_class(env, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName, methodDescriptor);
  if (methodId == NULL)
  {
    std::stringstream msg;
    msg << className << "#" << methodName << methodDescriptor;
    throw_java_exception(env, NO_SUCH_METHOD_EXCEPTION, msg.str().c_str());
  }
  return methodId;
}

jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
  {
    return JNI_ERR;
  } else
  {
    // need to add cls as global ref since they are treated as local refs
    jclass localKnnResultClass = find_class(env, KNN_RESULT);
    jclass localStringClass = find_class(env, STRING);

    JCLASS_KNN_RESULT = (jclass) env->NewGlobalRef(localKnnResultClass);
    JCLASS_STRING = (jclass) env->NewGlobalRef(localStringClass);

    JMETHODID_KNN_RESULT_CONSTRUCTOR = find_method_id(env, KNN_RESULT, "<init>", "([Ljava/lang/String;[FI)V");

    env->DeleteLocalRef(localKnnResultClass);
    env->DeleteLocalRef(localStringClass);

    return JNI_VERSION_1_6;
  }
}

void JNI_OnUnload(JavaVM *vm, void *reserved)
{
  JNIEnv* env;
  if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
  {
    return;
  } else
  {
    if (JCLASS_KNN_RESULT != NULL)
    {
      env->DeleteGlobalRef(JCLASS_KNN_RESULT);
      env->DeleteGlobalRef(JCLASS_STRING);

      JCLASS_KNN_RESULT = NULL;
      JCLASS_STRING = NULL;
      JMETHODID_KNN_RESULT_CONSTRUCTOR = NULL;
    }
  }
}

astdl::knn::KnnData* checkKnnDataPointer(JNIEnv *env, jlong ptr)
{
  astdl::knn::KnnData *knnData = (astdl::knn::KnnData*) ptr;

  // allocate GPU memory
  if (knnData == nullptr)
  {
    throw_java_exception(env, NULL_POINTER_EXCEPTION, "null pointer passed as KnnData, call initialize first!");
  }
  return knnData;
}

JNIEXPORT jlong JNICALL Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_initialize(JNIEnv *env, jclass clazz,
  jint maxK, jint batchSize, jint numGpus, jint jDataType)
{
  astdl::knn::DataType dataType;
  switch (jDataType) {
    case 0:
      dataType = astdl::knn::DataType::FP32;
      break;
    case 1:
      dataType = astdl::knn::DataType::FP16;
      break;
    default:
      std::stringstream msg("Unknown data type [");
      msg << jDataType << "]";
      throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  astdl::knn::KnnData *knnData = new astdl::knn::KnnData(numGpus, batchSize, maxK, dataType);

  return (jlong) knnData;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_load
(JNIEnv *env, jclass clazz, jobjectArray jFnames, jintArray jDevices, jchar keyValDelim, jchar vecElemDelim, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize fnamesLen = env->GetArrayLength(jFnames);
  jsize devicesLen = env->GetArrayLength(jDevices);
  if (fnamesLen != devicesLen)
  {
    std::stringstream msg;
    msg << "filenames.length (" << fnamesLen << ") != devices.length (" << devicesLen << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  std::map<int, DataReader*> dataReaders;

  jint *devices = env->GetIntArrayElements(jDevices, NULL);
  for (int i = 0; i < fnamesLen; ++i)
  {
    jint device = devices[i];

    jstring jFname = (jstring) env->GetObjectArrayElement(jFnames, i);
    const char* fnamePtr = env->GetStringUTFChars(jFname, NULL);
    std::string fname(fnamePtr);
    env->ReleaseStringUTFChars(jFname, fnamePtr);

    DataReader *reader = new TextFileDataReader(fname, keyValDelim, vecElemDelim);

    dataReaders.insert(
      { device, reader});
  }

  // we only read from jDevices so pass JNI_ABORT to free devices without copying it back to jDevices
  env->ReleaseIntArrayElements(jDevices, devices, JNI_ABORT);

  // TODO catch runtime_error and throw RuntimeException
  knnData->load(dataReaders);

  for(auto const& entry: dataReaders)
  {
    delete entry.second;
  }
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  delete knnData;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J
(JNIEnv *env, jclass clazz, jint k, jfloatArray jInputVectors, jint size, jint width, jfloatArray jScores, jobjectArray jKeys, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);

  jsize length = env->GetArrayLength(jInputVectors);
  int batchSize = length / width;  // batchSize is expected to equal maxBatchSize

  if (length % width != 0)
  {
    std::stringstream msg;
    msg << "feature size (" << width << ") does not divide data length (" << length << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if(batchSize != knnData->batchSize)
  {
    std::stringstream msg;
    msg << "length of input vectors (" << length << ") / feature size (" << width << ") != batch size (" << knnData->batchSize << ")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  }

  if(size > batchSize)
  {
    std::stringstream msg;
    msg << "active batch size (" << size << ") must be less than or equal to batch size (" << batchSize <<")";
    throw_java_exception(env, ILLEGAL_ARGUMENT_EXCEPTION, msg.str());
  } else
  {
    batchSize = size;
  }

  std::string *keys = new std::string[k * batchSize];
  jfloat *inputVectors = (jfloat*) env->GetPrimitiveArrayCritical(jInputVectors, NULL);
  jfloat *scores = (jfloat*) env->GetPrimitiveArrayCritical(jScores, NULL);

  astdl::knn::KnnExactGpu knnCuda(knnData);

  knnCuda.search(k, inputVectors, batchSize, keys, scores);

  env->ReleasePrimitiveArrayCritical(jInputVectors, (void*) inputVectors, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(jScores, scores, 0);

  for (int i = 0; i < batchSize; ++i)
  {
    for (int j = 0; j < k; ++j)
    {
      jstring key = env->NewStringUTF(keys[i * k + j].c_str());
      env->SetObjectArrayElement(jKeys, i * k + j, key);
    }
  }

  delete[] (keys);
}

JNIEXPORT jobject JNICALL Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_findKnn__I_3FIIJ(JNIEnv *env, jclass clazz,
  jint k, jfloatArray jInputVectors, jint size, jint width, jlong ptr)
{
  astdl::knn::KnnData *knnData = checkKnnDataPointer(env, ptr);
  int batchSize = knnData->batchSize;

  jfloatArray jScores = env->NewFloatArray(k * size);
  jobjectArray jKeys = env->NewObjectArray(k * size, JCLASS_STRING, NULL);

  Java_com_amazon_dsstne_knn_KNearestNeighborsCuda_findKnn__I_3FII_3F_3Ljava_lang_String_2J(env, clazz, k,
    jInputVectors, size, width, jScores, jKeys, ptr);

  jobject knnResult = env->NewObject(JCLASS_KNN_RESULT, JMETHODID_KNN_RESULT_CONSTRUCTOR, jKeys, jScores, k);

  if (knnResult == NULL)
  {
    std::stringstream msg;
    msg << "Unable to create new object " << KNN_RESULT;
    throw_java_exception(env, RUNTIME_EXCEPTION, msg.str().c_str());
  }

  return knnResult;
}

