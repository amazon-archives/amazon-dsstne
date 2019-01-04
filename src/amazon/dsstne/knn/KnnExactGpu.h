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

#ifndef LIBKNN_KNN_EXACT_GPU_H_
#define LIBKNN_KNN_EXACT_GPU_H_

#include "KnnData.h"

namespace astdl
{
namespace knn
{
class Knn
{
  public:

    virtual void search(int k, const float *inputs, std::string *keys, float *scores) = 0;

    virtual ~Knn()
    {

    }

  protected:
    KnnData *data;

    Knn(KnnData *data);
};

class KnnExactGpu: public Knn
{
  public:
    KnnExactGpu(KnnData *data);

    void search(int k, const float *inputs, int size, std::string *keys, float *scores);

    void search(int k, const float *inputs, std::string *keys, float *scores)
    {
        search(k, inputs, data->batchSize, keys, scores);
    }

};

/**
 * merges the top k results from each of the GPUs.
 *   batchSize - number of rows in g_scores[i], and g_indexes[i]
 *    k - number of columns in g_scores[i] and g_indexes[i]
 *    g_scores - top k scores (k x batchSize) from each GPU
 *    g_indexes - top k indexes (k x batchSize) from each GPU
 *    scores - where to store the merged top k scores (must malloc k x batchSize x sizeof(float))
 *    indexes - where to store the merged top k indexes (must malloc k x batchSize x sizeof(float))
 */
void mergeKnn(int k, int batchSize, int width, int numGpus, const std::vector<float*> &allScores,
    const std::vector<uint32_t*> &allIndexes, const std::vector<std::vector<std::string>> &allKeys, float *scores,
    std::string *keys);
} // namespace knn
} // namespace astdl

#endif /* LIBKNN_KNN_EXACT_GPU_H_ */
