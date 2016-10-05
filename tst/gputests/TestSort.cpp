// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"
// STL
#include <string>

<<<<<<< Updated upstream
#include "Utils.h"


=======
#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"
#include "Utils.h"


using namespace std;
>>>>>>> Stashed changes

void randData(NNFloat* pTarget, NNFloat* pOut, const size_t batch, const size_t nFeatures, const size_t stride) {
  memset(pTarget, 0, stride * batch * sizeof(NNFloat));
  memset(pOut, 0, stride * batch * sizeof(NNFloat));
  for (size_t i = 0; i < batch; i++) {

    for (size_t k = 0; k < nFeatures; k++) {
      pTarget[k] = rand(0, nFeatures - 1);
    }

    for (size_t o = 0; o < nFeatures; o++) {
      pOut[o] = rand(0.f, 1.f);
    }

    pTarget += stride;
    pOut += stride;
  }
}

bool testTopK(const size_t batch = 128, const size_t topK = 128, const size_t nFeatures = 1024) {

  cout << "TEST kCalculateTopK with parameters: " << "batch=" << batch << " topK=" << topK << " nFeatures=" << nFeatures << endl;
  bool ret = true;

  const float EPS = 1.e-6;
  const size_t STRIDE = ((nFeatures + 127) >> 7) << 7;
  timeval t0, t1;

  // allocate memory same way with main engine
  GpuBuffer<NNFloat>* pbKey = new GpuBuffer<NNFloat>(batch * topK, true);
  GpuBuffer<NNFloat>* pbFValue = new GpuBuffer<NNFloat>(batch * topK, true);
  GpuBuffer<unsigned int>* pbUIValue = new GpuBuffer<unsigned int>(batch * topK, true);

  GpuBuffer<NNFloat>* pbTarget = new GpuBuffer<NNFloat>(batch * STRIDE, true);
  GpuBuffer<NNFloat>* pbOutput = new GpuBuffer<NNFloat>(batch * STRIDE, true);

  cout << "1 TEST kCalculateTopK with 3 args" << endl;

  // prepare data
  {
    NNFloat* pTarget = pbTarget->_pSysData;
    NNFloat* pOut = pbOutput->_pSysData; // data from output layer (scores)

    randData(pTarget, pOut, batch, nFeatures, STRIDE);

    pbTarget->Upload(); // desired output
    pbOutput->Upload(); // copy of output layer + filtering

    memset(pbUIValue->_pSysData, 0, batch * topK * sizeof(unsigned int));
    pbUIValue->Upload();
  }
  // run test 1
  gettimeofday(&t0, NULL);
  kCalculateTopK(pbOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, batch, STRIDE, topK);
  gettimeofday(&t1, NULL);
  cout << "GPU sort: " << elapsed_time(t1, t0) << endl;

  //validate data
  {
    pbOutput->Download();
    pbTarget->Download();
    pbKey->Download();
    pbFValue->Download();
    pbUIValue->Download();

    vector<float> keys(nFeatures);
    vector<unsigned int> topK_vals(topK);
    vector<float> topK_keys(topK);

    NNFloat* pOutput = pbOutput->_pSysData;
    NNFloat* pKey = pbKey->_pSysData;
    unsigned int* pUIValue = pbUIValue->_pSysData;

    int countValueError = 0;
    float sumKeyError = 0.f;
    float cpuSort = 0.f;

    for (size_t i = 0; i < batch; i++) {

      gettimeofday(&t0, NULL);
      topKsort<NNFloat, unsigned int>(pOutput, NULL, nFeatures, &topK_keys[0], &topK_vals[0], topK);
      gettimeofday(&t1, NULL);
      cpuSort += elapsed_time(t1, t0);

      for (size_t k = 0; k < topK; k++) {
        unsigned int GPUvalue = pUIValue[k]; // index
        float GPUkey = pKey[k]; // score

        float CPUvalue = topK_vals[k];
        float CPUkey = topK_keys[k];

        if (fabs(GPUvalue - CPUvalue) > EPS) {
          countValueError++;
        }
        sumKeyError += fabs(GPUkey - CPUkey);
      }
      pKey += topK;
      pUIValue += topK;
      pOutput += STRIDE;
    }
    cout << "CPU sort: " << cpuSort << endl;

    if (countValueError && sumKeyError) {
      cout << "1 ERROR kCalculateTopK with 3 args; ";
      ret = false;
    } else {
      cout << "1 PASS kCalculateTopK with 3 args; "; // some error is accepted because bitonic sort belongs to not stable sorting
    }
    cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << endl;
  }

  cout << "2 TEST kCalculateTopK with 4 args" << endl;

  // prepare data
  {
    NNFloat* pTarget = pbTarget->_pSysData;
    NNFloat* pOut = pbOutput->_pSysData; // data from output layer (scores)

    randData(pTarget, pOut, batch, nFeatures, STRIDE);

    pbTarget->Upload(); // desired output
    pbOutput->Upload(); // copy of output layer + filtering
  }

  //run test
  kCalculateTopK(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, topK);

  //validate data
  {
    pbOutput->Download();
    pbTarget->Download();
    pbKey->Download();
    pbFValue->Download();

    vector<float> vals(nFeatures);
    vector<float> keys(nFeatures);
    vector<float> topK_vals(topK);
    vector<float> topK_keys(topK);

    NNFloat* pOutput = pbOutput->_pSysData;
    NNFloat* pTarget = pbTarget->_pSysData;
    NNFloat* pKey = pbKey->_pSysData;
    NNFloat* pValue = pbFValue->_pSysData;

    int countValueError = 0;
    float sumKeyError = 0;

    for (size_t i = 0; i < batch; i++) {

      topKsort<NNFloat, NNFloat>(pOutput, pTarget, nFeatures, &topK_keys[0], &topK_vals[0], topK);

      for (size_t k = 0; k < topK; k++) {
        unsigned int GPUvalue = pValue[k]; // index
        float GPUkey = pKey[k]; // score

        float CPUvalue = topK_vals[k];
        float CPUkey = topK_keys[k];

        if (fabs(GPUvalue - CPUvalue) > EPS) {
          countValueError++;
        }
        sumKeyError += fabs(GPUkey - CPUkey);
      }
      pKey += topK;
      pValue += topK;
      pOutput += STRIDE;
      pTarget += STRIDE;
    }

    if (countValueError && sumKeyError) {
      cout << "2 ERROR kCalculateTopK with 4 args; ";
      ret = false;
    } else {
      cout << "2 PASS kCalculateTopK with 4 args; ";
    }
    cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << endl;
  }

  int totalGPUMemory;
  int totalCPUMemory;
  getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
  cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << endl;
  cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << endl;

  delete pbKey;
  delete pbFValue;
  delete pbTarget;
  delete pbOutput;
  delete pbUIValue;

  return ret;
}

//----------------------------------------------------------------------------
class TestSort : public CppUnit::TestFixture
{
public:             // Interface
    void            TestCPU_GPUSort()
    {
      // Initialize GPU
      getGpu().SetRandomSeed(12345);
      getGpu().CopyConstants();
      {
        const size_t BATCH = 128;
        const size_t TOP_K = 128;
        const size_t N_FEATURES = 1024;
        bool result = testTopK(BATCH, TOP_K, N_FEATURES);
        CPPUNIT_ASSERT_MESSAGE("failed with N_FEATURES = 1024, TOP_K = 128", result);
      }
      {
        const size_t BATCH = 128;
        const size_t TOP_K = 128;
        const size_t N_FEATURES = 100000;
        bool result = testTopK(BATCH, TOP_K, N_FEATURES);
        CPPUNIT_ASSERT_MESSAGE("failed with N_FEATURES = 100000, TOP_K = 128", result);
      }
      {
        const size_t BATCH = 128;
        const size_t TOP_K = 64;
        const size_t N_FEATURES = 1024;
        bool result = testTopK(BATCH, TOP_K, N_FEATURES);
        CPPUNIT_ASSERT_MESSAGE("failed with N_FEATURES = 1024, TOP_K = 64", result);
      }
      {
        const size_t BATCH = 128;
        const size_t TOP_K = 32;
        const size_t N_FEATURES = 64;
        bool result = testTopK(BATCH, TOP_K, N_FEATURES);
        CPPUNIT_ASSERT_MESSAGE("failed with N_FEATURES = 64, TOP_K = 32", result);
      }
      {
        const size_t BATCH = 128;
        const size_t TOP_K = 1;
        const size_t N_FEATURES = 64;
        bool result = testTopK(BATCH, TOP_K, N_FEATURES);
        CPPUNIT_ASSERT_MESSAGE("failed with N_FEATURES = 64, TOP_K = 32", result);
      }
      //getGpu().Shutdown();
    }
    
public:
    CPPUNIT_TEST_SUITE(TestSort);
    CPPUNIT_TEST(TestCPU_GPUSort);
    CPPUNIT_TEST_SUITE_END();
    
};
