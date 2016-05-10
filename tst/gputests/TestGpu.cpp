#include <string>

// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"

#include "filterKernels.h"

class TestGpu : public CppUnit::TestFixture {

public:
    TestGpu() {
        getGpu().Startup(0, NULL);
    }

    ~TestGpu() {
        getGpu().Shutdown();
    }

    void TestApplyNodeFilter() {
        int outputKeySize = 6, filterSize = 3;
        NNFloat localOutputKey[outputKeySize] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        NNFloat expectedOutputKey[outputKeySize] = {7.0, 16.0, 27.0, 28.0, 40.0, 54.0};
        NNFloat localFilter[filterSize] = {7.0, 8.0, 9.0};
        NNFloat expectedFilter[filterSize] = {7.0, 8.0, 9.0};
        GpuBuffer<NNFloat> *deviceOutputKey = new GpuBuffer<NNFloat>(outputKeySize);
        GpuBuffer<NNFloat> *deviceFilter = new GpuBuffer<NNFloat>(filterSize);

        deviceOutputKey->Upload(localOutputKey);
        deviceFilter->Upload(localFilter);

        kApplyNodeFilter(deviceOutputKey->_pDevData, deviceFilter->_pDevData, filterSize, 2);

        deviceOutputKey->Download(localOutputKey);
        deviceFilter->Download(localFilter);

        delete deviceOutputKey;
        delete deviceFilter;

        for (int i = 0; i < outputKeySize; ++i) {
            CPPUNIT_ASSERT_EQUAL_MESSAGE("OutputKey is different", expectedOutputKey[i], localOutputKey[i]);
        }

        for (int i = 0; i < filterSize; ++i) {
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Filter is different", expectedFilter[i], localFilter[i]);
        }
    }

    CPPUNIT_TEST_SUITE(TestGpu);
    CPPUNIT_TEST(TestApplyNodeFilter);
    CPPUNIT_TEST_SUITE_END();
};
