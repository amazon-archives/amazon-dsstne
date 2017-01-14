// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"

// STL
#include <iostream>
#include <string>

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

class TestDotProduct: public CppUnit::TestFixture {

   void dotProductHelper(unsigned long randomSeed,
                         size_t xSize,
                         size_t ySize,
                         size_t zSize,
                         float ulp,
                         float randMin,
                         float randMax,
                         bool debugMode) {

        getGpu().SetRandomSeed(randomSeed);
        getGpu().CopyConstants();

        const size_t inputSize  = xSize * ySize * zSize;
        const size_t outputSize = xSize * ySize;

        GpuBuffer<NNFloat> output(outputSize, true);
        GpuBuffer<NNFloat> input1(inputSize, true);
        GpuBuffer<NNFloat> input2(inputSize, true);

        for (int i = 0; i < inputSize; i++) {
            input1._pSysData[i] = generateRandomNumber(randMin, randMax);
            input2._pSysData[i] = generateRandomNumber(randMin, randMax);
        }

        if (debugMode) {
            std::cout << "Input 1: ";
            for (int i = 0; i < inputSize; i++) {
                std::cout << input1._pSysData[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Input 2: ";
            for (int i = 0; i < inputSize; i++) {
                std::cout << input2._pSysData[i] << " ";
            }
            std::cout << std::endl;
        }

        input1.Upload();
        input2.Upload();

        kCalculateDotProduct(input1._pDevData, input2._pDevData, zSize, output._pDevData, outputSize);

        output.Download();

        if (debugMode) {
            std::cout << "Actual: ";
            for (int i = 0; i < outputSize; i++) {
                std::cout << output._pSysData[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Expected: ";
        }

        for (int i = 0; i < outputSize; i++) {
            float dotProduct = 0.f;
            for (int j = 0; j < zSize; j++) {
                const size_t pos = j + i * zSize;
                // When kernels.cu is compiled with the '-use_fast_math' option, the dot product calculation
                // will be performed using a fused multiply-add operation.
                dotProduct = std::fma(input1._pSysData[pos], input2._pSysData[pos], dotProduct);
            }
            if (debugMode) {
                std::cout << dotProduct << std::endl;
            }
            CPPUNIT_ASSERT(almostEqual(dotProduct, output._pSysData[i], 3));
        }

        if (debugMode) {
            std::cout << std::endl;
        }
    }

public:
    void testDotProduct() {
        dotProductHelper(12345, 1, 1, 1, 5, -1000.f, 1000.f, false);
        dotProductHelper(23456, 1, 1, 1, 5, -100.f, 100.f, false);
        dotProductHelper(34567, 1, 1, 1, 5, -10.f, 10.f, false);

        dotProductHelper(12345, 8, 8, 3, 5, -1000.f, 1000.f, false);
        dotProductHelper(23456, 8, 8, 3, 5, -100.f, 100.f, false);
        dotProductHelper(34567, 8, 8, 3, 5, -10.f, 10.f, false);

        dotProductHelper(12345, 256, 256, 10, 5, -1000.f, 1000.f, false);
        dotProductHelper(23456, 256, 256, 10, 5, -100.f, 100.f, false);
        dotProductHelper(34567, 256, 256, 10, 5, -10.f, 10.f, false);
    }

    CPPUNIT_TEST_SUITE(TestDotProduct);
    CPPUNIT_TEST(testDotProduct);
    CPPUNIT_TEST_SUITE_END();
};
