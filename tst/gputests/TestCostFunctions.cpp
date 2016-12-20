// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"
// STL
#include <string>

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

class TestCostFunctions: public CppUnit::TestFixture {
public:
    // Interface
    void testCostFunctions() {
        // Initialize GPU
        getGpu().SetRandomSeed(12345);
        getGpu().CopyConstants();

        // test data scaled marginal cross entropy
        {
            const size_t batch = 2;
            const string modelPath = std::string(TEST_DATA_PATH) + "validate_DataScaledMarginalCrossEntropy_02.json";
            DataParameters dataParameters;
            dataParameters.numberOfSamples = 1024;
            dataParameters.inpFeatureDimensionality = 2;
            dataParameters.outFeatureDimensionality = 2;
            bool result = validateNeuralNetwork(batch, modelPath, ClassificationAnalog, dataParameters, std::cout);
            CPPUNIT_ASSERT_MESSAGE("failed on DataScaledMarginalCrossEntropy", result);
        }

        // test marginal cross entropy
        {
            const size_t batch = 4;
            const string modelPath = std::string(TEST_DATA_PATH) + "validate_ScaledMarginalCrossEntropy_02.json";
            DataParameters dataParameters;
            dataParameters.numberOfSamples = 1024;
            dataParameters.inpFeatureDimensionality = 2;
            dataParameters.outFeatureDimensionality = 2;
            bool result = validateNeuralNetwork(batch, modelPath, Classification, dataParameters, std::cout);
            CPPUNIT_ASSERT_MESSAGE("failed on DataScaledMarginalCrossEntropy", result);
        }

        // test L2
        {
            const size_t batch = 4;
            const string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_02.json";
            DataParameters dataParameters;
            dataParameters.numberOfSamples = 1024;
            dataParameters.inpFeatureDimensionality = 1;
            dataParameters.outFeatureDimensionality = 1;
            dataParameters.W0 = -2.f;
            dataParameters.B0 = 3.f;
            bool result = validateNeuralNetwork(batch, modelPath, Regression, dataParameters, std::cout);
            CPPUNIT_ASSERT_MESSAGE("failed on DataScaledMarginalCrossEntropy", result);
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestCostFunctions);
    CPPUNIT_TEST(testCostFunctions);
    CPPUNIT_TEST_SUITE_END();
};
