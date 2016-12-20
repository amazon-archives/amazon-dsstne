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

using namespace std;

//----------------------------------------------------------------------------
class TestActivationFunctions: public CppUnit::TestFixture {
public:
    // Interface
    void testActivationFunctions() {
        // Initialize GPU
        //getGpu().SetRandomSeed(12345);
        //getGpu().CopyConstants();

        const size_t numberTests = 2;
        const string modelPaths[numberTests] = {
                        std::string(TEST_DATA_PATH) + std::string("validate_L2_LRelu_01.json"),
                        std::string(TEST_DATA_PATH) + std::string("validate_L2_LRelu_02.json")};
        const size_t batches[numberTests] =  {2,  2};

        for (size_t i = 0; i < numberTests; i++) {
            DataParameters dataParameters;
            dataParameters.numberOfSamples = 1024;
            dataParameters.inpFeatureDimensionality = 2;
            dataParameters.outFeatureDimensionality = 2;
            bool result = validateNeuralNetwork(batches[i], modelPaths[i], Classification, dataParameters, std::cout);
            cout << "batches " << batches[i] <<  ", model " << modelPaths[i] << endl;
            CPPUNIT_ASSERT_MESSAGE("failed on testActivationFunctions", result);
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestActivationFunctions);
    CPPUNIT_TEST(testActivationFunctions);
    CPPUNIT_TEST_SUITE_END();
};
