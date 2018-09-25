#include <cppunit/extensions/HelperMacros.h>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"

class TestNNDataSetDimensions : public CppUnit::TestFixture
{

CPPUNIT_TEST_SUITE(TestNNDataSetDimensions);

    CPPUNIT_TEST(testNumDimensions);

    CPPUNIT_TEST_SUITE_END();

 public:

    void testNumDimensions()
    {
        NNDataSetDimensions zero_d(1);
        NNDataSetDimensions one_d(2);
        NNDataSetDimensions two_d(2, 2);
        NNDataSetDimensions three_d(2, 2, 2);

        CPPUNIT_ASSERT_EQUAL(0U, zero_d._dimensions);
        CPPUNIT_ASSERT_EQUAL(1U, one_d._dimensions);
        CPPUNIT_ASSERT_EQUAL(2U, two_d._dimensions);
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestNNDataSetDimensions);
