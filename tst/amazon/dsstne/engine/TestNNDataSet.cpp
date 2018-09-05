#include <cppunit/extensions/HelperMacros.h>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"

class TestNNDataSet : public CppUnit::TestFixture
{

CPPUNIT_TEST_SUITE(TestNNDataSet);

    CPPUNIT_TEST(testCreateDenseDataset);
    CPPUNIT_TEST(testCreateDenseIndexedDataset);
    CPPUNIT_TEST_SUITE_END()
    ;

 public:

    void testCreateDenseDataset()
    {
        NNLayerDescriptor layerDescriptor;
        layerDescriptor._dimensions = 1;
        layerDescriptor._Nx = 128;
        layerDescriptor._Ny = 1;
        layerDescriptor._Nz = 1;
        layerDescriptor._Nw = 1;
        const NNLayer layer(layerDescriptor, 32);
        NNDataSet<int> dataset(32, layer);

//        uint32_t dp = dataset.GetDataPoint(0, 1, 0, 0);
//        std::cout << "data point = " << dp << std::endl;
        CPPUNIT_ASSERT(dataset._stride == 128);
        CPPUNIT_ASSERT(dataset._bIndexed == false);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(dataset._attributes == NNDataSetEnums::None);
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::UInt);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == 32);
        CPPUNIT_ASSERT(dataset._uniqueExamples == 32);
        CPPUNIT_ASSERT(dataset._localExamples == 32);
    }

    void testCreateDenseIndexedDataset()
    {
        NNLayerDescriptor layerDescriptor;
        layerDescriptor._dimensions = 1;
        layerDescriptor._Nx = 128;
        layerDescriptor._Ny = 1;
        layerDescriptor._Nz = 1;
        layerDescriptor._Nw = 1;
        const NNLayer layer(layerDescriptor, 32);

        uint32_t examples = 32;
        uint32_t uniqueExamples = 16;

        NNDataSet<uint32_t> dataset(examples, uniqueExamples, layer);

        CPPUNIT_ASSERT(dataset._stride == 128);
        CPPUNIT_ASSERT(dataset._bIndexed == true);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(dataset._attributes == NNDataSetEnums::Indexed);
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::UInt);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == examples);
        CPPUNIT_ASSERT(dataset._uniqueExamples == uniqueExamples);
        CPPUNIT_ASSERT(dataset._localExamples == examples);
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestNNDataSet);
