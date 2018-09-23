#include <cppunit/extensions/HelperMacros.h>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"

class TestNNDataSet : public CppUnit::TestFixture
{

CPPUNIT_TEST_SUITE(TestNNDataSet);

    CPPUNIT_TEST(testCreateDenseDataset);
    CPPUNIT_TEST(testCreateDenseIndexedDataset);
    CPPUNIT_TEST(testCreateSparseDataset);
    CPPUNIT_TEST(testCreateSparseWeightedDataset);
    CPPUNIT_TEST(testCreateSparseIndexedDataset);
    CPPUNIT_TEST(testCreateSparseWeightedIndexedDataset);

    CPPUNIT_TEST(testSetDenseData);
    CPPUNIT_TEST_EXCEPTION(testSetDenseData_Overflow, std::length_error);
    CPPUNIT_TEST_EXCEPTION(testSetDenseData_OnSparseDataset, std::runtime_error);

    CPPUNIT_TEST(testSetSparseData);
    CPPUNIT_TEST_EXCEPTION(testSetSparseData_Overflow, std::length_error);
    CPPUNIT_TEST_EXCEPTION(testSetSparseData_OnDenseDataset, std::runtime_error);

    CPPUNIT_TEST(testSetIndexedData);
    CPPUNIT_TEST_EXCEPTION(testSetIndexedData_Overflow, std::length_error);
    CPPUNIT_TEST_EXCEPTION(testSetIndexedData_OnNotIndexedDataset, std::runtime_error);

    CPPUNIT_TEST(testSetDataWeights);
    CPPUNIT_TEST_EXCEPTION(testSetDataWeights_Overflow, std::length_error);
    CPPUNIT_TEST_EXCEPTION(testSetDataWeights_OnNotWeightedDataset, std::runtime_error);

    CPPUNIT_TEST(testNNDataSetTypes);

    CPPUNIT_TEST_SUITE_END();

 private:
    NNDataSetDimensions datasetDim;
    uint32_t examples = 32;
    uint32_t uniqueExamples = 16;
    double sparseDensity = 0.1;

 public:

    void setUp()
    {
        datasetDim._dimensions = 1;
        datasetDim._width = 128;
        datasetDim._height = 1;
        datasetDim._length = 1;
    }

    void testCreateDenseDataset()
    {
        NNDataSet<uint32_t> dataset(32, datasetDim);

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
        CPPUNIT_ASSERT(dataset._sparseDataSize == 0);
    }

    void testCreateDenseIndexedDataset()
    {
        NNDataSet<uint32_t> dataset(examples, uniqueExamples, datasetDim);

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
        CPPUNIT_ASSERT(dataset._sparseDataSize == 0);
    }

    void testCreateSparseDataset()
    {
        bool isWeighted = false;
        NNDataSet<int> dataset(examples, sparseDensity, datasetDim, isWeighted);

        CPPUNIT_ASSERT(dataset._stride == 0);
        CPPUNIT_ASSERT(dataset._bIndexed == false);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(dataset._attributes == NNDataSetEnums::Sparse);
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::Int);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == examples);
        CPPUNIT_ASSERT(dataset._uniqueExamples == examples);
        CPPUNIT_ASSERT(dataset._localExamples == examples);
        CPPUNIT_ASSERT(dataset._sparseDataSize == (uint64_t ) (128.0 * 0.1 * 32.0));
    }

    void testCreateSparseWeightedDataset()
    {
        bool isWeighted = true;
        NNDataSet<int> dataset(examples, sparseDensity, datasetDim, isWeighted);

        CPPUNIT_ASSERT(dataset._stride == 0);
        CPPUNIT_ASSERT(dataset._bIndexed == false);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(dataset._attributes == (NNDataSetEnums::Sparse | NNDataSetEnums::Weighted));
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::Int);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == examples);
        CPPUNIT_ASSERT(dataset._uniqueExamples == examples);
        CPPUNIT_ASSERT(dataset._localExamples == examples);
        CPPUNIT_ASSERT(dataset._sparseDataSize == (uint64_t ) (128.0 * 0.1 * 32.0));
    }

    void testCreateSparseIndexedDataset()
    {
        size_t sparseDataSize = 128 * uniqueExamples / 10;
        bool isWeighted = false;
        NNDataSet<long> dataset(examples, uniqueExamples, sparseDataSize, datasetDim, isWeighted);

        CPPUNIT_ASSERT(dataset._stride == 0);
        CPPUNIT_ASSERT(dataset._bIndexed == true);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(dataset._attributes == (NNDataSetEnums::Sparse | NNDataSetEnums::Indexed));
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::LLInt);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == examples);
        CPPUNIT_ASSERT(dataset._uniqueExamples == uniqueExamples);
        CPPUNIT_ASSERT(dataset._localExamples == examples);
        CPPUNIT_ASSERT(dataset._sparseDataSize == sparseDataSize);
    }

    void testCreateSparseWeightedIndexedDataset()
    {
        size_t sparseDataSize = 128 * uniqueExamples / 10;
        bool isWeighted = true;
        NNDataSet<long> dataset(examples, uniqueExamples, sparseDataSize, datasetDim, isWeighted);

        CPPUNIT_ASSERT(dataset._stride == 0);
        CPPUNIT_ASSERT(dataset._bIndexed == true);
        CPPUNIT_ASSERT(dataset._bStreaming == false);
        CPPUNIT_ASSERT(dataset._dimensions == 1);
        CPPUNIT_ASSERT(
            dataset._attributes == (NNDataSetEnums::Sparse | NNDataSetEnums::Indexed | NNDataSetEnums::Weighted));
        CPPUNIT_ASSERT(dataset._dataType == NNDataSetEnums::DataType::LLInt);
        CPPUNIT_ASSERT(dataset._width == 128);
        CPPUNIT_ASSERT(dataset._height == 1);
        CPPUNIT_ASSERT(dataset._length == 1);
        CPPUNIT_ASSERT(dataset._examples == examples);
        CPPUNIT_ASSERT(dataset._uniqueExamples == uniqueExamples);
        CPPUNIT_ASSERT(dataset._localExamples == examples);
        CPPUNIT_ASSERT(dataset._sparseDataSize == (uint64_t ) (128.0 * 0.1 * 16.0));
    }

    void testSetDenseData()
    {
        NNDataSet<uint32_t> dataset(examples, datasetDim);

        uint32_t srcData[128];
        for (int i = 0; i < 128; ++i)
        {
            srcData[i] = i;
        }

        dataset.SetData(srcData, 0, 128);
        for (int i = 0; i < 128; ++i)
        {
            CPPUNIT_ASSERT(dataset.GetDataPoint(0, i, 0, 0) == (uint32_t ) i);
        }
    }

    void testSetDenseData_Overflow()
    {
        NNDataSet<uint32_t> dataset(examples, datasetDim);

        uint32_t srcData[128];
        dataset.SetData(srcData, 32 * 128, 128);
    }

    void testSetDenseData_OnSparseDataset()
    {
        NNDataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, false);
        uint32_t srcData[128];
        dataset.SetData(srcData, 31 * 128, 128);
    }

    void testSetSparseData()
    {
        NNDataSet<NNFloat> dataset(examples, sparseDensity, datasetDim, false);

        NNDataSetDimensions dim = dataset.GetDimensions();
        size_t sparseDataSize = (size_t) (((double) dim._height * dim._width * dim._length) * examples * sparseDensity);
        uint64_t sparseStart[examples];
        uint64_t sparseEnd[examples];
        NNFloat sparseData[sparseDataSize];
        uint32_t sparseIndex[sparseDataSize];

        size_t sparseExampleSize = (sparseDataSize + examples - 1) / examples;

        sparseStart[0] = 0;
        sparseEnd[0] = sparseDataSize - (sparseExampleSize * (examples - 1));
        for (uint32_t i = 1; i < examples; i++)
        {
            sparseStart[i] = sparseEnd[i - 1];
            sparseEnd[i] = sparseStart[i] + sparseExampleSize;
        }

        // data: 1,2,3,....
        for (uint32_t i = 0; i < sparseDataSize; ++i)
        {
            sparseData[i] = i + 1;
        }

        // index: 0,1,2,...
        for (size_t i = 0; i < sparseDataSize; ++i)
        {
            sparseIndex[i] = i;
        }

        dataset.SetSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);

        CPPUNIT_ASSERT_EQUAL(sparseEnd[0], dataset.GetSparseDataPoints(0));
        for (uint32_t i = 0; i < sparseEnd[0]; ++i)
        {
            CPPUNIT_ASSERT_EQUAL((NNFloat ) i + 1, dataset.GetSparseDataPoint(0, i));
            CPPUNIT_ASSERT_EQUAL(i, dataset.GetSparseIndex(0, i));
        }
    }

    void testSetSparseData_Overflow()
    {
        NNDataSet<NNFloat> dataset(examples, sparseDensity, datasetDim, false);

        NNDataSetDimensions dim = dataset.GetDimensions();
        size_t sparseDataSize = (size_t) (((double) dim._height * dim._width * dim._length) * examples * sparseDensity);
        uint64_t sparseStart[examples];
        uint64_t sparseEnd[examples];
        NNFloat sparseData[1];
        uint32_t sparseIndex[1];
        sparseEnd[examples - 1] = sparseDataSize + 1;
        dataset.SetSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);
    }

    void testSetSparseData_OnDenseDataset()
    {
        NNDataSet<NNFloat> dataset(examples, datasetDim);
        uint64_t sparseStart[examples];
        uint64_t sparseEnd[examples];
        NNFloat sparseData[1];
        uint32_t sparseIndex[1];
        dataset.SetSparseData(sparseStart, sparseEnd, sparseData, sparseIndex);
    }

    void testSetIndexedData()
    {
        NNDataSet<NNFloat> dataset(examples, uniqueExamples, datasetDim);
        uint32_t indexedData[uniqueExamples];
        for (uint32_t i = 0; i < uniqueExamples; ++i)
        {
            indexedData[i] = i;
        }

        dataset.SetIndexedData(indexedData, 0, 16);
        dataset.SetIndexedData(indexedData, 16, 16);

        for (uint32_t i = 0; i < examples; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(i % uniqueExamples, dataset._vIndex[i]);
        }
    }

    void testSetIndexedData_Overflow()
    {
        NNDataSet<NNFloat> dataset(examples, uniqueExamples, datasetDim);
        uint32_t indexedData[examples + 1];
        dataset.SetIndexedData(indexedData, 0, examples + 1);
    }

    void testSetIndexedData_OnNotIndexedDataset()
    {
        NNDataSet<NNFloat> dataset(examples, datasetDim);
        uint32_t indexedData[examples];
        dataset.SetIndexedData(indexedData, 0, examples + 1);
    }

    void testSetDataWeights()
    {
        NNDataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, true);

        NNFloat dataWeights[examples];
        for (uint32_t i = 0; i < examples; ++i)
        {
            dataWeights[i] = (NNFloat) i;
        }

        dataset.SetDataWeight(dataWeights, 0, examples);

        for(uint32_t i = 0; i < examples; ++i)
        {
            CPPUNIT_ASSERT_EQUAL((NNFloat) i, dataset._vSparseWeight[i]);
        }
    }

    void testSetDataWeights_Overflow()
    {
        NNDataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, true);

        NNFloat dataWeights[examples * 2];
        dataset.SetDataWeight(dataWeights, 0, examples * 2);
    }

    void testSetDataWeights_OnNotWeightedDataset()
    {
        NNDataSet<uint32_t> dataset(examples, sparseDensity, datasetDim, false);

        NNFloat dataWeights[examples];
        dataset.SetDataWeight(dataWeights, 0, examples);
    }

    void testNNDataSetTypes()
    {
        // ensure that we can instantiate the expected data types
        NNDataSet<NNFloat> floatDataset(examples, datasetDim);
        NNDataSet<double> doubleDataset(examples, datasetDim);
        NNDataSet<unsigned char> unsignedCharDataset(examples, datasetDim);
        NNDataSet<char> charDataset(examples, datasetDim);
        NNDataSet<uint32_t> unsingedIntDataset(examples, datasetDim);
        NNDataSet<uint64_t> unsingedLongDataset(examples, datasetDim);
        NNDataSet<int32_t> intDataset(examples, datasetDim);
        NNDataSet<int64_t> longDataset(examples, datasetDim);
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestNNDataSet);
