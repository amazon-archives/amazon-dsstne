#ifndef NNENUM_H
#define NNENUM_H

namespace NNDataSetEnums
{

enum Attributes
{
  Sparse = 1,                 // Sparse dataset
  Boolean = 2,                // All datapoints are 0/1
  Compressed = 4,             // Data uses type-specific compression
  Recurrent = 8,              // Data has a time dimension
  Mutable = 16,               // Data can be modified by running network backwards
  SparseIgnoreZero = 32,      // Only calculate errors and deltas on non-zero values
  Indexed = 64,               // Data set is indexed
  Weighted = 128              // Data points have weight associated with them
};

enum Kind
{
  Numeric = 0,
  Image = 1,
  Audio = 2
};

enum Sharding
{
  None = 0,
  Model = 1,
  Data = 2,
};

enum DataType
{
  UInt = 0,
  Int = 1,
  LLInt = 2,
  ULLInt = 3,
  Float = 4,
  Double = 5,
  RGB8 = 6,
  RGB16 = 7,
  UChar = 8,
  Char = 9
};

/**
 * Template specialization to get DataType enum.
 * Support types listed in NNTypes.cpp#LoadNetCDF()
 */
template<typename T> DataType getDataType()
{
  throw std::runtime_error("Default data type not defined");
}

template<> DataType getDataType<uint32_t>()
{
  return DataType::UInt;
}

template<> DataType getDataType<int>()
{
  return NNDataSetEnums::DataType::Int;
}

template<> DataType getDataType<long>()
{
  return NNDataSetEnums::DataType::LLInt;
}

template<> DataType getDataType<uint64_t>()
{
  return NNDataSetEnums::DataType::ULLInt;
}

template<> DataType getDataType<float>()
{
  return DataType::Float;
}

template<> DataType getDataType<double>()
{
  return DataType::Double;
}

template<> DataType getDataType<char>()
{
  return NNDataSetEnums::DataType::Char;
}

template<> DataType getDataType<uint8_t>()
{
  return NNDataSetEnums::DataType::UChar;
}

}  // namespace NNDataSetEnums

#endif
