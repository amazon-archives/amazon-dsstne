#ifndef NNENUM_H
#define NNENUM_H

namespace NNDataSetEnums {

    enum Attributes
    {
        Sparse = 1,                 // Sparse dataset
        Boolean = 2,                // All datapoints are 0/1
        Compressed = 4,             // Data uses type-specific compression
        Recurrent = 8,              // Data has a time dimension
        Mutable = 16,               // Data can be modified by running network backwards
        SparseIgnoreZero = 32       // Only calculate errors and deltas on non-zero values
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
}

#endif
