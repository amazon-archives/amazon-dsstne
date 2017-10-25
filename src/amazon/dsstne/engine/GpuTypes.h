/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __GPUTYPES_H__
#define __GPUTYPES_H__
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cstring>
#include <cstdint>
#include <assert.h>
#include <mpi.h>
#include <memory>

using namespace std;


#define VALIDATION


/* Enforce use of CUDA 5.0 due to GK110 issues with 4.2 */
#if defined(CUDA_VERSION) && (CUDA_VERSION < 5000)
#error "CUDA support requires the use of a 5.0 or later CUDA toolkit. Aborting compilation."
#endif

/* Control of single and double precision use:
   use_DPFP - use double-precision math and accumulate in 64-bit fixed point
   use_SPFP - use single-precision math and accumulate in 64-bit fixed point
   use_HPFP - Identical to SPFP except weights are stored as 16-bit floating point to save space

   By default, use_SPFP is active.  The double-precision mode exists to analyze
   numerical behavior while HPFP is a potential science experiment
   suggested by Leo Dirac.
*/

//#define use_DPFP
#define use_SPFP
//#define use_HPFP

// Enforce definition of one and only one precision mode
#if !(defined(use_DPFP) && !defined(use_HPFP) && !defined(use_SPFP)) && \
    !(defined(use_HPFP) && !defined(use_DPFP) && !defined(use_SPFP)) && \
    !(defined(use_SPFP) && !defined(use_DPFP) && !defined(use_HPFP))
#error "You must define one and only one precision mode (use_SPFP, use_HPFP, or use_DPFP). Aborting compilation."
#endif

#define ESCALE  (1ll << 30)
static const double ERRORSCALE              = ESCALE;
static const float ERRORSCALEF              = ESCALE;
static const double ONEOVERERRORSCALE       = 1.0 / (double)(ERRORSCALE);
static const float ONEOVERERRORSCALEF       = (float)(1.0 / (double)(ERRORSCALE));

typedef double                  __align__(8)    aligned_double;
typedef unsigned long int       __align__(8)    aligned_uli;
typedef long long int           __align__(8)    aligned_lli;
typedef unsigned long long int  __align__(8)    UllInt;

#if defined(use_DPFP)
typedef double                  __align__(8)    NNAccumulator;
typedef double                  __align__(8)    NNDouble;
typedef double                  __align__(8)    NNFloat;
typedef double2                 __align__(16)   NNDouble2;
typedef double4                 __align__(32)   NNDouble4;
typedef double2                 __align__(16)   NNFloat2;
typedef double4                 __align__(16)   NNFloat4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_NNFLOAT           = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_FLOAT;
#elif defined(use_SPFP)
typedef float                                   NNAccumulator;
typedef double                  __align__(8)    NNDouble;
typedef float                                   NNFloat;
typedef double2                 __align__(16)   NNDouble2;
typedef double4                 __align__(32)   NNDouble4;
typedef float2                  __align__(8)    NNFloat2;
typedef float4                  __align__(16)   NNFloat4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_NNFLOAT           = MPI_FLOAT;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_LONG_LONG_INT;
#else // use_HPFP
typedef float                                   NNAccumulator;
typedef double                  __align(8)__    NNDouble;
typedef float                                   NNFloat;
typedef double2                 __align(16)__   NNDouble2;
typedef double4                 __align(32)__   NNDouble4;
typedef float2                  __align(8)__    NNFloat2;
typedef float4                  __align(16)__   NNFloat4;
static const MPI_Datatype MPI_NNDOUBLE          = MPI_DOUBLE_PRECISION;
static const MPI_Datatype MPI_NNFLOAT           = MPI_FLOAT;
static const MPI_Datatype MPI_NNACCUMULATOR     = MPI_LONG_LONG_INT;
#endif

// Kernel dimensions - we only support SM 3.0 or better due to double-precision, 
// atomic operations, and a 32-bit blockDim.x
static const int SM_3X_THREADS_PER_BLOCK                        = 128;
static const int SM_5X_THREADS_PER_BLOCK                        = 128;
static const int SM_6X_THREADS_PER_BLOCK                        = 128;

#if (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

// Sparse kernel limits
static const uint32_t SM_6X_MAXSPARSE = 4608;
static const uint32_t SM_6X_MAXSPARSEANALOG = 2304;
static const uint32_t SM_5X_MAXSPARSE = 4608;
static const uint32_t SM_5X_MAXSPARSEANALOG = 2304;
static const uint32_t SM_3X_MAXSPARSE = 2304;
static const uint32_t SM_3X_MAXSPARSEANALOG = 1152;


static const bool bShadowedOutputBuffers                        = false;    // Turns off sysmem shadowing of really large buffers

#define FPSCALE  (1ll << 40)
#define DFSCALE (1ll << 44)


//#define GVERBOSE
//#define MEMTRACKING
//#define SYNCHRONOUS

#ifdef GVERBOSE
#ifndef MEMTRACKING
#define MEMTRACKING
#endif

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

#define LAUNCHERROR_BLOCKING(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        printf("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
    
#else   // GVERBOSE

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }
#endif

#define LAUNCHERROR_BLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }    
    
#endif  // GVERBOSE

#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        printf("%s %s\n", s, cudaGetErrorString(status)); \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }
    
#define CUDNNERROR(status, s) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("%s %s\n", s, cudnnGetErrorString(status)); \
        assert(0); \
        cudaThreadExit(); \
        exit(-1); \
    }    

// Contains information that needs to be accessible for GPU kernels and most static hyperparameters
struct GpuData {
    unsigned int            _warpSize;                  // Warp size
    unsigned int            _warpBits;                  // Warp bit count
    unsigned int            _warpMask;                  // Masks bits within a warp
    unsigned long long int* _pAccumulator;              // Accumulator for error calculations

    // Local response normalization settings
    float                   _LRN_k;                     // LRN offset
    int                     _LRN_n;                     // LRN spread
    float                   _LRN_alpha;                 // LRN scaling
    float                   _LRN_beta;                  // LRN exponent

    // Maxout parameters
    int                     _maxout_k;                  // Maxout neighborhood

    // Delta Boost parameters
    float                   _deltaBoost_one;            // Adjusts scaling of nonzero-valued outputs
    float                   _deltaBoost_zero;           // Adjusts scaling of zero-valued outputs

    // Scaled Marginal Cross Entropy parameters
    float                   _SMCE_oneTarget;            // Relaxed target for non-zero target values (Default 0.9)
    float                   _SMCE_zeroTarget;           // Relaxed target for zero target values (Default 0.1)
    float                   _SMCE_oneScale;             // Scaling factor for non-zero target values (Default 1.0)
    float                   _SMCE_zeroScale;            // Scaling factor for zero target values (Default 1.0)

    // Sparseness penalty for sparse hidden layers
    bool                    _bSparsenessPenalty;        // Controls whether sparse penalty should be applied to hidden layers
    float                   _sparsenessPenalty_p;       // Target sparseness probability for autoencoder
    float                   _sparsenessPenalty_beta;    // Sparse penalty weight on sparse hidden units

    // Denoising parameters for sparse input layers
    bool                    _bDenoising;                // Controls whether to apply denoising to nonzero inputs to sparse input layers
    float                   _denoising_p;               // Probability of denoising nonzero inputs to sparse input layers
    float                   _denoising_q;               // 1 / (1 - p) used to keep neuron weights constant

    // Example shuffling parameters
    bool                    _bShuffleIndices;           // Determines whether to directly look up examples or not
    unsigned int*           _pShuffleIndex;             // Index to shuffled training examples
    
    // Numeric limits
    uint32_t                _maxUint32_t;               // Language-constrained maximum 32-bit unsigned int value
    int32_t                 _maxInt32_t;                // Language-constrained maximum 32-bit int value
    uint64_t                _maxUint64_t;               // Language-constrained maximum 64-bit unsigned int value
    int64_t                 _maxInt64_t;                // Language-constrained maximum 64-bit int value
    float                   _maxFloat;                  // Language-constrained maximum 32-bit floating point value 
    float                   _minFloat;                  // Language-constrained minimum 32-bit floating point value     
};

template <typename T> struct GpuBuffer;
template <typename T> struct MultiGpuBuffer;
class NNNetwork;

struct GpuContext {
    enum SM_VERSION
    {
        SM_3X,
        SM_5X,
        SM_6X,
    };

   enum {
        PADDING                     = 32,
        PADDINGBITS                 = 5,
        PADDINGMASK                 = 0xffffffff - (PADDING - 1),
    };
   
    // Memory parameters
    GpuData                             _data;                      // All GPU data accessible from kernels (mostly device memory pointers)
    bool                                _bECCSupport;               // Flag for ECC support to detect Tesla versus consumer GPU
    bool                                _bCanMapHostMemory;         // Flag for pinned memory support
    aligned_lli                         _totalMemory;               // Total memory on GPU
    aligned_lli                         _totalCPUMemory;            // Approximate total allocated CPU memory
    aligned_lli                         _totalGPUMemory;            // Approximate total allocated CPU memory
    bool                                _bUnifiedMemory;            // Unified memory flag
    
    // SM/SMX parameters
    SM_VERSION                          _sm_version;                // SM revision
    unsigned int                        _threadsPerBlock;           // Default threads per block to launch
    unsigned int                        _warpSize;                  // Warp size (probably 32 but may change some day)
    unsigned int                        _warpBits;                  // Warp bit count
    unsigned int                        _warpMask;                  // Masks bits within a warp
    int                                 _numprocs;                  // Number of total processors in run
    int                                 _id;                        // Process ID
    int                                 _device;                    // Device ID

    // Fast sparse kernel limits
    uint32_t                            _maxSparse;                 // Maximum sparse boolean datapoints for sparse input layers
    uint32_t                            _maxSparseAnalog;           // Maximum sparse analog datapoints for sparse input layers

    // cuBLAS parameters
    cublasHandle_t                      _cuBLASHandle;              // Handle for cuBLAS state

    // cuRand parameters
    curandGenerator_t                   _RNG;                       // Handle for random number generator
    
    // cuDNN parameters
    cudnnHandle_t                       _cuDNNHandle;               // handle for cuDNN library   

    // Neural network parameters
    NNNetwork*                          _pNetwork;                  // Pointer to current neural network
    unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;   // Pointer to per-kernel fix point accumulator
    bool                                _bCPUValidate;              // Should CPU validate GPU calculations?
    float                               _acceptableError;           // Acceptable error between CPU and GPU
        
    // Single-node multi-gpu parameters
    bool                                _bSingleNode;               // Flag to indicate MPI run is all on one node
    bool                                _bP2P;                      // Flag to indicate P2P connectivity between all processes

    // Methods
    GpuContext();
    ~GpuContext();
    void GetMemoryUsage(int* gpuMemory, int* cpuMemory);
    void SetRandomSeed(unsigned long seed);
    void SetNeuralNetwork(NNNetwork* pNetwork);
    void Startup(int argc, char** argv);
    void Shutdown();
    void CopyConstants();
    void SetCPUValidate(bool bCPUValidate);

    // Static methods
    static unsigned int Pad(unsigned int x) { return (x + PADDING - 1) & PADDINGMASK; } 
};

extern struct GpuContext& getGpu();

template <typename T>
struct GpuBuffer
{
    unsigned long long int  _length;
    bool                    _bSysMem;
    bool                    _bManaged;
    T*                      _pSysData;
    T*                      _pDevData;
    GpuBuffer(int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(unsigned int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(unsigned long long int length, bool bSysMem = false, bool bManaged = false);
    GpuBuffer(size_t length, bool bSysMem = false, bool bManaged = false);
    virtual ~GpuBuffer();
    void Allocate();
    void Deallocate();
    void Upload(T* pBuff = NULL);
    void Download(T * pBuff = NULL);
    void Copy(T* pBuff);
};

template <typename T>
GpuBuffer<T>::GpuBuffer(int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();   
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();   
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned long long int length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();   
}

template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bManaged) : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();   
}

template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

template <typename T>
void GpuBuffer<T>::Allocate()
{
    cudaError_t status;

    // Force system memory shadowing on for managed buffers
    if (_bManaged)
        _bSysMem    = true;
         
#ifdef MEMTRACKING
    printf("Allocating %llu bytes of GPU memory", _length * sizeof(T));
    if (!_bSysMem)
    {
        printf(", unshadowed");
    }
    else if (_bManaged)
    {
        printf(", managed");
    }
    printf("\n");   
#endif
    
    // Allocate managed if managed
    if (_bManaged)
    {
        status = cudaMallocManaged((void **) &_pDevData, _length * sizeof(T), cudaMemAttachGlobal);
        getGpu()._totalGPUMemory           +=  _length * sizeof(T);
        _pSysData = _pDevData;
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMallocManaged)");
        memset(_pSysData, 0, _length * sizeof(T));
    }
    else
    {
        // Allocate in GPU space
        status = cudaMalloc((void **) &_pDevData, _length * sizeof(T));
        getGpu()._totalGPUMemory           +=  _length * sizeof(T);
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMalloc)");
        status = cudaMemset((void *) _pDevData, 0, _length * sizeof(T));
        RTERROR(status, "GpuBuffer::Allocate failed (cudaMemset)");

        // Allocate system memory
        if (_bSysMem)
        {
            _pSysData                           =  new T[_length];
            getGpu()._totalCPUMemory           +=  _length * sizeof(T);
            memset(_pSysData, 0, _length * sizeof(T));
        }
    }

#ifdef MEMTRACKING
    printf("Mem++: %llu %llu\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);     
#endif
}

template <typename T>
void GpuBuffer<T>::Deallocate()
{
    cudaError_t status;
    
    // Deallocate GPU memory    
    status = cudaFree(_pDevData);
    RTERROR(status, "GpuBuffer::Deallocate failed (cudaFree)");        
    getGpu()._totalGPUMemory           -=  _length * sizeof(T);
    
    // Delete system memory if present
    if (_bSysMem && !_bManaged)
    {
        delete[] _pSysData;
        getGpu()._totalCPUMemory           -=  _length * sizeof(T);   
    }
    
    _pSysData = NULL;
    _pDevData = NULL;
#ifdef MEMTRACKING    
    printf("Mem--: %lld %lld\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);     
#endif
}

template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
}

template <typename T>
void GpuBuffer<T>::Upload(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, _pSysData, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
}

template <typename T>
void GpuBuffer<T>::Download(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(pBuff, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pSysData, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
}

void verifySGEMM(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);
void verifySGEMMNT(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);
void verifySGEMMTN(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n);

#define SGEMM(A,B,C,m,n,k,alpha,beta,transf_A,transf_B) \
        cublasSgemm(getGpu()._cuBLASHandle, transf_B, transf_A, n, m, k, alpha, B, n, A, k, beta, C, n) 

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f,...)
#endif
#endif


