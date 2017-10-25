/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"


static const float cAcceptableError = 0.00001f;

static GpuContext gpu;
GpuContext& getGpu() { return gpu; }

static __always_inline int fls(int x)
{
    return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}



GpuContext::GpuContext() :
_bECCSupport(false),
_bCanMapHostMemory(false),
_bCPUValidate(false),
_bUnifiedMemory(false),
_acceptableError(cAcceptableError),
_totalCPUMemory(0),
_totalGPUMemory(0),
_numprocs(1),
_id(0),
_sm_version(SM_3X),
_warpSize(32),
_maxSparse(SM_3X_MAXSPARSE),
_maxSparseAnalog(SM_3X_MAXSPARSEANALOG),
_cuBLASHandle(0),
_cuDNNHandle(0),
_pbAccumulator()
{

}

GpuContext::~GpuContext()
{

}

void GpuContext::SetCPUValidate(bool bValidate)
{
    _bCPUValidate                       = bValidate;
}

void GpuContext::Startup(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id); 
    
    // Print MPI data
    printf("GpuContext::Startup: Process %d out of %d initialized.\n", _id, _numprocs);   

    // Initialize profiling if active
    if (getenv("CUDA_PROFILE") != 0) 
    {
        char profile_log[512];
        if (getenv("CUDA_PROFILE_LOG")) 
        {
            sprintf(profile_log, "%s%d", getenv("CUDA_PROFILE_LOG"), _id);
        } 
        else 
        {
            sprintf(profile_log, "cu%d.csv", _id);
        }
        setenv("CUDA_PROFILE_LOG", profile_log, 1);
        setenv("CUDA_PROFILE_CSV", "1", 1);
    }    
    
    // Select GPU: note we limit ourselves to one GPU per process by design
    // to avoid issues observed when mixing OpenMP and MPI.  Exhaustive testing
    // by Intel engineers attempting to do so indicated one GPU per process is currently
    // the more efficient way of handling this.  Ironic, right? (SML 05/30/14)
    int device                                      = -1;
    int gpuCount                                    = 0;
    cudaError_t status;
    cudaDeviceProp deviceProp;
    status = cudaGetDeviceCount(&gpuCount);
    RTERROR(status, "cudaGetDeviceCount failed");
    if (gpuCount == 0)
    {
        printf("GpuContext::Startup: No CUDA-capable devices found, exiting.\n");
        cudaThreadExit();
        Shutdown();
        exit(-1);
    }

    // Grab node names from all other processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int length;
    char myName[MPI_MAX_PROCESSOR_NAME + 1];
    unique_ptr<char[]> pName(new char[world_size * (MPI_MAX_PROCESSOR_NAME + 1)]);
    unique_ptr<int[]> pNameCount(new int[world_size]);
    unique_ptr<int[]> pNameDisp(new int[world_size]);
    MPI_Get_processor_name(myName, &length);
    strcpy(&pName[world_rank * (MPI_MAX_PROCESSOR_NAME + 1)], myName); 
    for (int i = 0; i < world_size; i++)
    {
        pNameCount[i]                               = MPI_MAX_PROCESSOR_NAME + 1;
        pNameDisp[i]                                = i * (MPI_MAX_PROCESSOR_NAME + 1);
    }
    MPI_Allgatherv(myName, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, pName.get(), pNameCount.get(), pNameDisp.get(),
            MPI_CHAR, MPI_COMM_WORLD);

    // Test for single node run
    bool bSingleNode                                = true;
    bool bP2P                                       = false;
    for (int i = 0; i < _numprocs; i++)
    {
        if (strcmp(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)], myName))
            bSingleNode                             = false;
    }

    // Activate zero-copy
    cudaSetDeviceFlags(cudaDeviceMapHost);


    // Select a GPU:   
    // First check for duplicate processes on current node
    int localCount                                  = 0;
    int offset                                      = 1;
    for (int i = 0; i < world_size; i++)
    {
        if (!strcmp(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)], myName))
        {
            localCount++;
            if (i < world_rank)
                offset++;
        }
    }
       
    if (localCount > 1)
    {
        // Choose nth gpu which is Kepler or better
        int pos = 0;
        while (offset > 0)
        {
            cudaGetDeviceProperties(&deviceProp, pos);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))        
            {
                device                              = pos;
                offset--;
            }   
            pos++;
            if (pos == gpuCount)
                pos                                 = 0;         
        }       
        char hostname[128];
        gethostname(hostname, 127);
        printf("GpuContext::Startup: Process %d running on device %d out of %d GPUs on %s\n", _id, device, gpuCount, hostname);
    }  
    else
    {
        // Generate list of compatible GPUs scored by GPU revision first and total memory second
        unique_ptr<int[]> pGPUList(new int[gpuCount]);
        unique_ptr<unsigned int[]> pGPUScore(new unsigned int[gpuCount]);
        int gpus                                    = 0;          
        for (int i = 0; i < gpuCount; i++)
        {
            cudaGetDeviceProperties(&deviceProp, i);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))
            {
                pGPUList[gpus]                      = i;
                pGPUScore[gpus]                     = (deviceProp.major << 24) + (deviceProp.totalGlobalMem >> 20);
                gpus++;
            }
        }

        // Select best GPU according to score
        if (gpus > 0)
        {
            // Bubble sort (go bubblesort go!) device list by score
            // Seriously, n < 8 and will never be more than 16 here
            // without a seismic shift in the PCIE standard and we'll
            // address that when the time comes (if ever because
            // power constraints).
            bool done                               = true;
            do
            {
                done                                = true;
                for (int i = 0; i < gpus - 1; i++)
                {
                    if (pGPUScore[i] < pGPUScore[i + 1])
                    {
                        done                        = false;
                        int gpu                     = pGPUList[i];
                        unsigned int score          = pGPUScore[i];
                        pGPUList[i]                 = pGPUList[i + 1];
                        pGPUScore[i]                = pGPUScore[i + 1];
                        pGPUList[i + 1]             = gpu;
                        pGPUScore[i + 1]            = score;
                    }
                }
            }
            while (!done);
        }
            
        // Let CUDA select any device from this list
        status                                      = cudaSetValidDevices(pGPUList.get(), gpus);
        RTERROR(status, "GpuContext::Startup: Error searching for compatible GPU");

        // Trick driver into creating a context on an available and valid GPU
        status                                      = cudaFree(0);
        RTERROR(status, "GpuContext::Startup: Error selecting compatible GPU");

        // Get device
        status                                      = cudaGetDevice(&device);
        RTERROR(status, "GpuContext::Startup: Error fetching current GPU");
    }           

    // Exit if no GPU available
    if (device == -1)
    {

        printf("GpuContext::Startup: No Kepler or later GPU located, exiting.\n");      
        cudaThreadExit();
        Shutdown();
        exit(-1);
    }
    

    // Finally set CUDA device
    status                                          = cudaSetDevice(device); 
    RTERROR(status, "GpuContext::Startup: Error setting CUDA device");  
    _device                                         = device;
    cudaThreadSynchronize();

    // Create local accumulator
    _pbAccumulator.reset(new GpuBuffer<unsigned long long int>((unsigned int)1, true));
    _data._pAccumulator                             = _pbAccumulator->_pDevData;

    // Grab GPU parameters
    cudaGetDeviceProperties(&deviceProp, _device);
    if (deviceProp.major == 3)
    {
        _sm_version                                 = SM_3X;
        _threadsPerBlock                            = SM_3X_THREADS_PER_BLOCK;
        _maxSparse                                  = SM_3X_MAXSPARSE;
        _maxSparseAnalog                            = SM_3X_MAXSPARSEANALOG;
    }
    else if (deviceProp.major == 5)
    {
        _sm_version                                 = SM_5X;
        _threadsPerBlock                            = SM_5X_THREADS_PER_BLOCK;
        _maxSparse                                  = SM_5X_MAXSPARSE;
        _maxSparseAnalog                            = SM_5X_MAXSPARSEANALOG;
    }
    else
    {
        _sm_version                                 = SM_6X;
        _threadsPerBlock                            = SM_6X_THREADS_PER_BLOCK;
        _maxSparse                                  = SM_6X_MAXSPARSE;
        _maxSparseAnalog                            = SM_6X_MAXSPARSEANALOG;       
        
    }
    _warpSize                                       = deviceProp.warpSize;
    _warpBits                                       = fls(_warpSize) - 1;
    _warpMask                                       = _warpSize - 1;
    _data._warpSize                                 = _warpSize;
    _data._warpBits                                 = _warpBits;
    _data._warpMask                                 = _warpMask;
    _bUnifiedMemory                                 = (deviceProp.concurrentManagedAccess != 0);
    
    // Determine language-specific type limits
    _data._maxUint32_t                              = numeric_limits<uint32_t>::max();
    _data._maxInt32_t                               = numeric_limits<int32_t>::max();
    _data._maxUint64_t                              = numeric_limits<uint64_t>::max();
    _data._maxInt64_t                               = numeric_limits<int64_t>::max();

    // Enumerate GPUs in use
    if (getGpu()._id == 0)
        printf("GpuContext::Startup: Enumerating GPUs in use.\n");
    for (size_t i = 0; i < getGpu()._numprocs; i++)
    {
        if (getGpu()._id == i)
            printf("Process: %lu, GPU: %s, running SM %d.%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    
    // Test for single node P2P run. P2P only works within a single node for
    // the foreseeable future (SML 9/25/15)
    printf("GpuContext::Startup: Single node flag on GPU for process %d is %d\n", _device, bSingleNode);
    if (bSingleNode)
    {
        bP2P                                        = true;
        unique_ptr<int[]> pDevice(new int[_numprocs]);
        pDevice[_id]                                = device;
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pDevice.get(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD);
        unique_ptr<int[]> pUnifiedAddressing(new int[_numprocs]);
        cudaGetDeviceProperties(&deviceProp, device);
        pUnifiedAddressing[_id]                     = deviceProp.unifiedAddressing;
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pUnifiedAddressing.get(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD);
        for (int i = 0; i < _numprocs; i++)
        {     
            if (pDevice[i] != device)
            {
                int canAccessPeer;
                printf("GpuContext::Startup: Testing P2P for processes %d and %d\n", device, pDevice[i]);
                cudaError_t status                  = cudaDeviceCanAccessPeer(&canAccessPeer, device, pDevice[i]);
                RTERROR(status, "cudaDeviceCanAccessPeer");
                if (canAccessPeer == 0)
                {
                    bP2P                            = false;
                }
                else
                {
                    status                          = cudaDeviceEnablePeerAccess(pDevice[i], 0);

                    // Ignore error that really isn't an error, just bad API design
                    if (status != cudaErrorPeerAccessAlreadyEnabled)
                    {
                        RTERROR(status, "cudaDeviceEnablePeerAccess");
                    }
                    else
                        cudaGetLastError();                        
                }
            }
            if (!pUnifiedAddressing[i])
                bSingleNode                         = false;
        }
    }
    _bSingleNode                                    = bSingleNode;
    _bP2P                                           = bP2P;
    printf("GpuContext::Startup: P2P support flags on GPU for process %d are %d %d\n", _device, _bP2P, _bSingleNode);  
    MPI_Allreduce(MPI_IN_PLACE, &_bP2P, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (!_bP2P)
    {
        if (_id == 0)
            printf("GpuContext::Startup: Not all GPUs can P2P between each other, falling back to MPI.\n");
    }
    MPI_Allreduce(MPI_IN_PLACE, &_bSingleNode, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (!_bSingleNode)
    {
        if (_id == 0)
            printf("GpuContext::Startup: P2P support only works within a single node, falling back to MPI.\n");
    }
    
     

    // Determine kernel call configuration and grab desired additional GPU properties
    cudaGetDeviceProperties(&deviceProp, device);
    _bECCSupport                                    = deviceProp.ECCEnabled || deviceProp.tccDriver || (strcasestr(deviceProp.name, "tesla") != NULL);
    _bCanMapHostMemory                              = deviceProp.canMapHostMemory;
    _totalMemory                                    = deviceProp.totalGlobalMem;

#ifdef GVERBOSE
    double memsize = (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0);
    printf("GpuContext::Startup: Using GPU %d, %s, SM %d.%d, %.1f MBytes of memory\n", device, deviceProp.name, deviceProp.major, deviceProp.minor, memsize);
#endif    

    // Initialize cuBLAS
    cublasStatus_t cstatus                          = cublasCreate(&_cuBLASHandle);
    if (cstatus != CUBLAS_STATUS_SUCCESS)
    {
        printf("GpuContext::Startup: Failed to initialize cuBLAS on GPU for process %d, exiting.\n", _device);
        Shutdown();
        exit(-1);
    }
    
    // Initialize cuDNN
    cudnnStatus_t cdstatus                          = cudnnCreate(&_cuDNNHandle);
    if (cdstatus != CUDNN_STATUS_SUCCESS)
    {
        printf("GpuContext::Startup: Failed to initialize cuDNN on GPU for process %d, exiting.\n", _device);
        Shutdown();
        exit(-1);
    }

    // Initialize cuRAND
    curandStatus_t crstatus                         = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);
    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        printf("GpuContext::Startup: Failed to initialize cuRand on GPU for process %d, exiting.\n", _device);
        Shutdown();
        exit(-1);
    }
    printf("GpuContext::Startup: GPU for process %d initialized.\n", device);
}

void GpuContext::CopyConstants()
{
    SetKernelsGpuData();
    SetKLossGpuData();
    SetKActivationGpuData();
    SetKDeltaGpuData();
}

void GpuContext::Shutdown()
{   
    // Delete kernel accumulator
    _pbAccumulator.reset();

    // Shut down cuBLAS if active
    printf("GpuContext::Shutdown: Shutting down cuBLAS on GPU for process %d\n", _device);
    cublasStatus_t cstatus                          = cublasDestroy(_cuBLASHandle);
    if (cstatus != CUBLAS_STATUS_SUCCESS)
    {
        printf("GpuContext::Shutdown: Failed to shut down cuBLAS on GPU for process %d.\n", _device);
    }
    printf("GpuContext::Shutdown: CuBLAS shut down on GPU for process %d\n", _device);

    // Shut down cuDNN if active
    printf("GpuContext::Shutdown: Shutting down cuDNN on GPU for process %d\n", _device);
    cudnnStatus_t cdstatus                          = cudnnDestroy(_cuDNNHandle);
    if (cdstatus != CUDNN_STATUS_SUCCESS)
    {
        printf("GpuContext::Shutdown: Failed to shut down cuDNN on GPU for process %d.\n", _device);
    }
    printf("GpuContext::Shutdown: CuDNN shut down on GPU for process %d\n", _device);


    // Shut down cuRand if active
    printf("GpuContext::Shutdown: Shutting down cuRand on GPU for process %d\n", _device);
    curandStatus_t crstatus                         = curandDestroyGenerator(_RNG);
    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        printf("GpuContext::Shutdown: Failed to shut down cuRand on GPU for process %d.\n", _device);
    }
    printf("GpuContext::Shutdown: CuRand shut down on GPU for process %d\n", _device);
    
    // Exit CUDA
    cudaThreadExit();

    // Shut down MPI
    MPI_Finalize();
    printf("GpuContext::Shutdown: Process %d out of %d finalized.\n", _id, _numprocs);
}


void GpuContext::SetNeuralNetwork(NNNetwork* pNetwork)
{
    _pNetwork                                       = pNetwork;
    _data._LRN_k                                    = pNetwork->_LRN_k;
    _data._LRN_n                                    = pNetwork->_LRN_n;
    _data._LRN_alpha                                = pNetwork->_LRN_alpha;
    _data._LRN_beta                                 = pNetwork->_LRN_beta;
    _data._maxout_k                                 = pNetwork->_maxout_k; 
    _data._bSparsenessPenalty                       = pNetwork->_bSparsenessPenalty;
    _data._sparsenessPenalty_p                      = pNetwork->_sparsenessPenalty_p;
    _data._sparsenessPenalty_beta                   = pNetwork->_sparsenessPenalty_beta;
    _data._bDenoising                               = pNetwork->_bDenoising;
    _data._denoising_p                              = pNetwork->_denoising_p;
    _data._denoising_q                              = 1.0f / (1.0f - pNetwork->_denoising_p);
    _data._deltaBoost_one                           = pNetwork->_deltaBoost_one;
    _data._deltaBoost_zero                          = pNetwork->_deltaBoost_zero;
    _data._SMCE_oneTarget                           = pNetwork->_SMCE_oneTarget;
    _data._SMCE_zeroTarget                          = pNetwork->_SMCE_zeroTarget;
    _data._SMCE_oneScale                            = pNetwork->_SMCE_oneScale;
    _data._SMCE_zeroScale                           = pNetwork->_SMCE_zeroScale;
    _data._bShuffleIndices                          = pNetwork->_bShuffleIndices && (pNetwork->_mode == Mode::Training);
    _data._pShuffleIndex                            = pNetwork->_pShuffleIndex;
    CopyConstants();
}

// Sets RNG seed across all GPUs
void GpuContext::SetRandomSeed(unsigned long seed)
{
    curandStatus_t crstatus                         = curandSetPseudoRandomGeneratorSeed(_RNG, seed + (unsigned long)_device * 76801ull);
    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        if (getGpu()._id == 0)
            printf("GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process %d, exiting.\n", _device);
        Shutdown();
        exit(-1);
    }
    srand(seed);
    
    // Report settings
    if (getGpu()._id == 0)
        printf("GpuContext::SetRandomSeed: Random seed set to %lu.\n", seed);
}


// Returns KB of memory in use on CPU and GPU
void GpuContext::GetMemoryUsage(int* gpuMemory, int* cpuMemory)
{
    *gpuMemory                                      = (int)(_totalGPUMemory / 1024ll);
    *cpuMemory                                      = (int)(_totalCPUMemory / 1024ll);
    return;
}

void verifySGEMM(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{

    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i * k;            
            NNFloat* pB         = vB.data() + j;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA++;
                pB             += n;
            }
            if (fabs(sum - vC[i * n + j]) > 0.000001f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    exit(-1);
}

void verifySGEMMNT(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{

    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());

    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i * k;            
            NNFloat* pB         = vB.data() + j * k;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA++;
                pB++;
            }
            if (fabs(sum - vC[i * n + j]) / (fabs(sum) + 0.00000000000001f) > 0.000002f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    printf("%u %u %u\n", m, k, n);
    exit(-1);
}

void verifySGEMMTN(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{
    printf("%u %u %u\n", m, k, n);  
    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i;            
            NNFloat* pB         = vB.data() + j;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA             += m;
                pB             += n;
            }
            if (fabs(sum - vC[i * n + j]) / (fabs(sum) + 0.00000000000001f) > 0.000005f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    printf("%u %u %u\n", m, k, n);    
    exit(-1);    
}
