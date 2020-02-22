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
_sm_major(0),
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



bool validate_path(vector< vector<nvlink> >& vMap, P2PRing& ring)
{
    // Restrict to paths starting at GPU 0
    if (ring.v[0] != 0)
        return false;
        
    vector<uint32_t>& v = ring.v;

    // Reject if there is no closed path
    int rank = (v.size() >= 2) ? vMap[v[0]][v[1]].rank : -1;

    // All connections within a ring must be the same rank
    for (size_t i = 0; i < v.size(); i++)
    {
        int j = (i + 1) % v.size();
        if (!vMap[v[i]][v[j]].bActive || (vMap[v[i]][v[j]].rank != rank))
            return false;
    }
    ring.rank = rank;

    return true;
}

void enumerate_ring(vector< vector<nvlink> >& vMap, P2PRing &ring, size_t l, size_t r, vector<P2PRing>& vRing)
{
   if (l == r)
   {
        if (validate_path(vMap, ring))
        {
            vRing.push_back(ring);
        }
   }
   else
   {
       for (size_t i = l; i <= r; i++)
       {
          swap(ring.v[l], ring.v[i]);
          enumerate_ring(vMap, ring, l+1, r, vRing);
          swap(ring.v[l], ring.v[i]); //backtrack
       }
   }
}


vector<P2PRing> getP2PRings(vector<uint32_t>& vDevice)
{
    vector<P2PRing> vP2PRings;

    
    vector< vector<nvlink> > vMap(vDevice.size());
    vector<P2PRing> vRing;
    for (size_t i = 0; i < vMap.size(); i++)
        vMap[i].resize(vDevice.size());
    
    // Enumerate all connections
    // Enumerates Device <-> Device links
    int maxRank = -1;
    vector<int> vRankCount(3);
    for (int i = 0; i < vDevice.size(); i++)
    {
        int device1 = vDevice[i];
        for (int j = 0; j < vDevice.size(); j++)
        {
            int device2 = vDevice[j];
            if (device1 == device2)
                continue;
            
            int accessSupported = 0;
            int rank = 0;
            cudaError_t status = cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);
            RTERROR(status, "cudaDeviceGetP2PAttribute");
            status = cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, device1, device2);
            RTERROR(status, "cudaDeviceGetP2PAttribute");
            if (accessSupported)
            {
                vMap[device1][device2].bActive = true;
                vMap[device2][device1].bActive = true;
                vMap[device1][device2].rank = rank;
                vMap[device2][device1].rank = rank;
                vMap[device1][device2].channels = rank;
                vMap[device2][device1].channels = rank;
                vMap[device1][device2].used = 0;
                vMap[device1][device2].used = 0;
                vRankCount[rank]++;
                cout << device1 << " " << device2 << " " << rank << endl;
                if (maxRank < rank)
                    maxRank = rank;
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        cout << "Total Rank: " << i << " " << vRankCount[i] << endl;
    }
    
    
    // Initialize simplest closed ring
    P2PRing ring;
    for (uint32_t i = 0; i < vDevice.size(); i++)
        ring.v.push_back(vDevice[i]);
    
    
    // Special case 2 GPUs
    if (vDevice.size() == 2)
    {
        if (vMap[vDevice[0]][vDevice[1]].bActive)
        {
            ring.rank = std::max(vMap[vDevice[0]][vDevice[1]].rank, 1);
            vP2PRings.push_back(ring);
        }
        return vP2PRings;
    }

    // Otherwise split high rank channels up into multiple rank 1 channels if < closed ring
    for (int i = 2; i < 3; i++)
    {
        cout << "Rank: " << i << " " << vRankCount[i] << endl;
        if ((vRankCount[i] < 2 * vDevice.size()) && (vRankCount[i] > 0))
        {
            cout << "Demoting rank " << i << " channels due to incomplete ring at this rank.\n";
            for (int j = 0; j < vDevice.size(); j++)
            {
                int device1 = vDevice[j];
                for (int k = 0; k < vDevice.size(); k++)
                {
                    int device2 = vDevice[k];
                    if (device1 == device2)
                        continue;
                    if (vMap[device1][device2].rank == i)
                    {
                        vMap[device1][device2].rank = 1;
                        vMap[device1][device2].channels += i;
                        vRankCount[1] += i;
                    }
                    if (vMap[device2][device1].rank == i)
                    {
                        vMap[device2][device1].rank = 1;
                        vMap[device2][device1].channels += i;                        
                        vRankCount[1] += i;
                    }
                }
            }
            vRankCount[i] = 0;
        }
    }
    for (int i = 0; i < 3; i++)
    {
        cout << "Rank: " << i << " " << vRankCount[i] << endl;
    }

    cout << "MR " << maxRank << endl;

    // Check for P2P without NVLINK
    if (maxRank <= 0)
    {
        ring.rank = 1;
        vP2PRings.push_back(ring);
    }
    else
    {
        // If NVLINK is present, try all permutations and report those that create a closed ring
        enumerate_ring(vMap, ring, (size_t)0, ring.v.size() - 1, vRing);
    }
    cout << vRing.size() << " feasible rings" << endl;

    for (size_t i = 0; i < vRing.size(); i++)
    {
        printf("RE %lu: ", i);
        for (size_t j = 0; j < vRing[i].v.size(); j++)
            printf("%u ", vRing[i].v[j]);
        printf("\n");
    }
    
    
    // Locate pairs of rings that use all connections
    vector< vector<nvlink> > vMap1, vMap2;
    for (size_t i = 0; i < vRing.size(); i++)
    {
        vMap1 = vMap;
        vector<uint32_t>& vi = vRing[i].v;
        for (size_t k = 0; k < vi.size(); k++)
        {
            size_t l = (k + 1) % vi.size();
            vMap1[vi[k]][vi[l]].used += vMap1[vi[k]][vi[l]].rank;
            if (vMap1[vi[k]][vi[l]].used == vMap1[vi[k]][vi[l]].channels)
                vMap1[vi[k]][vi[l]].bActive = false;
            vMap1[vi[l]][vi[k]].used += vMap1[vi[l]][vi[k]].rank;
            if (vMap1[vi[l]][vi[k]].used == vMap1[vi[l]][vi[k]].channels)
                vMap1[vi[l]][vi[k]].bActive = false;
        }

        for (size_t j = i + 1; j < vRing.size(); j++)
        {
            vMap2 = vMap1;
            vector<uint32_t>& vj = vRing[j].v;
            bool bSuccess = true;
            for (size_t k = 0; k < vi.size(); k++)
            {
                size_t l = (k + 1) % vi.size();
                if (!vMap2[vj[k]][vj[l]].bActive || !vMap2[vj[l]][vj[k]].bActive)
                {
                    bSuccess = false;
                    break;
                }
                vMap2[vj[k]][vj[l]].used += vMap2[vj[k]][vj[l]].rank;
                vMap2[vj[l]][vj[k]].used += vMap2[vj[l]][vj[k]].rank;
                if (vMap2[vj[k]][vj[l]].used == vMap2[vj[k]][vj[l]].channels)
                    vMap2[vj[k]][vj[l]].bActive = false;
                if (vMap2[vj[l]][vj[k]].used == vMap2[vj[l]][vj[k]].channels)     
                    vMap2[vj[l]][vj[k]].bActive = false;
            }        

            if (bSuccess)
            {
                //for (size_t k = 0; k < vi.size(); k++)
                //    cout << vi[k] << " ";
               // cout << endl;
               // for (size_t k = 0; k < vj.size(); k++)
               //     cout << vj[k] << " ";
               // cout << endl << endl;
                
                vP2PRings.push_back(vRing[i]);
                P2PRing ring = vRing[i];
                ring.v.resize(0);
                int pos = 0;
                do
                {
                    ring.v.push_back(vRing[i].v[pos]);
                    pos = (pos + vRing[i].v.size() - 1) % vRing[i].v.size();
                }
                while (pos != 0);
                vP2PRings.push_back(ring);

                vP2PRings.push_back(vRing[j]);
                ring = vRing[j];
                ring.v.resize(0);
                pos = 0;
                do
                {
                    ring.v.push_back(vRing[j].v[pos]);
                    pos = (pos + vRing[j].v.size() - 1) % vRing[j].v.size();
                }
                while (pos != 0);
                vP2PRings.push_back(ring);
                goto exit;
            }
        }
    }
exit:
    return vP2PRings;
}

bool EnableP2PAccess(int device1, int device2)
{
    // Don't bother if on same GPU
    if (device1 == device2)
        return true;

    int canAccessPeer;
    printf("Enable: Testing P2P access for devices %d and %d\n", device1, device2);
    cudaError_t status = cudaDeviceCanAccessPeer(&canAccessPeer, device1, device2);
    RTERROR(status, "cudaDeviceCanAccessPeer");
    
    // Signal failure if P2P is not possible
    if (canAccessPeer == 0)
    {
        return false;
    }
    
    // Turn on P2P access
    status = cudaDeviceEnablePeerAccess(device2, 0);

    // Ignore error that really isn't an error, just bad API design that
    // treats turning P2P access on more than once as an error
    if (status != cudaErrorPeerAccessAlreadyEnabled)
    {
        RTERROR(status, "cudaDeviceEnablePeerAccess");
    }
    else
    {
        cudaGetLastError();                        
    }

    // If we made it here, P2P access is on (or was already on)
    return true;
}

void GpuContext::Startup(int argc, char** argv)
{
    // Initialize MPI if not already initialized.
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(&argc, &argv);
    }
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
    //
    // As of Volta, this no longer holds (SML 10/01/17)
    int device                                      = -1;
    int gpuCount                                    = 0;
    cudaError_t status;
    cudaDeviceProp deviceProp;
    status = cudaGetDeviceCount(&gpuCount);
    RTERROR(status, "cudaGetDeviceCount failed");
    if (gpuCount == 0)
    {
        printf("GpuContext::Startup: No CUDA-capable devices found, exiting.\n");
        cudaDeviceReset();
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
    size_t localCount                               = 0;
    size_t localID                                  = 0;
    for (int i = 0; i < world_size; i++)
    {
        if (!strcmp(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)], myName))
        {
            if (i == _id)
                localID = localCount;
            localCount++;
        }
    }
    
    if (localCount > 1)
    {
        // Divide local processes amongst
        vector<uint32_t> vGPU;
        for (uint32_t i = 0; i < gpuCount; i++)
        {
            cudaGetDeviceProperties(&deviceProp, i);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))        
            {
                vGPU.push_back(i);
            }     
        }
        size_t gpus = vGPU.size();
        if (localCount <= gpus)
        {
            device = localID;
        }
        else
        {
            device = gpus * localID / localCount;
        }
        
        char hostname[128];
        gethostname(hostname, 127);
        printf("GpuContext::Startup: Process %d running on device %d out of %d GPUs on %s\n", _id, device, gpuCount, hostname);
    }  
    else
    {
        // Generate list of compatible GPUs scored by GPU revision first and total memory second
        vector<int> vGPUList(gpuCount);
        vector<uint32_t> vGPUScore(gpuCount);
        int gpus                                    = 0;          
        for (int i = 0; i < gpuCount; i++)
        {
            cudaGetDeviceProperties(&deviceProp, i);
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3))
            {
                vGPUList[gpus]                      = i;
                vGPUScore[gpus]                     = (deviceProp.major << 24) + (deviceProp.totalGlobalMem >> 20);
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
                    if (vGPUScore[i] < vGPUScore[i + 1])
                    {
                        done                        = false;
                        int gpu                     = vGPUList[i];
                        unsigned int score          = vGPUScore[i];
                        vGPUList[i]                 = vGPUList[i + 1];
                        vGPUScore[i]                = vGPUScore[i + 1];
                        vGPUList[i + 1]             = gpu;
                        vGPUScore[i + 1]            = score;
                    }
                }
            }
            while (!done);
        }
            
        // Let CUDA select any device from this list
        status                                      = cudaSetValidDevices(vGPUList.data(), gpus);
        RTERROR(status, "GpuContext::Startup: Error searching for compatible GPU");

        // Trick driver into creating a context on an available any valid GPU
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
        cudaDeviceReset();
        Shutdown();
        exit(-1);
    }
    

    // Finally set CUDA device
    status                                          = cudaSetDevice(device); 
    RTERROR(status, "GpuContext::Startup: Error setting CUDA device");  
    _device                                         = device;
    cudaDeviceSynchronize();

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
    _sm_major                                       = deviceProp.major;
    _warpSize                                       = deviceProp.warpSize;
    _warpBits                                       = fls(_warpSize) - 1;
    _warpMask                                       = _warpSize - 1;
    _data._warpSize                                 = _warpSize;
    _data._warpBits                                 = _warpBits;
    _data._warpMask                                 = _warpMask;
    _bUnifiedMemory                                 = (deviceProp.managedMemory != 0);
    
    // Determine language-specific type limits
    _data._maxUint32_t                              = numeric_limits<uint32_t>::max();
    _data._maxInt32_t                               = numeric_limits<int32_t>::max();
    _data._maxUint64_t                              = numeric_limits<uint64_t>::max();
    _data._maxInt64_t                               = numeric_limits<int64_t>::max();

    // Enumerate GPUs in use
    if (_id == 0)
        printf("GpuContext::Startup: Enumerating GPUs in use.\n");
    for (size_t i = 0; i < _numprocs; i++)
    {
        if (_id == i)
            printf("Process: %lu, GPU: %s, running SM %d.%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    
    // For P2P runs, insure process i can P2P to device on process i - 1 and process i + 1
    printf("GpuContext::Startup: Single node flag on GPU for process %d is %d\n", _device, bSingleNode);
    if (bSingleNode)
    {  
        bP2P                                        = true;
        vector<int> vProcessDevice(_numprocs);
        vProcessDevice[_id]                         = device;
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vProcessDevice.data(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD);
        cudaGetDeviceProperties(&deviceProp, device);
        bSingleNode                                 = (deviceProp.unifiedAddressing > 0);

        // Calculate rings from active devices
        if (_id == 0)
        {
            
            // Collect unique GPUs
            set<int> sDevice;
            for (int i = 0; i < _numprocs; i++)
            {
                sDevice.insert(vProcessDevice[i]);
            }
            
            vector<unsigned int> vDevice;
            for (auto d : sDevice)
            {
                cout << "D " << d << endl;
                vDevice.push_back(d);
            }
            
            // Calculate rings from device list if >1 device
            vector<P2PRing> vP2PRings;
            if (sDevice.size() > 1)
            {
                vP2PRings = getP2PRings(vDevice);
                cout << vP2PRings.size() << " rings\n";
                for (auto r : vP2PRings)
                {
                    cout << r.rank << ": ";
                    for (auto x : r.v)
                        cout << x << " ";
                    cout << endl;
                }
            }
            else
            {
                // Only using one GPU
                P2PRing p;
                p.v.push_back(vProcessDevice[0]);
                p.rank = 1;
                vP2PRings.push_back(p);
            }  
                            
            // Map rings over to process IDs
            for (int i = 0; i < vP2PRings.size(); i++)
            {
                P2PRing p;
                p.rank = vP2PRings[i].rank;
                for (auto d : vP2PRings[i].v)
                {
                    for (uint32_t j = 0; j < _numprocs; j++)
                    {
                        if (vProcessDevice[j] == d)
                            p.v.push_back(j);
                    }                 
                }
                _vP2PRings.push_back(p);   
            }
            cout << _vP2PRings.size() << " process rings\n";
            for (auto r : _vP2PRings)
            {
                cout << r.rank << ": ";
                for (auto x : r.v)
                    cout << x << " ";
                cout << endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
       
        // Broadcast ring dimensions to other GPUs
        size_t rings = _vP2PRings.size();
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &rings, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
        cout << " TOTAL " << rings << endl;
        _vP2PRings.resize(rings);
        
        // Broadcast all rings and calculate offsets
        _totalP2PRank = 0;
        for (size_t i = 0; i < rings; i++)
        {
            _vP2PRings[i].v.resize(_numprocs);
            MPI_Bcast(_vP2PRings[i].v.data(), _numprocs, MPI_UINT32_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(&_vP2PRings[i].rank, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
            _vP2PRings[i].offset = _totalP2PRank;
            _totalP2PRank += _vP2PRings[i].rank;
        }
        MPI_Barrier(MPI_COMM_WORLD);        
        
        // Enable P2P access for GPUs adjacent to each other on any ring
        if (_vP2PRings.size() > 0)
        {
            bool bP2PTemp = true;
            for (int i = 0; i < _vP2PRings.size(); i++)
            {
                // Create stream for each ring;
                cudaError_t status = cudaStreamCreate(&_vP2PRings[i].stream);
                RTERROR(status, "GpuContext::Startup: failed to create cuda stream.");
                
                // Find position on current ring
                for (uint32_t j = 0; j < _numprocs; j++)
                {
                    if (_vP2PRings[i].v[j] == _id)
                    {
                        // Enable P2P on the GPUs of each adjacent process on the ring
                        uint32_t minusDevice = vProcessDevice[_vP2PRings[i].v[(j + _numprocs - 1) % _numprocs]];
                        bP2PTemp &= EnableP2PAccess(device, minusDevice);
                        uint32_t plusDevice = vProcessDevice[_vP2PRings[i].v[(j + 1) % _numprocs]];
                        bP2PTemp &= EnableP2PAccess(device, plusDevice);
                        _vP2PRings[i].position = j;
                    }
                }
            }
            bP2P = bP2PTemp;
        }
        else
        {
            bP2P = false;
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

void GpuContext::SetFastMath(bool flag)
{
    cublasMath_t mathMode = flag ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasStatus_t cstatus = CUBLAS_STATUS_SUCCESS;
    if (_sm_major >= 7)
    {
        cstatus = cublasSetMathMode(_cuBLASHandle, mathMode);
        if (cstatus != CUBLAS_STATUS_SUCCESS)
        {
            printf("GpuContext::SetFastMath: failed to set math mode\n");
        }
    }
    else
    {
        printf("GpuContext::SetFastMath: failed to set math mode because GPU SM revision is <7.0\n");
    }
}

void GpuContext::Shutdown()
{   
    // Delete kernel accumulator
    _pbAccumulator.reset();
    
    // Delete P2P Streams
    for (size_t i = 0; i < _vP2PRings.size(); i++)
    {
        cudaError_t status = cudaStreamDestroy(_vP2PRings[i].stream);
        if (status != cudaSuccess)
        {
            printf("GpuContext::Shutdown: Error destroying cuda stream.");
        }
    }

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
    cudaDeviceReset();

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
