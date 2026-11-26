#include "cudart_internal.h"
#include "acuda/context.h"

extern acuda_context_t g_acuda_ctx; // defined in driver if needed; use driver API via cu* 
// We'll use driver API queries where possible

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    int n = 0;
    r = map_cu(cuDeviceGetCount(&n));
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    *count = n;
    return cudaSuccess;
}

/* Frozen pre-CUDA-10 cudaDeviceProp layout, which is what the unsuffixed
 * cudaGetDeviceProperties symbol must write: binaries compiled against
 * CUDA <= 9.x (PhysX 2.8.4 is a CUDA 6.5 build) reserve this layout on
 * their stack and read totalGlobalMem at offset 0x100.  The current
 * cudaDeviceProp is 956 bytes with uuid/luid right after the name, so
 * writing it into an old caller both returns garbage and overruns the
 * caller's object by 68 bytes.  cudaGetDeviceProperties_v2 keeps the
 * current layout for CUDA 10+ callers. */
typedef struct {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
} acuda_legacy_device_prop_t;

typedef char acuda_legacy_device_prop_size_check[
    (sizeof(acuda_legacy_device_prop_t) == 592) ? 1 : -1];

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    struct cudaDeviceProp cur;
    CUresult cr = acuda_query_device_properties(&cur, device);
    r = map_cu(cr);
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    acuda_legacy_device_prop_t old;
    memset(&old, 0, sizeof(old));
    memcpy(old.name, cur.name, sizeof(old.name));
    old.totalGlobalMem = cur.totalGlobalMem;
    old.sharedMemPerBlock = cur.sharedMemPerBlock;
    old.regsPerBlock = cur.regsPerBlock;
    old.warpSize = cur.warpSize;
    old.memPitch = cur.memPitch;
    old.maxThreadsPerBlock = cur.maxThreadsPerBlock;
    memcpy(old.maxThreadsDim, cur.maxThreadsDim, sizeof(old.maxThreadsDim));
    memcpy(old.maxGridSize, cur.maxGridSize, sizeof(old.maxGridSize));
    old.totalConstMem = cur.totalConstMem;
    old.major = cur.major;
    old.minor = cur.minor;
    old.textureAlignment = cur.textureAlignment;
    old.texturePitchAlignment = cur.texturePitchAlignment;
    /* Fields the current cudaDeviceProp dropped but the old ABI still
     * exposes.  The clock rates come from the driver (hwinfo/ADL), not a
     * constant. */
    {
        int av = 0;
        if (cuDeviceGetAttribute(&av, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                 device) == CUDA_SUCCESS)
            old.clockRate = av;
        if (cuDeviceGetAttribute(&av, 15 /* GPU_OVERLAP */,
                                 device) == CUDA_SUCCESS)
            old.deviceOverlap = av;
        if (cuDeviceGetAttribute(&av, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
                                 device) == CUDA_SUCCESS)
            old.kernelExecTimeoutEnabled = av;
        if (cuDeviceGetAttribute(&av, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
                                 device) == CUDA_SUCCESS)
            old.computeMode = av;
        if (cuDeviceGetAttribute(&av, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                 device) == CUDA_SUCCESS)
            old.memoryClockRate = av;
        if (cuDeviceGetAttribute(&av,
                                 CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                                 device) == CUDA_SUCCESS)
            old.memoryBusWidth = av;
        if (cuDeviceGetAttribute(&av, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                                 device) == CUDA_SUCCESS)
            old.l2CacheSize = av;
        if (cuDeviceGetAttribute(
                &av, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                device) == CUDA_SUCCESS)
            old.maxThreadsPerMultiProcessor = av;
        if (cuDeviceGetAttribute(&av,
                                 37 /* MAX_TEXTURE1D_LINEAR */,
                                 device) == CUDA_SUCCESS)
            old.maxTexture1DLinear = av;
    }
    old.multiProcessorCount = cur.multiProcessorCount;
    old.integrated = cur.integrated;
    old.canMapHostMemory = cur.canMapHostMemory;
    old.maxTexture1D = cur.maxTexture1D;
    old.maxTexture1DMipmap = cur.maxTexture1DMipmap;
    memcpy(old.maxTexture2D, cur.maxTexture2D, sizeof(old.maxTexture2D));
    memcpy(old.maxTexture2DMipmap, cur.maxTexture2DMipmap,
           sizeof(old.maxTexture2DMipmap));
    memcpy(old.maxTexture2DLinear, cur.maxTexture2DLinear,
           sizeof(old.maxTexture2DLinear));
    memcpy(old.maxTexture2DGather, cur.maxTexture2DGather,
           sizeof(old.maxTexture2DGather));
    memcpy(old.maxTexture3D, cur.maxTexture3D, sizeof(old.maxTexture3D));
    memcpy(old.maxTexture3DAlt, cur.maxTexture3DAlt,
           sizeof(old.maxTexture3DAlt));
    old.maxTextureCubemap = cur.maxTextureCubemap;
    old.surfaceAlignment = cur.surfaceAlignment;
    old.concurrentKernels = cur.concurrentKernels;
    old.ECCEnabled = cur.ECCEnabled;
    old.pciBusID = cur.pciBusID;
    old.pciDeviceID = cur.pciDeviceID;
    old.pciDomainID = cur.pciDomainID;
    old.tccDriver = cur.tccDriver;
    old.asyncEngineCount = cur.asyncEngineCount;
    old.unifiedAddressing = cur.unifiedAddressing;
    old.memoryBusWidth = cur.memoryBusWidth;
    old.l2CacheSize = cur.l2CacheSize;
    old.maxThreadsPerMultiProcessor = cur.maxThreadsPerMultiProcessor;
    old.streamPrioritiesSupported = cur.streamPrioritiesSupported;
    old.globalL1CacheSupported = cur.globalL1CacheSupported;
    old.localL1CacheSupported = cur.localL1CacheSupported;
    old.sharedMemPerMultiprocessor = cur.sharedMemPerMultiprocessor;
    old.regsPerMultiprocessor = cur.regsPerMultiprocessor;
    old.managedMemory = cur.managedMemory;
    old.isMultiGpuBoard = cur.isMultiGpuBoard;
    old.multiGpuBoardGroupID = cur.multiGpuBoardGroupID;
    old.hostNativeAtomicSupported = cur.hostNativeAtomicSupported;
    old.pageableMemoryAccess = cur.pageableMemoryAccess;
    old.concurrentManagedAccess = cur.concurrentManagedAccess;
    old.computePreemptionSupported = cur.computePreemptionSupported;
    old.canUseHostPointerForRegisteredMem =
        cur.canUseHostPointerForRegisteredMem;
    memcpy(prop, &old, sizeof(old));
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS)
        return cudaErrorNoDevice;
    if (device < 0 || device >= n)
        return cudaErrorInvalidDevice;
    CUcontext ctx = NULL;
    if (cuDevicePrimaryCtxRetain(&ctx, device) != CUDA_SUCCESS)
        return cudaErrorInvalidDevice;
    g_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    *device = g_device;
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion)
        return cudaErrorInvalidValue;
    int v = 0;
    if (cuDriverGetVersion(&v) != CUDA_SUCCESS)
        return cudaErrorInitializationError;
    *driverVersion = v;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion)
        return cudaErrorInvalidValue;
    *runtimeVersion = ACUDA_RT_VERSION;
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    return ensure_init();
}

cudaError_t cudaDeviceSynchronize(void) {
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    r = map_cu(cuCtxSynchronize());
    set_last(r);
    return r;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit) {
    (void)limit;
    if (!pValue)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    *pValue = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    (void)limit;
    (void)value;
    return ensure_init();
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    static int stored = 1;
    if (!mode)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    int old = stored;
    stored = *mode;
    *mode = old;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop,
                                       int device) {
    /* CUDA 10+ layout: the caller holds the current (956-byte) struct. */
    if (!prop)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    CUresult cr = acuda_query_device_properties(prop, device);
    r = map_cu(cr);
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attrib, int device) {
    if (!value)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    int v = 0;
    CUresult cr = cuDeviceGetAttribute(&v, attrib, device);
    r = map_cu(cr);
    set_last(r);
    if (r != cudaSuccess)
        return r;
    *value = v;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (!pciBusId || len <= 0)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    return map_cu(cuDeviceGetPCIBusId(pciBusId, len, device));
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice) {
    if (!canAccessPeer)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    *canAccessPeer = (device == peerDevice) ? 1 : 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)flags;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS)
        return cudaErrorNoDevice;
    if (peerDevice < 0 || peerDevice >= n)
        return cudaErrorInvalidDevice;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    if (leastPriority)
        *leastPriority = 0;
    if (greatestPriority)
        *greatestPriority = 0;
    return cudaSuccess;
}

cudaError_t cudaProfilerStart(void) {
    return ensure_init();
}

cudaError_t cudaProfilerStop(void) {
    return ensure_init();
}
