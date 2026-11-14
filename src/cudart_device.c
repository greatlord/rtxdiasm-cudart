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

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop)
        return cudaErrorInvalidValue;
    cudaError_t r = ensure_init();
    if (r != cudaSuccess)
        return r;
    memset(prop, 0, sizeof(*prop));
    char name[256] = { 0 };
    if (cuDeviceGetName(name, sizeof(name), device) != CUDA_SUCCESS)
        return cudaErrorInvalidDevice;
    memcpy(prop->name, name, sizeof(prop->name) - 1);
    int major = 0, minor = 0;
    cuDeviceComputeCapability(&major, &minor, device);
    size_t total = 0;
    cuDeviceTotalMem(&total, device);
    prop->major = major;
    prop->minor = minor;
    prop->totalGlobalMem = total;
    prop->sharedMemPerBlock = 65536;
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->memPitch = 2147483647u;
    prop->maxThreadsPerBlock = 256;
    prop->maxThreadsDim[0] = 256;
    prop->maxThreadsDim[1] = 65535;
    prop->maxThreadsDim[2] = 65535;
    prop->maxGridSize[0] = 65535;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 2000000;
    prop->totalConstMem = 65536;
    prop->textureAlignment = 512;
    prop->texturePitchAlignment = 512;
    prop->deviceOverlap = 1;
    prop->multiProcessorCount = 16;
    prop->kernelExecTimeoutEnabled = 1;
    prop->integrated = 0;
    prop->canMapHostMemory = 1;
    prop->computeMode = 0;
    prop->maxTexture1D = 134217728;
    prop->maxTexture1DMipmap = 32768;
    prop->maxTexture1DLinear = 134217728;
    prop->maxTexture2D[0] = prop->maxTexture2D[1] = 65536;
    prop->maxTexture2DMipmap[0] = prop->maxTexture2DMipmap[1] = 32768;
    prop->maxTexture2DLinear[0] = 65536;
    prop->maxTexture2DLinear[1] = 65536;
    prop->maxTexture2DLinear[2] = 2097152;
    prop->maxTexture2DGather[0] = prop->maxTexture2DGather[1] = 32768;
    prop->maxTexture3D[0] = prop->maxTexture3D[1] = 16384;
    prop->maxTexture3D[2] = 16384;
    prop->maxTexture3DAlt[0] = prop->maxTexture3DAlt[1] = 8192;
    prop->maxTexture3DAlt[2] = 8192;
    prop->maxTextureCubemap = 32768;
    prop->maxTexture1DLayered[0] = 32768;
    prop->maxTexture1DLayered[1] = 2048;
    prop->maxTexture2DLayered[0] = prop->maxTexture2DLayered[1] = 32768;
    prop->maxTexture2DLayered[2] = 2048;
    prop->maxTextureCubemapLayered[0] = 32768;
    prop->maxTextureCubemapLayered[1] = 2046;
    prop->maxSurface1D = 32768;
    prop->maxSurface2D[0] = prop->maxSurface2D[1] = 32768;
    prop->maxSurface3D[0] = prop->maxSurface3D[1] = 32768;
    prop->maxSurface3D[2] = 32768;
    prop->maxSurface1DLayered[0] = 32768;
    prop->maxSurface1DLayered[1] = 2048;
    prop->maxSurface2DLayered[0] = prop->maxSurface2DLayered[1] = 32768;
    prop->maxSurface2DLayered[2] = 2048;
    prop->maxSurfaceCubemap = 32768;
    prop->maxSurfaceCubemapLayered[0] = 32768;
    prop->maxSurfaceCubemapLayered[1] = 2046;
    prop->surfaceAlignment = 512;
    prop->concurrentKernels = 1;
    prop->ECCEnabled = 0;
    prop->pciBusID = 0;
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;
    prop->tccDriver = 0;
    prop->asyncEngineCount = 1;
    prop->unifiedAddressing = 1;
    prop->memoryClockRate = 1600000;
    prop->memoryBusWidth = 256;
    prop->l2CacheSize = 4 << 20;
    prop->persistingL2CacheMaxSize = 0;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->streamPrioritiesSupported = 0;
    prop->globalL1CacheSupported = 1;
    prop->localL1CacheSupported = 1;
    prop->sharedMemPerMultiprocessor = 65536;
    prop->regsPerMultiprocessor = 65536;
    prop->managedMemory = 0;
    prop->isMultiGpuBoard = 0;
    prop->multiGpuBoardGroupID = 0;
    prop->hostNativeAtomicSupported = 0;
    prop->singleToDoublePrecisionPerfRatio = 0;
    prop->pageableMemoryAccess = 1;
    prop->concurrentManagedAccess = 0;
    prop->computePreemptionSupported = 0;
    prop->canUseHostPointerForRegisteredMem = 0;
    prop->cooperativeLaunch = 0;
    prop->cooperativeMultiDeviceLaunch = 0;
    prop->sharedMemPerBlockOptin = 65536;
    prop->pageableMemoryAccessUsesHostPageTables = 1;
    prop->directManagedMemAccessFromHost = 0;
    prop->maxBlocksPerMultiProcessor = 32;
    prop->accessPolicyMaxWindowSize = 0;
    prop->reservedSharedMemPerBlock = 0;
    prop->hostRegisterSupported = 1;
    prop->sparseCudaArraySupported = 0;
    prop->hostRegisterReadOnlySupported = 0;
    prop->timelineSemaphoreInteropSupported = 1;
    prop->memoryPoolsSupported = 1;
    prop->gpuDirectRDMASupported = 0;
    prop->gpuDirectRDMAFlushWritesOptions = 0;
    prop->gpuDirectRDMAWritesOrdering = 0;
    prop->memoryPoolSupportedHandleTypes = 0;
    prop->deferredMappingCudaArraySupported = 0;
    prop->ipcEventSupported = 0;
    prop->clusterLaunch = 0;
    prop->unifiedFunctionPointers = 0;
    {
        int v = 0;
        if (cuDeviceGetAttribute(&v, 33, device) == CUDA_SUCCESS) prop->pciBusID = v;
        v = 0;
        if (cuDeviceGetAttribute(&v, 34, device) == CUDA_SUCCESS) prop->pciDeviceID = v;
        v = 0;
        if (cuDeviceGetAttribute(&v, 50, device) == CUDA_SUCCESS) prop->pciDomainID = v;
    }
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
    return cudaGetDeviceProperties(prop, device);
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
