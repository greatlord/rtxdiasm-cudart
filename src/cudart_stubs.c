#include "cudart_internal.h"

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args) {
    (void)config;
    (void)func;
    (void)args;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    set_last(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaGraphCreate(void *pGraph, unsigned int flags) {
    (void)pGraph;
    (void)flags;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    return cudaErrorNotSupported;
}

cudaError_t cudaMemPoolSetAccess(void *pool, void *desc, size_t n) {
    (void)pool; (void)desc; (void)n;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaSuccess : r;
}

cudaError_t cudaDeviceRegisterAsyncNotification(void *a, void *b, void *c) {
    (void)a; (void)b; (void)c;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaDeviceUnregisterAsyncNotification(void *a) {
    (void)a;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

#define ASTUB(name) \
    cudaError_t name(void) { \
        cudaError_t r = check_ctx(); \
        if (r != cudaSuccess) \
            return r; \
        return cudaErrorNotSupported; \
    }
ASTUB(cudaGraphDestroy)
ASTUB(cudaGraphExecDestroy)
ASTUB(cudaGraphLaunch)
ASTUB(cudaUserObjectCreate)
ASTUB(cudaUserObjectRelease)
ASTUB(cudaStreamBeginCapture)
ASTUB(cudaStreamEndCapture)
ASTUB(cudaStreamBeginCaptureToGraph)
ASTUB(cudaIpcGetMemHandle)
ASTUB(cudaIpcOpenMemHandle)
ASTUB(cudaIpcCloseMemHandle)
ASTUB(cudaIpcGetEventHandle)
ASTUB(cudaIpcOpenEventHandle)
ASTUB(cudaGraphConditionalHandleCreate)
ASTUB(cudaGraphGetNodes)
ASTUB(cudaGraphInstantiateWithFlags)
ASTUB(cudaGraphNodeGetDependencies)
ASTUB(cudaGraphRetainUserObject)
ASTUB(cudaDeviceGetGraphMemAttribute)
ASTUB(cudaDeviceSetGraphMemAttribute)
ASTUB(cudaDeviceGraphMemTrim)
ASTUB(cudaStreamUpdateCaptureDependencies)
ASTUB(cudaStreamCopyAttributes)
#undef ASTUB

cudaError_t cudaGraphAddNode_v2(void *a, void *b, void *c, void *d, size_t e) {
    (void)a; (void)b; (void)c; (void)d; (void)e;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaStreamUpdateCaptureDependencies_v2(void *a, void *b, void *c,
                                                   size_t d, unsigned e) {
    (void)a; (void)b; (void)c; (void)d; (void)e;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaGraphAddNode(void *node, void *graph, const void *deps,
                              const void *dep_data, size_t n_deps,
                              void *params) {
    (void)node; (void)graph; (void)deps; (void)dep_data; (void)n_deps;
    (void)params;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaGraphInstantiate(void *exec_out, void *graph,
                                 unsigned long long flags) {
    (void)exec_out; (void)graph; (void)flags;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaKernelSetAttributeForDevice(void *kernel, int attr, int value,
                                            int device) {
    (void)kernel; (void)attr; (void)value; (void)device;
    cudaError_t r = check_ctx();
    return r == cudaSuccess ? cudaErrorNotSupported : r;
}

cudaError_t cudaLibraryLoadData(void *library, const void *code,
                                void *jitOptions, void **jitOptionValues,
                                unsigned int numJitOptions,
                                void *libraryOptions,
                                void **libraryOptionValues,
                                unsigned int numLibraryOptions) {
    return map_cu(cuLibraryLoadData((void **)library, code, jitOptions,
                                    jitOptionValues, numJitOptions,
                                    libraryOptions, libraryOptionValues,
                                    numLibraryOptions));
}

cudaError_t cudaLibraryLoadFromFile(void *library, const char *fileName,
                                    void *jitOptions, void **jitOptionValues,
                                    unsigned int numJitOptions,
                                    void *libraryOptions,
                                    void **libraryOptionValues,
                                    unsigned int numLibraryOptions) {
    return map_cu(cuLibraryLoadFromFile(
        (void **)library, fileName, jitOptions, jitOptionValues, numJitOptions,
        libraryOptions, libraryOptionValues, numLibraryOptions));
}

cudaError_t cudaLibraryUnload(void *library) {
    return map_cu(cuLibraryUnload(library));
}

cudaError_t cudaLibraryGetKernel(void *kernel, void *library,
                                 const char *name) {
    return map_cu(cuLibraryGetKernel((void **)kernel, library, name));
}

cudaError_t cudaLibraryGetGlobal(void **dptr, size_t *bytes, void *library,
                                 const char *name) {
    return map_cu(cuLibraryGetGlobal((CUdeviceptr *)dptr, bytes,
                                     library, name));
}

cudaError_t cudaLibraryGetManaged(void **dptr, size_t *bytes, void *library,
                                  const char *name) {
    return map_cu(cuLibraryGetManaged((CUdeviceptr *)dptr, bytes,
                                      library, name));
}

cudaError_t cudaLibraryGetUnifiedFunction(void **fptr, void *library,
                                          const char *symbol) {
    return map_cu(cuLibraryGetUnifiedFunction(fptr, library, symbol));
}

cudaError_t cudaLibraryGetKernelCount(unsigned int *count, void *library) {
    return map_cu(cuLibraryGetKernelCount(count, library));
}

cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                    unsigned long long flags,
                                    int *driverStatus) {
    (void)flags;
    if (!symbol || !funcPtr)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUresult cr = cuGetProcAddress(symbol, funcPtr, ACUDA_RT_VERSION, 0);
    if (driverStatus)
        *driverStatus = (cr == CUDA_SUCCESS)
                            ? cudaDriverEntryPointSuccess
                            : cudaDriverEntryPointSymbolNotFound;
    return cudaSuccess;
}

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol,
                                             void **funcPtr,
                                             unsigned int cudaVersion,
                                             unsigned long long flags,
                                             int *driverStatus) {
    (void)cudaVersion;
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}


/* Graphics interop (no Vulkan/GL/D3D interop wired yet) */
typedef struct cudaGraphicsResource_st *cudaGraphicsResource_t;
cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
    (void)stream;
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (count < 0 || (count > 0 && !resources)) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    set_last(cudaErrorNotSupported); return cudaErrorNotSupported;
}
cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
    (void)stream;
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (count < 0 || (count > 0 && !resources)) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    set_last(cudaErrorNotSupported); return cudaErrorNotSupported;
}
cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource) {
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (!devPtr || !size || !resource) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    *devPtr = NULL; *size = 0;
    set_last(cudaErrorNotSupported); return cudaErrorNotSupported;
}
cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
    (void)arrayIndex; (void)mipLevel;
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (!array || !resource) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    *array = NULL;
    set_last(cudaErrorNotSupported); return cudaErrorNotSupported;
}
cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) {
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (!mipmappedArray || !resource) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    *mipmappedArray = NULL;
    set_last(cudaErrorNotSupported); return cudaErrorNotSupported;
}

/* CUDA Logs */
typedef struct { void *cb; void *userData; } cudart_log_cb_t;
static cudart_log_cb_t g_cudart_log_cbs[16];
cudaError_t cudaLogsRegisterCallback(void *cb, void *userData) {
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (!cb) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    for (int i=0;i<16;i++) if (g_cudart_log_cbs[i].cb==cb && g_cudart_log_cbs[i].userData==userData) return cudaSuccess;
    for (int i=0;i<16;i++) if (!g_cudart_log_cbs[i].cb){ g_cudart_log_cbs[i].cb=cb; g_cudart_log_cbs[i].userData=userData; return cudaSuccess; }
    return cudaSuccess;
}
cudaError_t cudaLogsUnregisterCallback(void *cb) {
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (!cb) { set_last(cudaErrorInvalidValue); return cudaErrorInvalidValue; }
    for (int i=0;i<16;i++) if (g_cudart_log_cbs[i].cb==cb){ g_cudart_log_cbs[i].cb=NULL; g_cudart_log_cbs[i].userData=NULL; }
    return cudaSuccess;
}
cudaError_t cudaLogsCurrent(void **log, unsigned int flags) {
    (void)flags;
    cudaError_t r = check_ctx(); if (r != cudaSuccess) return r;
    if (log) *log = NULL;
    return cudaSuccess;
}
