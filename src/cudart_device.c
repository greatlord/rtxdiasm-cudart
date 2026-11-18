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
    CUresult cr = acuda_query_device_properties(prop, device);
    r = map_cu(cr);
    if (r != cudaSuccess) {
        set_last(r);
        return r;
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
