#include "cudart_internal.h"

#if defined(_WIN32)
#include <direct.h>
#define acuda_mkdir(path) _mkdir(path)
#define ACUDA_DEFAULT_DUMP_DIR "raw-cubin"
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#define acuda_mkdir(path) mkdir((path), 0755)
#define ACUDA_DEFAULT_DUMP_DIR "/home/magnusolsen/git/aicode-ptx/raw-cubin"
#endif

int g_init;
int g_device;
ACUDA_CUDART_TLS cudaError_t g_last;

void set_last(cudaError_t e) {
    if (e != cudaSuccess)
        g_last = e;
}

cudaError_t map_cu(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidPtx;
    case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorInvalidValue;
    case CUDA_ERROR_ALREADY_MAPPED: return cudaErrorAlreadyMapped;
    case CUDA_ERROR_NOT_MAPPED: return cudaErrorNotMapped;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidHandle;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;
    }
}

cudaError_t ensure_init(void) {
    if (g_init)
        return cudaSuccess;
    if (cuInit(0) != CUDA_SUCCESS)
        return cudaErrorInitializationError;
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS || n <= 0)
        return cudaErrorNoDevice;
    CUcontext ctx = NULL;
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != CUDA_SUCCESS)
        return cudaErrorInitializationError;
    g_init = 1;
    g_device = 0;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t check_ctx(void) {
    return ensure_init();
}
