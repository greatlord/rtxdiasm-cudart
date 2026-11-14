#include "cudart_internal.h"

cudaError_t cudaGetLastError(void) {
    cudaError_t e = g_last;
    g_last = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) {
    return g_last;
}

const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorNoDevice: return "no CUDA-capable device is detected";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorInvalidPtx: return "a PTX compilation failed or the PTX is unsupported";
    case cudaErrorInvalidHandle: return "invalid resource handle";
    case cudaErrorSymbolNotFound: return "named symbol not found";
    case cudaErrorNotReady: return "device not ready";
    case cudaErrorIllegalAddress: return "an illegal memory access was encountered";
    case cudaErrorLaunchFailure: return "unspecified launch failure";
    case cudaErrorNotSupported: return "operation not supported";
    default: break;
    }
    const char *s = NULL;
    if (cuGetErrorString(error, &s) != CUDA_SUCCESS || !s)
        return "unknown error";
    return s;
}

const char *cudaGetErrorName(cudaError_t error) {
    switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorNoDevice: return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidPtx: return "cudaErrorInvalidPtx";
    case cudaErrorInvalidHandle: return "cudaErrorInvalidHandle";
    case cudaErrorSymbolNotFound: return "cudaErrorSymbolNotFound";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    default: break;
    }
    const char *s = NULL;
    if (cuGetErrorName(error, &s) != CUDA_SUCCESS || !s)
        return "cudaErrorUnknown";
    return s;
}
