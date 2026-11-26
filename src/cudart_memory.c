#include "cudart_internal.h"

/* Win32 x86: ensure driver host-memory APIs are seen with stdcall. */
CUresult ACUDA_CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult ACUDA_CUDAAPI cuMemFreeHost(void *p);
CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer(void **pdptr, void *p, unsigned int Flags);
CUresult ACUDA_CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p);
CUresult ACUDA_CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                            CUdeviceptr dptr);

static struct {
    void *ptr;
    size_t size;
} *alloc_tab;
static size_t alloc_tab_n, alloc_tab_cap;
static size_t g_live_bytes;

static void alloc_track(void *ptr, size_t size) {
    if (alloc_tab_n == alloc_tab_cap) {
        size_t ncap = alloc_tab_cap ? alloc_tab_cap * 2 : 16;
        void *np = malloc(ncap * sizeof(*alloc_tab));
        if (!np)
            return;
        memcpy(np, alloc_tab, alloc_tab_n * sizeof(*alloc_tab));
        free(alloc_tab);
        alloc_tab = np;
        alloc_tab_cap = ncap;
    }
    alloc_tab[alloc_tab_n].ptr = ptr;
    alloc_tab[alloc_tab_n].size = size;
    alloc_tab_n++;
    g_live_bytes += size;
}

static size_t alloc_untrack(void *ptr) {
    for (size_t i = 0; i < alloc_tab_n; i++) {
        if (alloc_tab[i].ptr == ptr) {
            size_t size = alloc_tab[i].size;
            memmove(&alloc_tab[i], &alloc_tab[i + 1],
                    sizeof(*alloc_tab) * (alloc_tab_n - i - 1));
            alloc_tab_n--;
            g_live_bytes -= size;
            return size;
        }
    }
    return 0;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr)
        return cudaErrorInvalidValue;
    *devPtr = NULL;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (size == 0)
        return cudaErrorInvalidValue;
    CUdeviceptr d = 0;
    r = map_cu(cuMemAlloc_v2(&d, size));
    set_last(r);
    if (r != cudaSuccess)
        return r;
    alloc_track((void *)(uintptr_t)d, size);
    *devPtr = (void *)(uintptr_t)d;
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!devPtr)
        return cudaErrorInvalidValue;
    alloc_untrack(devPtr);
    r = map_cu(cuMemFree_v2((CUdeviceptr)(uintptr_t)devPtr));
    set_last(r);
    return r;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    (void)stream;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    if (!devPtr)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread) {
        r = map_cu(cuStreamSynchronize((CUstream)stream));
        if (r != cudaSuccess) {
            set_last(r);
            return r;
        }
    } else {
        r = map_cu(cuCtxSynchronize());
        if (r != cudaSuccess) {
            set_last(r);
            return r;
        }
    }
    return cudaFree(devPtr);
}



static int ptr_is_device(const void *p) {
    /* Arena allocations are tracked by the driver; host allocations are not.
     * cuMemGetAddressRange reports the allocation size for tracked pointers
     * and 0 for everything else. */
    size_t sz = 0;
    if (!p)
        return 0;
    if (cuMemGetAddressRange(NULL, &sz, (CUdeviceptr)(uintptr_t)p) !=
        CUDA_SUCCESS)
        return 0;
    return sz != 0;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
    return cudaMemcpyAsync(dst, src, count, kind, NULL);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if ((!dst && count) || (!src && count))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    CUstream s = map_stream(stream);
    (void)s;
    CUresult cr;
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr, "acuda-trace ?? memcpy dst=%p src=%p count=%zu kind=%d\n",
                dst, src, count, (int)kind);
    if (kind == cudaMemcpyDefault) {
        /* PhysX 2.8.4 uses cudaMemcpyDefault for particle uploads.  Classify
         * both sides; a device pointer is one the driver's arena tracks. */
        int sd = ptr_is_device(src);
        int dd = ptr_is_device(dst);
        if (dd && sd)
            kind = cudaMemcpyDeviceToDevice;
        else if (dd)
            kind = cudaMemcpyHostToDevice;
        else if (sd)
            kind = cudaMemcpyDeviceToHost;
        else
            kind = cudaMemcpyHostToHost;
    }
    switch (kind) {
    case cudaMemcpyHostToHost:
        memmove(dst, src, count);
        return cudaSuccess;
    case cudaMemcpyHostToDevice:
        cr = cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, count);
        break;
    case cudaMemcpyDeviceToHost:
        cr = cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, count);
        break;
    case cudaMemcpyDeviceToDevice:
        cr = cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst,
                             (CUdeviceptr)(uintptr_t)src, count);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    r = map_cu(cr);
    set_last(r);
    return r;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    return cudaMemsetAsync(devPtr, value, count, NULL);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!devPtr)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    CUstream s = map_stream(stream);
    if (s) {
        r = map_cu(cuStreamSynchronize(s));
        if (r != cudaSuccess) {
            set_last(r);
            return r;
        }
    }
    r = map_cu(cuMemsetD8_v2((CUdeviceptr)(uintptr_t)devPtr,
                             (unsigned char)value, count));
    set_last(r);
    return r;
}

cudaError_t cudaMemGetInfo(size_t *freeBytes, size_t *totalBytes) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    size_t total = 0;
    if (cuDeviceTotalMem(&total, g_device) != CUDA_SUCCESS)
        return cudaErrorUnknown;
    if (totalBytes)
        *totalBytes = total;
    if (freeBytes)
        *freeBytes = total > g_live_bytes ? total - g_live_bytes : 0;
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr, "ACuda-cudart: cudaMemGetInfo free=%zu total=%zu live=%zu\n",
                freeBytes ? *freeBytes : (size_t)0, total, g_live_bytes);
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attrs,
                                     const void *ptr) {
    if (!attrs)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!ptr)
        return cudaErrorInvalidValue;
    memset(attrs, 0, sizeof(*attrs));
    if (ptr_is_device(ptr)) {
        attrs->type = cudaMemoryTypeDevice;
        attrs->device = g_device;
        attrs->devicePointer = (void *)ptr;
        attrs->hostPointer = NULL;
    } else {
        attrs->type = cudaMemoryTypeHost;
        attrs->device = g_device;
        attrs->devicePointer = (void *)ptr;
        attrs->hostPointer = (void *)ptr;
    }
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
    if (!ptr)
        return cudaErrorInvalidValue;
    *ptr = NULL;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUresult cr = cuMemHostAlloc(ptr, size, flags);
    if (cr != CUDA_SUCCESS)
        return map_cu(cr);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
    if (!ptr)
        return cudaSuccess;
    return map_cu(cuMemFreeHost(ptr));
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    return cudaHostAlloc(ptr, size, 0);
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
    if (!pDevice)
        return cudaErrorInvalidValue;
    CUresult r = cuMemHostGetDevicePointer(pDevice, pHost, flags);
    return map_cu(r);
}

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    if (!pFlags)
        return cudaErrorInvalidValue;
    CUresult r = cuMemHostGetFlags(pFlags, pHost);
    return map_cu(r);
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    (void)flags;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, int attr,
                                    void *value) {
    (void)memPool;
    if (!value)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (attr == cudaMemPoolAttrUsedMemCurrent) {
        *(uint64_t *)value = (uint64_t)g_live_bytes;
        return cudaSuccess;
    }
    return cudaErrorInvalidValue;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, int attr,
                                    void *value) {
    (void)memPool;
    (void)attr;
    (void)value;
    return check_ctx();
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *pool_out, int device) {
    if (!pool_out)
        return cudaErrorInvalidValue;
    *pool_out = NULL;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUmemPool pool = NULL;
    r = map_cu(cuDeviceGetDefaultMemPool(&pool, device));
    set_last(r);
    if (r != cudaSuccess)
        return r;
    *pool_out = (cudaMemPool_t)pool;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    r = map_cu(cuDeviceSetMemPool(device, (CUmemPool)pool));
    set_last(r);
    return r;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    r = map_cu(cuMemPoolTrimTo((CUmemPool)pool, minBytesToKeep));
    set_last(r);
    return r;
}
