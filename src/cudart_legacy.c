// Legacy CUDA 6.x runtime entry points used by PhysX 2.8.4.
//
// PhysXCore.dll (the CUDA 6.5 build) imports the pre-CUDA-7 launch API
// (cudaConfigureCall/cudaSetupArgument/cudaLaunch), legacy texture
// references and a few old memory/cache calls.  A cudart that does not
// export them cannot satisfy the PE import table, so the loader refuses
// the module and FluidMark stays at "initialization: 0%".
//
// The launch trio is implemented for real (per-thread argument buffer,
// the same call-configuration stack the generated launcher stub uses);
// the texture-reference binds are accepted as no-ops because ACuda's
// texture path is the driver texref API that PhysX calls directly, and
// cache configuration has no analogue in Vulkan.
#include "cudart_internal.h"

#if defined(_WIN32)
#define ACUDA_TLS __declspec(thread)
#else
#define ACUDA_TLS __thread
#endif

extern unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                            size_t sharedMem, void *stream);
extern cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                              size_t *sharedMem, void *stream);

#define ACUDA_LEGACY_MAX_ARGS 64
#define ACUDA_LEGACY_ARG_BYTES 8192

static ACUDA_TLS unsigned char g_launch_args[ACUDA_LEGACY_ARG_BYTES];
static ACUDA_TLS size_t g_launch_off[ACUDA_LEGACY_MAX_ARGS];
static ACUDA_TLS int g_launch_nargs;

static int legacy_trace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *s = getenv("ACUDA_CUDART_TRACE");
        cached = (s && *s) ? 1 : 0;
    }
    return cached;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                              cudaStream_t stream) {
    if (legacy_trace())
        fprintf(stderr,
                "ACuda-cudart: cudaConfigureCall grid=%u,%u,%u block=%u,%u,%u "
                "shm=%zu\n",
                gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                blockDim.z, sharedMem);
    g_launch_nargs = 0;
    if (__cudaPushCallConfiguration(gridDim, blockDim, sharedMem,
                                    (void *)stream) != 0) {
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    set_last(cudaSuccess);
    return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    if (size > ACUDA_LEGACY_ARG_BYTES ||
        offset > ACUDA_LEGACY_ARG_BYTES - size ||
        g_launch_nargs >= ACUDA_LEGACY_MAX_ARGS) {
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (size && !arg) {
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (size)
        memcpy(g_launch_args + offset, arg, size);
    g_launch_off[g_launch_nargs++] = offset;
    if (legacy_trace())
        fprintf(stderr,
                "ACuda-cudart: cudaSetupArgument size=%zu off=%zu nargs=%d\n",
                size, offset, g_launch_nargs);
    set_last(cudaSuccess);
    return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func) {
    if (legacy_trace())
        fprintf(stderr, "ACuda-cudart: cudaLaunch func=%p nargs=%d\n", func,
                g_launch_nargs);
    dim3 gridDim, blockDim;
    size_t sharedMem = 0;
    void *stream = NULL;
    cudaError_t r =
        __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    void *args[ACUDA_LEGACY_MAX_ARGS];
    for (int i = 0; i < g_launch_nargs; i++)
        args[i] = g_launch_args + g_launch_off[i];
    /* cudaSetupArgument may be called out of parameter order; cuLaunchKernel
     * wants the pointers in parameter order, which is ascending offset. */
    for (int i = 1; i < g_launch_nargs; i++) {
        void *key = args[i];
        size_t key_off = g_launch_off[i];
        int j = i - 1;
        while (j >= 0 && g_launch_off[j] > key_off) {
            args[j + 1] = args[j];
            g_launch_off[j + 1] = g_launch_off[j];
            j--;
        }
        args[j + 1] = key;
        g_launch_off[j + 1] = key_off;
    }
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                            (cudaStream_t)stream);
}

cudaError_t cudaThreadSynchronize(void) {
    return cudaDeviceSynchronize();
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
                                    size_t count, size_t offset,
                                    cudaMemcpyKind kind, cudaStream_t stream) {
    /* NVIDIA's runtime is asynchronous here; PhysX uploads its kernel
     * parameters from a device buffer every frame.  Blocking in the driver
     * (memcpy_d2d takes a stream synchronise) stalls the PhysX worker
     * thread, so use the driver's asynchronous copy. */
    return cudart_memcpy_to_symbol_async(symbol, src, count, offset, (int)kind,
                                         stream);
}

cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    (void)cacheConfig;
    set_last(cudaSuccess);
    return cudaSuccess;
}

static CUarray_format legacy_cu_format(const cudaChannelFormatDesc *d,
                                       unsigned *nch) {
    int bits = d->x;
    *nch = (d->x ? 1u : 0u) + (d->y ? 1u : 0u) + (d->z ? 1u : 0u) +
           (d->w ? 1u : 0u);
    switch (d->f) {
    case cudaChannelFormatKindSigned:
        return bits == 8 ? CU_AD_FORMAT_SIGNED_INT8
                         : bits == 16 ? CU_AD_FORMAT_SIGNED_INT16
                                      : CU_AD_FORMAT_SIGNED_INT32;
    case cudaChannelFormatKindFloat:
        return bits == 16 ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT;
    case cudaChannelFormatKindUnsigned:
    default:
        return bits == 8 ? CU_AD_FORMAT_UNSIGNED_INT8
                         : bits == 16 ? CU_AD_FORMAT_UNSIGNED_INT16
                                      : CU_AD_FORMAT_UNSIGNED_INT32;
    }
}

cudaError_t cudaBindTexture(size_t *offset, const void *texref,
                            const void *devPtr,
                            const cudaChannelFormatDesc *desc, size_t size) {
    /* Bind the module's driver texref.  The launch path looks the texref up
     * by name and needs a bound texture object (CUDA_ERROR_INVALID_HANDLE
     * otherwise), so this cannot be a no-op. */
    cudaError_t r = check_ctx();
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    fatbin_t *fb = NULL;
    const char *name = NULL;
    int normalized = 0;
    if (!find_tex(texref, &fb, &name, &normalized) || !name || !fb) {
        if (getenv("ACUDA_TRACE"))
            fprintf(stderr, "ACuda-cudart: cudaBindTexture(%p) NOT REGISTERED\n",
                    texref);
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    CUmodule mod = NULL;
    CUresult cr = module_for_fatbin(fb, &mod);
    if (cr != CUDA_SUCCESS) {
        set_last(map_cu(cr));
        return map_cu(cr);
    }
    CUtexref tr = NULL;
    cr = cuModuleGetTexRef(&tr, mod, name);
    if (cr != CUDA_SUCCESS) {
        if (getenv("ACUDA_TRACE"))
            fprintf(stderr,
                    "ACuda-cudart: cudaBindTexture(%s) cuModuleGetTexRef -> %d\n",
                    name, (int)cr);
        set_last(map_cu(cr));
        return map_cu(cr);
    }
    if (desc) {
        unsigned nch = 0;
        CUarray_format fmt = legacy_cu_format(desc, &nch);
        cr = cuTexRefSetFormat(tr, fmt, (int)nch);
        if (cr != CUDA_SUCCESS) {
            set_last(map_cu(cr));
            return map_cu(cr);
        }
    }
    if (normalized)
        (void)cuTexRefSetFlags(tr, 0x01 /* CU_TRSF_NORMALIZED_COORDINATES */);
    size_t off = 0;
    cr = cuTexRefSetAddress_v2(&off, tr, (CUdeviceptr)(uintptr_t)devPtr, size);
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr,
                "ACuda-cudart: cudaBindTexture(%s) ptr=%p size=%zu -> off=%zu st=%d\n",
                name, devPtr, size, off, (int)cr);
    if (cr != CUDA_SUCCESS) {
        set_last(map_cu(cr));
        return map_cu(cr);
    }
    if (offset)
        *offset = off;
    set_last(cudaSuccess);
    return cudaSuccess;
}

cudaError_t cudaUnbindTexture(const void *texref) {
    (void)texref;
    set_last(cudaSuccess);
    return cudaSuccess;
}
