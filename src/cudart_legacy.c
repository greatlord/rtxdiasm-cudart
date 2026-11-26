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
    /* The copy path is host-visible device memory, so an asynchronous
     * transfer is already ordered by cuMemcpy; the stream only affects
     * completion ordering, which cudaDeviceSynchronize covers. */
    (void)stream;
    return cudaMemcpyToSymbol(symbol, src, count, offset, (int)kind);
}

cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    (void)cacheConfig;
    set_last(cudaSuccess);
    return cudaSuccess;
}

cudaError_t cudaBindTexture(size_t *offset, const void *texref,
                            const void *devPtr,
                            const cudaChannelFormatDesc *desc, size_t size) {
    /* ACuda binds textures through the driver texref API (cuTexRefSetAddress),
     * which PhysX calls directly.  Accept the legacy runtime bind so the
     * module loads; there is no image to create here. */
    (void)texref;
    (void)devPtr;
    (void)desc;
    (void)size;
    if (offset)
        *offset = 0;
    set_last(cudaSuccess);
    return cudaSuccess;
}

cudaError_t cudaUnbindTexture(const void *texref) {
    (void)texref;
    set_last(cudaSuccess);
    return cudaSuccess;
}

void __cudaRegisterTexture(void **fatCubinHandle, const void *hostVar,
                           const void **deviceAddress, const char *deviceName,
                           int dim, int norm, int ext) {
    (void)fatCubinHandle;
    (void)hostVar;
    (void)deviceAddress;
    (void)deviceName;
    (void)dim;
    (void)norm;
    (void)ext;
}
