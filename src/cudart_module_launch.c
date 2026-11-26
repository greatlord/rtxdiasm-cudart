#include "cudart_internal.h"

#include <time.h>

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

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    if (!func) {
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (gridDim.x == 0 || gridDim.y == 0 || gridDim.z == 0 ||
        blockDim.x == 0 || blockDim.y == 0 || blockDim.z == 0) {
        set_last(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    const char *deviceName = NULL;
    fatbin_t *fb = NULL;
    if (!find_func_fb(func, &deviceName, &fb) || !deviceName || !fb) {
        const char *dump_dir = ACUDA_DEFAULT_DUMP_DIR;
        acuda_mkdir(dump_dir);
        for (size_t i = 0; i < g_nfatbins; i++) {
            for (size_t f = 0; f < g_fatbins[i]->nfuncs; f++) {
                if (g_fatbins[i]->funcs[f].hostFun == func &&
                    g_fatbins[i]->orig_blob) {
                    char name[512];
                    {
                        time_t t = time(NULL);
                        snprintf(name, sizeof(name), "%s/fatbin_%p_%ld.bin",
                                 dump_dir, func, (long)t);
                    }
                    FILE *fp = fopen(name, "wb");
                    if (fp) {
                        const unsigned char *b =
                            (const unsigned char *)g_fatbins[i]->orig_blob;
                        fwrite(b, 1, 64 * 1024 * 1024, fp);
                        fclose(fp);
                    }
                    break;
                }
            }
        }
        set_last(cudaErrorSymbolNotFound);
        return cudaErrorSymbolNotFound;
    }
    CUmodule mod = NULL;
    CUresult cr = module_for_fatbin(fb, &mod);
    if (cr != CUDA_SUCCESS) {
        r = map_cu(cr);
        set_last(r);
        return r;
    }
    CUfunction fn = NULL;
    cr = cuModuleGetFunction(&fn, mod, deviceName);
    if (cr != CUDA_SUCCESS) {
        r = map_cu(cr);
        set_last(r);
        return r;
    }
    cr = cuLaunchKernel(fn, gridDim.x, gridDim.y, gridDim.z,
                        blockDim.x, blockDim.y, blockDim.z, sharedMem,
                        (CUstream)(stream == cudaStreamPerThread ? NULL
                                                                  : stream),
                        args, NULL);
    r = map_cu(cr);
    set_last(r);
    return r;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                  const void *func) {
    if (!attr)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!func)
        return cudaErrorInvalidValue;
    const char *deviceName = NULL;
    if (!find_func_fb(func, &deviceName, NULL) || !deviceName)
        return cudaErrorSymbolNotFound;
    memset(attr, 0, sizeof(*attr));
    attr->sharedSizeBytes = 0;
    attr->constSizeBytes = 0;
    attr->localSizeBytes = 0;
    attr->maxThreadsPerBlock = 256;
    attr->numRegs = 32;
    attr->ptxVersion = 80;
    attr->binaryVersion = 80;
    attr->cacheModeCA = 0;
    attr->maxDynamicSharedSizeBytes = 0;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
    if (!numBlocks)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!func || blockSize <= 0)
        return cudaErrorInvalidValue;
    (void)dynamicSMemSize;
    int by_threads = 2048 / blockSize;
    int by_regs = 65536 / (32 * blockSize);
    int n = by_threads < by_regs ? by_threads : by_regs;
    if (n < 1)
        n = 1;
    if (n > 32)
        n = 32;
    *numBlocks = n;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
    (void)flags;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                        blockSize,
                                                        dynamicSMemSize);
}

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
    (void)func;
    (void)gridDim;
    (void)blockDim;
    (void)args;
    (void)sharedMem;
    (void)stream;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    return cudaErrorNotSupported;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *userData),
                               void *userData) {
    if (!fn)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (stream && stream != cudaStreamPerThread) {
        r = cudaStreamSynchronize(stream);
        if (r != cudaSuccess)
            return r;
    } else {
        r = cudaDeviceSynchronize();
        if (r != cudaSuccess)
            return r;
    }
    fn(userData);
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!func)
        return cudaErrorInvalidValue;
    const char *deviceName = NULL;
    fatbin_t *fb = NULL;
    if (!find_func_fb(func, &deviceName, &fb) || !deviceName || !fb)
        return cudaErrorSymbolNotFound;
    CUmodule mod = NULL;
    CUresult cr = module_for_fatbin(fb, &mod);
    if (cr != CUDA_SUCCESS)
        return map_cu(cr);
    CUfunction fn = NULL;
    cr = cuModuleGetFunction(&fn, mod, deviceName);
    if (cr != CUDA_SUCCESS)
        return map_cu(cr);
    return map_cu(cuFuncSetAttribute(fn, attr, value));
}

// CUDA 12.8+ launch stubs push the launch configuration here and pop it
// back immediately before calling __cudaLaunchKernel.  The stack must be
// shared by both entry points (and per host thread); separate function
// statics would make every pop fail with cudaErrorInvalidValue, which the
// generated stub treats as "skip the launch".
typedef struct {
    dim3 grid, block;
    size_t shared;
    cudaStream_t stream;
} acuda_call_config_t;

#if defined(_WIN32)
#define ACUDA_TLS __declspec(thread)
#else
#define ACUDA_TLS __thread
#endif

static ACUDA_TLS acuda_call_config_t cc_stack[16];
static ACUDA_TLS unsigned cc_depth;

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem, void *stream) {
    if (cc_depth < 16) {
        cc_stack[cc_depth].grid = gridDim;
        cc_stack[cc_depth].block = blockDim;
        cc_stack[cc_depth].shared = sharedMem;
        cc_stack[cc_depth].stream = (cudaStream_t)stream;
        cc_depth++;
        return 0;
    }
    return 1;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (cc_depth == 0)
        return cudaErrorInvalidValue;
    cc_depth--;
    if (gridDim)
        *gridDim = cc_stack[cc_depth].grid;
    if (blockDim)
        *blockDim = cc_stack[cc_depth].block;
    if (sharedMem)
        *sharedMem = cc_stack[cc_depth].shared;
    if (stream)
        *(void **)stream = (void *)cc_stack[cc_depth].stream;
    return cudaSuccess;
}

void __cudaInitModule(void) {}

cudaError_t __cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem,
                               cudaStream_t stream) {
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t __cudaGetKernel(void **kernel, const void *func) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!kernel || !func)
        return cudaErrorInvalidValue;
    *kernel = (void *)func;
    return cudaSuccess;
}
