#include "cudart_internal.h"

/* Provided by ACuda driver bridge (linked via acuda_driver/libcuda).
 * cdecl: the rest of this target is compiled with /Gz (stdcall) on x86 to
 * match the CUDA ABI, but the driver side uses the default convention. */
#if defined(_MSC_VER) && defined(_M_IX86)
#define ACUDA_BRIDGE_CALL __cdecl
#else
#define ACUDA_BRIDGE_CALL
#endif
char *ACUDA_BRIDGE_CALL acuda_rtxdisam_cubin_to_ptx_for_target(
    const unsigned char *data, size_t size, const char *entry,
    const char *target, char *err, size_t errlen);

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

static uint32_t fb_rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t fb_rd64(const unsigned char *p) {
    return (uint64_t)fb_rd32(p) | ((uint64_t)fb_rd32(p + 4) << 32);
}

static const char *lift_target(char *buf, size_t len) {
    int major = 0, minor = 0;
    if (cuDeviceComputeCapability(&major, &minor, 0) == CUDA_SUCCESS)
        snprintf(buf, len, "sm_%d%d", major, minor);
    else
        buf[0] = '\0';
    return buf;
}

int fatbin_lift_ptx(const void *fatCubin, const char *entry, char **ptx,
                    size_t *ptx_len) {
    *ptx = NULL;
    *ptx_len = 0;
    const unsigned char *w = (const unsigned char *)fatCubin;
    if (!w)
        return -1;
    char target[16];
    lift_target(target, sizeof(target));
    uint32_t magic = fb_rd32(w);
    if (magic == 0xBA55ED50u) {
        uint64_t files_size = fb_rd64(w + 8);
        if (files_size > ((uint64_t)1 << 30))
            return -1;
        char err[1024];
        char *out = acuda_rtxdisam_cubin_to_ptx_for_target(
            w, 16 + (size_t)files_size, entry, target, err, sizeof(err));
        if (!out) {
            if (entry)
                fprintf(stderr, "ACuda: lift %s: %s\n",
                        entry ? entry : "(auto)", err);
            return -1;
        }
        *ptx = out;
        *ptx_len = strlen(out);
        return 0;
    }
    if (magic != 0x466243B1u)
        return -1;
    uint32_t version = fb_rd32(w + 4);
    if (version == 1) {
        const unsigned char *data = NULL;
        memcpy(&data, w + 8, sizeof(data));
        if (!data || fb_rd32(data) != 0xBA55ED50u)
            return -1;
        uint64_t files_size = fb_rd64(data + 8);
        if (files_size > ((uint64_t)1 << 30))
            return -1;
        char err[1024];
        char *out = acuda_rtxdisam_cubin_to_ptx_for_target(
            data, 16 + (size_t)files_size, entry, target, err, sizeof(err));
        if (!out) {
            if (entry)
                fprintf(stderr, "ACuda: lift %s: %s\n",
                        entry ? entry : "(auto)", err);
            return -1;
        }
        *ptx = out;
        *ptx_len = strlen(out);
        return 0;
    }
    if (version == 2) {
        const void *const *arr = NULL;
        memcpy(&arr, w + 16, sizeof(arr));
        if (!arr)
            return -1;
        for (size_t i = 0; i < 4096; i++) {
            const unsigned char *sub = NULL;
            memcpy(&sub, &arr[i], sizeof(sub));
            if (!sub)
                break;
            if (fb_rd32(sub) != 0x466243B1u || fb_rd32(sub + 4) != 1)
                continue;
            const unsigned char *data = NULL;
            memcpy(&data, sub + 8, sizeof(data));
            if (!data || fb_rd32(data) != 0xBA55ED50u)
                continue;
            uint64_t files_size = fb_rd64(data + 8);
            if (files_size > ((uint64_t)1 << 30))
                continue;
            char err[1024];
            char *out = acuda_rtxdisam_cubin_to_ptx_for_target(
                data, 16 + (size_t)files_size, entry, target, err, sizeof(err));
            if (out) {
                *ptx = out;
                *ptx_len = strlen(out);
                return 0;
            }
            if (entry)
                fprintf(stderr, "ACuda: lift %s: %s\n",
                        entry ? entry : "(auto)", err);
        }
    }
    return -1;
}

static struct {
    const char *ptx;
    char name[256];
    CUmodule mod;
} *mod_cache;
static size_t mod_cache_n, mod_cache_cap;

CUresult module_for(const char *ptx, const char *name, CUmodule *out) {
    for (size_t i = 0; i < mod_cache_n; i++) {
        if (mod_cache[i].ptx == ptx && strcmp(mod_cache[i].name, name) == 0) {
            *out = mod_cache[i].mod;
            return CUDA_SUCCESS;
        }
    }
    CUmodule mod = NULL;
    CUresult cr = cuModuleLoadData(&mod, ptx);
    if (cr != CUDA_SUCCESS)
        return cr;
    if (mod_cache_n == mod_cache_cap) {
        size_t ncap = mod_cache_cap ? mod_cache_cap * 2 : 8;
        void *np = malloc(ncap * sizeof(*mod_cache));
        if (!np)
            return CUDA_ERROR_OUT_OF_MEMORY;
        memcpy(np, mod_cache, mod_cache_n * sizeof(*mod_cache));
        free(mod_cache);
        mod_cache = np;
        mod_cache_cap = ncap;
    }
    mod_cache[mod_cache_n].ptx = ptx;
    strncpy(mod_cache[mod_cache_n].name, name, 255);
    mod_cache[mod_cache_n].name[255] = '\0';
    mod_cache[mod_cache_n].mod = mod;
    mod_cache_n++;
    *out = mod;
    return CUDA_SUCCESS;
}

CUresult module_for_fatbin(fatbin_t *fb, CUmodule *out) {
    if (!fb || !out)
        return CUDA_ERROR_INVALID_VALUE;
    /* Single loader: nvcuda parses the wrapper/fatbin and selects the image
     * (PTX or SASS), exactly as it does when driven by NVIDIA's cudart. */
    if (fb->orig_blob)
        return module_for((const char *)fb->orig_blob,
                          "__acuda_fatbin_module__", out);
    for (size_t i = 0; i < fb->nentries; i++) {
        if (fb->entries[i].ptx) {
            CUresult cr =
                module_for(fb->entries[i].ptx, "__acuda_fatbin_module__", out);
            if (cr == CUDA_SUCCESS)
                return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}
