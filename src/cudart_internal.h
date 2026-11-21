#ifndef ACUDA_CUDART_INTERNAL_H
#define ACUDA_CUDART_INTERNAL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver.h"
#include "acuda/runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACUDA_RT_VERSION 13040

/* Cubin/fatbin backend selector, shared with the ACuda driver:
 *   0 (default) - C fatbin parser first, rtxdisam (C#) SASS lift when the
 *                 container has no PTX.
 *   1           - C fatbin parser only; never call the SASS lifter.
 *   2           - rtxdisam (C#) only; skip the C fatbin parser.
 * Unset, empty or invalid values mean 0. */
static inline int cudart_cubin_backend(void) {
    const char *s = getenv("ACUDA_CUBIN_BACKEND");
    if (s && *s) {
        long v = strtol(s, NULL, 0);
        if (v == 1 || v == 2)
            return (int)v;
    }
    return 0;
}

extern int g_init;
extern int g_device;
extern cudaError_t g_last;

void set_last(cudaError_t e);
cudaError_t map_cu(CUresult r);
cudaError_t ensure_init(void);
cudaError_t check_ctx(void);
static inline CUstream map_stream(cudaStream_t s) {
    if (!s || s == cudaStreamPerThread)
        return NULL;
    return (CUstream)s;
}

typedef struct {
    char *ptx;
    size_t ptx_len;
} fat_entry_t;

typedef struct {
    const void *hostVar;
    char deviceName[256];
    size_t size;
    int constant;
    int global;
} fat_var_t;

typedef struct {
    fat_entry_t *entries;
    size_t nentries;
    struct {
        const void *hostFun;
        char deviceName[256];
        size_t entry;
    } *funcs;
    size_t nfuncs;
    fat_var_t *vars;
    size_t nvars;
    void *orig_blob;
    size_t orig_blob_len;
    int sass_only;
} fatbin_t;

extern fatbin_t **g_fatbins;
extern size_t g_nfatbins;
extern size_t g_fatcap;

int find_func(const void *func, const char **deviceName, const char **ptx);
int find_var(const void *hostVar, const char **deviceName, size_t *size, fatbin_t **out_fb);
void fatbins_add(fatbin_t *fb);
void fatbins_remove(fatbin_t *fb);
size_t fatbin_add_ptx(fatbin_t *fb, char *ptx, size_t ptx_len);
int fatbin_lift_ptx(const void *fatCubin, const char *entry, char **ptx, size_t *ptx_len);
CUresult module_for(const char *ptx, const char *name, CUmodule *out);
CUresult module_for_fatbin(fatbin_t *fb, CUmodule *out);

#ifdef __cplusplus
}
#endif

#endif
