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
    fat_entry_t *entries;
    size_t nentries;
    struct {
        const void *hostFun;
        char deviceName[256];
        size_t entry;
    } *funcs;
    size_t nfuncs;
    void *orig_blob;
    size_t orig_blob_len;
    int sass_only;
} fatbin_t;

extern fatbin_t **g_fatbins;
extern size_t g_nfatbins;
extern size_t g_fatcap;

int find_func(const void *func, const char **deviceName, const char **ptx);
void fatbins_add(fatbin_t *fb);
void fatbins_remove(fatbin_t *fb);
size_t fatbin_add_ptx(fatbin_t *fb, char *ptx, size_t ptx_len);
int fatbin_lift_ptx(const void *fatCubin, const char *entry, char **ptx, size_t *ptx_len);
CUresult module_for(const char *ptx, const char *name, CUmodule *out);

#ifdef __cplusplus
}
#endif

#endif
