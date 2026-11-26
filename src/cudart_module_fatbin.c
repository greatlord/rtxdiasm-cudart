#include "cudart_internal.h"

#include "rtxdiasm_fatbin.h"

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

static fatbin_t *fatbin_of(void **handle) { return (fatbin_t *)handle; }

void **__cudaRegisterFatBinary(void *fatCubin) {
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr, "ACuda-cudart: __cudaRegisterFatBinary(%p)\n", fatCubin);
    fatbin_t *fb = calloc(1, sizeof(*fb));
    if (!fb)
        return NULL;
    if (fatCubin) {
        fb->orig_blob = fatCubin;
        fb->orig_blob_len = 0;
        {
            const char *dump_env = getenv("ACUDA_DUMP_FATBIN");
            if (dump_env && *dump_env) {
                const char *dump_dir = getenv("ACUDA_FATBIN_DUMP_DIR");
                if (!dump_dir || !*dump_dir)
                    dump_dir = ACUDA_DEFAULT_DUMP_DIR;
                acuda_mkdir(dump_dir);
                static int dump_cnt = 0;
                char dname[512];
                snprintf(dname, sizeof(dname), "%s/reg_%d_%p.bin", dump_dir,
                         dump_cnt++, fatCubin);
                FILE *dfp = fopen(dname, "wb");
                if (dfp) {
                    const unsigned char *b = (const unsigned char *)fatCubin;
                    fwrite(b, 1, 64 * 1024, dfp);
                    fclose(dfp);
                }
            }
        }
        /* The image is handed to the driver untouched: nvcuda owns fatbin
         * parsing (wrapper -> fatbin -> PTX/SASS), the same single loader the
         * real cudart drives.  Registering the blob here keeps one
         * implementation instead of a second, divergent one in this DLL. */
        fb->orig_blob_len = 0;
    }
    fb->sass_only = fb->nentries == 0 && fb->orig_blob != NULL;
    fb->vars = NULL;
    fb->nvars = 0;
    fb->texs = NULL;
    fb->ntexs = 0;
    fatbins_add(fb);
    return (void **)fb;
}

void __cudaRegisterTexture(void **fatCubinHandle, const void *hostVar,
                           const void **deviceAddress, const char *deviceName,
                           int dim, int norm, int ext) {
    (void)deviceAddress;
    (void)dim;
    (void)ext;
    fatbin_t *fb = fatbin_of(fatCubinHandle);
    if (!fb || !hostVar || !deviceName)
        return;
    void *nt = realloc(fb->texs, sizeof(*fb->texs) * (fb->ntexs + 1));
    if (!nt)
        return;
    fb->texs = nt;
    fb->texs[fb->ntexs].hostVar = hostVar;
    strncpy(fb->texs[fb->ntexs].deviceName, deviceName, 255);
    fb->texs[fb->ntexs].deviceName[255] = '\0';
    fb->texs[fb->ntexs].normalized = norm;
    fb->ntexs++;
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr, "ACuda-cudart: __cudaRegisterTexture(%s) norm=%d\n",
                deviceName, norm);
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) { (void)fatCubinHandle; }

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid, void *bDim,
                            void *gDim, int *wSize) {
    if (getenv("ACUDA_TRACE"))
        fprintf(stderr, "ACuda-cudart: __cudaRegisterFunction(%s)\n",
                deviceName ? deviceName : "?");
    (void)deviceFun;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bDim;
    (void)gDim;
    (void)wSize;
    fatbin_t *fb = fatbin_of(fatCubinHandle);
    if (!fb || !hostFun || !deviceName)
        return;
    size_t entry = 0;
    void *np = realloc(fb->funcs, sizeof(*fb->funcs) * (fb->nfuncs + 1));
    if (!np)
        return;
    fb->funcs = np;
    fb->funcs[fb->nfuncs].hostFun = hostFun;
    strncpy(fb->funcs[fb->nfuncs].deviceName, deviceName, 255);
    fb->funcs[fb->nfuncs].deviceName[255] = '\0';
    fb->funcs[fb->nfuncs].entry = entry;
    fb->nfuncs++;
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
    (void)deviceAddress;
    (void)ext;
    fatbin_t *fb = fatbin_of(fatCubinHandle);
    if (!fb || !hostVar || !deviceName)
        return;
    void *nv = realloc(fb->vars, sizeof(*fb->vars) * (fb->nvars + 1));
    if (!nv)
        return;
    fb->vars = nv;
    fb->vars[fb->nvars].hostVar = (const void *)hostVar;
    strncpy(fb->vars[fb->nvars].deviceName, deviceName, 255);
    fb->vars[fb->nvars].deviceName[255] = '\0';
    fb->vars[fb->nvars].size = size;
    fb->vars[fb->nvars].constant = constant;
    fb->vars[fb->nvars].global = global;
    fb->nvars++;
    if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
        fprintf(stderr, "ACuda-cudart: RegisterVar host=%p name=%s size=%zu const=%d global=%d\n",
                (const void *)hostVar, deviceName, size, constant, global);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    fatbin_t *fb = fatbin_of(fatCubinHandle);
    if (!fb)
        return;
    fatbins_remove(fb);
    for (size_t i = 0; i < fb->nentries; i++)
        free(fb->entries[i].ptx);
    free(fb->entries);
    free(fb->funcs);
    free(fb->vars);
    free(fb);
}

fatbin_t **g_fatbins;
size_t g_nfatbins;
size_t g_fatcap;

int find_func_fb(const void *func, const char **deviceName, fatbin_t **out_fb) {
    for (size_t i = 0; i < g_nfatbins; i++) {
        for (size_t f = 0; f < g_fatbins[i]->nfuncs; f++) {
            if (g_fatbins[i]->funcs[f].hostFun == func) {
                if (deviceName)
                    *deviceName = g_fatbins[i]->funcs[f].deviceName;
                if (out_fb)
                    *out_fb = g_fatbins[i];
                return 1;
            }
        }
    }
    return 0;
}

int find_func(const void *func, const char **deviceName, const char **ptx) {
    for (size_t i = 0; i < g_nfatbins; i++) {
        for (size_t f = 0; f < g_fatbins[i]->nfuncs; f++) {
            if (g_fatbins[i]->funcs[f].hostFun == func) {
                size_t e = g_fatbins[i]->funcs[f].entry;
                if (e < g_fatbins[i]->nentries &&
                    g_fatbins[i]->entries[e].ptx) {
                    *deviceName = g_fatbins[i]->funcs[f].deviceName;
                    *ptx = g_fatbins[i]->entries[e].ptx;
                    return 1;
                }
                return 0;
            }
        }
    }
    return 0;
}

int find_var(const void *hostVar, const char **deviceName, size_t *size, fatbin_t **out_fb) {
    for (size_t i = 0; i < g_nfatbins; i++) {
        for (size_t v = 0; v < g_fatbins[i]->nvars; v++) {
            if (g_fatbins[i]->vars[v].hostVar == hostVar) {
                if (deviceName) *deviceName = g_fatbins[i]->vars[v].deviceName;
                if (size) *size = g_fatbins[i]->vars[v].size;
                if (out_fb) *out_fb = g_fatbins[i];
                return 1;
            }
        }
    }
    return 0;
}

int find_tex(const void *hostVar, fatbin_t **out_fb, const char **name,
             int *normalized) {
    for (size_t i = 0; i < g_nfatbins; i++) {
        for (size_t t = 0; t < g_fatbins[i]->ntexs; t++) {
            if (g_fatbins[i]->texs[t].hostVar == hostVar) {
                if (out_fb) *out_fb = g_fatbins[i];
                if (name) *name = g_fatbins[i]->texs[t].deviceName;
                if (normalized) *normalized = g_fatbins[i]->texs[t].normalized;
                return 1;
            }
        }
    }
    return 0;
}

void fatbins_add(fatbin_t *fb) {
    if (g_nfatbins == g_fatcap) {
        size_t ncap = g_fatcap ? g_fatcap * 2 : 4;
        fatbin_t **nl = malloc(sizeof(*nl) * ncap);
        if (!nl)
            return;
        memcpy(nl, g_fatbins, sizeof(*nl) * g_nfatbins);
        free(g_fatbins);
        g_fatbins = nl;
        g_fatcap = ncap;
    }
    g_fatbins[g_nfatbins++] = fb;
}

void fatbins_remove(fatbin_t *fb) {
    for (size_t i = 0; i < g_nfatbins; i++) {
        if (g_fatbins[i] == fb) {
            memmove(&g_fatbins[i], &g_fatbins[i + 1],
                    sizeof(*g_fatbins) * (g_nfatbins - i - 1));
            g_nfatbins--;
            return;
        }
    }
}

size_t fatbin_add_ptx(fatbin_t *fb, char *ptx, size_t ptx_len) {
    fat_entry_t *ne = realloc(fb->entries,
                              sizeof(*fb->entries) * (fb->nentries + 1));
    if (!ne) {
        free(ptx);
        return (size_t)-1;
    }
    fb->entries = ne;
    fb->entries[fb->nentries].ptx = ptx;
    fb->entries[fb->nentries].ptx_len = ptx_len;
    return fb->nentries++;
}
