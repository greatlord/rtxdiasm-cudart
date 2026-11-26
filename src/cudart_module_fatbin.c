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

static int looks_like_ptx_text(const char *text, size_t len) {
    size_t i = 0;
    while (i < len && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' ||
                       text[i] == '\r'))
        i++;
    if (i + 1 < len && text[i] == '/' && text[i + 1] == '/')
        return 1;
    return i + 8 <= len && memcmp(text + i, ".version", 8) == 0;
}

static int fatbin_extract_ptx(const unsigned char *data, size_t len,
                              char **ptx, size_t *ptx_len) {
    *ptx = NULL;
    *ptx_len = 0;
    if (len < 16)
        return -1;
    uint32_t magic = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
                     ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
    if (magic == 0x466243b1u) {
        size_t off = 16;
        while (off + 32 <= len) {
            uint16_t kind = (uint16_t)data[off] | ((uint16_t)data[off + 1] << 8);
            uint64_t size = 0, start = 0;
            memcpy(&size, data + off + 16, 8);
            memcpy(&start, data + off + 24, 8);
            if (kind == 1 && size > 0 && start + size <= len && start >= 16) {
                char *copy = malloc(size + 1);
                if (!copy)
                    return -1;
                memcpy(copy, data + start, size);
                copy[size] = '\0';
                if (looks_like_ptx_text(copy, size)) {
                    *ptx = copy;
                    *ptx_len = size;
                    return 0;
                }
                free(copy);
            }
            if (size > len)
                break;
            off += 32;
        }
        return -1;
    }
    if (magic == 0xBA55ED50u && len >= 16) {
        /* Shared with acuda via rtxdiasm-fatbin-parse for 100% identical
         * raw/LZ4/ZSTD PTX extraction from fatbin blobs. */
        if (rtxdiasm_fatbin_extract_first_ptx(data, len, ptx, ptx_len) ==
            RTXDIASM_FATBIN_OK)
            return 0;
        return -1;
    }
    if ((data[0] == 0x01 && data[1] == 0x5A && data[2] == 0xE5 &&
         data[3] == 0x1E) ||
        magic == 0x1EE55A01u) {
        for (size_t i = 0; i + 64 < len; i++) {
            if (looks_like_ptx_text((const char *)data + i, len - i)) {
                size_t j = i;
                while (j < len && data[j] != 0)
                    j++;
                char *copy = malloc(j - i + 1);
                if (!copy)
                    return -1;
                memcpy(copy, data + i, j - i);
                copy[j - i] = '\0';
                *ptx = copy;
                *ptx_len = j - i;
                return 0;
            }
        }
        return -1;
    }
    return -1;
}

static fatbin_t *fatbin_of(void **handle) { return (fatbin_t *)handle; }

/* __fatBinC_Wrapper_t for CUDA 2.x-4.x compilers:
 *   uint32 magic == 0x466243B1; uint32 version == 1;
 *   const void *data;        <- fatbin image (0xBA55ED50)
 *   const char *filename;
 * The image handed to __cudaRegisterFatBinary is the wrapper, not the
 * fatbin, so unwrap it before any parsing. */
static const unsigned char *fatbin_unwrap(const unsigned char *blob,
                                          size_t *len_out) {
    uint32_t magic = 0, version = 0;
    if (!blob)
        return NULL;
    memcpy(&magic, blob, 4);
    if (magic != 0x466243B1u)
        return blob;
    memcpy(&version, blob + 4, 4);
    if (version != 1)
        return blob;
    const void *inner = NULL;
    memcpy(&inner, blob + 8, sizeof(inner));
    if (!inner)
        return blob;
    uint32_t imagic = 0;
    memcpy(&imagic, inner, 4);
    if (imagic != 0xBA55ED50u)
        return blob;
    uint64_t files_size = 0;
    memcpy(&files_size, (const unsigned char *)inner + 8, 8);
    if (files_size == 0 || files_size > ((uint64_t)1 << 30))
        return blob;
    if (len_out)
        *len_out = 16 + (size_t)files_size;
    return (const unsigned char *)inner;
}

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
        size_t len = 0;
        const unsigned char *image =
            fatbin_unwrap((const unsigned char *)fatCubin, &len);
        if (len == 0) {
            /* Not a version-1 wrapper: treat it as a text blob. */
            len = strlen((const char *)fatCubin);
        }
        if (looks_like_ptx_text((const char *)image, len)) {
            fb->entries = malloc(sizeof(*fb->entries));
            if (!fb->entries) {
                free(fb);
                return NULL;
            }
            fb->entries[0].ptx = malloc(len + 1);
            if (!fb->entries[0].ptx) {
                free(fb->entries);
                free(fb);
                return NULL;
            }
            memcpy(fb->entries[0].ptx, image, len + 1);
            fb->entries[0].ptx_len = len;
            fb->nentries = 1;
        } else {
            char *ptx = NULL;
            size_t ptx_len = 0;
            if (cudart_cubin_backend() != 2) {
                if (len) {
                    fatbin_extract_ptx(image, len, &ptx, &ptx_len);
                } else {
                    for (size_t probe = 512; probe <= 1 << 20; probe *= 2) {
                        if (fatbin_extract_ptx(fatCubin, probe, &ptx, &ptx_len) == 0)
                            break;
                    }
                }
            }
            if (!ptx && cudart_cubin_backend() != 1)
                fatbin_lift_ptx(fatCubin, NULL, &ptx, &ptx_len);
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr,
                        "ACuda-cudart: fatbin image=%p len=%zu ptx=%p ptx_len=%zu\n",
                        (const void *)image, len, (void *)ptx, ptx_len);
            if (ptx) {
                fb->entries = malloc(sizeof(*fb->entries));
                if (!fb->entries) {
                    free(ptx);
                    free(fb);
                    return NULL;
                }
                fb->entries[0].ptx = ptx;
                fb->entries[0].ptx_len = ptx_len;
                fb->nentries = 1;
            }
        }
    }
    fb->sass_only = fb->nentries == 0 && fb->orig_blob != NULL;
    fb->vars = NULL;
    fb->nvars = 0;
    fatbins_add(fb);
    return (void **)fb;
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
    if (fb->sass_only && fb->orig_blob && cudart_cubin_backend() != 1) {
        char *ptx = NULL;
        size_t ptx_len = 0;
        if (fatbin_lift_ptx(fb->orig_blob, deviceName, &ptx, &ptx_len) == 0) {
            size_t added = fatbin_add_ptx(fb, ptx, ptx_len);
            if (added != (size_t)-1)
                entry = added;
        }
    }
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
