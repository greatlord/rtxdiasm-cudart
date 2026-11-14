#include "cudart_internal.h"

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
        uint16_t version = (uint16_t)data[4] | ((uint16_t)data[5] << 8);
        uint16_t header_size = (uint16_t)data[6] | ((uint16_t)data[7] << 8);
        uint64_t fatbin_size = 0;
        memcpy(&fatbin_size, data + 8, 8);
        if (version != 1 || header_size < 16)
            return -1;
        size_t pos = header_size;
        size_t end = header_size + (size_t)fatbin_size;
        if (end < pos || end > len)
            end = len;
        while (pos + 64 <= end) {
            const unsigned char *e = data + pos;
            uint16_t kind = (uint16_t)e[0] | ((uint16_t)e[1] << 8);
            uint32_t eheader_size = (uint32_t)e[4] | ((uint32_t)e[5] << 8) |
                                    ((uint32_t)e[6] << 16) | ((uint32_t)e[7] << 24);
            uint32_t payload_size = (uint32_t)e[8] | ((uint32_t)e[9] << 8) |
                                    ((uint32_t)e[10] << 16) | ((uint32_t)e[11] << 24);
            uint32_t compressed_size = (uint32_t)e[16] | ((uint32_t)e[17] << 8) |
                                       ((uint32_t)e[18] << 16) | ((uint32_t)e[19] << 24);
            uint64_t flags = 0;
            memcpy(&flags, e + 40, 8);
            if (eheader_size < 64 || eheader_size > 4096)
                break;
            size_t pay = pos + eheader_size;
            size_t disk =
                payload_size > compressed_size ? payload_size : compressed_size;
            if (pay + disk > end || disk == 0) {
                pos += eheader_size + (payload_size ? payload_size : disk);
                if (pos <= pay)
                    break;
                continue;
            }
            if (kind == 1 && !(flags & (0x2000ull | 0x8000ull)) &&
                payload_size > 0) {
                size_t n = payload_size;
                while (n > 0 && data[pay + n - 1] == 0)
                    n--;
                char *copy = malloc(n + 1);
                if (!copy)
                    return -1;
                memcpy(copy, data + pay, n);
                copy[n] = '\0';
                if (looks_like_ptx_text(copy, n)) {
                    *ptx = copy;
                    *ptx_len = n;
                    return 0;
                }
                free(copy);
            }
            size_t step = eheader_size + disk;
            if (step == 0)
                break;
            pos += step;
        }
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

void **__cudaRegisterFatBinary(void *fatCubin) {
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
        size_t len = strlen((const char *)fatCubin);
        if (looks_like_ptx_text((const char *)fatCubin, len)) {
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
            memcpy(fb->entries[0].ptx, fatCubin, len + 1);
            fb->entries[0].ptx_len = len;
            fb->nentries = 1;
        } else {
            char *ptx = NULL;
            size_t ptx_len = 0;
            for (size_t probe = 512; probe <= 1 << 20; probe *= 2) {
                if (fatbin_extract_ptx(fatCubin, probe, &ptx, &ptx_len) == 0)
                    break;
            }
            if (!ptx)
                fatbin_lift_ptx(fatCubin, NULL, &ptx, &ptx_len);
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
    fatbins_add(fb);
    return (void **)fb;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) { (void)fatCubinHandle; }

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid, void *bDim,
                            void *gDim, int *wSize) {
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
    if (fb->sass_only && fb->orig_blob) {
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
    (void)fatCubinHandle;
    (void)hostVar;
    (void)deviceAddress;
    (void)deviceName;
    (void)ext;
    (void)size;
    (void)constant;
    (void)global;
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
