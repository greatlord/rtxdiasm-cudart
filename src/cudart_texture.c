#include "cudart_internal.h"
#include "acuda/driver.h"

#ifndef ACUDA_CUDAAPI
#if defined(_WIN32) && defined(_M_IX86)
#define ACUDA_CUDAAPI __stdcall
#else
#define ACUDA_CUDAAPI
#endif
#endif
extern CUresult ACUDA_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
extern CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                                   const void *srcHost,
                                                   size_t ByteCount,
                                                   void *hStream);
extern CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                                   CUdeviceptr srcDevice,
                                                   size_t ByteCount,
                                                   void *hStream);

static int chan_count(const cudaChannelFormatDesc *d) {
    return (d->x ? 1 : 0) + (d->y ? 1 : 0) + (d->z ? 1 : 0) + (d->w ? 1 : 0);
}

static int to_cu_format(const cudaChannelFormatDesc *d, CUarray_format *f,
                        unsigned *nch) {
    int n = chan_count(d);
    *nch = (unsigned)n;
    if (d->f == cudaChannelFormatKindFloat && d->x == 32) {
        *f = CU_AD_FORMAT_FLOAT;
        return 0;
    }
    if (d->f == cudaChannelFormatKindUnsigned && d->x == 32) {
        *f = CU_AD_FORMAT_UNSIGNED_INT32;
        return 0;
    }
    if (d->f == cudaChannelFormatKindUnsigned && d->x == 8) {
        *f = CU_AD_FORMAT_UNSIGNED_INT8;
        return 0;
    }
    return -1;
}

cudaError_t cudaMallocArray(cudaArray_t *array,
                            const cudaChannelFormatDesc *desc, size_t width,
                            size_t height, unsigned int flags) {
    if (!array || !desc || width == 0)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUarray_format f;
    unsigned n = 0;
    if (to_cu_format(desc, &f, &n) != 0)
        return cudaErrorInvalidValue;
    CUDA_ARRAY_DESCRIPTOR ad = { width, height, f, n };
    CUarray a = NULL;
    if (cuArrayCreate(&a, &ad) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    *array = (cudaArray_t)(void *)a;
    (void)flags;
    return cudaSuccess;
}

cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                              const cudaChannelFormatDesc *desc,
                              struct cudaExtent_st extent,
                              unsigned int flags) {
    if (!array || !desc || extent.width == 0)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUarray_format f;
    unsigned n = 0;
    if (to_cu_format(desc, &f, &n) != 0)
        return cudaErrorInvalidValue;
    CUDA_ARRAY3D_DESCRIPTOR d = {
        .Width = extent.width,
        .Height = extent.height ? extent.height : 1,
        .Depth = extent.depth ? extent.depth : 1,
        .Format = f,
        .NumChannels = n,
        .Flags = 0,
    };
    if (flags & cudaArrayLayered)
        d.Flags |= CUDA_ARRAY3D_LAYERED;
    if (flags & cudaArrayCubemap)
        d.Flags |= CUDA_ARRAY3D_CUBEMAP;
    if (flags & cudaArraySurfaceLoadStore)
        d.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
    CUarray a = NULL;
    if (cuArray3DCreate(&a, &d) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    *array = (cudaArray_t)(void *)a;
    return cudaSuccess;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                     const cudaChannelFormatDesc *desc,
                                     struct cudaExtent_st extent,
                                     unsigned int numLevels,
                                     unsigned int flags) {
    if (!mipmappedArray || !desc || extent.width == 0 || numLevels == 0)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUarray_format f;
    unsigned n = 0;
    if (to_cu_format(desc, &f, &n) != 0)
        return cudaErrorInvalidValue;
    CUDA_ARRAY3D_DESCRIPTOR d = {
        .Width = extent.width,
        .Height = extent.height ? extent.height : 1,
        .Depth = extent.depth ? extent.depth : 1,
        .Format = f,
        .NumChannels = n,
        .Flags = 0,
    };
    if (flags & cudaArrayLayered)
        d.Flags |= CUDA_ARRAY3D_LAYERED;
    if (flags & cudaArrayCubemap)
        d.Flags |= CUDA_ARRAY3D_CUBEMAP;
    CUmipmappedArray m = NULL;
    if (cuMipmappedArrayCreate(&m, &d, numLevels) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    *mipmappedArray = (cudaMipmappedArray_t)(void *)m;
    return cudaSuccess;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
    if (!array)
        return cudaSuccess;
    return map_cu(cuArrayDestroy((CUarray)(void *)array));
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
    if (!mipmappedArray)
        return cudaSuccess;
    return map_cu(cuMipmappedArrayDestroy(
        (CUmipmappedArray)(void *)mipmappedArray));
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                                       cudaMipmappedArray_const_t m,
                                       unsigned int level) {
    if (!levelArray || !m)
        return cudaErrorInvalidValue;
    CUarray a = NULL;
    CUresult r = cuMipmappedArrayGetLevel(&a,
                                          (CUmipmappedArray)(void *)m, level);
    if (r != CUDA_SUCCESS)
        return map_cu(r);
    *levelArray = (cudaArray_t)(void *)a;
    return cudaSuccess;
}

static cudaError_t to_cu_resdesc(const cudaResourceDesc *pResDesc,
                                 CUDA_RESOURCE_DESC *rd) {
    memset(rd, 0, sizeof(*rd));
    rd->resType = (CUresourcetype)pResDesc->resType;
    if (pResDesc->resType == cudaResourceTypeArray) {
        rd->res.array.hArray = (CUarray)(void *)pResDesc->res.array.array;
    } else if (pResDesc->resType == cudaResourceTypeMipmappedArray) {
        rd->res.mipmap.hMipmappedArray =
            (CUmipmappedArray)(void *)pResDesc->res.mipmap.mipmap;
    } else if (pResDesc->resType == cudaResourceTypeLinear) {
        CUarray_format cf;
        unsigned nch = 0;
        if (to_cu_format(&pResDesc->res.linear.desc, &cf, &nch) != 0)
            return cudaErrorNotSupported;
        rd->res.linear.devPtr =
            (CUdeviceptr)(uintptr_t)pResDesc->res.linear.devPtr;
        rd->res.linear.format = cf;
        rd->res.linear.numChannels = nch;
        rd->res.linear.sizeInBytes = pResDesc->res.linear.sizeInBytes;
    } else if (pResDesc->resType == cudaResourceTypePitch2D) {
        CUarray_format cf;
        unsigned nch = 0;
        if (to_cu_format(&pResDesc->res.pitch2D.desc, &cf, &nch) != 0)
            return cudaErrorNotSupported;
        rd->res.pitch2D.devPtr =
            (CUdeviceptr)(uintptr_t)pResDesc->res.pitch2D.devPtr;
        rd->res.pitch2D.format = cf;
        rd->res.pitch2D.numChannels = nch;
        rd->res.pitch2D.width = pResDesc->res.pitch2D.width;
        rd->res.pitch2D.height = pResDesc->res.pitch2D.height;
        rd->res.pitch2D.pitchInBytes = pResDesc->res.pitch2D.pitchInBytes;
    } else {
        return cudaErrorNotSupported;
    }
    return cudaSuccess;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const cudaResourceDesc *pResDesc,
                                    const cudaTextureDesc *pTexDesc,
                                    const cudaResourceViewDesc *pResViewDesc) {
    if (!pTexObject || !pResDesc || !pTexDesc)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUDA_RESOURCE_DESC rd;
    r = to_cu_resdesc(pResDesc, &rd);
    if (r != cudaSuccess)
        return r;
    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof(td));
    for (int i = 0; i < 3; i++)
        td.addressMode[i] = (CUaddress_mode)pTexDesc->addressMode[i];
    td.filterMode = (CUfilter_mode)pTexDesc->filterMode;
    td.mipmapFilterMode = (CUfilter_mode)pTexDesc->mipmapFilterMode;
    for (int i = 0; i < 4; i++)
        td.borderColor[i] = pTexDesc->borderColor[i];
    td.maxAnisotropy = pTexDesc->maxAnisotropy;
    if (pTexDesc->normalizedCoords)
        td.flags |= 0x2;
    CUDA_RESOURCE_VIEW_DESC vd;
    CUDA_RESOURCE_VIEW_DESC *pvd = NULL;
    if (pResViewDesc) {
        memset(&vd, 0, sizeof(vd));
        vd.format = (CUresourceViewFormat)pResViewDesc->format;
        vd.width = pResViewDesc->width;
        vd.height = pResViewDesc->height;
        vd.depth = pResViewDesc->depth;
        vd.firstMipmapLevel = pResViewDesc->firstMipmapLevel;
        vd.lastMipmapLevel = pResViewDesc->lastMipmapLevel;
        vd.firstLayer = pResViewDesc->firstLayer;
        vd.lastLayer = pResViewDesc->lastLayer;
        pvd = &vd;
    }
    CUtexObject to = 0;
    CUresult cr = cuTexObjectCreate(&to, &rd, &td, pvd);
    if (cr != CUDA_SUCCESS)
        return map_cu(cr);
    *pTexObject = (cudaTextureObject_t)to;
    return cudaSuccess;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    return map_cu(cuTexObjectDestroy((CUtexObject)texObject));
}

cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *pResDesc,
                                             cudaTextureObject_t texObject) {
    if (!pResDesc)
        return cudaErrorInvalidValue;
    CUDA_RESOURCE_DESC rd;
    CUresult r = cuTexObjectGetResourceDesc(&rd, (CUtexObject)texObject);
    if (r != CUDA_SUCCESS)
        return map_cu(r);
    memset(pResDesc, 0, sizeof(*pResDesc));
    pResDesc->resType = (cudaResourceType)rd.resType;
    if (rd.resType == CU_RESOURCE_TYPE_ARRAY)
        pResDesc->res.array.array = (cudaArray_t)(void *)rd.res.array.hArray;
    else if (rd.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY)
        pResDesc->res.mipmap.mipmap =
            (cudaMipmappedArray_t)(void *)rd.res.mipmap.hMipmappedArray;
    return cudaSuccess;
}

cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *pTexDesc,
                                            cudaTextureObject_t texObject) {
    if (!pTexDesc)
        return cudaErrorInvalidValue;
    CUDA_TEXTURE_DESC td;
    CUresult r = cuTexObjectGetTextureDesc(&td, (CUtexObject)texObject);
    if (r != CUDA_SUCCESS)
        return map_cu(r);
    memset(pTexDesc, 0, sizeof(*pTexDesc));
    for (int i = 0; i < 3; i++)
        pTexDesc->addressMode[i] = (cudaTextureAddressMode)td.addressMode[i];
    pTexDesc->filterMode = (cudaTextureFilterMode)td.filterMode;
    pTexDesc->mipmapFilterMode = (cudaTextureFilterMode)td.mipmapFilterMode;
    for (int i = 0; i < 4; i++)
        pTexDesc->borderColor[i] = td.borderColor[i];
    pTexDesc->maxAnisotropy = td.maxAnisotropy;
    return cudaSuccess;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                    const cudaResourceDesc *pResDesc) {
    if (!pSurfObject || !pResDesc)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUDA_RESOURCE_DESC rd;
    r = to_cu_resdesc(pResDesc, &rd);
    if (r != cudaSuccess)
        return r;
    CUsurfObject so = 0;
    CUresult cr = cuSurfObjectCreate(&so, &rd);
    if (cr != CUDA_SUCCESS)
        return map_cu(cr);
    *pSurfObject = (cudaSurfaceObject_t)so;
    return cudaSuccess;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    return map_cu(cuSurfObjectDestroy((CUsurfObject)surfObject));
}

cudaError_t cudaGetSurfaceObjectResourceDesc(
    cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
    if (!pResDesc)
        return cudaErrorInvalidValue;
    CUDA_RESOURCE_DESC rd;
    CUresult r = cuSurfObjectGetResourceDesc(&rd, (CUsurfObject)surfObject);
    if (r != CUDA_SUCCESS)
        return map_cu(r);
    memset(pResDesc, 0, sizeof(*pResDesc));
    pResDesc->resType = (cudaResourceType)rd.resType;
    if (rd.resType == CU_RESOURCE_TYPE_ARRAY)
        pResDesc->res.array.array = (cudaArray_t)(void *)rd.res.array.hArray;
    else if (rd.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY)
        pResDesc->res.mipmap.mipmap =
            (cudaMipmappedArray_t)(void *)rd.res.mipmap.hMipmappedArray;
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset,
                                size_t hOffset, const void *src, size_t spitch,
                                size_t width, size_t height,
                                cudaMemcpyKind kind) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!dst || !src || width == 0)
        return cudaErrorInvalidValue;
    CUDA_MEMCPY2D m;
    memset(&m, 0, sizeof(m));
    if (kind == cudaMemcpyDeviceToDevice) {
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = (CUdeviceptr)(uintptr_t)src;
    } else {
        m.srcMemoryType = CU_MEMORYTYPE_HOST;
        m.srcHost = src;
    }
    m.srcPitch = spitch;
    m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    m.dstArray = (CUarray)(void *)dst;
    m.dstXInBytes = wOffset;
    m.dstY = hOffset;
    m.WidthInBytes = width;
    m.Height = height;
    return map_cu(cuMemcpy2D(&m));
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                     size_t hOffset, const void *src,
                                     size_t spitch, size_t width,
                                     size_t height,
                                     cudaMemcpyKind kind,
                                     cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width,
                               height, kind);
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                              const void *src, size_t count,
                              cudaMemcpyKind kind) {
    return cudaMemcpy2DToArray(dst, wOffset, hOffset, src, count, count, 1, kind);
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset,
                                   size_t hOffset, const void *src,
                                   size_t count, cudaMemcpyKind kind,
                                   cudaStream_t stream) {
    (void)stream;
    return cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src,
                                size_t wOffset, size_t hOffset, size_t count,
                                cudaMemcpyKind kind) {
    return cudaMemcpy2DFromArray(dst, count, src, wOffset, hOffset, count, 1, kind);
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src,
                                     size_t wOffset, size_t hOffset,
                                     size_t count, cudaMemcpyKind kind,
                                     cudaStream_t stream) {
    (void)stream;
    return cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst,
                                   size_t hOffsetDst, cudaArray_const_t src,
                                   size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t count, cudaMemcpyKind kind) {
    return cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src,
                                    wOffsetSrc, hOffsetSrc, count, 1, kind);
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                  cudaArray_const_t src, size_t wOffset,
                                  size_t hOffset, size_t width, size_t height,
                                  cudaMemcpyKind kind) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!dst || !src || width == 0)
        return cudaErrorInvalidValue;
    CUDA_MEMCPY2D m;
    memset(&m, 0, sizeof(m));
    m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    m.srcArray = (CUarray)(void *)src;
    m.srcXInBytes = wOffset;
    m.srcY = hOffset;
    if (kind == cudaMemcpyDeviceToDevice) {
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        m.dstDevice = (CUdeviceptr)(uintptr_t)dst;
    } else {
        m.dstMemoryType = CU_MEMORYTYPE_HOST;
        m.dstHost = dst;
    }
    m.dstPitch = dpitch;
    m.WidthInBytes = width;
    m.Height = height;
    return map_cu(cuMemcpy2D(&m));
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                       cudaArray_const_t src, size_t wOffset,
                                       size_t hOffset, size_t width,
                                       size_t height,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width,
                                 height, kind);
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst,
                                     size_t hOffsetDst, cudaArray_const_t src,
                                     size_t wOffsetSrc, size_t hOffsetSrc,
                                     size_t width, size_t height,
                                     cudaMemcpyKind kind) {
    (void)kind;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!dst || !src || width == 0)
        return cudaErrorInvalidValue;
    CUDA_MEMCPY2D m;
    memset(&m, 0, sizeof(m));
    m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    m.srcArray = (CUarray)(void *)src;
    m.srcXInBytes = wOffsetSrc;
    m.srcY = hOffsetSrc;
    m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    m.dstArray = (CUarray)(void *)dst;
    m.dstXInBytes = wOffsetDst;
    m.dstY = hOffsetDst;
    m.WidthInBytes = width;
    m.Height = height;
    return map_cu(cuMemcpy2D(&m));
}

static void cudart_fill_memcpy3d(CUDA_MEMCPY3D *m,
                                 const struct cudaMemcpy3DParms_st *p,
                                 int src) {
    if (src) {
        if (p->srcArray) {
            m->srcMemoryType = CU_MEMORYTYPE_ARRAY;
            m->srcArray = (CUarray)(void *)p->srcArray;
        } else if (p->srcPtr.ptr) {
            m->srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m->srcDevice = (CUdeviceptr)(uintptr_t)p->srcPtr.ptr;
        } else {
            m->srcMemoryType = CU_MEMORYTYPE_HOST;
            m->srcHost = p->srcPtr.ptr;
        }
        m->srcXInBytes = p->srcPos.x;
        m->srcY = p->srcPos.y;
        m->srcZ = p->srcPos.z;
        m->srcPitch = p->srcPtr.pitch;
        m->srcHeight = p->srcPtr.ysize;
    } else {
        if (p->dstArray) {
            m->dstMemoryType = CU_MEMORYTYPE_ARRAY;
            m->dstArray = (CUarray)(void *)p->dstArray;
        } else if (p->dstPtr.ptr) {
            m->dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m->dstDevice = (CUdeviceptr)(uintptr_t)p->dstPtr.ptr;
        } else {
            m->dstMemoryType = CU_MEMORYTYPE_HOST;
            m->dstHost = p->dstPtr.ptr;
        }
        m->dstXInBytes = p->dstPos.x;
        m->dstY = p->dstPos.y;
        m->dstZ = p->dstPos.z;
        m->dstPitch = p->dstPtr.pitch;
        m->dstHeight = p->dstPtr.ysize;
    }
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms_st *p) {
    if (!p)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUDA_MEMCPY3D m;
    memset(&m, 0, sizeof(m));
    if (p->kind == cudaMemcpyHostToDevice) {
        m.srcMemoryType = CU_MEMORYTYPE_HOST;
        m.srcHost = p->srcPtr.ptr;
        cudart_fill_memcpy3d(&m, p, 0);
    } else if (p->kind == cudaMemcpyDeviceToHost) {
        cudart_fill_memcpy3d(&m, p, 1);
        m.dstMemoryType = CU_MEMORYTYPE_HOST;
        m.dstHost = p->dstPtr.ptr;
        m.dstPitch = p->dstPtr.pitch;
        m.dstHeight = p->dstPtr.ysize;
        m.dstXInBytes = p->dstPos.x;
        m.dstY = p->dstPos.y;
        m.dstZ = p->dstPos.z;
    } else {
        cudart_fill_memcpy3d(&m, p, 1);
        cudart_fill_memcpy3d(&m, p, 0);
    }
    m.WidthInBytes = p->extent.width;
    m.Height = p->extent.height;
    m.Depth = p->extent.depth ? p->extent.depth : 1;
    return map_cu(cuMemcpy3D(&m));
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms_st *p,
                              cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy3D(p);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (width == 0 || height == 0)
        return cudaSuccess;
    if (!dst || !src || width > dpitch || width > spitch)
        return cudaErrorInvalidValue;
    for (size_t row = 0; row < height; row++) {
        r = cudaMemcpyAsync((char *)dst + row * dpitch,
                            (const char *)src + row * spitch, width, kind,
                            stream);
        if (r != cudaSuccess)
            return r;
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream) {
    (void)stream;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (dstDevice != srcDevice)
        return cudaErrorNotSupported;
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

static cudaError_t cudart_resolve_symbol(const void *symbol, CUdeviceptr *dptr, size_t *sym_size) {
    const char *name = NULL;
    size_t reg_size = 0;
    fatbin_t *fb = NULL;
    if (!symbol || !dptr)
        return cudaErrorInvalidValue;
    if (!find_var(symbol, &name, &reg_size, &fb) || !name || !fb) {
        if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
            fprintf(stderr, "ACuda-cudart: resolve_symbol host=%p NOT FOUND\n", symbol);
        return cudaErrorSymbolNotFound;
    }
    CUmodule mod = NULL;
    CUresult cr = module_for_fatbin(fb, &mod);
    if (cr != CUDA_SUCCESS) {
        if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
            fprintf(stderr, "ACuda-cudart: resolve_symbol host=%p name=%s module load failed %d\n", symbol, name, (int)cr);
        return map_cu(cr);
    }
    CUdeviceptr dp = 0;
    size_t sz = 0;
    cr = cuModuleGetGlobal_v2(&dp, &sz, mod, name);
    if (cr != CUDA_SUCCESS) {
        if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
            fprintf(stderr, "ACuda-cudart: resolve_symbol host=%p name=%s cuModuleGetGlobal failed %d\n", symbol, name, (int)cr);
        return map_cu(cr);
    }
    if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
        fprintf(stderr, "ACuda-cudart: resolve_symbol host=%p name=%s dp=0x%llx sz=%zu\n",
                symbol, name, (unsigned long long)dp, sz);
    *dptr = dp;
    if (sym_size)
        *sym_size = sz ? sz : reg_size;
    return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                               size_t count, size_t offset, int kind) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!symbol)
        return cudaErrorInvalidValue;
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    r = cudart_resolve_symbol(symbol, &dptr, &sym_size);
    if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
        fprintf(stderr, "ACuda-cudart: cudaMemcpyToSymbol host=%p -> dp=0x%llx sym_size=%zu count=%zu off=%zu kind=%d resolve=%d\n",
                symbol, (unsigned long long)dptr, sym_size, count, offset, kind, (int)r);
    if (r != cudaSuccess)
        return r;
    if (count == 0)
        count = sym_size > offset ? sym_size - offset : 0;
    if (offset + count > sym_size)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    switch ((cudaMemcpyKind)kind) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDefault:
        r = map_cu(cuMemcpyHtoD_v2(dptr + (CUdeviceptr)offset, src, count));
        break;
    case cudaMemcpyDeviceToDevice:
        r = map_cu(cuMemcpyDtoD_v2(dptr + (CUdeviceptr)offset,
                                   (CUdeviceptr)(uintptr_t)src, count));
        break;
    default:
        r = cudaErrorInvalidValue;
        break;
    }
    set_last(r);
    return r;
}

cudaError_t cudart_memcpy_to_symbol_async(const void *symbol, const void *src,
                                          size_t count, size_t offset, int kind,
                                          cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!symbol)
        return cudaErrorInvalidValue;
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    r = cudart_resolve_symbol(symbol, &dptr, &sym_size);
    if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
        fprintf(stderr, "ACuda-cudart: cudaMemcpyToSymbolAsync host=%p -> dp=0x%llx sym_size=%zu count=%zu off=%zu kind=%d resolve=%d\n",
                symbol, (unsigned long long)dptr, sym_size, count, offset, kind, (int)r);
    if (r != cudaSuccess)
        return r;
    if (count == 0)
        count = sym_size > offset ? sym_size - offset : 0;
    if (offset + count > sym_size)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    switch ((cudaMemcpyKind)kind) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDefault:
        r = map_cu(cuMemcpyHtoDAsync_v2(dptr + (CUdeviceptr)offset, src, count,
                                        (void *)stream));
        break;
    case cudaMemcpyDeviceToDevice:
        /* Ordered copy: PhysX fills the source from a preceding kernel in the
         * same stream, so the block must be complete before this memmove.  The
         * stream timeline deadlock that forced the async path is fixed by the
         * single-context change. */
        r = map_cu(cuMemcpyDtoD_v2(dptr + (CUdeviceptr)offset,
                                   (CUdeviceptr)(uintptr_t)src, count));
        break;
    default:
        r = cudaErrorInvalidValue;
        break;
    }
    set_last(r);
    return r;
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                 size_t count, size_t offset, int kind) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!symbol)
        return cudaErrorInvalidValue;
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    r = cudart_resolve_symbol(symbol, &dptr, &sym_size);
    if (getenv("ACUDA_TRACE") || getenv("ACUDA_DEBUG_SYMBOLS"))
        fprintf(stderr, "ACuda-cudart: cudaMemcpyFromSymbol host=%p -> dp=0x%llx sym_size=%zu count=%zu off=%zu kind=%d resolve=%d\n",
                symbol, (unsigned long long)dptr, sym_size, count, offset, kind, (int)r);
    if (r != cudaSuccess)
        return r;
    if (count == 0)
        count = sym_size > offset ? sym_size - offset : 0;
    if (offset + count > sym_size)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    switch ((cudaMemcpyKind)kind) {
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDefault:
        r = map_cu(cuMemcpyDtoH_v2(dst, dptr + (CUdeviceptr)offset, count));
        break;
    case cudaMemcpyDeviceToDevice:
        r = map_cu(cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst,
                                   dptr + (CUdeviceptr)offset, count));
        break;
    default:
        r = cudaErrorInvalidValue;
        break;
    }
    set_last(r);
    return r;
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!devPtr || !symbol)
        return cudaErrorInvalidValue;
    CUdeviceptr dptr = 0;
    r = cudart_resolve_symbol(symbol, &dptr, NULL);
    if (r != cudaSuccess)
        return r;
    *devPtr = (void *)(uintptr_t)dptr;
    return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!size || !symbol)
        return cudaErrorInvalidValue;
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    r = cudart_resolve_symbol(symbol, &dptr, &sym_size);
    if (r != cudaSuccess)
        return r;
    *size = sym_size;
    return cudaSuccess;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    (void)ptr;
    (void)size;
    (void)flags;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    return cudaErrorNotSupported;
}

cudaError_t cudaHostUnregister(void *ptr) {
    (void)ptr;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    return cudaErrorNotSupported;
}


cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                            enum cudaChannelFormatKind f) {
    cudaChannelFormatDesc d;
    d.x = x; d.y = y; d.z = z; d.w = w; d.f = f;
    return d;
}

cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc, cudaArray_t array) {
    if (!desc) return cudaErrorInvalidValue;
    memset(desc, 0, sizeof(*desc));
    if (!array) return cudaErrorInvalidValue;
    {
        CUDA_ARRAY_DESCRIPTOR ad;
        if (cuArrayGetDescriptor(&ad, (CUarray)(void *)array) == CUDA_SUCCESS) {
            unsigned n = ad.NumChannels; int bits = 32; enum cudaChannelFormatKind f = (enum cudaChannelFormatKind)cudaChannelFormatKindFloat;
            switch (ad.Format) {
            case CU_AD_FORMAT_UNSIGNED_INT8: bits=8; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_UNSIGNED_INT16: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_UNSIGNED_INT32: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_SIGNED_INT8: bits=8; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_SIGNED_INT16: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_SIGNED_INT32: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_HALF: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindFloat; break;
            case CU_AD_FORMAT_FLOAT: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindFloat; break;
            default: break;
            }
            desc->x = n>=1?bits:0; desc->y = n>=2?bits:0; desc->z = n>=3?bits:0; desc->w = n>=4?bits:0; desc->f = f;
            return cudaSuccess;
        }
    }
    {
        CUDA_ARRAY3D_DESCRIPTOR ad3;
        if (cuArray3DGetDescriptor(&ad3, (CUarray)(void *)array) == CUDA_SUCCESS) {
            unsigned n = ad3.NumChannels; int bits = 32; enum cudaChannelFormatKind f = (enum cudaChannelFormatKind)cudaChannelFormatKindFloat;
            switch (ad3.Format) {
            case CU_AD_FORMAT_UNSIGNED_INT8: bits=8; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_UNSIGNED_INT16: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_UNSIGNED_INT32: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindUnsigned; break;
            case CU_AD_FORMAT_SIGNED_INT8: bits=8; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_SIGNED_INT16: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_SIGNED_INT32: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindSigned; break;
            case CU_AD_FORMAT_HALF: bits=16; f=(enum cudaChannelFormatKind)cudaChannelFormatKindFloat; break;
            case CU_AD_FORMAT_FLOAT: bits=32; f=(enum cudaChannelFormatKind)cudaChannelFormatKindFloat; break;
            default: break;
            }
            desc->x = n>=1?bits:0; desc->y = n>=2?bits:0; desc->z = n>=3?bits:0; desc->w = n>=4?bits:0; desc->f = f;
            return cudaSuccess;
        }
    }
    return cudaErrorInvalidValue;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc_st *pResViewDesc,
                                                 cudaTextureObject_t texObject) {
    if (!pResViewDesc) return cudaErrorInvalidValue;
    CUDA_RESOURCE_VIEW_DESC vd;
    memset(&vd, 0, sizeof(vd));
    CUresult r = cuTexObjectGetResourceViewDesc(&vd, (CUtexObject)texObject);
    if (r != CUDA_SUCCESS) return map_cu(r);
    memset(pResViewDesc, 0, sizeof(*pResViewDesc));
    pResViewDesc->format = (cudaResourceViewFormat)vd.format;
    pResViewDesc->width = vd.width;
    pResViewDesc->height = vd.height;
    pResViewDesc->depth = vd.depth;
    pResViewDesc->firstMipmapLevel = vd.firstMipmapLevel;
    pResViewDesc->lastMipmapLevel = vd.lastMipmapLevel;
    pResViewDesc->firstLayer = vd.firstLayer;
    pResViewDesc->lastLayer = vd.lastLayer;
    return cudaSuccess;
}
