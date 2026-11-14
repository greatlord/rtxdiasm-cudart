#include "cudart_internal.h"

static CUevent map_event(cudaEvent_t e) { return (CUevent)e; }

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    return cudaStreamCreateWithFlags(pStream, 0);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
    if (!pStream)
        return cudaErrorInvalidValue;
    *pStream = NULL;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUstream s = NULL;
    r = map_cu(cuStreamCreate(&s, flags));
    set_last(r);
    if (r != cudaSuccess)
        return r;
    *pStream = (cudaStream_t)s;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
    (void)priority;
    return cudaStreamCreateWithFlags(pStream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!stream || stream == cudaStreamPerThread)
        return cudaSuccess;
    r = map_cu(cuStreamDestroy_v2((CUstream)stream));
    set_last(r);
    return r;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    r = map_cu(cuStreamSynchronize(map_stream(stream)));
    set_last(r);
    return r;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!stream || stream == cudaStreamPerThread)
        return cudaSuccess;
    r = map_cu(cuStreamQuery((CUstream)stream));
    set_last(r);
    return r;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    r = map_cu(cuStreamWaitEvent(map_stream(stream), map_event(event),
                                 flags));
    set_last(r);
    return r;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
    (void)flags;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!callback)
        return cudaErrorInvalidValue;
    if (stream && stream != cudaStreamPerThread)
        r = cudaStreamSynchronize(stream);
    else
        r = cudaDeviceSynchronize();
    if (r != cudaSuccess) {
        set_last(r);
        return r;
    }
    callback(stream, (int)cudaSuccess, userData);
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus) {
    (void)stream;
    if (!pCaptureStatus)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    *pCaptureStatus = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                        int *captureStatus_out,
                                        unsigned long long *id_out) {
    (void)stream;
    (void)id_out;
    if (!captureStatus_out)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    *captureStatus_out = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, int *status_out,
                                     unsigned long long *id_out, void *graph_out,
                                     const void **deps_out,
                                     const void **edge_out,
                                     size_t *num_deps_out) {
    (void)stream;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!status_out)
        return cudaErrorInvalidValue;
    *status_out = 0;
    if (id_out)
        *id_out = 0;
    if (graph_out)
        *(void **)graph_out = NULL;
    if (deps_out)
        *deps_out = NULL;
    if (edge_out)
        *edge_out = NULL;
    if (num_deps_out)
        *num_deps_out = 0;
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    return cudaEventCreateWithFlags(event, 0);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (!event)
        return cudaErrorInvalidValue;
    *event = NULL;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    CUevent e = NULL;
    r = map_cu(cuEventCreate(&e, flags));
    set_last(r);
    if (r != cudaSuccess)
        return r;
    *event = (cudaEvent_t)e;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!event)
        return cudaErrorInvalidValue;
    r = map_cu(cuEventDestroy_v2(map_event(event)));
    set_last(r);
    return r;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!event)
        return cudaErrorInvalidValue;
    r = map_cu(cuEventRecord(map_event(event), map_stream(stream)));
    set_last(r);
    return r;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                     unsigned int flags) {
    (void)flags;
    return cudaEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!event)
        return cudaErrorInvalidValue;
    r = map_cu(cuEventSynchronize(map_event(event)));
    set_last(r);
    return r;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    if (!event)
        return cudaErrorInvalidValue;
    r = map_cu(cuEventQuery(map_event(event)));
    set_last(r);
    return r;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end) {
    (void)ms;
    (void)start;
    (void)end;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    return cudaErrorNotSupported;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
    if (!priority)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    (void)hStream;
    *priority = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
    if (!flags)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    (void)hStream;
    *flags = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream,
                                        int *captureStatus_out,
                                        unsigned long long *id_out,
                                        void **graph_out,
                                        const void **dependencies_out,
                                        size_t *numDependencies_out,
                                        void **edgeData_out) {
    (void)stream;
    (void)graph_out;
    (void)dependencies_out;
    (void)numDependencies_out;
    (void)edgeData_out;
    if (!captureStatus_out)
        return cudaErrorInvalidValue;
    cudaError_t r = check_ctx();
    if (r != cudaSuccess)
        return r;
    *captureStatus_out = 0;
    if (id_out)
        *id_out = 0;
    return cudaSuccess;
}
