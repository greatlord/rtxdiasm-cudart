# CUDA Runtime API compatibility (`libcudart.so`, Phase 5c)

`bin/*/libcudart.so.13.40` (`SOVERSION 13`, version-scripted
`libcudart.so.13` nodes) implements the runtime subset torch needs to
import and initialize: device count/props/set, malloc/free (+async via
pools), memcpy/memset (+async, 1D/2D), streams/events (+priority/flags/
callbacks/capture-status), modules via fatbin registration, launches,
func attributes, occupancy estimates, mempools, peer stubs, driver
entry points, profiler/call-config stubs. Layout-critical types
(`cudaDeviceProp`, `cudaFuncAttributes`, enums) match the CUDA 13
headers field-for-field (verified against the pip wheel headers).

## How it maps

No GPU state lives here: everything forwards to `libcuda.so`
(`DT_NEEDED`), so processes using both APIs share one state. Lazy
per-process init (primary context on device 0). Sticky last-error:
**every** error return (including launch lookup failures) is recorded,
so torch surfaces `named symbol not found` loudly instead of reading
back unwritten memory (see `docs/torch-zeros.md`).
`CUdeviceptr` handles cross as plain addresses (arena-backed).
`cudaStreamLegacy`/`cudaStreamPerThread` map to the null stream.
`__cudaPushCallConfiguration` returns 0 on success (torch's generated
stubs test the return and skip the launch on nonzero — returning the
depth silently disabled every aten launch).
`CUdeviceptr` handles cross as plain addresses (arena-backed).
`cudaStreamLegacy`/`cudaStreamPerThread` map to the null stream.
Version script `src/cudart/cudart.map` provides `@@libcudart.so.13`
nodes (torch references versioned symbols and refuses to load without
them). `-Bsymbolic` (same interposition reason as the driver) and
per-library instrumentation copies for `ACUDA_TRACE`.

## Fatbin registration

`__cudaRegisterFatBinary` accepts raw PTX text (tests, NVRTC route) or
real fatbin containers. PTX payloads are extracted first. If a
container is SASS-only, the optional `rtxdisam_native` library is
dynamically loaded to lift the SASS to PTX; `RTXDISAM_LIBRARY` overrides
the library path. Multi-entry images are lifted per registered device
function, so each entry gets its own PTX body. The reported device
compute capability (`sm_XX`) is used to prefer a matching SASS member in
multi-architecture fatbins. If the library is missing, registration
leaves the image unloadable and lookup fails cleanly.
`__cudaRegisterFunction/Var`, `__cudaUnregisterFatBinary` maintained
in-process; launches resolve host function → (PTX, entry) → cached
driver module → `cuLaunchKernel`. `__device__` variables are recorded
but not backed (`MemcpyToSymbol`/`GetSymbolAddress` return
`NOT_SUPPORTED`).

## Divergences

- `cudaEventElapsedTime`: no timestamps → `NOT_SUPPORTED`.
- Async copies complete inline (ordered, not overlapped).
- `cudaHostAlloc` is pageable (correct, not pinned);
  `cudaHostRegister` unsupported.
- Cooperative launches, graphs, IPC, streams capture:
  present-for-linking stubs returning `NOT_SUPPORTED`.
- Texture arrays and texture objects are backed by the driver surface
  (see `docs/driver-api.md`): `cudaMallocArray`/`cudaMalloc3DArray`/
  `cudaMallocMipmappedArray`, `cudaFreeArray`/`cudaFreeMipmappedArray`,
  `cudaGetMipmappedArrayLevel`, `cudaMemcpy2DToArray`/`FromArray` (+Async),
  `cudaMemcpy3D` (+Async), and `cudaCreateTextureObject`/`Destroy`/
  `GetTextureObject{Resource,Texture}Desc`. Supported channel formats
  are the same subset as the driver (RGBA8 UNORM, RGBA32F, R32F,
  R32UINT, depth). Linear and pitch2D resources snapshot the device
buffer into a 1D/2D image; `cudaTextureDesc.normalizedCoords` selects
normalized or unnormalized sampling.
- Surface objects use the same storage-capable arrays:
  `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, and
  `cudaGetSurfaceObjectResourceDesc` map to `cuSurfObjectCreate`/
  `Destroy`/`GetResourceDesc`.
- `cudaDeviceReset`: no-op success.
- Occupancy/`numRegs` are documented estimates.
