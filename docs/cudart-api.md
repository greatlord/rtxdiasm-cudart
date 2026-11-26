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

## Calling convention (32-bit Windows)

NVIDIA's `cudart32_*.dll` exports the runtime API with the `__stdcall` calling
convention (`CUDARTAPI` in the CUDA headers), so the callee pops the
arguments. Callers are compiled against NVIDIA's headers and depend on it;
it is part of the ABI, not an implementation detail.

- `include/acuda/runtime_api.h` declares every entry point with `CUDARTAPI`
  (`__stdcall` on `_M_IX86`, empty on other targets).
- The Win32 target is compiled with `/Gz` so the definitions in `src/*.c`
  match the header without annotating each definition separately.
- Helper libraries called by the shim (`rtxdiasm_fatbin_*`,
  `acuda_rtxdisam_cubin_to_ptx_for_target`) are pinned to `__cdecl`, since
  they are built without `/Gz`.
- Exported names stay undecorated: `WINDOWS_EXPORT_ALL_SYMBOLS` generates a
  `.def` that aliases the decorated symbols (`cudaMalloc = _cudaMalloc@8`).

A cdecl export silently corrupts the caller: after every call the caller's
stack is left off by the argument bytes because neither side pops them. With
PhysX 2.8.4 this appeared as `LoadLibrary("PhysXCore.dll")` failing with
`ERROR_DLL_INIT_FAILED` (1114) and an access violation at `eip=00000002`
inside a static initializer - a `ret` eventually jumped through a shifted
stack. Matching the convention fixed GPU detection and simulation.

## Fatbin ownership and context model

Two design rules keep PhysX 2.8.4 stable with this shim:

- **One fatbin loader.** `__cudaRegisterFatBinary` stores the
  `__fatBinC_Wrapper_t` and hands it to `cuModuleLoadData` unchanged. nvcuda
  parses the wrapper, the `0xBA55ED50` container and the PTX/SASS member, so
  there is exactly one implementation choosing the image. A second extractor
  inside the runtime could pick a different member than the driver and
  produced divergent (hanging) kernels.
- **One context.** Every driver entry point, including the CUDA 13.x
  interface shim in `driver_export_dd9e98a4.c`, uses the single shared context
  created by `need_actx()`. Retaining a primary context on top of it created a
  second context: launches were recorded on one stream timeline while
  memcpy/synchronise waited on the other, deadlocking PhysX's worker thread.

Legacy texture references are real bindings: `cudaBindTexture` resolves the
module registered by `__cudaRegisterTexture` and calls
`cuModuleGetTexRef`/`cuTexRefSetFormat`/`cuTexRefSetAddress`. The launch path
resolves hidden texref parameters by name and fails with
`CUDA_ERROR_INVALID_HANDLE` when no texture object is bound.
