# rtxdiasm_guard

Standalone CUDA Runtime API shim (`cudart`) project extracted from ACuda/rtxdisam.

## Outputs

This project produces platform-specific `cudart` libraries with the following names:

- **Windows 64-bit:** `cudart64_13.dll`
- **Windows 32-bit:** `cudart32_13.dll`
- **Linux:** `libcudart.so`, `libcudart.so.13`, `libcudart.so.13.0.96`

## Build

```sh
cmake -S . -B build
cmake --build build --config Release
```

When integrated into ACuda, the build can provide `acuda_driver` and `ACUDA_TRACE_SOURCE`/warning settings to match the monorepo build.

## Notes

- Implements the CUDA Runtime API surface in `include/acuda/runtime_api.h` and forwards to the ACuda CUDA Driver API (`acuda/driver.h` / `acuda_driver`).
- Runtime version is aligned to the requested library naming (`13.0.96` series, soname/major `13`).

## Copyright and authorship

Copyright 2026 by Magnus Olsen.

This code had assistance by models DeepFlash 4.1, Muse Spark 1.3, and Iceman.

## License

MIT (see `LICENSE`).
