# Third party libraries

Please keep categories (`##` level) listed alphabetically and matching their
respective folder names. Use two empty lines to separate categories for
readability.


## accesskit

- Upstream: https://github.com/AccessKit/accesskit-c
- Version: 0.18.0 (0c52a8ce2357bbeb927f90dc9a1c19c8ec1bd2c3, 2025)
- License: MIT

Files extracted from upstream source:

- `accesskit.h`
- `LICENSE-MIT`


## amd-fsr

- Upstream: https://github.com/GPUOpen-Effects/FidelityFX-FSR
- Version: 1.0.2 (a21ffb8f6c13233ba336352bdff293894c706575, 2021)
- License: MIT

Files extracted from upstream source:

- `ffx_a.h` and `ffx_fsr1.h` from `ffx-fsr`
- `license.txt`


## amd-fsr2

- Upstream: https://github.com/GPUOpen-Effects/FidelityFX-FSR2
- Version: 2.2.1 (1680d1edd5c034f88ebbbb793d8b88f8842cf804, 2023)
- License: MIT

Files extracted from upstream source:

- `ffx_*.cpp` and `ffx_*.h` from `src/ffx-fsr2-api`
- `shaders` folder from `src/ffx-fsr2-api` with `ffx_*.hlsl` files excluded
- `LICENSE.txt`

Patches:

- `0001-build-fixes.patch` ([GH-81197](https://github.com/godotengine/godot/pull/81197))
- `0002-godot-fsr2-options.patch` ([GH-81197](https://github.com/godotengine/godot/pull/81197))


## angle

- Upstream: https://chromium.googlesource.com/angle/angle/
- Version: git (chromium/5907, 430a4f559cbc2bcd5d026e8b36ee46ddd80e9651, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- `include/*`
- `LICENSE`


## astcenc

- Upstream: https://github.com/ARM-software/astc-encoder
- Version: 5.3.0 (bf32abd05eccaf3042170b2a85cebdf0bfee5873, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- `astcenc_*` and `astcenc.h` files from `Source`
- `LICENSE.txt`


## basis_universal

- Upstream: https://github.com/BinomialLLC/basis_universal
- Version: git (b1110111d4a93c7dd7de93ce3d9ed8fcdfd114f2, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- `encoder/` and `transcoder/` folders, with the following files removed from `encoder`:
  `3rdparty/{qoi.h,tinydds.h,tinyexr.cpp,tinyexr.h}`
- `LICENSE`

Patches:

- `0001-external-zstd-pr344.patch` ([GH-73441](https://github.com/godotengine/godot/pull/73441))
- `0002-external-tinyexr.patch` ([GH-97582](https://github.com/godotengine/godot/pull/97582))
- `0003-remove-tinydds-qoi.patch` ([GH-97582](https://github.com/godotengine/godot/pull/97582))
- `0004-clang-warning-exclude.patch` ([GH-111346](https://github.com/godotengine/godot/pull/111346))
- `0005-unused-typedef.patch` ([GH-111445](https://github.com/godotengine/godot/pull/111445))
- `0006-explicit-includes.patch` ([GH-111557](https://github.com/godotengine/godot/pull/111557))


## brotli

- Upstream: https://github.com/google/brotli
- Version: 1.2.0 (028fb5a23661f123017c060daa546b55cf4bde29, 2025)
- License: MIT

Files extracted from upstream source:

- `common/`, `dec/` and `include/` folders from `c/`,
  minus the `dictionary.bin*` files
- `LICENSE`


## certs

- Upstream: Mozilla, via https://github.com/bagder/ca-bundle
- Version: git (cc4096bef208d35e2884571046c75a726185c358, 2025)
- License: MPL 2.0

Files extracted from upstream source:

- `ca-bundle.crt`


## clipper2

- Upstream: https://github.com/AngusJohnson/Clipper2
- Version: 1.5.4 (ef88ee97c0e759792e43a2b2d8072def6c9244e8, 2025)
- License: BSL 1.0

Files extracted from upstream source:

- `CPP/Clipper2Lib/` folder (in root)
- `LICENSE`

Patches:

- `0001-disable-exceptions.patch` ([GH-80796](https://github.com/godotengine/godot/pull/80796))
- `0002-llvm-21-header.patch` ([GH-113850](https://github.com/godotengine/godot/pull/113850))


## cvtt

- Upstream: https://github.com/elasota/ConvectionKernels
- Version: git (350416daa4e98f1c17ffc273b134d0120a2ef230, 2022)
- License: MIT

Files extracted from upstream source:

- All `.cpp` and `.h` files except the folders `MakeTables` and `etc2packer`
- `LICENSE.txt`

Patches:

- `0001-revert-bc6h-reorg.patch` ([GH-73715](https://github.com/godotengine/godot/pull/73715))


## d3d12ma

- Upstream: https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator
- Version: 2.1.0-development (4d16e802e0b9451c9d3c27cd308928c13b73acd6, 2023)
- License: MIT

Files extracted from upstream source:

- `src/D3D12MemAlloc.cpp`, `src/D3D12MemAlloc.natvis`
- `include/D3D12MemAlloc.h`
- `LICENSE.txt`, `NOTICES.txt`

Patches:

- `0001-mingw-support.patch` ([GH-83452](https://github.com/godotengine/godot/pull/83452))


## directx_headers

- Upstream: https://github.com/microsoft/DirectX-Headers
- Version: 1.618.2 (dde59d560da2760fec612d6634124edc2a26b82f, 2025)
- License: MIT

Files extracted from upstream source:

- `include/directx/*.h`
- `include/dxguids/*.h`
- `LICENSE`

Patches:

- `0001-win7-8-dynamic-load.patch` ([GH-88496](https://github.com/godotengine/godot/pull/88496))


## doctest

- Upstream: https://github.com/onqtam/doctest
- Version: 2.4.12 (1da23a3e8119ec5cce4f9388e91b065e20bf06f5, 2025)
- License: MIT

Files extracted from upstream source:

- `doctest/doctest.h` as `doctest.h`
- `LICENSE.txt`

Patches:

- `0001-ciso646-version.patch` ([GH-105913](https://github.com/godotengine/godot/pull/105913))


## dr_libs

- Upstream: https://github.com/mackron/dr_libs
- Version: mp3-0.7.2 (547c211a87a06a42bf62c1366616aa14b57dd429, 2025)
- License: Public Domain or Unlicense or MIT-0

Files extracted from upstream source:

- `dr_mp3.h`
- `LICENSE`

`dr_bridge.h` is a Godot file and should be preserved on updates.


## embree

- Upstream: https://github.com/embree/embree
- Version: 4.4.0 (ff9381774dc99fea81a932ad276677aad6a3d4dd, 2025)
- License: Apache 2.0

Files extracted from upstream:

- All `.cpp` files listed in `modules/raycast/godot_update_embree.py`
- All header files in the directories listed in `modules/raycast/godot_update_embree.py`
- All config files listed in `modules/raycast/godot_update_embree.py`
- `LICENSE.txt`

Patches:

- `0001-disable-exceptions.patch` ([GH-48050](https://github.com/godotengine/godot/pull/48050))
- `0002-godot-config.patch` ([GH-88783](https://github.com/godotengine/godot/pull/88783))
- `0003-emscripten-nthreads.patch` ([GH-69799](https://github.com/godotengine/godot/pull/69799))
- `0004-mingw-no-cpuidex.patch` ([GH-92488](https://github.com/godotengine/godot/pull/92488))
- `0005-mingw-llvm-arm64.patch` ([GH-93364](https://github.com/godotengine/godot/pull/93364))
- `0006-explicit-includes.patch` ([GH-111557](https://github.com/godotengine/godot/pull/111557))

The `modules/raycast/godot_update_embree.py` script can be used to pull the
relevant files from the latest Embree release and apply patches automatically.


## enet

- Upstream: https://github.com/lsalzman/enet
- Version: 1.3.18 (2662c0de09e36f2a2030ccc2c528a3e4c9e8138a, 2024)
- License: MIT

Files extracted from upstream source:

- All `.c` files in the main directory (except `unix.c` and `win32.c`)
- The `include/enet/` folder as `enet/` (except `unix.h` and `win32.h`)
- `LICENSE` file
- Added 3 files `enet_godot.cpp`, `enet/enet_godot.h`, and `enet/enet_godot_ext.h`,
  providing ENet socket implementation using Godot classes, allowing IPv6 and DTLS.

Patches:

- `0001-godot-socket.patch` ([GH-7985](https://github.com/godotengine/godot/pull/7985))

Important: Building against a system wide ENet is possible, but will limit its
functionality to IPv4 only and no DTLS. We recommend against it.


## etcpak

- Upstream: https://github.com/wolfpld/etcpak
- Version: 2.0 (a43d6925bee49277945cf3e311e4a022ae0c2073, 2024)
- License: BSD-3-Clause

Files extracted from upstream source:

- Only the files relevant for compression (i.e. `Process*.cpp` and their deps):
  ```
  Dither.{cpp,hpp} ForceInline.hpp Math.hpp ProcessCommon.hpp ProcessRGB.{cpp,hpp}
  ProcessDxtc.{cpp,hpp} Tables.{cpp,hpp} Vector.hpp
  ```
- The files `DecodeRGB.{cpp.hpp}` are based on the code from the original repository.
- `AUTHORS.txt` and `LICENSE.txt`

Patches:

- `0001-remove-bc7enc.patch` ([GH-101362](https://github.com/godotengine/godot/pull/101362))


## fonts

- `DroidSans*.woff2`:
  * Upstream: https://android.googlesource.com/platform/frameworks/base/+/master/data/fonts/
  * Version: ? (pre-2014 commit when DroidSansJapanese.ttf was obsoleted)
  * License: Apache 2.0
- `Inter*.woff2`:
  * Upstream: https://github.com/rsms/inter
  * Version: v4.1 (e3a3d4c57d5ecc01453a575621882a384c1995a3, 2024)
  * License: OFL-1.1
- `JetBrainsMono_Regular.woff2`:
  * Upstream: https://github.com/JetBrains/JetBrainsMono
  * Version: 2.304 (cd5227bd1f61dff3bbd6c814ceaf7ffd95e947d9, 2023)
  * License: OFL-1.1
- `NotoSansBengali*.woff2`:
  * Upstream: https://github.com/notofonts/bengali
  * Version: 2.003 (020a5701f6fc6a363d5eccbae45e37714c0ad686, 2022)
  * License: OFL-1.1
- `NotoSansDevanagari*.woff2`:
  * Upstream: https://github.com/notofonts/devanagari
  * Version: 2.004 (f8f27e49da0ec9e5e38ecf3628671f05b24dd955, 2023)
  * License: OFL-1.1
- `NotoSansGeorgian*.woff2`:
  * Upstream: https://github.com/notofonts/georgian
  * Version: 2.002 (243ec9aa1d4ec58cc42120d30faac1a102fbfeb9, 2022)
  * License: OFL-1.1
- `NotoSansHebrew*.woff2`:
  * Upstream: https://github.com/notofonts/hebrew
  * Version: 2.003 (caa7ab0614fb5b37cc003d9bf3d7d3e765331110, 2022)
  * License: OFL-1.1
- `NotoSansMalayalam*.woff2`:
  * Upstream: https://github.com/notofonts/malayalam
  * Version: 2.104 (0fd65e553a6af3dc1c09ed39dfe8933e01c17b32, 2023)
  * License: OFL-1.1
- `NotoSansOriya*.woff2`:
  * Upstream: https://github.com/notofonts/oriya
  * Version: 2.005 (9377f242b247df12d0bf4cecd93b9c4b18036fbd, 2023)
  * License: OFL-1.1
- `NotoSansSinhala*.woff2`:
  * Upstream: https://github.com/notofonts/sinhala
  * Version: 2.006 (66e5a2ed9797e575222d6e7c5b3710c7bf68be79, 2022)
  * License: OFL-1.1
- `NotoSansTamil*.woff2`:
  * Upstream: https://github.com/notofonts/tamil
  * Version: 2.004 (f34a08d1ae3fa810581f63410296d971bdcd62dc, 2023)
  * License: OFL-1.1
- `NotoSansTelugu*.woff2`:
  * Upstream: https://github.com/notofonts/telugu
  * Version: 2.004 (68a6a8170cba5b2e9b45029ef36994961e8f614c, 2023)
  * License: OFL-1.1
- `NotoSansThai*.woff2`:
  * Upstream: https://github.com/notofonts/thai
  * Version: 2.001 (09af528011390f35abf15cf86068dae208f512c4, 2022)
  * License: OFL-1.1
- `OpenSans_SemiBold.woff2`:
  * Upstream: https://github.com/googlefonts/opensans
  * Version: git (bd7e37632246368c60fdcbd374dbf9bad11969b6, 2023)
  * License: OFL-1.1
- `Vazirmatn*.woff2`:
  * Upstream: https://github.com/rastikerdar/vazirmatn
  * Version: 33.003 (83629f877e8f084cc07b47030b5d3a0ff06c76ec, 2022)
  * License: OFL-1.1

All fonts are converted from the unhinted `.ttf` sources using the
`https://github.com/google/woff2` tool.

Use UI font variant if available, because it has tight vertical metrics and good
for UI.


## freetype

- Upstream: https://gitlab.freedesktop.org/freetype/freetype
- Version: 2.14.1 (526ec5c47b9ebccc4754c85ac0c0cdf7c85a5e9b, 2025)
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- `src/` folder, minus the `dlg` and `tools` subfolders
  * These files can be removed: `.dat`, `.diff`, `.mk`, `.rc`, `README*`
  * In `src/gzip/`, keep only `ftgzip.c`
- `include/` folder, minus the `dlg` subfolder
- `LICENSE.TXT` and `docs/FTL.TXT`


## glad

- Upstream: https://github.com/Dav1dde/glad
- Version: 2.0.4 (d08b1aa01f8fe57498f04d47b5fa8c48725be877, 2023)
- License: CC0 1.0 and Apache 2.0

Files extracted from upstream source:
- `LICENSE`

Files generated from [upstream web instance](https://gen.glad.sh/):
- `EGL/eglplatform.h`
- `KHR/khrplatform.h`
- `egl.c`
- `glad/egl.h`
- `gl.c`
- `glad/gl.h`
- `glx.c`
- `glad/glx.h`

See the permalinks in `glad/gl.h` and `glad/glx.h` to regenerate the files with
a new version of the web instance.

Patches:

- `0001-enable-both-gl-and-gles.patch` ([GH-72831](https://github.com/godotengine/godot/pull/72831))


## glslang

- Upstream: https://github.com/KhronosGroup/glslang
- Version: vulkan-sdk-1.3.283.0 (e8dd0b6903b34f1879520b444634c75ea2deedf5, 2024)
- License: glslang

Version should be kept in sync with the one of the used Vulkan SDK (see `vulkan`
section).

Files extracted from upstream source:

- `glslang/` folder (except the `glslang/HLSL` and `glslang/ExtensionHeaders`
  subfolders), `SPIRV/` folder
  * Remove C interface code: `CInterface/` folders, files matching `"*_c[_\.]*"`
- Run `cmake . && make` and copy generated `include/glslang/build_info.h`
  to `glslang/build_info.h`
- `LICENSE.txt`
- Unnecessary files like `CMakeLists.txt` or `updateGrammar` removed

Patches:

- `0001-apple-disable-absolute-paths.patch` ([GH-92010](https://github.com/godotengine/godot/pull/92010))
- `0002-gcc15-include-fix.patch` ([GH-102022](https://github.com/godotengine/godot/pull/102022))


## graphite

- Upstream: https://github.com/silnrsi/graphite
- Version: 1.3.14 (27572742003b93dc53dc02c01c237b72c6c25f54, 2022)
- License: MIT

Files extracted from upstream source:

- The `include` folder
- The `src` folder (minus `CMakeLists.txt` and `files.mk`)
- `COPYING`


## grisu2

- Upstream: https://github.com/simdjson/simdjson/blob/master/src/to_chars.cpp
- Version: git (667d0ed3c77f55cbda2082b034168d69898d1f88, 2025)
- License: Apache and MIT

Files extracted from upstream source:

- The `src/to_chars.cpp` file renamed to `grisu2.h` and slightly modified.

Patches:

- `0001-godot-changes.patch` ([GH-98750](https://github.com/godotengine/godot/pull/98750))


## harfbuzz

- Upstream: https://github.com/harfbuzz/harfbuzz
- Version: 12.2.0 (0426759f2b03e479495e987608f10e0866e70180, 2025)
- License: MIT

Files extracted from upstream source:

- `AUTHORS`, `COPYING`, `THANKS`
- From the `src` folder, recursively:
  - All the `.cc`, `.h`, `.hh` files
  - Except `main.cc`, `harfbuzz*.cc`, `harfrust.cc`, `failing-alloc.c`, `test*.cc`, `hb-wasm*.*`, `hb-harfrust.cc`, `wasm/*`, `ms-use/*`, `rust/*`


## icu4c

- Upstream: https://github.com/unicode-org/icu
- Version: 78.1 (049e0d6a420629ac7db77256987d083a563287b5, 2025)
- License: Unicode

Files extracted from upstream source:

- The `common` folder
- `scriptset.*`, `ucln_in.*`, `uspoof.cpp` and `uspoof_impl.*` from the `i18n` folder
- `uspoof.h` from the `i18n/unicode` folder
- `LICENSE`

Files generated from upstream source:

- The `icudt_godot.dat` built with the provided `godot_data.json` config file (see
  https://github.com/unicode-org/icu/blob/master/docs/userguide/icu_data/buildtool.md
  for instructions).

1. Download and extract both `icu4c-{version}-src.tgz` and `icu4c-{version}-data.zip`
  (replace `data` subfolder from the main source archive)
2. Build ICU with default options: `./runConfigureICU {PLATFORM} && make`
3. Reconfigure ICU with custom data config:
   `ICU_DATA_FILTER_FILE={GODOT_SOURCE}/thirdparty/icu4c/godot_data.json ./runConfigureICU {PLATFORM} --with-data-packaging=common`
4. Delete `data/out` folder and rebuild data: `cd data && rm -rf ./out && make`
5. Copy `source/data/out/icudt{ICU_VERSION}l.dat` to the `{GODOT_SOURCE}/thirdparty/icu4c/icudt_godot.dat`


## jolt_physics

- Upstream: https://github.com/jrouwe/JoltPhysics
- Version: 5.4.0 (036ea7b1d717b3e713ac9d8cbd47118fb9cd5d60, 2025)
- License: MIT

Files extracted from upstream source:

- All files in `Jolt/`, except `Jolt/Jolt.cmake` and any files dependent on `ENABLE_OBJECT_STREAM`, as seen in `Jolt/Jolt.cmake`
- `LICENSE`


## libbacktrace

- Upstream: https://github.com/ianlancetaylor/libbacktrace
- Version: git (4d2dd0b172f2c9192f83ba93425f868f2a13c553, 2022)
- License: BSD-3-Clause

Files extracted from upstream source:

- `*.{c,h}` files for Windows platform, i.e. remove the following:
  * `allocfail.c`, `instrumented_alloc.c`, `*test*.{c,h}`
  * `elf.c`, `macho.c`, `mmap.c`, `mmapio.c`, `nounwind.c`, `unknown.c`, `xcoff.c`
- `LICENSE`

Patches:

- `0001-big-files-support.patch` ([GH-100281](https://github.com/godotengine/godot/pull/100281))


## libjpeg-turbo

- Upstream: https://github.com/libjpeg-turbo/libjpeg-turbo
- Version: 3.1.3 (af9c1c268520a29adf98cad5138dafe612b3d318, 2025)
- License: BSD-3-Clause and IJG

Files extracted from upstream source:

- `src/*.{c,h}` except for:
  * `cdjpeg.c cjpeg.c djpeg.c example.c jcdiffct.c jclhuff.c jclossls.c jcstest.c jddiffct.c jdlhuff.c jdlossls.c jlossls.h jpegtran.c rdbmp.c rdcolmap.c rdgif.c rdjpgcom.c rdppm.c rdswitch.c rdtarga.c strtest.c tjbench.c tjcomp.c tjdecomp.c tjtran.c tjunittest.c tjutil.c wrbmp.c wrgif.c wrjpgcom.c wrppm.c wrtarga.c`
- `LICENSE.md`
- `README.ijg`

Patches:

- `0001-cmake-generated-headers.patch` ([GH-104347](https://github.com/godotengine/godot/pull/104347))
- `0002-disable-16bitlossless.patch` ([GH-104347](https://github.com/godotengine/godot/pull/104347))
- `0003-remove-bmp-ppm-support.patch` ([GH-104347](https://github.com/godotengine/godot/pull/104347))


## libktx

- Upstream: https://github.com/KhronosGroup/KTX-Software
- Version: 4.4.2 (4d6fc70eaf62ad0558e63e8d97eb9766118327a6, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- `LICENSE.md`
- `include/` minus `.clang-format`
- `external/dfdutils/LICENSE.adoc` as `LICENSE.dfdutils.adoc` (in root)
- `external/dfdutils/LICENSES/Apache-2.0.txt` as `Apache-2.0.txt` (in root)
- `external/dfdutils/{KHR/,dfd.h,colourspaces.c,createdfd.c,interpretdfd.c,printdfd.c,queries.c,dfd2vk.inl,vk2dfd.*}`
- `lib/{basis_sgd.h,formatsize.h,gl_format.h,ktxint.h,uthash.h,vk_format.h,vkformat_enum.h,checkheader.c,swap.c,hashlist.c,vkformat_check*.c,vkformat_typesize.c,basis_transcode.cpp,miniz_wrapper.cpp,filestream.*,memstream.*,texture*}`
- `other_include/KHR/`
- `utils/unused.h`

Patches:

- `0001-external-basisu.patch` ([GH-76572](https://github.com/godotengine/godot/pull/76572))
- `0002-disable-astc-block-ext.patch` ([GH-76572](https://github.com/godotengine/godot/pull/76572))
- `0003-basisu-1.60.patch` ([GH-103968](https://github.com/godotengine/godot/pull/103968))


## libogg

- Upstream: https://www.xiph.org/ogg
- Version: 1.3.6 (be05b13e98b048f0b5a0f5fa8ce514d56db5f822, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*.{c,h}`
- `include/ogg/*.h` in `ogg/` (run `configure` to generate `config_types.h`)
- `COPYING`


## libpng

- Upstream: http://libpng.org/pub/png/libpng.html
- Version: 1.6.53 (4e3f57d50f552841550a36eabbb3fbcecacb7750, 2025)
- License: libpng/zlib

Files extracted from upstream source:

- All `.c` and `.h` files of the main directory, apart from `example.c` and `pngtest.c`
- `arm/`, `intel/`, `loongarch/`, and `powerpc/` folders, except `arm/filter_neon.S` and `.editorconfig` files
- `scripts/pnglibconf.h.prebuilt` as `pnglibconf.h`
- `LICENSE`


## libtheora

- Upstream: https://www.theora.org
- Version: 1.2.0 (8e4808736e9c181b971306cc3f05df9e61354004, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- All `.c` and `.h` files in `lib/`, except `arm/` and `c64x/` folders
- All `.h` files in `include/theora/` as `theora/`
- `COPYING` and `LICENSE`


## libvorbis

- Upstream: https://www.xiph.org/vorbis
- Version: 1.3.7 (0657aee69dec8508a0011f47f3b69d7538e9d262, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- `lib/*` except from: `lookups.pl`, `Makefile.*`
- `include/vorbis/*.h` as `vorbis/`
- `COPYING`


## libwebp

- Upstream: https://chromium.googlesource.com/webm/libwebp/
- Version: 1.6.0 (4fa21912338357f89e4fd51cf2368325b59e9bd9, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/` and `sharpyuv/` except from `.am`, `.rc` and `.in` files
- `AUTHORS`, `COPYING`, `PATENTS`

Patches:

- `0001-msvc-node-debug-rename.patch` ([GH-75769](https://github.com/godotengine/godot/pull/75769))
- `0002-msvc-arm64-fpstrict.patch` ([GH-94655](https://github.com/godotengine/godot/pull/94655))
- `0003-clang-cl-sse2-sse41-avx2.patch` ([GH-92316](https://github.com/godotengine/godot/pull/92316))


## linuxbsd_headers

See `linuxbsd_headers/README.md`.


## manifold

- Upstream: https://github.com/elalish/manifold
- Version: 3.3.2 (798d83c8d7fabcddd23c1617097b95ba40f2597c, 2025)
- License: Apache 2.0

File extracted from upstream source:

- `src/` and `include/`, except from `CMakeLists.txt`, `cross_section.h` and `meshIO.{cpp,h}`
- `AUTHORS`, `LICENSE`


## lzfse

- Upstream: https://github.com/lzfse/lzfse
- Version: git (e634ca58b4821d9f3d560cdc6df5dec02ffc93fd, 2017)
- License: BSD-3-Clause


## mbedtls

- Upstream: https://github.com/Mbed-TLS/mbedtls
- Version: 3.6.5 (e185d7fd85499c8ce5ca2a54f5cf8fe7dbe3f8df, 2025)
- License: Apache 2.0

File extracted from upstream release tarball:

- All `.h` from `include/mbedtls/` to `thirdparty/mbedtls/include/mbedtls/`
  and all `.h` from `include/psa/` to `thirdparty/mbedtls/include/psa/`
- From `library/` to `thirdparty/mbedtls/library/`:
  - All `.c` and `.h` files
  - Except `bignum_mod.c`, `block_cipher.c`, `ecp_curves_new.c`, `lmots.c`,
    `lms.c`
- The `LICENSE` file (edited to keep only the Apache 2.0 variant)
- Added 2 files `godot_core_mbedtls_platform.c` and `godot_core_mbedtls_config.h`
  providing configuration for light bundling with core
- Added 2 files `godot_module_mbedtls_config.h` and `threading_alt.h`
  to customize the build configuration when bundling the full library

Patches:

- `0001-msvc-2019-psa-redeclaration.patch` ([GH-90535](https://github.com/godotengine/godot/pull/90535))


## meshoptimizer

- Upstream: https://github.com/zeux/meshoptimizer
- Version: 1.0 (73583c335e541c139821d0de2bf5f12960a04941, 2025)
- License: MIT

Files extracted from upstream repository:

- All files in `src/`
- `LICENSE.md`


## mingw-std-threads

- Upstream: https://github.com/meganz/mingw-std-threads
- Version: git (c931bac289dd431f1dd30fc4a5d1a7be36668073, 2023)
- License: BSD-2-clause

Files extracted from upstream repository:

- `LICENSE`
- `mingw.condition_variable.h`
- `mingw.invoke.h`
- `mingw.mutex.h`
- `mingw.shared_mutex.h`
- `mingw.thread.h`

Patches:

- `0001-disable-exceptions.patch` ([GH-85039](https://github.com/godotengine/godot/pull/85039))
- `0002-clang-std-replacements-leak.patch` ([GH-85208](https://github.com/godotengine/godot/pull/85208))
- `0003-explicit-includes.patch` ([GH-111557](https://github.com/godotengine/godot/pull/111557))


## miniupnpc

- Upstream: https://github.com/miniupnp/miniupnp
- Version: 2.3.3 (bf4215a7574f88aa55859db9db00e3ae58cf42d6, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- `miniupnpc/src/` as `src/`
- `miniupnpc/include/` as `include/miniupnpc/`
- Remove the following test or sample files:
  `listdevices.c,minihttptestserver.c,miniupnpcmodule.c,upnpc.c,upnperrors.*,test*`
- `LICENSE`
- `src/miniupnpcstrings.h` was created manually for Godot (it is usually generated
  by CMake). Bump the version number for miniupnpc in that file when upgrading.


## minizip

- Upstream: https://github.com/madler/zlib
- Version: 1.3.1.2 (570720b0c24f9686c33f35a1b3165c1f568b96be, 2025)
- License: zlib

Files extracted from the upstream source:

- From `contrib/minizip`:
  `{crypt.h,ints.h,ioapi.{c,h},skipset.h,unzip.{c,h},zip.{c,h}}`
  `MiniZip64_info.txt`

Patches:

- `0001-godot-seek.patch` ([GH-10428](https://github.com/godotengine/godot/pull/10428))


## misc

Collection of single-file libraries used in Godot components.

- `bcdec.h`
  * Upstream: https://github.com/iOrange/bcdec
  * Version: git (3b29f8f44466c7d59852670f82f53905cf627d48, 2024)
  * License: MIT
- `cubemap_coeffs.h`
  * Upstream: https://research.activision.com/publications/archives/fast-filtering-of-reflection-probes
    File coeffs_const_8.txt (retrieved April 2020)
  * License: MIT
- `fastlz.{c,h}`
  * Upstream: https://github.com/ariya/FastLZ
  * Version: 0.5.0 (4f20f54d46f5a6dd4fae4def134933369b7602d2, 2020)
  * License: MIT
- `FastNoiseLite.h`
  * Upstream: https://github.com/Auburn/FastNoiseLite
  * Version: 1.1.0 (f7af54b56518aa659e1cf9fb103c0b6e36a833d9, 2023)
  * License: MIT
  * Patches:
    - `FastNoiseLite-0001-namespace-warnings.patch` ([GH-88526](https://github.com/godotengine/godot/pull/88526))
- `ifaddrs-android.{cc,h}`
  * Upstream: https://chromium.googlesource.com/external/webrtc/stable/talk/+/master/base/ifaddrs-android.h
  * Version: git (5976650443d68ccfadf1dea24999ee459dd2819d, 2013)
  * License: BSD-3-Clause
  * Patches:
    - `ifaddrs-android-0001-complete-struct.patch` ([GH-34101](https://github.com/godotengine/godot/pull/34101))
- `mikktspace.{c,h}`
  * Upstream: https://archive.blender.org/wiki/index.php/Dev:Shading/Tangent_Space_Normal_Maps/
  * Version: 1.0 (2011)
  * License: zlib
- `nvapi_minimal.h`
  * Upstream: http://download.nvidia.com/XFree86/nvapi-open-source-sdk
  * Version: R525
  * License: MIT
  * Modifications: Created from upstream `nvapi.h` by removing unnecessary code.
- `ok_color.h`
  * Upstream: https://github.com/bottosson/bottosson.github.io/blob/master/misc/ok_color.h
  * Version: git (d69831edb90ffdcd08b7e64da3c5405acd48ad2c, 2022)
  * License: MIT
  * Modifications: License included in header.
- `ok_color_shader.h`
  * https://www.shadertoy.com/view/7sK3D1
  * Version: 2021-09-13
  * License: MIT
- `pcg.{cpp,h}`
  * Upstream: http://www.pcg-random.org
  * Version: minimal C implementation, http://www.pcg-random.org/download.html
  * License: Apache 2.0
- `polypartition.{cpp,h}`
  * Upstream: https://github.com/ivanfratric/polypartition (`src/polypartition.{cpp,h}`)
  * Version: git (7bdffb428b2b19ad1c43aa44c714dcc104177e84, 2021)
  * License: MIT
  * Patches:
    - `polypartition-0001-godot-types.patch` (2185c018f)
    - `polypartition-0002-shadow-warning.patch` ([GH-66808](https://github.com/godotengine/godot/pull/66808))
- `qoa.{c,h}`
  * Upstream: https://github.com/phoboslab/qoa
  * Version: git (ae07b57deb98127a5b40916cb57775823d7437d2, 2025)
  * License: MIT
  * Modifications: Added implementation through `qoa.c`.
- `r128.{c,h}`
  * Upstream: https://github.com/fahickman/r128
  * Version: git (6fc177671c47640d5bb69af10cf4ee91050015a1, 2023)
  * License: Public Domain or Unlicense
- `smaz.{c,h}`
  * Upstream: https://github.com/antirez/smaz
  * Version: git (2f625846a775501fb69456567409a8b12f10ea25, 2012)
  * License: BSD-3-Clause
  * Modifications: License included in header.
  * Patches:
    - `smaz-0001-write-string-warning.patch` ([GH-8572](https://github.com/godotengine/godot/pull/8572))
- `smolv.{cpp,h}`
  * Upstream: https://github.com/aras-p/smol-v
  * Version: git (9dd54c379ac29fa148cb1b829bb939ba7381d8f4, 2024)
  * License: Public Domain or MIT
- `stb_rect_pack.h`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.01 (af1a5bc352164740c1cc1354942b1c6b72eacb8a, 2021)
  * License: Public Domain or Unlicense or MIT
- `yuv2rgb.h`
  * Upstream: http://wss.co.uk/pinknoise/yuv2rgb/ (to check)
  * Version: ?
  * License: BSD


## msdfgen

- Upstream: https://github.com/Chlumsky/msdfgen
- Version: 1.13 (1874bcf7d9624ccc85b4bc9a85d78116f690f35b, 2025)
- License: MIT

Files extracted from the upstream source:

- `msdfgen.h`
- Files in `core/` folder, minus `export-svg.*` and `save-*.*` files
- `LICENSE.txt`

Patches:

- `0001-remove-unused-save-features.patch` ([GH-113965](https://github.com/godotengine/godot/issues/113965))


## openxr

- Upstream: https://github.com/KhronosGroup/OpenXR-SDK
- Version: 1.1.54 (c15d38cb4bb10a5b7e075f74493ff13896e2597a, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- `include/`
- `src/common/`
- `src/loader/`
- `src/*.{c,h}`
- `src/external/jsoncpp/include/`
- `src/external/jsoncpp/src/lib_json/`
- `src/external/jsoncpp/{AUTHORS,LICENSE}`
- `LICENSE` and `COPYING.adoc`

Exclude:

- `src/external/android-jni-wrappers` and `src/external/jnipp` (not used yet)
- Obsolete `src/xr_generated_dispatch_table.{c,h}`
- All CMake stuff: `cmake/`, `CMakeLists.txt` and `*.cmake`
- All Gradle stuff: `*gradle*`, `AndroidManifest.xml`
- All following files (and their `.license` files):
  `*.{def,expsym,in,json,map,pom,rc,txt}`
- All dotfiles

Additional:
- Update `openxrLoaderVersion` in `platform/android/java/app/config.gradle`


## pcre2

- Upstream: http://www.pcre.org
- Version: 10.47 (f454e231fe5006dd7ff8f4693fd2b8eb94333429, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- Files listed in the file `NON-AUTOTOOLS-BUILD` steps 1-4
- All `.h` files in `src/` apart from `pcre2posix.h`, `pcre2_printint_inc.h`, `pcre2test_inc.h`
- `src/pcre2_compile_cgroup.c`
- `src/pcre2_match_next.c`
- `src/pcre2_{jit_char,jit_match,jit_misc,jit_simd,ucptables}_inc.h`
- `deps/sljit/sljit_src`
- `AUTHORS.md` and `LICENCE.md`


## recastnavigation

- Upstream: https://github.com/recastnavigation/recastnavigation
- Version: 1.6.0 (6dc1667f580357e8a2154c28b7867bea7e8ad3a7, 2023)
- License: zlib

Files extracted from upstream source:

- `Recast/` folder without `CMakeLists.txt`
- `License.txt`


## re-spirv

- Upstream: https://github.com/renderbag/re-spirv
- Version: git (5af3b606e6aaf11bae8dc7b5cd236c943e24865e, 2025)
- License: MIT

Files extracted from upstream source:

- `re-spirv.cpp`
- `re-spirv.h`
- `LICENSE`


## rvo2

For 2D in `rvo2_2d` folder

- Upstream: https://github.com/snape/RVO2
- Version: git (f7c5380235f6c9ac8d19cbf71fc94e2d4758b0a3, 2021)
- License: Apache 2.0

For 3D in `rvo2_3d` folder

- Upstream: https://github.com/snape/RVO2-3D
- Version: git (bfc048670a4e85066e86a1f923d8ea92e3add3b2, 2021)
- License: Apache 2.0

Files extracted from upstream source:

- All `.cpp` and `.h` files in the `src/` folder except for `Export.h` and `RVO.h`
- `LICENSE`

Important: Nearly all files have Godot-made changes and renames
to make the 2D and 3D rvo libraries compatible with each other
and solve conflicts and also enrich the feature set originally
proposed by these libraries and better integrate them with Godot.


## smaa

- Upstream: https://github.com/iryoku/smaa
- Version: git (71c806a838bdd7d517df19192a20f0c61b3ca29d, 2013)
- License: MIT

Files extracted from upstream source:

- `LICENSE`
- Textures generated using the Python scripts in the `Scripts` folder


## sdl

- Upstream: https://github.com/libsdl-org/SDL
- Version: 3.2.28 (7f3ae3d57459e59943a4ecfefc8f6277ec6bf540, 2025)
- License: Zlib
- Vendored: hidapi 0.14.0, license BSD-3-Clause

Files extracted from upstream source:

- See `thirdparty/sdl/update-sdl.sh`

Patches:

- `0001-remove-unnecessary-subsystems.patch` ([GH-106218](https://github.com/godotengine/godot/pull/106218))
- `0003-std-include.patch` ([GH-108144](https://github.com/godotengine/godot/pull/108144))
- `0004-errno-include.patch` ([GH-108354](https://github.com/godotengine/godot/pull/108354))
- `0005-fix-libudev-dbus.patch` ([GH-108373](https://github.com/godotengine/godot/pull/108373))
- `0006-fix-cs-environ.patch` ([GH-109283](https://github.com/godotengine/godot/pull/109283))


## spirv-cross

- Upstream: https://github.com/KhronosGroup/SPIRV-Cross
- Version: git (d7440cbc6c50332600fdf21c45e6a5df0b07e54c, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- All `.cpp`, `.hpp` and `.h` files, minus `main.cpp`, `spirv_cross_c.*`, `spirv_hlsl.*`, `spirv_cpp.*`
- `include/` folder
- `LICENSE` and `LICENSES/` folder, minus `CC-BY-4.0.txt`

Versions of this SDK do not have to match the `vulkan` section, as this SDK is required
to generate Metal source from Vulkan SPIR-V.


## spirv-headers

- Upstream: https://github.com/KhronosGroup/SPIRV-Headers
- Version: vulkan-sdk-1.4.328.1 (01e0577914a75a2569c846778c2f93aa8e6feddd, 2025)

Files extracted from upstream source:
- `include/spirv/unified1` folder with only `spirv.h` and `spirv.hpp`
- `LICENSE`


## spirv-reflect

- Upstream: https://github.com/KhronosGroup/SPIRV-Reflect
- Version: vulkan-sdk-1.3.283.0 (ee5b57fba6a986381f998567761bbc064428e645, 2024)
- License: Apache 2.0

Version should be kept in sync with the one of the used Vulkan SDK (see `vulkan`
section).

Files extracted from upstream source:

- `spirv_reflect.h`, `spirv_reflect.c`
- `include/` folder
- `LICENSE`

Patches:

- `0001-specialization-constants.patch` ([GH-50325](https://github.com/godotengine/godot/pull/50325))
- `0002-zero-size-for-sc-sized-arrays.patch` ([GH-94985](https://github.com/godotengine/godot/pull/94985))
- `0003-spirv-headers.patch` ([GH-111452](https://github.com/godotengine/godot/pull/111452))


## swappy-frame-pacing

- Upstream: https://android.googlesource.com/platform/frameworks/opt/gamesdk/ via https://github.com/godotengine/godot-swappy
- Version: git (1198bb06b041e2df5d42cc5cf18fac81fcefa03f, 2025)
- License: Apache 2.0

Files extracted from upstream source:

- `include/common/`
- `include/swappy/{swappy_common.h,swappyVk.h}`
- `LICENSE`


## thorvg

- Upstream: https://github.com/thorvg/thorvg
- Version: 0.15.16 (e15069de7afcc5e853edf1561e69d9b8383e2c6c, 2025)
- License: MIT

Files extracted from upstream source:

- See `thorvg/update-thorvg.sh` for extraction instructions.
  Set the version number and run the script.

Patches:

- `0001-revert-tvglines-bezier-precision.patch` ([GH-96658](https://github.com/godotengine/godot/pull/96658))
- `0002-use-heap-alloc.patch` ([GH-109530](https://github.com/godotengine/godot/pull/109530))
- `0003-explicit-includes.patch` ([GH-111557](https://github.com/godotengine/godot/pull/111557))


## tinyexr

- Upstream: https://github.com/syoyo/tinyexr
- Version: 1.0.12 (735ff73ce5959cf005eb99ce517c9bcecab89dfb, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`

Patches:

- `0001-external-zlib.patch` ([GH-55115](https://github.com/godotengine/godot/pull/55115))


## ufbx

- Upstream: https://github.com/ufbx/ufbx
- Version: 0.20.0 (a63ff0a47485328880b3300e7bcdf01413343a45, 2025)
- License: MIT

Files extracted from upstream source:

- `ufbx.{c,h}`
- `LICENSE`


## vhacd

- Upstream: https://github.com/kmammou/v-hacd
- Version: git (1a49edf29c69039df15286181f2f27e17ceb9aef, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- From `src/VHACD_Lib/`: `inc`, `public` and `src`
- `LICENSE`

Patches:

- `0001-bullet-namespace.patch` ([GH-27929](https://github.com/godotengine/godot/pull/27929))
- `0002-fpermissive-fix.patch` ([GH-27929](https://github.com/godotengine/godot/pull/27929))
- `0003-fix-musl-build.patch` ([GH-34250](https://github.com/godotengine/godot/pull/34250))
- `0004-fix-msvc-arm-build.patch` ([GH-34331](https://github.com/godotengine/godot/pull/34331))
- `0005-fix-scale-calculation.patch` ([GH-38506](https://github.com/godotengine/godot/pull/38506))
- `0006-gcc13-include-fix.patch` ([GH-77949](https://github.com/godotengine/godot/pull/77949))


## volk

- Upstream: https://github.com/zeux/volk
- Version: vulkan-sdk-1.3.283.0 (3a8068a57417940cf2bf9d837a7bb60d015ca2f1, 2024)
- License: MIT

Version should be kept in sync with the one of the used Vulkan SDK (see `vulkan`
section).

Files extracted from upstream source:

- `volk.h`, `volk.c`
- `LICENSE.md`


## vulkan

- Upstream: https://github.com/KhronosGroup/Vulkan-Headers
- Version: vulkan-sdk-1.3.283.0 (eaa319dade959cb61ed2229c8ea42e307cc8f8b3, 2024)
- License: Apache 2.0

Unless there is a specific reason to package a more recent version, please stick
to tagged SDK releases. All Vulkan libraries and headers should be kept in sync so:

- Update Vulkan SDK components to the matching tag (see "vulkan")
- Update volk (see "volk")
- Update glslang (see "glslang")
- Update spirv-reflect (see "spirv-reflect")

Files extracted from upstream source:

- `include/`
- `LICENSE.md`

`vk_enum_string_helper.h` is taken from the matching `Vulkan-Utility-Libraries`
SDK release: https://github.com/KhronosGroup/Vulkan-Utility-Libraries/blob/main/include/vulkan/vk_enum_string_helper.h

`vk_mem_alloc.h` is taken from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
Version: 3.1.0 (009ecd192c1289c7529bff248a16cfe896254816, 2024)
`vk_mem_alloc.cpp` is a Godot file and should be preserved on updates.

Patches:

- `0001-VKEnumStringHelper-godot-vulkan.patch` ([GH-97510](https://github.com/godotengine/godot/pull/97510))
- `0002-VMA-godot-vulkan.patch` ([GH-97510](https://github.com/godotengine/godot/pull/97510))
- `0003-VMA-add-vmaCalculateLazilyAllocatedBytes.patch` ([GH-99257](https://github.com/godotengine/godot/pull/99257))


## wayland

- Upstream: https://gitlab.freedesktop.org/wayland/wayland
- Version: 1.24.0 (736d12ac67c20c60dc406dc49bb06be878501f86, 2025)
- License: MIT

Files extracted from upstream source:

- `protocol/wayland.xml`
- `COPYING`


# wayland-protocols

- Upstream: https://gitlab.freedesktop.org/wayland/wayland-protocols
- Version: 1.46 (6141e1154303dadd5c3e480bc4a16e26f1dcb2af, 2025)
- License: MIT

Files extracted from upstream source:

- `stable/tablet/README`
- `stable/tablet/tablet-unstable-v2.xml`
- `stable/viewporter/README`
- `stable/viewporter/viewporter.xml`
- `stable/xdg-shell/README`
- `stable/xdg-shell/xdg-shell.xml`
- `staging/fractional-scale/README`
- `staging/fractional-scale/fractional-scale-v1.xml`
- `staging/xdg-activation/README`
- `staging/xdg-activation/xdg-activation-v1.xml`
- `staging/xdg-system-bell/xdg-system-bell-v1.xml`
- `unstable/idle-inhibit/README`
- `unstable/idle-inhibit/idle-inhibit-unstable-v1.xml`
- `unstable/pointer-constraints/README`
- `unstable/pointer-constraints/pointer-constraints-unstable-v1.xml`
- `unstable/pointer-gestures/README`
- `unstable/pointer-gestures/pointer-gestures-unstable-v1.xml`
- `unstable/primary-selection/README`
- `unstable/primary-selection/primary-selection-unstable-v1.xml`
- `unstable/relative-pointer/README`
- `unstable/relative-pointer/relative-pointer-unstable-v1.xml`
- `unstable/text-input/README`
- `unstable/text-input/text-input-unstable-v3.xml`
- `unstable/xdg-decoration/README`
- `unstable/xdg-decoration/xdg-decoration-unstable-v1.xml`
- `unstable/xdg-foreign/README`
- `unstable/xdg-foreign/xdg-foreign-unstable-v1.xml`
- `COPYING`

The following files are extracted from thirdparty sources:

- `mesa/wayland-drm.xml`: https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.3.0/src/egl/wayland/wayland-drm/wayland-drm.xml


## wslay

- Upstream: https://github.com/tatsuhiro-t/wslay
- Version: 1.1.1+git (0e7d106ff89ad6638090fd811a9b2e4c5dda8d40, 2022)
- License: MIT

File extracted from upstream release tarball:

- Run `cmake .` to generate `config.h` and `wslayver.h`
  Contents might need tweaking for Godot, review diff
- All `.c` and `.h` files from `lib/`
- All `.h` in `lib/includes/wslay/` as `wslay/`
- `COPYING`

Patches:

- `0001-msvc-build-fix.patch` ([GH-30263](https://github.com/godotengine/godot/pull/30263))


## xatlas

- Upstream: https://github.com/jpcy/xatlas
- Version: git (f700c7790aaa030e794b52ba7791a05c085faf0c, 2022)
- License: MIT

Files extracted from upstream source:

- `source/xatlas/xatlas.{cpp,h}`
- `LICENSE`


## zlib

- Upstream: https://github.com/madler/zlib
- Version: 1.3.1.2 (570720b0c24f9686c33f35a1b3165c1f568b96be, 2025)
- License: zlib

Files extracted from upstream source:

- All `.c` and `.h` files, except `gz*.c` and `infback.c`
- `LICENSE`


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.5.7 (f8745da6ff1ad1e7bab384bd1f9d742439278e99, 2025)
- License: BSD-3-Clause

Files extracted from upstream source:

- `lib/{common/,compress/,decompress/,zstd.h,zstd_errors.h}`
- `LICENSE`
