# Third party libraries

Please keep categories (`##` level) listed alphabetically and matching their
respective folder names. Use two empty lines to separate categories for
readability.


## amd-fsr

- Upstream: https://github.com/GPUOpen-Effects/FidelityFX-FSR
- Version: 1.0.2 (a21ffb8f6c13233ba336352bdff293894c706575, 2021)
- License: MIT

Files extracted from upstream source:

- `ffx_a.h` and `ffx_fsr1.h` from `ffx-fsr`
- `license.txt`


## angle

- Upstream: https://chromium.googlesource.com/angle/angle/
- Version: git (chromium/5907, 430a4f559cbc2bcd5d026e8b36ee46ddd80e9651, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- `include/*`
- `LICENSE`


## astcenc

- Upstream: https://github.com/ARM-software/astc-encoder
- Version: 4.4.0 (5a5b5a1ef60dd47c27c28c66c118d22c40e3197e, 2023)
- License: Apache 2.0

Files extracted from upstream source:

- `astcenc_*` and `astcenc.h` files from `Source`
- `LICENSE.txt`


## basis_universal

- Upstream: https://github.com/BinomialLLC/basis_universal
- Version: 1.16.4 (900e40fb5d2502927360fe2f31762bdbb624455f, 2023)
- License: Apache 2.0

Files extracted from upstream source:

- `encoder/` and `transcoder/` folders
- `LICENSE`

Applied upstream PR https://github.com/BinomialLLC/basis_universal/pull/344 to
fix build with our own copy of zstd (patch in `patches`).


## brotli

- Upstream: https://github.com/google/brotli
- Version: git (ed1995b6bda19244070ab5d331111f16f67c8054, 2023)
- License: MIT

Files extracted from upstream source:

- `common/`, `dec/` and `include/` folders from `c/`,
  minus the `dictionary.bin*` files
- `LICENSE`


## certs

- Upstream: Mozilla, via https://github.com/bagder/ca-bundle
- Version: git (3aaca635bad074a0ce5c15fa8aa0dff47f5c639a, 2023)
- License: MPL 2.0


## cvtt

- Upstream: https://github.com/elasota/ConvectionKernels
- Version: git (350416daa4e98f1c17ffc273b134d0120a2ef230, 2022)
- License: MIT

Files extracted from upstream source:

- all .cpp, .h, and .txt files except the folders MakeTables and etc2packer.

Changes related to BC6H packing and unpacking made upstream in
https://github.com/elasota/cvtt/commit/2e4b6b2747aec11f4cc6dd09ef43fa8ce769f6e2
have been removed as they caused massive quality regressions. Apply the patches
in the `patches/` folder when syncing on newer upstream commits.


## doctest

- Upstream: https://github.com/onqtam/doctest
- Version: 2.4.11 (ae7a13539fb71f270b87eb2e874fbac80bc8dda2, 2023)
- License: MIT

Files extracted from upstream source:

- `doctest/doctest.h` as `doctest.h`
- `LICENSE.txt`


## embree

- Upstream: https://github.com/embree/embree
- Version: 3.13.5 (698442324ccddd11725fb8875275dc1384f7fb40, 2022)
- License: Apache 2.0

Files extracted from upstream:

- All cpp files listed in `modules/raycast/godot_update_embree.py`
- All header files in the directories listed in `modules/raycast/godot_update_embree.py`

The `modules/raycast/godot_update_embree.py` script can be used to pull the
relevant files from the latest Embree release and apply some automatic changes.

Some changes have been made in order to remove exceptions and fix minor build errors.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments. Apply the patches in the `patches/` folder when syncing on newer upstream
commits.


## enet

- Upstream: http://enet.bespin.org
- Version: git (ea4607a90dbfbcf4da2669ea998585253d8e70b1, 2023)
- License: MIT

Files extracted from upstream source:

- all .c files in the main directory (except unix.c win32.c)
- the include/enet/ folder as enet/ (except unix.h win32.h)
- LICENSE file

Important: enet.h, host.c, protocol.c have been slightly modified
to be usable by Godot's socket implementation and allow IPv6 and DTLS.
Apply the patches in the `patches/` folder when syncing on newer upstream
commits.

Three files (godot.cpp, enet/godot.h, enet/godot_ext.h) have been added to provide
enet socket implementation using Godot classes.

It is still possible to build against a system wide ENet but doing so
will limit its functionality to IPv4 only.


## etcpak

- Upstream: https://github.com/wolfpld/etcpak
- Version: 1.0 (153f0e04a18b93c277684b577365210adcf8e11c, 2022)
- License: BSD-3-Clause

Files extracted from upstream source:

- Only the files relevant for compression (i.e. `Process*.cpp` and their deps):
  ```
  Dither.{cpp,hpp} ForceInline.hpp Math.hpp ProcessCommon.hpp ProcessRGB.{cpp,hpp}
  ProcessDxtc.{cpp,hpp} Tables.{cpp,hpp} Vector.hpp
  ```
- `AUTHORS.txt` and `LICENSE.txt`


## fonts

- `DroidSans*.woff2`:
  * Upstream: https://android.googlesource.com/platform/frameworks/base/+/master/data/fonts/
  * Version: ? (pre-2014 commit when DroidSansJapanese.ttf was obsoleted)
  * License: Apache 2.0
- `JetBrainsMono_Regular.woff2`:
  * Upstream: https://github.com/JetBrains/JetBrainsMono
  * Version: 2.304 (cd5227bd1f61dff3bbd6c814ceaf7ffd95e947d9, 2023)
  * License: OFL-1.1
- `NotoNaskhArabicUI*.woff2`:
  * Upstream: https://github.com/notofonts/arabic
  * Version: 2.014 (133ccaebf922ca080a7eef22998611ac3c242df9, 2022)
  * License: OFL-1.1
- `NotoSans*.woff2`:
  * Upstream: https://github.com/notofonts/latin-greek-cyrillic
  * Version: 2.012 (9ea0c8d37bff0c0067b03777f40aa04f2bf78f99, 2023)
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
  * Upstream: https://fonts.google.com/specimen/Open+Sans
  * Version: 1.10 (downloaded from Google Fonts in February 2021)
  * License: Apache 2.0
- All fonts are converted from the unhinted `.ttf` sources using `https://github.com/google/woff2` tool.
- Comment: Use UI font variant if available, because it has tight vertical metrics and good for UI.


## freetype

- Upstream: https://www.freetype.org
- Version: 2.13.2 (920c5502cc3ddda88f6c7d85ee834ac611bb11cc, 2023)
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

See the permalinks in `glad/gl.h` and `glad/glx.h` to regenrate the files with
a new version of the web instance.

Some changes have been made in order to allow loading OpenGL and OpenGLES APIs at the same time.
See the patches in the `patches` directory.


## glslang

- Upstream: https://github.com/KhronosGroup/glslang
- Version: 12.2.0 / sdk-1.3.250.0 (d1517d64cfca91f573af1bf7341dc3a5113349c0, 2023)
- License: glslang

Version should be kept in sync with the one of the used Vulkan SDK (see `vulkan`
section). Check Vulkan-ValidationLayers at the matching SDK tag for the known
good glslang commit: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/scripts/known_good.json

When updating, also review that our `modules/glslang/glslang_resource_limits.h`
copy of `DefaultTBuiltInResource` is in sync with the one defined upstream in
`StandAlone/ResourceLimits.cpp`.

Files extracted from upstream source:

- `glslang` (except `glslang/HLSL` and `glslang/ExtensionHeaders`),
  `OGLCompilersDLL`, `SPIRV`, w/o `CInterface` folders (depend on `StandAlone`)
- Run `cmake . && make` and copy generated `include/glslang/build_info.h`
  to `glslang/build_info.h`
- `LICENSE.txt`
- Unnecessary files like `CMakeLists.txt`, `*.m4` and `updateGrammar` removed.


## graphite

- Upstream: https://github.com/silnrsi/graphite
- Version: 1.3.14 (27572742003b93dc53dc02c01c237b72c6c25f54, 2022)
- License: MIT

Files extracted from upstream source:

- the `include` folder
- the `src` folder (minus `CMakeLists.txt` and `files.mk`)
- `COPYING`


## harfbuzz

- Upstream: https://github.com/harfbuzz/harfbuzz
- Version: 8.1.1 (1d665c2b521512cdd56964138fc601debd1f1177, 2023)
- License: MIT

Files extracted from upstream source:

- `AUTHORS`, `COPYING`, `THANKS`
- from the `src` folder, recursively
  - all the `*.c`, `*.cc`, `*.h`, `*.hh` files
  - _except_ `main.cc`, `harfbuzz*.cc`, `failing-alloc.c`, `test*.cc`, `hb-wasm*.*`


## icu4c

- Upstream: https://github.com/unicode-org/icu
- Version: 73.2 (680f521746a3bd6a86f25f25ee50a62d88b489cf, 2023)
- License: Unicode

Files extracted from upstream source:

- the `common` folder
- `scriptset.*`, `ucln_in.*`, `uspoof.cpp"` and `uspoof_impl.cpp` from the `i18n` folder
- `uspoof.h` from the `i18n/unicode` folder
- `LICENSE`

Files generated from upstream source:

- the `icudt73l.dat` built with the provided `godot_data.json` config file (see
  https://github.com/unicode-org/icu/blob/master/docs/userguide/icu_data/buildtool.md
  for instructions).

- Step 1: Download and extract both `icu4c-{version}-src.tgz` and `icu4c-{version}-data.zip` (replace `data` subfolder from the main source archive).
- Step 2: Build ICU with default options - `./runConfigureICU {PLATFORM} && make`.
- Step 3: Reconfigure ICU with custom data config - `ICU_DATA_FILTER_FILE={GODOT_SOURCE}/thirdparty/icu4c/godot_data.json ./runConfigureICU {PLATFORM} --with-data-packaging=common`.
- Step 4: Delete `data/out` folder and rebuild data - `cd data && rm -rf ./out && make`.
- Step 5: Copy `source/data/out/icudt73l.dat` to the `{GODOT_SOURCE}/thirdparty/icu4c/icudt73l.dat`.


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 2.00 (aeb7d3b463aa8228b87a28013c15ee50a7e6fcf3, 2020)
- License: Public domain or MIT

Files extracted from upstream source:

- `jpgd*.{c,h}`
- `jpge*.{c,h}`


## libktx

- Upstream: https://github.com/KhronosGroup/KTX-Software
- Version: 4.1.0 (d7255fe73cd53b856731ceb9f2c279181d0dbbca, 2023)
- License: Apache-2.0

Files extracted from upstream source:

- `LICENSE.md`
- `include/*`
- `lib/dfdutils/{LICENSES/Apache-2.0.txt,KHR,*.c,*.h,*.inl}`
- `lib/{basis_sgd.h,basis_transcode.cpp,checkheader.c,filestream.*,formatsize.h,gl_format.h,hashlist.c,ktxint.h,memstream.*,swap.c,texture*,uthash.h,vk_format.h,vkformat_enum.h}`
- `utils/unused.h`
- `other_include/KHR/*`

Some Godot-specific changes are applied via `godot.patch`.


## libogg

- Upstream: https://www.xiph.org/ogg
- Version: 1.3.5 (e1774cd77f471443541596e09078e78fdc342e4f, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*.{c,h}`
- `include/ogg/*.h` in `ogg/` (run `configure` to generate `config_types.h`)
- `COPYING`


## libpng

- Upstream: http://libpng.org/pub/png/libpng.html
- Version: 1.6.40 (f135775ad4e5d4408d2e12ffcc71bb36e6b48551, 2023)
- License: libpng/zlib

Files extracted from upstream source:

- all .c and .h files of the main directory, except from
  `example.c` and `pngtest.c`
- `arm/`, `intel/` and `powerpc/` folders
- `scripts/pnglibconf.h.prebuilt` as `pnglibconf.h`
- `LICENSE`


## libtheora

- Upstream: https://www.theora.org
- Version: git (7180717276af1ebc7da15c83162d6c5d6203aabf, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c, .h in lib/, except arm/ and c64x/ folders
- all .h files in include/theora/ as theora/
- COPYING and LICENSE


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
- Version: 1.3.2 (ca332209cb5567c9b249c86788cb2dbf8847e760, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/` and `sharpyuv/` except from: `.am`, `.rc` and `.in` files
- `AUTHORS`, `COPYING`, `PATENTS`

Patch `godot-node-debug-fix.patch` workarounds shadowing of godot's Node class in the MSVC debugger.


## mbedtls

- Upstream: https://github.com/Mbed-TLS/mbedtls
- Version: 2.28.4 (aeb97a18913a86f051afab11b2c92c6be0c2eb83, 2023)
- License: Apache 2.0

File extracted from upstream release tarball:

- All `*.h` from `include/mbedtls/` to `thirdparty/mbedtls/include/mbedtls/` except `config_psa.h` and `psa_util.h`.
- All `*.c` and `*.h` from `library/` to `thirdparty/mbedtls/library/` except those starting with `psa_*`.
- The `LICENSE` file.
- Applied the patch in `patches/windows-arm64-hardclock.diff`.
  Applied the patch in `aesni-no-arm-intrinsics.patch` to fix MSVC ARM build.
- Added 2 files `godot_core_mbedtls_platform.c` and `godot_core_mbedtls_config.h`
  providing configuration for light bundling with core.
- Added the file `godot_module_mbedtls_config.h` to customize the build configuration when bundling the full library.


## meshoptimizer

- Upstream: https://github.com/zeux/meshoptimizer
- Version: git (4a287848fd664ae1c3fc8e5e008560534ceeb526, 2022)
- License: MIT

Files extracted from upstream repository:

- All files in `src/`.
- `LICENSE.md`.

An [experimental upstream feature](https://github.com/zeux/meshoptimizer/tree/simplify-attr),
has been backported. On top of that, it was modified to report only distance error metrics
instead of a combination of distance and attribute errors. Patches for both changes can be
found in the `patches` directory.


## minimp3

- Upstream: https://github.com/lieff/minimp3
- Version: git (afb604c06bc8beb145fecd42c0ceb5bda8795144, 2021)
- License: CC0 1.0

Files extracted from upstream repository:

- `minimp3.h`
- `minimp3_ex.h`
- `LICENSE`

Some changes have been made in order to fix Windows on ARM build errors, and
to solve some MSVC warnings. See the patches in the `patches` directory.


## miniupnpc

- Upstream: https://github.com/miniupnp/miniupnp
- Version: 2.2.5 (58837ef586278d18cbebee50be758835ed4be79a, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- Copy `miniupnpc/src` and `miniupnpc/include` to `thirdparty/miniupnpc`
- Remove the following test or sample files:
  `listdevices.c minihttptestserver.c miniupnpcmodule.c upnpc.c upnperrors.* test*`
- `LICENSE`

The only modified file is `src/miniupnpcstrings.h`, which was created for Godot
(it is usually autogenerated by cmake). Bump the version number for miniupnpc in
that file when upgrading.


## minizip

- Upstream: https://www.zlib.net
- Version: 1.3 (zlib contrib, 2023)
- License: zlib

Files extracted from the upstream source:

- From `contrib/minizip`:
  `{crypt.h,ioapi.{c,h},unzip.{c,h},zip.{c,h}}`
  `MiniZip64_info.txt`

Important: Some files have Godot-made changes for use in core/io.
They are marked with `/* GODOT start */` and `/* GODOT end */`
comments and a patch is provided in the `patches` folder.


## misc

Collection of single-file libraries used in Godot components.

- `clipper.{cpp,hpp}`
  * Upstream: https://sourceforge.net/projects/polyclipping
  * Version: 6.4.2 (2017) + Godot changes (added optional exceptions handling)
  * License: BSL-1.0
- `cubemap_coeffs.h`
  * Upstream: https://research.activision.com/publications/archives/fast-filtering-of-reflection-probes
    File coeffs_const_8.txt (retrieved April 2020)
  * License: MIT
- `fastlz.{c,h}`
  * Upstream: https://github.com/ariya/FastLZ
  * Version: 0.5.0 (4f20f54d46f5a6dd4fae4def134933369b7602d2, 2020)
  * License: MIT
- `hq2x.{cpp,h}`
  * Upstream: https://github.com/brunexgeek/hqx
  * Version: TBD, file structure differs
  * License: Apache 2.0
- `ifaddrs-android.{cc,h}`
  * Upstream: https://chromium.googlesource.com/external/webrtc/stable/talk/+/master/base/ifaddrs-android.h
  * Version: git (5976650443d68ccfadf1dea24999ee459dd2819d, 2013)
  * License: BSD-3-Clause
- `mikktspace.{c,h}`
  * Upstream: https://archive.blender.org/wiki/index.php/Dev:Shading/Tangent_Space_Normal_Maps/
  * Version: 1.0 (2011)
  * License: zlib
- `FastNoiseLite.h}`
  * Upstream: https://github.com/Auburn/FastNoiseLite
  * Version: git (6be3d6bf7fb408de341285f9ee8a29b67fd953f1, 2022) + custom changes
  * License: MIT
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
  * Modifications: Change from STL to Godot types (see provided patch).
  * License: MIT
- `r128.h`
  * Upstream: https://github.com/fahickman/r128
  * Version: 1.4.4 (cf2e88fc3e7d7dfe99189686f914874cd0bda15e, 2020)
  * License: Public Domain or Unlicense
- `smaz.{c,h}`
  * Upstream: https://github.com/antirez/smaz
  * Version: git (2f625846a775501fb69456567409a8b12f10ea25, 2012)
  * License: BSD-3-Clause
  * Modifications: use `const char*` instead of `char*` for input string
- `smolv.h`
  * Upstream: https://github.com/aras-p/smol-v
  * Version: git (4b52c165c13763051a18e80ffbc2ee436314ceb2, 2020)
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
- Version: 1.10 (64a91eec3ca3787e6f78b4c99fcd3052ad3e37c0, 2021)
- License: MIT

Files extracted from the upstream source:

- `msdfgen.h`
- Files in `core/` folder.
- `LICENSE.txt`


## nvapi

- Upstream: http://download.nvidia.com/XFree86/nvapi-open-source-sdk
- Version: R525
- License: MIT

- `nvapi_minimal.h` was created by using `nvapi.h` from upstream and removing unnecessary code.


## oidn

- Upstream: https://github.com/OpenImageDenoise/oidn
- Version: 1.1.0 (c58c5216db05ceef4cde5a096862f2eeffd14c06, 2019)
- License: Apache 2.0

Files extracted from upstream source:

- common/* (except tasking.* and CMakeLists.txt)
- core/*
- include/OpenImageDenoise/* (except version.h.in)
- LICENSE.txt
- mkl-dnn/include/*
- mkl-dnn/src/* (except CMakeLists.txt)
- weights/rtlightmap_hdr.tza
- scripts/resource_to_cpp.py

Modified files:
Modifications are marked with `// -- GODOT start --` and `// -- GODOT end --`.
Patch files are provided in `oidn/patches/`.

- core/autoencoder.cpp
- core/autoencoder.h
- core/common.h
- core/device.cpp
- core/device.h
- core/transfer_function.cpp

- scripts/resource_to_cpp.py (used in modules/denoise/resource_to_cpp.py)


## openxr

- Upstream: https://github.com/KhronosGroup/OpenXR-SDK
- Version: 1.0.28 (f5beb0131f1bea8701ace744d1b50df9049bf331, 2023)
- License: Apache 2.0

Files extracted from upstream source:

- include/
- src/common/
- src/loader/
- src/*.{c,h}
- src/external/jsoncpp/include/
- src/external/jsoncpp/src/lib_json/
- LICENSE and COPYING.adoc

Exclude:

- src/external/android-jni-wrappers and src/external/jnipp (not used yet)
- All CMake stuff: cmake/, CMakeLists.txt and *.cmake
- All Gradle stuff: *gradle*, AndroidManifest.xml
- All following files (and their .license files): *.{def,expsym,in,json,map,pom,rc,txt}
- All dotfiles


## pcre2

- Upstream: http://www.pcre.org
- Version: 10.42 (52c08847921a324c804cabf2814549f50bce1265, 2022)
- License: BSD-3-Clause

Files extracted from upstream source:

- Files listed in the file NON-AUTOTOOLS-BUILD steps 1-4
- All .h files in src/ apart from pcre2posix.h
- src/pcre2_jit_match.c
- src/pcre2_jit_misc.c
- src/sljit/
- AUTHORS and LICENCE

A sljit patch from upstream was backported to fix macOS < 11.0 compilation
in 10.40, it can be found in the `patches` folder.


## recastnavigation

- Upstream: https://github.com/recastnavigation/recastnavigation
- Version: 1.6.0 (6dc1667f580357e8a2154c28b7867bea7e8ad3a7, 2023)
- License: zlib

Files extracted from upstream source:

- `Recast/` folder without `CMakeLists.txt`
- License.txt


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

- All .cpp and .h files in the `src/` folder except for Export.h and RVO.h
- LICENSE

Important: Nearly all files have Godot-made changes and renames
to make the 2D and 3D rvo libraries compatible with each other
and solve conflicts and also enrich the feature set originally
proposed by these libraries and better integrate them with Godot.


## spirv-reflect

- Upstream: https://github.com/KhronosGroup/SPIRV-Reflect
- Version: sdk-1.3.250.0 (1fd43331f0bd77cc0f421745781f79a14d8f2bb1, 2023)
- License: Apache 2.0

Now tracks Vulkan SDK releases, so keep it in sync with volk / vulkan.

Files extracted from upstream source:

- `spirv_reflect.{c,h}`
- `include` folder
- `LICENSE`

Some downstream changes have been made and are identified by
`// -- GODOT begin --` and `// -- GODOT end --` comments.
They can be reapplied using the patches included in the `patches`
folder.


## squish

- Upstream: https://sourceforge.net/projects/libsquish
- Version: 1.15 (r104, 2017)
- License: MIT

Files extracted from upstream source:

- all .cpp, .h and .inl files

Important: Some files have Godot-made changes.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments and a patch is provided in the squish/ folder.


## tinyexr

- Upstream: https://github.com/syoyo/tinyexr
- Version: 1.0.5 (3627ab3060592468d49547b4cdf5353e9e2b50dc, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`

The `tinyexr.cc` file was modified to include `zlib.h` which we provide,
instead of `miniz.h` as an external dependency.


## thorvg

- Upstream: https://github.com/thorvg/thorvg
- Version: 0.10.0 (b8c605583fd7de73209a93a1238e1ba72cce2e8f, 2023)
- License: MIT

Files extracted from upstream source:

See `thorvg/update-thorvg.sh` for extraction instructions. Set the version
number and run the script.


## vhacd

- Upstream: https://github.com/kmammou/v-hacd
- Version: git (1a49edf29c69039df15286181f2f27e17ceb9aef, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- From `src/VHACD_Lib/`: `inc`, `public` and `src`
- `LICENSE`

Some downstream changes have been made and are identified by
`// -- GODOT start --` and `// -- GODOT end --` comments.
They can be reapplied using the patches included in the `vhacd`
folder.


## volk

- Upstream: https://github.com/zeux/volk
- Version: sdk-1.3.250.0 (b3bc21e584f97400b6884cb2a541a56c6a5ddba3, 2023)
- License: MIT

Unless there is a specific reason to package a more recent version, please stick
to tagged releases. All Vulkan libraries and headers should be kept in sync so:

- Update Vulkan SDK components to the matching tag (see "vulkan").
- Update glslang (see "glslang").
- Update spirv-reflect (see "spirv-reflect").

Files extracted from upstream source:

- `volk.h`, `volk.c`
- `LICENSE.md`


## vulkan

- Upstream: https://github.com/KhronosGroup/Vulkan-Headers
- Version: sdk-1.3.250 (bae9700cd9425541a0f6029957f005e5ad3ef660, 2023)
- License: Apache 2.0

The vendored version should be kept in sync with volk, see above.

Files extracted from upstream source:

- `include/`
- `LICENSE.txt`

`vk_enum_string_helper.h` is taken from the matching `Vulkan-ValidationLayers`
SDK release: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/layers/vulkan/generated/vk_enum_string_helper.h

`vk_mem_alloc.h` is taken from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
Version: 3.0.1 (2022-06-10), commit `cfdc0f8775ab3258a3b9c4e47d8ce4b6f52a5441`
`vk_mem_alloc.cpp` is a Godot file and should be preserved on updates.

Patches in the `patches` directory should be re-applied after updates.


## wslay

- Upstream: https://github.com/tatsuhiro-t/wslay
- Version: 1.1.1+git (0e7d106ff89ad6638090fd811a9b2e4c5dda8d40, 2022)
- License: MIT

File extracted from upstream release tarball:

- Run `cmake .` to generate `config.h` and `wslayver.h`.
  Contents might need tweaking for Godot, review diff.
- All `*.c` and `*.h` files from `lib/`
- All `*.h` in `lib/includes/wslay/` as `wslay/`
- `wslay/wslay.h` has a small Godot addition to fix MSVC build.
  See `patches/msvcfix.diff`
- `COPYING`


## xatlas

- Upstream: https://github.com/jpcy/xatlas
- Version: git (f700c7790aaa030e794b52ba7791a05c085faf0c, 2022)
- License: MIT

Files extracted from upstream source:

- `source/xatlas/xatlas.{cpp,h}`
- `LICENSE`


## zlib

- Upstream: https://www.zlib.net
- Version: 1.3 (2023)
- License: zlib

Files extracted from upstream source:

- All `*.c` and `*.h` files, minus `infback.c`
- `LICENSE`


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.5.5 (63779c798237346c2b245c546c40b72a5a5913fe, 2023)
- License: BSD-3-Clause

Files extracted from upstream source:

- `lib/{common/,compress/,decompress/,zstd.h,zstd_errors.h}`
- `LICENSE`
