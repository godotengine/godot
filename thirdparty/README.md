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


## basis_universal

- Upstream: https://github.com/BinomialLLC/basis_universal
- Version: git (1531cfaf9ed5232248a0a45736686a849ca3befc, 2022)
- License: Apache 2.0

Files extracted from upstream source:

- `encoder/` and `transcoder/` folders
- `LICENSE`


## brotli

- Upstream: https://github.com/google/brotli
- Version: git (f4153a09f87cbb9c826d8fc12c74642bb2d879ea, 2022)
- License: MIT

Files extracted from upstream source:

- `common/`, `dec/` and `include/` folders
- `LICENSE`


## certs

- Upstream: Mozilla, via https://github.com/bagder/ca-bundle
- Version: git (8b263a18fca98ea371e54227837321c5cdaa1ba7, 2021)
- License: MPL 2.0


## cvtt

- Upstream: https://github.com/elasota/ConvectionKernels
- Version: git (dc2dbbe0ae2cf2be06ef56d1021e2222a56c7fe2, 2021)
- License: MIT

Files extracted from upstream source:

- all .cpp, .h, and .txt files except the folders MakeTables and etc2packer.


## doctest

- Upstream: https://github.com/onqtam/doctest
- Version: 2.4.9 (b7c21ec5ceeadb4951b00396fc1e4642dd347e5f, 2022)
- License: MIT

Files extracted from upstream source:

- `doctest/doctest.h` as `doctest.h`
- `LICENSE.txt`


## embree

- Upstream: https://github.com/embree/embree
- Version: 3.13.1 (12b99393438a4cc9e478e33459eed78bec6233fd, 2021)
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
- Version: 1.3.17 (e0e7045b7e056b454b5093cb34df49dc4cee0bee, 2020)
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
- Version: 1.0 (a77d5a37ddf48034cee8aeb9e8792a623c265b4c, 2022)
- License: BSD-3-Clause

Files extracted from upstream source:

- Only the files relevant for compression (i.e. `Process*.cpp` and their deps):
  ```
  Dither.{cpp,hpp} ForceInline.hpp Math.hpp ProcessCommon.hpp ProcessRGB.{cpp,hpp}
  ProcessDxtc.{cpp,hpp} Tables.{cpp,hpp} Vector.hpp
  ```
- `AUTHORS.txt` and `LICENSE.txt`


## fonts

- `NotoSans*.woff2`, `NotoNaskhArabicUI_*.woff2`:
  * Upstream: https://github.com/googlefonts/noto-fonts
  * Version: v2017-10-24-phase3-second-cleanup
  * License: OFL-1.1
  * Comment: Use UI font variant if available, because it has tight vertical metrics and
    good for UI.
- `JetBrainsMono_Regular.woff2`:
	* Upstream: https://github.com/JetBrains/JetBrainsMono
  * Version: 2.242
  * License: OFL-1.1
- `DroidSans*.woff2`:
  * Upstream: https://android.googlesource.com/platform/frameworks/base/+/master/data/fonts/
  * Version: ? (pre-2014 commit when DroidSansJapanese.ttf was obsoleted)
  * License: Apache 2.0
- `OpenSans_SemiBold.woff2`:
  * Upstream: https://fonts.google.com/specimen/Open+Sans
  * Version: 1.10 (downloaded from Google Fonts in February 2021)
  * License: Apache 2.0
- All fonts are converted from the `.ttf` sources using `https://github.com/google/woff2` tool.


## freetype

- Upstream: https://www.freetype.org
- Version: 2.12.1 (e8ebfe988b5f57bfb9a3ecb13c70d9791bce9ecf, 2022)
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- `src/` folder, minus the `dlg` and `tools` subfolders
  * These files can be removed: `.dat`, `.diff`, `.mk`, `.rc`, `README*`
- `include/` folder, minus the `dlg` subfolder
- `LICENSE.TXT` and `docs/FTL.TXT`

Some changes have been made in order to prevent LTO from removing code.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments. Apply the patches in the `patches/` folder when syncing on newer upstream
commits.


## glslang

- Upstream: https://github.com/KhronosGroup/glslang
- Version: 11.8.0 (c34bb3b6c55f6ab084124ad964be95a699700d34, 2022)
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
- Version: 1.3.14 (80c52493ef42e6fe605a69dcddd2a691cd8a1380, 2021)
- License: MPL-2.0

Files extracted from upstream source:

- the `include` folder
- the `src` folder (minus `CMakeLists.txt` and `files.mk`)
- `COPYING`


## harfbuzz

- Upstream: https://github.com/harfbuzz/harfbuzz
- Version: 4.3.0 (aee123fc83388b8f5acfb301d87bd92eccc5b843, 2022)
- License: MIT

Files extracted from upstream source:

- the `src` folder
- `AUTHORS`, `COPYING`, `THANKS`


## icu4c

- Upstream: https://github.com/unicode-org/icu
- Version: 71.1 (c205e7ee49a7086a28b9c275fcfdac9ca3dc815d, 2022)
- License: Unicode

Files extracted from upstream source:

- the `common` folder
- `LICENSE`

Files generated from upstream source:

- the `icudt71l.dat` built with the provided `godot_data.json` config file (see
  https://github.com/unicode-org/icu/blob/master/docs/userguide/icu_data/buildtool.md
  for instructions).

- Step 1: Build ICU with default options - `./runConfigureICU {PLATFORM} && make`.
- Step 2: Reconfigure ICU with custom data config - `ICU_DATA_FILTER_FILE={GODOT_SOURCE}/thirdparty/icu4c/godot_data.json ./runConfigureICU {PLATFORM} --with-data-packaging=common`.
- Step 3: Delete `data/out` folder and rebuild data - `cd data && rm -rf ./out && make`.
- Step 4: Copy `source/data/out/icudt71l.dat` to the `{GODOT_SOURCE}/thirdparty/icu4c/icudt71l.dat`.


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 2.00 (aeb7d3b463aa8228b87a28013c15ee50a7e6fcf3, 2020)
- License: Public domain or MIT

Files extracted from upstream source:

- `jpgd*.{c,h}`
- `jpge*.{c,h}`


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
- Version: 1.6.37 (a40189cf881e9f0db80511c382292a5604c3c3d1, 2019)
- License: libpng/zlib

Files extracted from upstream source:

- all .c and .h files of the main directory, except from
  `example.c` and `pngtest.c`
- the arm/ folder
- `scripts/pnglibconf.h.prebuilt` as `pnglibconf.h`
- `LICENSE`


## libtheora

- Upstream: https://www.theora.org
- Version: 1.1.1 (2010)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c, .h in lib/
- all .h files in include/theora/ as theora/
- COPYING and LICENSE

Upstream patches included in the `patches` directory have been applied
on top of the 1.1.1 source (not included in any stable release yet).


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
- Version: 1.2.2 (b0a860891dcd4c0c2d7c6149e5cccb6eb881cc21, 2022)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: `.am`, `.rc` and `.in` files
- `AUTHORS`, `COPYING`, `PATENTS`


## mbedtls

- Upstream: https://tls.mbed.org/
- Version: 2.16.12 (cf4667126010c665341f9e50ef691b7ef8294188, 2021)
- License: Apache 2.0

File extracted from upstream release tarball:

- All `*.h` from `include/mbedtls/` to `thirdparty/mbedtls/include/mbedtls/` except `config_psa.h` and `psa_util.h`.
- All `*.c` and `*.h` from `library/` to `thirdparty/mbedtls/library/` except those starting with `psa_*`.
- `LICENSE` and `apache-2.0.txt` files.
- Applied the patch in `patches/1453.diff` (upstream PR:
  https://github.com/ARMmbed/mbedtls/pull/1453).
- Added 2 files `godot_core_mbedtls_platform.c` and `godot_core_mbedtls_config.h`
  providing configuration for light bundling with core.


## meshoptimizer

- Upstream: https://github.com/zeux/meshoptimizer
- Version: git (ea4558d1c0f217f1d67ed7fe0b07896ece88ae18, 2022)
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


## miniupnpc

- Upstream: https://github.com/miniupnp/miniupnp
- Version: 2.2.3 (2df8120326ed4246e049a7a6de707539604cd514, 2021)
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
- Version: 1.2.12 (zlib contrib, 2022)
- License: zlib

Files extracted from the upstream source:

- contrib/minizip/{crypt.h,ioapi.{c,h},unzip.{c,h},zip.{c,h}}

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
- Version: 1.9.2 (64a91eec3ca3787e6f78b4c99fcd3052ad3e37c0, 2021)
- License: MIT

Files extracted from the upstream source:

- `msdfgen.h`
- Files in `core/` folder.
- `LICENSE.txt` and `CHANGELOG.md`


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
- Version: 1.0.23 (885a90f8934d84121344ba8e4aa5159d5b496e08, 2022)
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
- All following files (and their .license files): *.{def,in,json,map,pom,rc}


## pcre2

- Upstream: http://www.pcre.org
- Version: 10.40 (3103b8f20a3b9944b177e812fde29fbfb8b90558, 2022)
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
- Version: git (5a870d427e47abd4a8e4ce58a95582ec049434d5, 2022)
- License: zlib

Files extracted from upstream source:

- `Recast/` folder without `CMakeLists.txt`
- License.txt


## rvo2

- Upstream: https://github.com/snape/RVO2-3D
- Version: git (bfc048670a4e85066e86a1f923d8ea92e3add3b2, 2021)
- License: Apache 2.0

Files extracted from upstream source:

- All .cpp and .h files in the `src/` folder except for Export.h, RVO.h, RVOSimulator.cpp and RVOSimulator.h
- LICENSE

Important: Some files have Godot-made changes; so to enrich the features
originally proposed by this library and better integrate this library with
Godot. See the patch in the `patches` folder for details.


## spirv-reflect

- Upstream: https://github.com/KhronosGroup/SPIRV-Reflect
- Version: git (1aceb6af56e74b92a00378842dda5c5a73f49a4b, 2022)
- License: Apache 2.0

Does not track Vulkan SDK releases closely, but try to package a commit newer
than the matching glslang and Vulkan headers, just in case.

Files extracted from upstream source:

- `spirv_reflect.{c,h}`
- `include` folder
- `LICENSE`

Some downstream changes have been made and are identified by
`// -- GODOT begin --` and `// -- GODOT end --` comments.
They can be reapplied using the patch included in the `patches`
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
- Version: 1.0.1 (67010eae802211202d0797f4df2b809f4ba7442c, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`

The `tinyexr.cc` file was modified to include `zlib.h` which we provide,
instead of `miniz.h` as an external dependency.


## thorvg

- Upstream: https://github.com/Samsung/thorvg
- Version: 0.8.1 (c4ccb1078f4390ec749ab8e05ba7e9e35f81285f, 2022)
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
- Version: 1.3.204 (92ba7c9f112a82cecf452ebf4b7c46f149a5799e, 2022)
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
- Version: 1.3.204 (1dace16d8044758d32736eb59802d171970e9448, 2022)
- License: Apache 2.0

The vendored version should be kept in sync with volk, see above.

Files extracted from upstream source:

- `include/`
- `LICENSE.txt`

`vk_enum_string_helper.h` is taken from the matching `Vulkan-ValidationLayers`
SDK release: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/layers/generated/vk_enum_string_helper.h

`vk_mem_alloc.h` is taken from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
Version: 3.0.1 (2022-06-10), commit `cfdc0f8775ab3258a3b9c4e47d8ce4b6f52a5441`
`vk_mem_alloc.cpp` is a Godot file and should be preserved on updates.

Patches in the `patches` directory should be re-applied after updates.


## wslay

- Upstream: https://github.com/tatsuhiro-t/wslay
- Version: 1.1.1+git (45d22583b488f79d5a4e598cc7675c191c5ab53f, 2021)
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
- Version: git (16ace528acd2cf1f16a7c0dde99c42c486488dbe, 2022)
- License: MIT

Files extracted from upstream source:

- `source/xatlas/xatlas.{cpp,h}`
- `LICENSE`


## zlib

- Upstream: https://www.zlib.net
- Version: 1.2.12 (2022)
- License: zlib

Files extracted from upstream source:

- all .c and .h files


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.5.0 (a488ba114ec17ea1054b9057c26a046fc122b3b6, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- `lib/{common/,compress/,decompress/,zstd.h,zstd_errors.h}`
- `LICENSE`
