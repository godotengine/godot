# Third party libraries

Please keep categories (`##` level) listed alphabetically and matching their
respective folder names. Use two empty lines to separate categories for
readability.


## basis_universal

- Upstream: https://github.com/BinomialLLC/basis_universal
- Version: git (ba1c3e40f1d434ebaf9a167b44e9b11d2bf0f765, 2021)
- License: Apache 2.0

Files extracted from upstream source:

- `encoder/` and `transcoder/` folders
- `LICENSE`


## bullet

- Upstream: https://github.com/bulletphysics/bullet3
- Version: 3.17 (ebe1916b90acae8b13cd8c6b637d8327cdc64e94, 2021)
- License: zlib

Files extracted from upstream source:

- `src/*` apart from CMakeLists.txt and premake4.lua files
- `LICENSE.txt`, and `VERSION` as `VERSION.txt`

Includes some patches in the `patches` folder which have been sent upstream.


## certs

- Upstream: Mozilla, via https://github.com/bagder/ca-bundle
- Version: git (8b263a18fca98ea371e54227837321c5cdaa1ba7, 2021)
- License: MPL 2.0


## cvtt

- Upstream: https://github.com/elasota/cvtt
- Version: 1.0.0-beta4 (cc8472a04ba110fe999c686d07af40f7839051fd, 2018)
- License: MIT

Files extracted from upstream source:

- all .cpp, .h, and .txt files in ConvectionKernels/


## doctest

- Upstream: https://github.com/onqtam/doctest
- Version: 2.4.4 (97d5a9447e66cd5e107b7a6c463be4a468a40496, 2020)
- License: MIT

Extracted from .zip provided. Extracted license and header only.


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
- Version: git (7c3cb6fe708d4ae330b0ab2af1ad472bae2a37a2, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- Only the files relevant for compression (i.e. `Process*.cpp` and their deps):
  ```
  Dither.{cpp,hpp} ForceInline.hpp Math.hpp ProcessCommon.hpp ProcessRGB.{cpp,hpp}
  ProcessDxtc.{cpp,hpp} Tables.{cpp,hpp} Vector.hpp
  ```
- `AUTHORS.txt` and `LICENSE.txt`


## fonts

- `NotoSans*.ttf`, `NotoNaskhArabicUI_*.ttf`:
  * Upstream: https://github.com/googlefonts/noto-fonts
  * Version: v2017-10-24-phase3-second-cleanup
  * License: OFL-1.1
  * Comment: Use UI font variant if available, because it has tight vertical metrics and
    good for UI.
- `Hack_Regular.ttf`:
  * Upstream: https://github.com/source-foundry/Hack
  * Version: 3.003 (2018)
  * License: MIT + Bitstream Vera License
- `DroidSans*.ttf`:
  * Upstream: https://android.googlesource.com/platform/frameworks/base/+/master/data/fonts/
  * Version: ? (pre-2014 commit when DroidSansJapanese.ttf was obsoleted)
  * License: Apache 2.0
- `OpenSans_SemiBold.ttf`:
  * Upstream: https://fonts.google.com/specimen/Open+Sans
  * Version: 1.10 (downloaded from Google Fonts in February 2021)
  * License: Apache 2.0


## freetype

- Upstream: https://www.freetype.org
- Version: 2.10.4 (6a2b3e4007e794bfc6c91030d0ed987f925164a8, 2020)
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- the `src/` folder, stripped of the `Jamfile` files and the `tools` subfolder
- the `include/` folder
- `docs/{FTL.TXT,LICENSE.TXT}`


## glslang

- Upstream: https://github.com/KhronosGroup/glslang
- Version: 11.6.0 (2fb89a0072ae7316af1c856f22663fde4928128a, 2021)
- License: glslang

Version should be kept in sync with the one of the used Vulkan SDK (see `vulkan`
section). Check Vulkan-ValidationLayers at the matching SDK tag for the known
good glslang commit: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/scripts/known_good.json

Files extracted from upstream source:

- `glslang` (except `glslang/HLSL`), `OGLCompilersDLL`, `SPIRV`
- `StandAlone/{DirStackFileIncluder.h,ResourceLimits.{cpp,h}}`
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
- Version: 3.1.1 (cd5c6cd0419ac5e4de975d6c476fb760bf06d2ce, 2021)
- License: MIT

Files extracted from upstream source:

- the `src` folder
- `AUTHORS`, `COPYING`, `THANKS`


## icu4c

- Upstream: https://github.com/unicode-org/icu
- Version: 70.1 (a56dde820dc35665a66f2e9ee8ba58e75049b668, 2021)
- License: Unicode

Files extracted from upstream source:

- the `common` folder
- `LICENSE`

Files generated from upstream source:

- the `icudt70l.dat` built with the provided `godot_data.json` config file (see
  https://github.com/unicode-org/icu/blob/master/docs/userguide/icu_data/buildtool.md
  for instructions).

- Step 1: Build ICU with default options - `./runConfigureICU {PLATFORM} && make`.
- Step 2: Reconfigure ICU with custom data config - `ICU_DATA_FILTER_FILE={GODOT_SOURCE}/thirdparty/icu4c/godot_data.json ./runConfigureICU {PLATFORM} --with-data-packaging=common`.
- Step 3: Delete `data/out` folder and rebuild data - `cd data && rm -rf ./out && make`.
- Step 4: Copy `source/data/out/icudt70l.dat` to the `{GODOT_SOURCE}/thirdparty/icu4c/icudt70l.dat`.


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 2.00 (aeb7d3b463aa8228b87a28013c15ee50a7e6fcf3, 2020)
- License: Public domain or MIT

Files extracted from upstream source:

- `jpgd*.{c,h}`


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
- Version: 1.2.1 (9ce5843dbabcfd3f7c39ec7ceba9cbeb213cbfdf, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: `.am`, `.rc` and `.in` files
- `AUTHORS`, `COPYING`, `PATENTS`

Important: The files `utils/bit_reader_utils.{c,h}` have Godot-made
changes to ensure they build for Javascript/HTML5. Those
changes are marked with `// -- GODOT --` comments.


## mbedtls

- Upstream: https://tls.mbed.org/
- Version: 2.16.11 (aa1d4e097342af799ba80dfb13640efef498227c, 2021)
- License: Apache 2.0

File extracted from upstream release tarball:

- All `*.h` from `include/mbedtls/` to `thirdparty/mbedtls/include/mbedtls/`.
- All `*.c` from `library/` to `thirdparty/mbedtls/library/`.
- `LICENSE` and `apache-2.0.txt` files.
- Applied the patch in `patches/1453.diff` (upstream PR:
  https://github.com/ARMmbed/mbedtls/pull/1453).
- Applied the patch in `patches/padlock.diff`. This disables VIA padlock
  support which defines a symbol `unsupported` which clashes with a
  pre-defined symbol.
- Applied the patch in `patches/pr4948-fix-clang12-opt.patch`. Upstream bugfix
  from PR 4948 to fix a bug caused by Clang 12 optimizations.
- Applied the patch in `patches/pr4819-faster-base64.patch`. This fixes a certs
  parsing speed regression since 2.16.10 (upstream PR:
  https://github.com/ARMmbed/mbedtls/pull/4819).
- Added 2 files `godot_core_mbedtls_platform.c` and `godot_core_mbedtls_config.h`
  providing configuration for light bundling with core.


## meshoptimizer

- Upstream: https://github.com/zeux/meshoptimizer
- Version: git (f5d83e879c48f8664783a69b4f50711d27549b66, 2021)
- License: MIT

Files extracted from upstream repository:

- All files in `src/`.
- `LICENSE.md`.

An [experimental upstream feature](https://github.com/zeux/meshoptimizer/tree/simplify-attr),
has been backported. On top of that, it was modified to report only distance error metrics 
instead of a combination of distance and attribute errors. Patches for both changes can be
found in the `patches` directory.


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

- Upstream: http://www.zlib.net
- Version: 1.2.11 (zlib contrib, 2017)
- License: zlib

Files extracted from the upstream source:

- contrib/minizip/{crypt.h,ioapi.{c,h},zip.{c,h},unzip.{c,h}}

Important: Some files have Godot-made changes for use in core/io.
They are marked with `/* GODOT start */` and `/* GODOT end */`
comments and a patch is provided in the minizip/ folder.


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
- `open-simplex-noise.{c,h}`
  * Upstream: https://github.com/smcameron/open-simplex-noise-in-c
  * Version: git (826f1dd1724e6fb3ff45f58e48c0fbae864c3403, 2020) + custom changes
  * License: Public Domain or Unlicense
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
  * Version: 1.00 (2bb4a0accd4003c1db4c24533981e01b1adfd656, 2019)
  * License: Public Domain or Unlicense or MIT
- `yuv2rgb.h`
  * Upstream: http://wss.co.uk/pinknoise/yuv2rgb/ (to check)
  * Version: ?
  * License: BSD


## msdfgen

- Upstream: https://github.com/Chlumsky/msdfgen
- Version: 1.9.1 (1b3b6b985094e6f12751177490add3ad11dd91a9, 2010)
- License: MIT

Files extracted from the upstream source:

- `msdfgen.h`
- Files in `core/` folder.
- `LICENSE.txt` and `CHANGELOG.md`


## nanosvg

- Upstream: https://github.com/memononen/nanosvg
- Version: git (ccdb1995134d340a93fb20e3a3d323ccb3838dd0, 2021)
- License: zlib

Files extracted from the upstream source:

- All .h files in `src/`
- LICENSE.txt


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


## pcre2

- Upstream: http://www.pcre.org
- Version: 10.39 (35fee4193b852cb504892352bd0155de10809889, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- Files listed in the file NON-AUTOTOOLS-BUILD steps 1-4
- All .h files in src/ apart from pcre2posix.h
- src/pcre2_jit_match.c
- src/pcre2_jit_misc.c
- src/sljit/
- AUTHORS and LICENCE


## pvrtccompressor

- Upstream: https://bitbucket.org/jthlim/pvrtccompressor (dead link)
  Unofficial backup fork: https://github.com/LibreGamesArchive/PVRTCCompressor
- Version: hg (cf7177748ee0dcdccfe89716dc11a47d2dc81af5, 2015)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .cpp and .h files apart from `main.cpp`
- LICENSE.TXT


## recastnavigation

- Upstream: https://github.com/recastnavigation/recastnavigation
- Version: git (57610fa6ef31b39020231906f8c5d40eaa8294ae, 2019)
- License: zlib

Files extracted from upstream source:

- `Recast/` folder without `CMakeLists.txt`
- License.txt


## rvo2

- Upstream: https://github.com/snape/RVO2-3D
- Version: 1.0.1 (e3883f288a9e55ecfed3633a01af3e12778c6acf, 2016)
- License: Apache 2.0

Files extracted from upstream source:

- All .cpp and .h files in the `src/` folder except for RVO.h, RVOSimulator.cpp and RVOSimulator.h
- LICENSE

Important: Some files have Godot-made changes; so to enrich the features
originally proposed by this library and better integrate this library with
Godot. Please check the file to know what's new.


## spirv-reflect

- Upstream: https://github.com/KhronosGroup/SPIRV-Reflect
- Version: git (cc937caab141d889c9c9dff572c5a6854d5cf9b4, 2021)
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
- Version: 1.2.190 (760a782f295a66de7391d6ed573d65e3fb1c8450, 2021)
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
- Version: 1.2.190 (9e62d027636cd7210f60d934f56107ed6e1579b8, 2021)
- License: Apache 2.0

The vendored version should be kept in sync with volk, see above.

Files extracted from upstream source:

- `include/`
- `LICENSE.txt`

`vk_enum_string_helper.h` is taken from the matching `Vulkan-ValidationLayers`
SDK release: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/layers/generated/vk_enum_string_helper.h

`vk_mem_alloc.h` is taken from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
Version: 3.0.0-development (2021-07-07), branch `feature-small-buffers`, commit `cfea2f72851f9ee4a399769f18865047b83711f1`
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
- Version: git (ec707faeac3b95e6b416076a9509718cce105b6a, 2021)
- License: MIT

Files extracted from upstream source:

- `source/xatlas/xatlas.{cpp,h}`
- `LICENSE`


## zlib

- Upstream: http://www.zlib.net
- Version: 1.2.11 (2017)
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

