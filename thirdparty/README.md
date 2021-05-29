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
- Version: 3.08 (df09fd9ed37e365ceae884ca7f620b61607dae2e, 2020)
- License: zlib

Files extracted from upstream source:

- src/* apart from CMakeLists.txt and premake4.lua files
- LICENSE.txt


## certs

- Upstream: Mozilla, via https://apps.fedoraproject.org/packages/ca-certificates
- Version: 2018.2.26 (2018)
- License: MPL 2.0

File extracted from a recent Fedora install:
/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
(It can't be extracted directly from the package,
as it's generated on the user's system.)


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
- Version: 3.13.0 (7c53133eb21424f7f0ae1e25bf357e358feaf6ab, 2021)
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
- Version: git (f27daea656ff77671580f838a889e33049430ebd, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- Only the files relevant for compression (i.e. `Process*.cpp` and their deps):
  ```
  Dither.{cpp,hpp} ForceInline.hpp Math.hpp ProcessCommon.hpp ProcessRGB.{cpp,hpp}
  ProcessDxtc.{cpp,hpp} Tables.{cpp,hpp} Vector.hpp
  ```
- `AUTHORS.txt` and `LICENSE.txt`


## fonts

- `NotoSans*.ttf`, `NotoNaskhArabicUI_Regular.ttf`:
  * Upstream: https://github.com/googlei18n/noto-fonts
  * Version: 1.06 (2017)
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
- `Tamsyn*.png`:
  * Upstream: http://www.fial.com/~scott/tamsyn-font/
  * Version: 1.11 (2015)
  * License: Tamsyn
  * Comment: Extracted "0..9,A..F" characters for hex code printing.


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
- Version: git (dd69df7f3dac26362e10b0f38efb9e47990f7537, 2020)
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
- Unnecessary files like `CMakeLists.txt` and `updateGrammar` removed.


## graphite

- Upstream: https://github.com/silnrsi/graphite
- Version: 1.3.14 (92f59dcc52f73ce747f1cdc831579ed2546884aa, 2020)
- License: MPL-2.0

Files extracted from upstream source:

- the `include` folder
- the `src` folder
- `COPYING`, `ChangeLog`


## harfbuzz

- Upstream: https://github.com/harfbuzz/harfbuzz
- Version: 2.8.0 (03538e872a0610a65fad692b33d3646f387cf578, 2021)
- License: MIT

Files extracted from upstream source:

- the `src` folder
- `AUTHORS`, `COPYING`, `NEWS`, `THANKS`


## icu4c

- Upstream: https://github.com/unicode-org/icu
- Version: 69.1 (0e7b4428866f3133b4abba2d932ee3faa708db1d, 2021)
- License: Unicode

Files extracted from upstream source:

- the `common` folder
- `LICENSE`

Files generated from upstream source:

- the `icudt69l.dat` built with the provided `godot_data.json` config file (see
  https://github.com/unicode-org/icu/blob/master/docs/userguide/icu_data/buildtool.md
  for instructions)


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 2.00 (aeb7d3b463aa8228b87a28013c15ee50a7e6fcf3, 2020)
- License: Public domain or MIT

Files extracted from upstream source:

- `jpgd*.{c,h}`


## libogg

- Upstream: https://www.xiph.org/ogg
- Version: git (c8fca6b4a02d695b1ceea39b330d4406001c03ed, 2019)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*.{c,h}`
- `include/ogg/*.h` in ogg/
- COPYING


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


## libsimplewebm

- Upstream: https://github.com/zaps166/libsimplewebm
- Version: git (fe57fd3cfe6c0af4c6af110b1f84a90cf191d943, 2019)
- License: MIT (main), BSD-3-Clause (libwebm)

This contains libwebm, but the version in use is updated from the one used by libsimplewebm,
and may have *unmarked* alterations from that.

Files extracted from upstream source:

- all the .cpp, .hpp files in the main folder except `example.cpp`
- LICENSE

Important: Some files have Godot-made changes.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments.


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
- Version: 1.3.6 (2018)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: `lookups.pl`, `Makefile.*`
- `include/vorbis/*.h` as vorbis/
- COPYING


## libvpx

- Upstream: https://chromium.googlesource.com/webm/libvpx/
- Version: 1.6.0 (2016)
- License: BSD-3-Clause

Files extracted from upstream source:

TODO.

Important: File `libvpx/vpx_dsp/x86/vpx_subpixel_8t_intrin_avx2.c` has
Godot-made change marked with `// -- GODOT --` comments.

The files `libvpx/third_party/android/cpu-features.{c,h}` were copied
from the Android NDK r18.


## libwebp

- Upstream: https://chromium.googlesource.com/webm/libwebp/
- Version: 1.1.0 (d7844e9762b61c9638c263657bd49e1690184832, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: .am, .rc and .in files
- AUTHORS, COPYING, PATENTS

Important: The files `utils/bit_reader_utils.{c,h}` have Godot-made
changes to ensure they build for Javascript/HTML5. Those
changes are marked with `// -- GODOT --` comments.


## mbedtls

- Upstream: https://tls.mbed.org/
- Version: 2.16.10 (d61fa61bef06b64132e3490543c81b8ee40fbee3, 2021)
- License: Apache 2.0

File extracted from upstream release tarball:

- All `*.h` from `include/mbedtls/` to `thirdparty/mbedtls/include/mbedtls/`
- All `*.c` from `library/` to `thirdparty/mbedtls/library/`
- LICENSE and apache-2.0.txt files
- Applied the patch in `thirdparty/mbedtls/patches/1453.diff` (PR 1453).
  Soon to be merged upstream. Check it out at next update.
- Applied the patch in `thirdparty/mbedtls/patches/padlock.diff`. This disables
  VIA padlock support which defines a symbol `unsupported` which clashes with
  a pre-defined symbol.
- Added 2 files `godot_core_mbedtls_platform.{c,h}` providing configuration
  for light bundling with core.


## meshoptimizer

- Upstream: https://github.com/zeux/meshoptimizer
- Version: git (f5d83e879c48f8664783a69b4f50711d27549b66, 2021)
- License: MIT

Files extracted from upstream repository:

- All files in `src/`.
- `LICENSE.md`.

An [experimental upstream feature](https://github.com/zeux/meshoptimizer/tree/simplify-attr),
has been backported, see patch in `patches` directory.


## miniupnpc

- Upstream: https://github.com/miniupnp/miniupnp
- Version: 2.2.2 (81029a860baf1f727903e5b85307903b3f40cbc8, 2021)
- License: BSD-3-Clause

Files extracted from upstream source:

- All `*.c` and `*.h` files from `miniupnpc` to `thirdparty/miniupnpc/miniupnpc`
- Remove the following test or sample files:
  `listdevices.c minihttptestserver.c miniupnpcmodule.c upnpc.c upnperrors.* test* wingenminiupnpcstrings.c`
- `LICENSE`

The only modified file is `miniupnpcstrings.h`, which was created for Godot
(it is usually autogenerated by cmake). Bump the version number for miniupnpc in that
file when upgrading.

Note: The following upstream patch has been applied, remove this notice on next update.
https://github.com/miniupnp/miniupnp/commit/3a08dd4b89af2e9effa22a136bac86f2f306fd79


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
- `easing_equations.cpp`
  * Upstream: http://robertpenner.com/easing/ via https://github.com/jesusgollonet/ofpennereasing (modified to fit Godot types)
  * Version: git (af72c147c3a74e7e872aa28c7e2abfcced04fdce, 2008) + Godot types and style changes
  * License: BSD-3-Clause
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
- `stb_rect_pack.h`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.00 (2bb4a0accd4003c1db4c24533981e01b1adfd656, 2019)
  * License: Public Domain or Unlicense or MIT
- `stb_vorbis.c`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.20 (314d0a6f9af5af27e585336eecea333e95c5a2d8, 2020)
  * License: Public Domain or Unlicense or MIT
- `yuv2rgb.h`
  * Upstream: http://wss.co.uk/pinknoise/yuv2rgb/ (to check)
  * Version: ?
  * License: BSD


## nanosvg

- Upstream: https://github.com/memononen/nanosvg
- Version: git (3e403ec72a9145cbbcc6c63d94a4caf079aafec2, 2020)
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


## opus

- Upstream: https://opus-codec.org
- Version: 1.1.5 (opus) and 0.8 (opusfile) (2017)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c and .h files in src/ (both opus and opusfile)
- all .h files in include/ (both opus and opusfile) as opus/
- remove unused `opus_demo.c`,
- remove `http.c`, `wincerts.c` and `winerrno.h` (part of
  unused libopusurl)
- celt/ and silk/ subfolders
- COPYING


## pcre2

- Upstream: http://www.pcre.org
- Version: 10.36 (r1288, 2020)
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
- Version: git (c0ce03a43ca77fedb5abfd1976ae2fd0eeb0e611, 2021)
- License: Apache 2.0

Files extracted from upstream source:

- `spirv_reflect.{c,h}`
- `include` folder
- `LICENSE`


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
- Version: 1.0.0 (e4b7840d9448b7d57a88384ce26143004f3c0c71, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`


## vhacd

- Upstream: https://github.com/kmammou/v-hacd
- Version: git (b07958e18e01d504e3af80eeaeb9f033226533d7, 2019)
- License: BSD-3-Clause

Files extracted from upstream source:

- From `src/VHACD_Lib/`: `inc`, `public` and `src`
- `LICENSE`

Some downstream changes have been made and are identified by
`// -- GODOT start --` and `// -- GODOT end --` comments.
They can be reapplied using the patches included in the `vhacd`
folder.


## vulkan

- Upstream: https://github.com/KhronosGroup/Vulkan-Loader
- Version: sdk-1.2.162.0 (7a313093b5c4af964d50a5a64e73d7df6152ea3f, 2020)
- License: Apache 2.0

Unless there is a specific reason to package a more recent version, please stick
to Vulkan SDK releases (prefixed by `sdk-`) for all components.

NOTE: Use `scripts/update_deps.py --ref <version>` in the Loader git repository
to retrieve the `Vulkan-Headers` repository matching the loader version.

Files extracted from upstream source:

- `Vulkan-Headers/include/` as `include/`
- All `.c` and `.h` files in `loader/` and `loader/generated/`, put in a common
  `loader/` folder
- `LICENSE.txt`

`vk_enum_string_helper.h` is taken from the matching `Vulkan-ValidationLayers`
SDK release: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/layers/generated/vk_enum_string_helper.h

`vk_mem_alloc.h` is taken from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
Version: 2.3.0 (2019)
`vk_mem_alloc.cpp` and `android/vk_mem_alloc.cpp` are Godot files and should be preserved on updates.

Patches in the `patches` directory should be re-applied after updates.


## wslay

- Upstream: https://github.com/tatsuhiro-t/wslay
- Version: 1.1.1 (c9a84aa6df8512584c77c8cd15be9536b89c35aa, 2020)
- License: MIT

File extracted from upstream release tarball:

- All `*.c` and `*.h` in `lib/` and `lib/includes/`
- `wslay.h` has a small Godot addition to fix MSVC build.
  See `thirdparty/wslay/msvcfix.diff`


## xatlas

- Upstream: https://github.com/jpcy/xatlas
- Version: git (5571fc7ef0d06832947c0a935ccdcf083f7a9264, 2020)
- License: MIT

Files extracted from upstream source:

- `xatlas.{cpp,h}`
- `LICENSE`


## zlib

- Upstream: http://www.zlib.net
- Version: 1.2.11 (2017)
- License: zlib

Files extracted from upstream source:

- all .c and .h files


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.4.8 (97a3da1df009d4dc67251de0c4b1c9d7fe286fc1, 2020)
- License: BSD-3-Clause

Files extracted from upstream source:

- lib/{common/,compress/,decompress/,zstd.h}
- LICENSE
