# Third party libraries

Please keep categories (`##` level) listed alphabetically and matching their
respective folder names. Use two empty lines to separate categories for
readability.


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


## etc2comp

- Upstream: https://github.com/google/etc2comp
- Version: git (9cd0f9cae0f32338943699bb418107db61bb66f2, 2017)
- License: Apache 2.0

Files extracted from upstream source:

- all .cpp and .h files in EtcLib/
- README.md, LICENSE, AUTHORS

Important: Some files have Godot-made changes.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments.


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


## freetype

- Upstream: https://www.freetype.org
- Version: 2.10.4 (6a2b3e4007e794bfc6c91030d0ed987f925164a8, 2020)
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- the `src/` folder, stripped of the `Jamfile` files and the `tools` subfolder
- the `include/` folder
- `docs/{FTL.TXT,LICENSE.TXT}`


## glad

- Upstream: https://github.com/Dav1dde/glad
- Version: 0.1.34 (a5ca31c88a4cc5847ea012629aff3690f261c7c4, 2020)
- License: MIT

The files we package are automatically generated.
See the header of glad.c for instructions on how to generate them for
the GLES version Godot targets.


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
- `smaz.{c,h}`
  * Upstream: https://github.com/antirez/smaz
  * Version: git (2f625846a775501fb69456567409a8b12f10ea25, 2012)
  * License: BSD-3-Clause
  * Modifications: use `const char*` instead of `char*` for input string
- `stb_rect_pack.h`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.00
  * License: Public Domain (Unlicense) or MIT
- `stb_vorbis.c`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.20 (314d0a6f9af5af27e585336eecea333e95c5a2d8, 2020)
  * License: Public Domain or Unlicense or MIT
- `triangulator.{cpp,h}`
  * Upstream: https://github.com/ivanfratric/polypartition (`src/polypartition.cpp`)
  * Version: TBD, class was renamed
  * License: MIT
- `yuv2rgb.h`
  * Upstream: http://wss.co.uk/pinknoise/yuv2rgb/ (to check)
  * Version: ?
  * License: BSD


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

common/* (except tasking.* and CMakeLists.txt)
core/*
include/OpenImageDenoise/* (except version.h.in)
LICENSE.txt
mkl-dnn/include/*
mkl-dnn/src/* (except CMakeLists.txt)
weights/rtlightmap_hdr.tza
scripts/resource_to_cpp.py

Modified files:
Modifications are marked with `// -- GODOT start --` and `// -- GODOT end --`.
Patch files are provided in `oidn/patches/`.

core/autoencoder.cpp
core/autoencoder.h
core/common.h
core/device.cpp
core/device.h
core/transfer_function.cpp

scripts/resource_to_cpp.py (used in modules/denoise/resource_to_cpp.py)


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
