# Third party libraries


## b2d_convexdecomp

- Upstream: https://github.com/erincatto/Box2D/tree/master/Contributions/Utilities/ConvexDecomposition
- Version: git (25615e0, 2015) with modifications
- License: zlib

The files were adapted to Godot by removing the dependency on b2Math (replacing
it by b2Glue.h) and commenting out some verbose printf calls.
Upstream code has not changed in 10 years, no need to keep track of changes.


## bullet

- Upstream: https://github.com/bulletphysics/bullet3
- Version: git (d05ad4b, 2017)
- License: zlib

Files extracted from upstream source:

- src/* apart from CMakeLists.txt and premake4.lua files
- LICENSE.txt


## certs

- Upstream: Mozilla, via https://apps.fedoraproject.org/packages/ca-certificates
- Version: 2018.2.22
- License: MPL 2.0

File extracted from a recent Fedora install:
/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
(It can't be extracted directly from the package,
as it's generated on the user's system.)


## enet

- Upstream: http://enet.bespin.org
- Version: 1.3.13
- License: MIT

Files extracted from upstream source:

- all .c files in the main directory (except unix.c win32.c)
- the include/enet/ folder as enet/ (except unix.h win32.h)
- LICENSE file

Important: enet.h, host.c, protocol.c have been slightly modified
to be usable by godot socket implementation and allow IPv6.
Two files (godot.cpp and enet/godot.h) have been added to provide
enet socket implementation using Godot classes.
It is still possible to build against a system wide ENet but doing so
will limit it's functionality to IPv4 only.
Check the diff of enet.h, protocol.c, and host.c with the 1.3.13
tarball before the next update.


## etc2comp

- Upstream: https://github.com/google/etc2comp
- Version: git (9cd0f9c, 2017)
- License: Apache 2.0

Files extracted from upstream source:

- all .cpp and .h files in EtcLib/
- README.md, LICENSE, AUTHORS


## fonts

### Noto Sans

- Upstream: https://github.com/googlei18n/noto-fonts
- Version: 1.06
- License: OFL-1.1

Use UI font variant if available, because it has tight vertical metrics and good for UI.

### Hack Regular

- Upstream: https://github.com/source-foundry/Hack
- Version: 3.000
- License: MIT + Bitstream Vera License

### DroidSans*.ttf

- Upstream: https://android.googlesource.com/platform/frameworks/base/+/master/data/fonts/
- Version: ? (pre-2014 commit when DroidSansJapanese.ttf was obsoleted)
- License: Apache 2.0


## freetype

- Upstream: https://www.freetype.org
- Version: 2.8.1
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- the src/ folder, stripped of the `Jamfile` files
- the include/ folder
- `docs/{FTL.TXT,LICENSE.TXT}`


## glad

- Upstream: https://github.com/Dav1dde/glad
- Version: 0.1.16a0
- License: MIT

The files we package are automatically generated.
See the header of glad.c for instructions on how to generate them for
the GLES version Godot targets.


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 1.04
- License: Public domain

Files extracted from upstream source:

- `jpgd.{c,h}`


## libogg

- Upstream: https://www.xiph.org/ogg
- Version: 1.3.3
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*.c`
- `include/ogg/*.h` in ogg/
- COPYING


## libpng

- Upstream: http://libpng.org/pub/png/libpng.html
- Version: 1.6.34
- License: libpng/zlib

Files extracted from upstream source:

- all .c and .h files of the main directory, except from
  `example.c` and `pngtest.c`
- the arm/ folder
- `scripts/pnglibconf.h.prebuilt` as `pnglibconf.h`


## libsimplewebm

- Upstream: https://github.com/zaps166/libsimplewebm
- Version: git (05cfdc2, 2016)
- License: MIT, BSD-3-Clause

Files extracted from upstream source:

TODO.


## libtheora

- Upstream: https://www.theora.org
- Version: 1.1.1
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c, .h in lib/
- all .h files in include/theora/ as theora/
- COPYING and LICENSE


## libvorbis

- Upstream: https://www.xiph.org/vorbis
- Version: 1.3.5
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: `lookups.pl`, `Makefile.*`
- `include/vorbis/*.h` as vorbis/
- COPYING


## libvpx

- Upstream: https://chromium.googlesource.com/webm/libvpx/
- Version: 1.6.0
- License: BSD-3-Clause

Files extracted from upstream source:

TODO.

Important: File `libvpx/vpx_dsp/x86/vpx_subpixel_8t_intrin_avx2.c` has
Godot-made change marked with `// -- GODOT --` comments.


## libwebp

- Upstream: https://chromium.googlesource.com/webm/libwebp/
- Version: 0.6.1
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: .am, .rc and .in files
- AUTHORS, COPYING, PATENTS

Important: The files `utils/bit_reader_utils.{c,h}` have Godot-made
changes to ensure they build for Javascript/HTML5. Those
changes are marked with `// -- GODOT --` comments.


## libwebsockets

- Upstream: https://github.com/warmcat/libwebsockets
- Version: 2.4.1
- License: LGPLv2.1 + static linking exception

File extracted from upstream source:
- Everything in `lib/` except `http2/`, `event-libs/`.
  - From `misc/` exclude `lws-genhash.c`, `lws-ring.c`, `romfs.{c,h}`, `smtp.c`.
  - From `plat/` exclude `lws-plat-{esp*,optee}.c`.
  - From `server/` exclude `access-log.c`, `cgi.c`, `daemonize.c`, `lws-spa.c`, 
`peer-limits.c`, `rewrite.c`
- Also copy `win32helpers/` from `win32port/`
- `mbedtls_wrapper/include/platform/ssl_port.h` has a small change to check for OSX (missing `malloc.h`).
  The bug is fixed in upstream master via `LWS_HAVE_MALLOC_H`, but not in the 2.4.1 branch (as the file structure has changed).

Important: `lws_config.h` and `lws_config_private.h` contains custom 
Godot build configurations, check them out when updating.

## mbedTLS

- Upstream: https://tls.mbed.org/
- Version: 2.7.0
- License: Apache 2.0

File extracted from upstream release tarball `mbedtls-2.7.0-apache.tgz`:
- All `*.h` from `include/mbedtls/` to `thirdparty/include/mbedtls/`
- All `*.c` from `library/` to `thirdparty/library/`

## minizip

- Upstream: http://www.zlib.net
- Version: 1.2.4 (zlib contrib)
- License: zlib

Files extracted from the upstream source:

- contrib/minizip/{crypt.h,ioapi.{c,h},zip.{c,h},unzip.{c,h}}

Important: Some files have Godot-made changes for use in core/io.
They are marked with `/* GODOT start */` and `/* GODOT end */`
comments and a patch is provided in the minizip/ folder.


## misc

Collection of single-file libraries used in Godot components.

### core

- `aes256.{cpp,h}`
  * Upstream: http://www.literatecode.com/aes256
  * Version: latest, as of April 2017
  * License: ISC
- `base64.{c,h}`
  * Upstream: http://episec.com/people/edelkind/c.html
  * Version: latest, as of April 2017
  * License: Public Domain
- `fastlz.{c,h}`
  * Upstream: https://github.com/ariya/FastLZ
  * Version: git (f121734, 2007)
  * License: MIT
- `hq2x.{cpp,h}`
  * Upstream: https://github.com/brunexgeek/hqx
  * Version: TBD, file structure differs
  * License: Apache 2.0
- `md5.{cpp,h}`
  * Upstream: http://www.efgh.com/software/md5.htm
  * Version: TBD, might not be latest from above URL
  * License: RSA Message-Digest License
- `pcg.{cpp,h}`
  * Upstream: http://www.pcg-random.org
  * Version: minimal C implementation, http://www.pcg-random.org/download.html
  * License: Apache 2.0
- `sha256.{c,h}`
  * Upstream: https://github.com/ilvn/SHA256
  * Version: git (35ff823, 2015)
  * License: ISC
- `smaz.{c,h}`
  * Upstream: https://github.com/antirez/smaz
  * Version: git (150e125, 2009)
  * License: BSD-3-Clause
  * Modifications: use `const char*` instead of `char*` for input string
- `triangulator.{cpp,h}`
  * Upstream: https://github.com/ivanfratric/polypartition (`src/polypartition.cpp`)
  * Version: TBD, class was renamed
  * License: MIT

### modules

- `curl_hostcheck.{c,h}`
  * Upstream: https://curl.haxx.se/
  * Version: ? (2013)
  * License: MIT
- `yuv2rgb.h`
  * Upstream: http://wss.co.uk/pinknoise/yuv2rgb/ (to check)
  * Version: ?
  * License: BSD

### scene

- `mikktspace.{c,h}`
  * Upstream: https://wiki.blender.org/index.php/Dev:Shading/Tangent_Space_Normal_Maps
  * Version: 1.0
  * License: zlib
- `stb_truetype.h`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.17
  * License: Public Domain (Unlicense) or MIT
- `stb_vorbis.c`
  * Upstream: https://github.com/nothings/stb
  * Version: 1.11
  * License: Public Domain (Unlicense) or MIT


## nanosvg

- Upstream: https://github.com/memononen/nanosvg
- Version: 9a74da4 (git)
- License: zlib

Files extracted from the upstream source:

- All .h files in `src/`
- LICENSE.txt

## opus

- Upstream: https://opus-codec.org
- Version: 1.1.5 (opus) and 0.8 (opusfile)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c and .h files in src/ (both opus and opusfile),
  except `opus_demo.c`
- all .h files in include/ (both opus and opusfile) as opus/
- celt/ and silk/ subfolders
- COPYING


## pcre2

- Upstream: http://www.pcre.org/
- Version: 10.23
- License: BSD-3-Clause

Files extracted from upstream source:

- Files listed in NON-AUTOTOOLS-BUILD steps 1-4
- All .h files in src/
- src/pcre2_jit_*.c and src/sljit/*
- AUTHORS and COPYING


## pvrtccompressor

- Upstream: https://bitbucket.org/jthlim/pvrtccompressor
- Version: hg (cf71777, 2015)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .cpp and .h files apart from `main.cpp`
- LICENSE.TXT


## recastnavigation

- Upstream: https://github.com/recastnavigation/recastnavigation
- version: git (ef3ea40f, 2017)
- License: zlib

Files extracted from upstream source:

- `Recast/` folder
- License.txt


## rtaudio

- Upstream: http://www.music.mcgill.ca/~gary/rtaudio/
- Version: 4.1.2
- License: MIT-like

Files extracted from upstream source:

- `RtAudio.{cpp,h}`


## squish

- Upstream: https://sourceforge.net/projects/libsquish
- Version: 1.15
- License: MIT

Files extracted from upstream source:

- all .cpp, .h and .inl files

Important: Some files have Godot-made changes.
They are marked with `// -- GODOT start --` and `// -- GODOT end --`
comments and a patch is provided in the squish/ folder.


## thekla_atlas

- Upstream: https://github.com/Thekla/thekla_atlas
- Version: git (80a1430, 2017)
- License: MIT

Files extracted from the upstream source:

- Relevant sources from src/
- License.txt

Important: Some files have Godot-made changes, those
changes are marked with `// -- GODOT --` comments.


## tinyexr

- Upstream: https://github.com/syoyo/tinyexr
- Version: git (e385dad, 2018)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`


## zlib

- Upstream: http://www.zlib.net
- Version: 1.2.11
- License: zlib

Files extracted from upstream source:

- all .c and .h files


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.3.3
- License: BSD-3-Clause

Files extracted from upstream source:

- lib/{common/,compress/,decompress/,zstd.h}
- LICENSE
