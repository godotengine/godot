# Third party libraries


## b2d_convexdecomp

- Upstream: https://github.com/erincatto/Box2D (Contributions/Utilities/ConvexDecomposition)
- Version: TBD
- License: zlib


## certs

- Upstream: ?

TODO.


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
- Version: 9cd0f9c (git)
- License: Apache

Files extracted from upstream source:

- all .cpp and .h files in EtcLib/
- README.md, LICENSE, AUTHORS


## fonts

### Noto Sans

- Upstream: https://github.com/googlei18n/noto-fonts
- Version: 1.06
- License: OFL-1.1

Use UI font if exists, because it has tight vertical metrics and good for UI.

### Hack Regular

- Upstream: https://github.com/source-foundry/Hack
- Version: 3.000
- License: MIT + Bitstream Vera License

### DroidSans*.ttf

- Upstream: ?


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
- Version: 1.3.2
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
- Version: 05cfdc2 (git)
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

- Upstream: http://www.webmproject.org/code/
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
  * Upstream: https://code.google.com/archive/p/fastlz
  * Version: svn (r12)
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
  * License: BSD 3-clause
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

### poshlib

- Upstream: http://poshlib.hookatooka.com/poshlib/trac.cgi (username guest, password guest123)
- Version: 1.3.002
- License: MIT

Files extracted from the upstream source:

- posh.c
- posh.h

(no license file was included with the upstream distribution)

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

## thekla_atlas

- Upstream: https://github.com/Thekla/thekla_atlas
- Version: 80a1430 (git)
- License: zlib

Files extracted from the upstream source:

- Relevant sources from src/
- License.txt

Important: Some files have Godot-made changes, those
changes are marked with `// -- GODOT --` comments.


## nanosvg

- Upstream: https://github.com/memononen/nanosvg
- Version: 9a74da4 (git)
- License: zlib

Files extracted from the upstream source:

- All .h files in `src/`
- LICENSE.txt


## openssl

- Upstream: https://www.openssl.org
- Version: 1.0.2l
- License: OpenSSL license / BSD-like

Files extracted from the upstream source:

- Our `openssl/`: contains the headers installed in /usr/include/openssl;
  gather them in the source tarball with `make links` and
  `cp -f include/openssl/*.h ../openssl/openssl/`
- Our `crypto/`: copy of upstream `crypto/`, with some cleanup (see below).
- Our `ssl/`: copy of upstream `ssl/`, with some cleanup (see below).
- Cleanup:
  ```
  find \( -name "Makefile" -o -name "*.S" -o -name "*.bat" -o -name "*.bc" \
    -o -name "*.com" -o -name "*.cnf" -o -name "*.ec" -o -name "*.fre" \
    -o -name "*.gcc" -o -name "*.in" -o -name "*.lnx" -o -name "*.m4" \
    -o -name "*.pl" -o -name "*.pod" -o -name "*.s" -o -name "*.sh" \
    -o -name "*.sol" -o -name "*test*" \) -delete
  cd openssl; for file in *.h; do find ../{crypto,ssl} -name "$file" -delete; done
  ```
  For the rest check the `git status` and decide.
- e_os.h
- Apply the Godot-specific patches in the `patches/` folder.


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
- Version: hg commit cf71777 - 2015-01-08
- License: BSD-3-Clause

Files extracted from upstream source:

- all .cpp and .h files apart from `main.cpp`
- LICENSE.TXT


## recastnavigation

- Upstream: https://github.com/recastnavigation/recastnavigation
- version: git commit ef3ea40f - 2016-02-06
- License: zlib


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


## tinyexr

- Upstream: https://github.com/syoyo/tinyexr
- Version: 0.9.5+ (git 9f784ca - 24 October 2017)
- License: BSD-3-Clause

Files extracted from upstream source:

- `tinyexr.{cc,h}`


## zlib

- Upstream: http://www.zlib.net/
- Version: 1.2.11
- License: zlib

Files extracted from upstream source:

- all .c and .h files


## zstd

- Upstream: https://github.com/facebook/zstd
- Version: 1.3.2
- License: BSD-3-Clause

Files extracted from upstream source:

- lib/{common/,compress/,decompress/,zstd.h}
- README.md, LICENSE
