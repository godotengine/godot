# Third party libraries


## enet

- Upstream: http://enet.bespin.org
- Version: 1.3.13
- License: MIT

Files extracted from upstream source:

- all .c files in the main directory
- the include/enet/ folder as enet/
- LICENSE file

Important: Some files have been modified by Godot developers so that they work
for all platforms (especially WinRT). Check the diff with the 1.3.13 tarball
before the next update.


## freetype

- Upstream: https://www.freetype.org
- Version: 2.6.5
- License: FreeType License (BSD-like)

Files extracted from upstream source:

- the src/ folder, stripped of the `Jamfile` files
- the include/ folder
- `docs/{FTL.TXT,LICENSE.TXT}`


## glew

- Upstream: http://glew.sourceforge.net
- Version: 1.13.0
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/glew.c`
- include/GL/ as GL/
- LICENSE.txt


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 1.04
- License: Public domain

Files extracted from upstream source:

- `jpgd.{c,h}`


## libmpcdec

- Upstream: https://www.musepack.net
- Version: SVN somewhere between SV7 and SV8 (r475)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c and .h files in libmpcdec/
- include/mpc as mpc/
- COPYING from libmpcdec/


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
- Version: 1.6.23
- License: libpng/zlib

Files extracted from upstream source:

- all .c and .h files of the main directory, except from
  `example.c` and `pngtest.c`
- the arm/ folder
- `scripts/pnglibconf.h.prebuilt` as `pnglibconf.h`


## libvorbis

- Upstream: https://www.xiph.org/vorbis
- Version: 1.3.5
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: `lookups.pl`, `Makefile.*`
- `include/vorbis/*.h` as vorbis/
- COPYING


## libwebp

- Upstream: https://chromium.googlesource.com/webm/libwebp/
- Version: 0.5.1
- License: BSD-3-Clause

Files extracted from upstream source:

- `src/*` except from: .am and .in, files, extras/, `webp/extras.h`
- AUTHORS, COPYING, PATENTS

Important: The files `utils/bit_reader.{c,h}` have Godot-made
changes to ensure they build for Javascript/HTML5. Those
changes are marked with `// -- GODOT --` comments.


## openssl

- Upstream: https://www.openssl.org
- Version: 1.2.0h
- License: OpenSSL license / BSD-like

Files extracted from the upstream source:

TODO.


## opus

- Upstream: https://opus-codec.org
- Version: 1.1.2 (opus) and 0.7 (opusfile)
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c and .h files in src/ (both opus and opusfile),
  except `opus_demo.c`
- all .h files in include/ (both opus and opusfile)
- COPYING


## pvrtccompressor

- Upstream: https://bitbucket.org/jthlim/pvrtccompressor
- Version: hg commit cf71777 - 2015-01-08
- License: BSD-3-Clause

Files extracted from upstream source:

- all .cpp and .h files apart from `main.cpp`
- LICENSE.TXT


## rg-etc1

- Upstream: https://github.com/richgel999/rg-etc1
- Version: 1.04
- License: zlib

Files extracted from upstream source:

- `rg_etc1.{cpp,h}`


## rtaudio

- Upstream: http://www.music.mcgill.ca/~gary/rtaudio/
- Version: 4.1.2
- License: MIT-like

Files extracted from upstream source:

- `RtAudio.{cpp,h}`


## squish

- Upstream: https://sourceforge.net/projects/libsquish
- Version: 1.14
- License: MIT

Files extracted from upstream source:

- all .cpp, .h and .inl files


## theora

- Upstream: https://www.theora.org
- Version: 1.1.1
- License: BSD-3-Clause

Files extracted from upstream source:

- all .c, .h in lib/
- all .h files in include/theora/ as theora/
- COPYING and LICENSE


## zlib

- Upstream: http://www.zlib.net/
- Version: 1.2.8
- License: zlib

Files extracted from upstream source:

- all .c and .h files apart from `gz*`
