# Third party libraries


## enet

- Upstream: http://enet.bespin.org
- Version: 1.3.13
- License: MIT

Files extracted from upstream source:

- all *.c files in the main directory
- the include/enet/ folder as enet/
- LICENSE file

Important: Some files have been modified by Godot developers so that they work
for all platforms (especially WinRT). Check the diff with the 1.3.13 tarball
before the next update.


## jpeg-compressor

- Upstream: https://github.com/richgel999/jpeg-compressor
- Version: 1.04
- License: Public domain

Files extracted from upstream source:

- jpgd.{c,h}


## libpng

- Upstream: http://libpng.org/pub/png/libpng.html
- Version: 1.6.23
- License: libpng/zlib

Files extracted from upstream source:

- all .c and .h files of the main directory, except from:
  * example.c
  * pngtest.c
- the arm/ folder
- scripts/pnglibconf.h.prebuilt as pnglibconf.h
