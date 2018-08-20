FastLZ - lightning-fast lossless compression library

Author: Ariya Hidayat
Official website: http://www.fastlz.org

FastLZ is distributed using the MIT license, see file LICENSE
for details.

FastLZ consists of two files: fastlz.h and fastlz.c. Just add these 
files to your project in order to use FastLZ. For information on 
compression and decompression routines, see fastlz.h.

A simple file compressor called 6pack is included as an example 
on how to use FastLZ. The corresponding decompressor is 6unpack. 

To compile using GCC:

  gcc -o 6pack 6pack.c fastlz.c
  gcc -o 6unpack 6unpack.c fastlz.c

To compile using MinGW:

  mingw32-gcc -o 6pack 6pack.c fastlz.c
  mingw32-gcc -o 6unpack 6unpack.c fastlz.c

To compile using Microsoft Visual C++:

  cl 6pack.c fastlz.c
  cl 6unpack.c fastlz.c

To compile using Borland C++:

  bcc32 6pack.c fastlz.c
  bcc32 6unpack.c fastlz.c

To compile using OpenWatcom C/C++:

  cl386 6pack.c fastlz.c
  cl386 6unpack.c fastlz.c

To compile using Intel C++ compiler for Windows:

  icl 6pack.c fastlz.c
  icl 6unpack.c fastlz.c

To compile using Intel C++ compiler for Linux:

  icc -o 6pack 6pack.c fastlz.c
  icc -o 6unpack 6unpack.c fastlz.c

To compile 6pack using LCC-Win32:

  lc 6pack.c fastlz.c
  lc 6unpack.c fastlz.c

To compile 6pack using Pelles C:

  pocc 6pack.c
  pocc 6unpack.c
  pocc fastlz.c
  polink 6pack.obj fastlz.obj
  polink 6unpack.obj fastlz.obj

For speed optimization, always use proper compile flags for optimization options.
Typical compiler flags are given below:

* GCC (pre 4.2): -march=pentium -O3 -fomit-frame-pointer -mtune=pentium
* GCC 4.2 or later: -march=pentium -O3 -fomit-frame-pointer -mtune=generic
* Digital Mars C/C++: -o+all -5
* Intel C++ (Windows): /O3 /Qipo  
* Intel C++ (Linux): -O2 -march=pentium -mtune=pentium
* Borland C++: -O2 -5
* LCC-Win32: -O
* Pelles C: /O2

