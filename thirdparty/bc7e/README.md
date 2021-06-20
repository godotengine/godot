bc7enc - Fast BC1-7 GPU texture encoders with Rate Distortion Optimization (RDO)

This repo contains fast texture encoders for BC1-7. All formats support a simple post-processing transform on the encoded texture data designed to trade off quality for smaller compressed file sizes using LZ compression. Significant (10-50%) size reductions are possible. The BC7 encoder also supports a "reduced entropy" mode using the -e option which causes the output to be biased/weighted in various ways which minimally impact quality, which results in 5-10% smaller file sizes with no slowdowns in encoding time.

Currently, the entropy reduction transform is tuned for Deflate, LZHAM, or LZMA. The method used to control the rate-distortion tradeoff is the classic Lagrangian multiplier RDO method, modified to favor MSE on very smooth blocks. Rate is approximated using a fixed Deflate model. The post-processing transform applied to the encoded texture data tries to introduce the longest match it can into every encoded output block. It also tries to continue matches between blocks and (specifically for codecs like LZHAM/LZMA/Zstd) it tries to utilize REP0 (repeat) matches.

You can see examples of the RDO BC7 encoder's current output [here](https://richg42.blogspot.com/2021/02/more-rdo-bc7-encoding.html). Some examples on how to use the command line tool are on my blog, [here](https://richg42.blogspot.com/2021/02/how-to-use-bc7encrdo.html).

This repo contains both [bc7e.ispc](https://github.com/BinomialLLC/bc7e) and its distantly related but weaker 4 mode only non-ispc variant, bc7enc.cpp. By default, if you set SUPPORT_BC7E=TRUE when running cmake, you get bc7e.ispc, otherwise you get bc7enc.cpp. (The -C option forces bc7enc.cpp.) bc7e supports all BC7 modes and features, but doesn't yet support reduced entropy BC7 encoding. bc7enc.cpp supports optional reduced entropy encoding (using -e with the command line tool). RDO BC7 is supported when using either encoder, however.

The next major focus will be improving the default smooth block handling and improving rate distorton performance.

This repo was originally derived from [bc7enc](https://github.com/richgel999/bc7enc) and [bc7e](https://github.com/BinomialLLC/bc7e). Note this repo contains the latest version of bc7e.ispc, which has a determinism bug fix.

**Note: If you use this software in a product, attribution / credits is requested but not required. Thanks!**

### Compiling

This build has been tested with MSVC 2019 x64 and clang 6.0.0 under Ubuntu v18.04.

To compile with bc7e.ispc (on Linux this requires [Intel's ISPC compiler](https://ispc.github.io/downloads.html) to be in your path - recommended):

```
cmake -D SUPPORT_BC7E=TRUE .
make
```

To compile without BC7E:

```
cmake .
make
```

Note the MSVC and Linux builds enable OpenMP for faster compression.

### Examples

The [.DDS](https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide) output files can be loaded/viewed using tools like [AMD Compressonator](https://gpuopen.com/compressonator/).

To encode to non-RDO BC7 using BC7E, highest quality, linear RGB(A) metrics:

```
./bc7enc blah.png
```

To encode to non-RDO BC7 using BC7E, highest quality, using perceptual (scaled YCbCr) colorspace error metrics:

```
./bc7enc blah.png -s
```

To encode to RDO BC7 using BC7E, highest quality, lambda=.5, linear metrics (perceptual colorspace metrics are always automatically disabled when -z is specified), with a balance of encoding performance vs. RDO efficiency:

```
./bc7enc blah.png -z.5
```

To encode to RDO BC7 using BC7E, lower baseline quality (-u4) for faster encoding, lambda=.5, and with faster encoding (only inject one match vs two, with a tiny RDO lookback window size of 16 bytes):

```
./bc7enc blah.png -u4 -z.5 -ze -zc16
```

To encode to non-RDO BC7 using entropy reduced or quantized/weighted BC7 (no slowdown vs. non-RDO bc7enc.cpp for BC7, slightly reduced quality, but 5-10% better LZ compression, only uses 2 or 4 BC7 modes):

```
./bc7enc blah.png -C -e
```

To encode to RDO BC7 using the entropy reduction transform combined with reduced entropy BC7 encoding, with a slightly larger window size than the default which is 128 bytes:

```
./bc7enc -zc256 blah.png -C -e -z1.0
```

Same as before, but higher compression (allow 2 matches per block instead of 1):

```
./bc7enc -zc256 blah.png -C -e -z1.0 -zn
```

Same, except disable ultra-smooth block handling:

```
./bc7enc -zc256 blah.png -C -e -z1.0 -zu
```

To encode to RDO BC7 using the entropy reduction transform at lower quality, combined with reduced entropy BC7 encoding, with a slightly larger window size than the default which is 128 bytes:

```
./bc7enc -zc256 blah.png -C -e -z2.0
```

To encode to RDO BC7 using the entropy reduction transform at higher effectivenes using a larger window size, without using reduced entropy BC7 encoding:

```
./bc7enc -zc1024 blah.png -z1.0
```

To encode to RDO BC7 using the entropy reduction transform at higher effectivenes using a larger window size, with a manually specified max smooth block max error scale:

```
./bc7enc -zc1024 blah.png -z2.0 -zb30.0
```

To encode to RDO BC7 using the entropy reduction transform at higher effectivenes using a larger window size, using only mode 6 (more block artifacts, but better rate-distortion performance as measured by PSNR):

```
./bc7enc -zc1024 blah.png -6 -z1.0 -e
```

To encode to BC1:
```
./bc7enc -1 blah.png
```

To encode to BC1 with Rate Distortion Optimization (RDO) at lambda=1.0:
```
./bc7enc -1 -z1.0 blah.png
```

The -z option controls lambda, or the rate vs. distortion tradeoff. 0 = maximum quality, higher values=lower bitrates but lower quality. Try values [.25-8].

To encode to BC1 with RDO, with RDO debug output, to monitor the percentage of blocks impacted:
```
./bc7enc -1 -z1.0 -zd blah.png
```

To encode to BC1 with RDO with a higher then default smooth block scale factor:
```
./bc7enc -1 -z1.0 -zb40.0 blah.png
```

Use -zb1.0 to disable smooth block error scaling completely, which increases RDO performance but can result in noticeable artifacts on smooth/flat blocks at higher lambdas.

Use -zc# to control the RDO window size in bytes. Good values to try are 16-8192. 
Use -zt to disable RDO multithreading. 

To encode to BC1 with RDO at the highest achievable quality/effectiveness (this is extremely slow):

```
./bc7enc -1 -z1.0 -zc32768 blah.png
```

This sets the window size to 32KB (the highest setting that makes sense for Deflate). Window sizes of 2KB (the default) to 8KB are way faster and in practice are almost as effective. The maximum window size setting supported by the command line tool is 64KB, but this would be very slow.

For even higher quality per bit (this is incredibly slow):
```
./bc7enc -1 -z1.0 -zc32768 -zm blah.png
```

### Dependencies
There are no 3rd party code or library dependencies. utils.cpp/.h is only needed by the example command line tool. It uses C++11. The individual .cpp files are designed to be easily dropped into other codebases.

For RDO post-processing of any block-based format: ert.cpp/.h. You provide this function an array of encoded blocks, an array of source/original 32bpp blocks, some parameters, and a pointer to a block decoder function for your format as a callback. It must return false if the passed in block data is invalid. (Make sure you *really* validate the block's data, because the ERT post-processor will inevitably call your callback with invalid blocks.) This transform works on most other texture formats, such as ETC1/2, EAC, and ASTC. The ERT works on block sizes ranging from 1x1 to 12x12. This file has no other dependencies apart from utils.cpp/h.

For BC1-5 encoding/decoding: rgbcx.cpp/.h

For BC7 encoding: bc7enc.cpp/.h

For BC7 decoding: bc7decomp.cpp/.h

