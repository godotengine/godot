# fpng
fpng is a very fast C++ .PNG image reader/writer for 24/32bpp images. It's a [single source file](src/fpng.h) with no dependencies on any other library. fpng.cpp was written to see just how fast you can write .PNG's without sacrificing too much compression. The files written by fpng conform to the [PNG standard](https://www.w3.org/TR/PNG/), are readable using any PNG decoder, and load or validate successfully using libpng, wuffs, lodepng, stb_image, and [pngcheck](http://www.libpng.org/pub/png/apps/pngcheck.html). PNG files written using fpng can also be read using fpng significantly faster than other PNG libraries, due to its explicit use of [Length-Limited Prefix Codes](https://create.stephan-brumme.com/length-limited-prefix-codes/) and an [optimized decoder](https://fastcompression.blogspot.com/2015/10/huffman-revisited-part-4-multi-bytes.html) that exploits the properties of these codes.

fpng.cpp compression compared to stb_image_write.h: 12-19x faster with roughly 5-11% avg. smaller files. 

fpng.cpp decompression compared to stb_image.h: 2.5-3x faster (on fpng compressed PNG's)

fpng.cpp compared to libpng: ~23x faster compression, 2.5-3x faster decompression (on fpng compressed PNG's)

fpng.cpp compared to Wuffs decompression: roughly 10% faster decompression (on fpng compressed PNG's - note Wuffs decompression is in general *extremely* fast)

Here's an example image encoded by fpng (a downsampled version of "bridge" from [here](http://imagecompression.info/test_images/)):
![fpng encoded "bridge" image](https://github.com/richgel999/fpng/blob/main/example.png)

A real-world benchmark using an assortment of 303 24/32bpp test images used for GPU texture compression benchmarks (mps="megapixels/second", sorted by compression rate):

```
                 comp_size  avg_comp_mps  avg_decomp_mps
fpng_1_pass:     293.10 MB  110.16 mps    162.01 mps
qoi:             300.84 MB  83.90 mps     138.18 mps
fpng_2_pass:     275.73 MB  68.32 mps     165.73 mps
lodepng:         220.40 MB  6.21 mps      27.66 mps
stb_image:       311.41 MB  5.76 mps      50.00 mps
```

A real-world benchmark using the 184 QOI test images (note 182 of the qoi test images don't have alpha channels, so this is almost entirely a 24bpp test):

```
                 comp_size  avg_comp_mps  avg_decomp_mps
fpng_1_pass:     392.45 MB  115.17 mps    161.92 mps
qoi:             359.55 MB  88.22 mps     156.24 mps
fpng_2_pass:     374.76 MB  71.29 mps     164.12 mps
stb_image:       425.64 MB  5.71 mps      52.18 mps
lodepng:         300.14 MB  5.20 mps      29.63 mps
```

An artificial benchmark using the 184 QOI test images, but with the green channel swizzled into alpha and all images compressed as 32bpp (to easily create a correlated alpha channel, common in video game textures):

```
                 comp_size  avg_comp_mps  avg_decomp_mps
qoi:             697.20 MB  154.43 mps    160.30 mps
fpng_1_pass:     540.61 MB  93.10 mps     128.43 mps
fpng_2_pass:     487.99 MB  59.12 mps     136.46 mps
stb_image:       486.44 MB  4.63 mps      46.25 mps
lodepng:         352.10 MB  4.25 mps      28.84 mps
```

A well-behaved lossless compressor should output files roughly up to 1/3rd larger in this test. QOI's compressed output files are 1.94x larger vs. the 24bpp variants (697.20MB vs. 359.55MB), which is significantly more expansion than I would expect.

Benchmarks were made using the included fpng_test tool to generate .CSV files, MSVC 2019, on a Xeon E5-2690 3.00 GHz. The above benchmarks were made before SSE adler32/crc32 functions were added to the encoder. With 24bpp images and MSVC2022 the encoder is now around 15% faster.

## Building

To build, compile from the included .SLN with Visual Studio 2019/2022 or use cmake to generate a .SLN file. For Linux/OSX, use

```
  cmake -DSSE=1 .
  make
```

Remove "-DSSE=1" on non-x86/x64 systems. The test executable will be in the "bin" or "bin_osx" subdirectory.

Tested with MSVC 2022/2019/gcc 7.5.0/clang 6.0. I have only tested fpng.cpp on little endian systems. The code is there for big endian, and it should work, but it needs testing.

## Testing

From the "bin" directory, run "fpng_test.exe" or "./fpng_test" like this:

```fpng_test.exe <image_filename.png>```

For two pass compression (slower compression, usually faster decompression, smaller average file size):

```fpng_test.exe -s <image_filename.png>```

To generate .CSV output only:

```fpng_test.exe -c <image_filename.png>```

There will be several output files written to the current directory: stbi.png, lodepng.png, qoi.qoi, and fpng.png. Statistics or .CSV data will be printed to stdout, and errors to stderr.

The test app decompresses fpng's output using lodepng, stb_image, and the fpng decoder to validate the compressed data. The compressed output has also been validated using [pngcheck](http://www.libpng.org/pub/png/apps/pngcheck.html).

## Using fpng 

To use fpng.cpp in other programs, copy fpng.cpp/.h into your project. Alternatively, `#include "fpng.cpp"` and `#include "fpng.h"` in one place, and then `#include "fpng.h"` everywhere else. 

There are a few optional compile-time defines you can use to configure fpng, particularly `FPNG_NO_SSE`. With gcc/clang on x86/x64, to get SSE you must compile with "-msse4.1 -mpclmul". Also, the code has only been tested with `-fno-strict-aliasing` (same as the Linux kernel, and MSVC's default). See the top of fpng.cpp for a list of the optional defines.

### Initialization

**Call `fpng::fpng_init()` once before using fpng** so it can detect if the CPU supports SSE 4.1+pclmul (for fast CRC-32 and Adler32). Otherwise, it'll always use the slower scalar fallbacks.

### Encoding

Call one of these C-style functions in the "fpng" namespace:

```
namespace fpng {
  bool fpng_encode_image_to_memory(const void* pImage, uint32_t w, uint32_t h, uint32_t num_chans, std::vector<uint8_t>& out_buf, uint32_t flags = 0);
  bool fpng_encode_image_to_file(const char* pFilename, const void* pImage, uint32_t w, uint32_t h, uint32_t num_chans, uint32_t flags = 0);
}
```

`num_chans` must be 3 or 4. There must be ```w*3*h``` or ```w*4*h``` bytes pointed to by ```pImage```. The image row pitch is always ```w*3``` or ```w*4``` bytes. There is no automatic determination if the image actually uses an alpha channel, so if you call it with 4 you will always get a 32bpp .PNG file.

### Decoding

**The included fast decoder will only decode PNG files created by fpng.** However, it has a full PNG chunk parser, and when it detects PNG files not written by fpng it returns the error code `FPNG_DECODE_NOT_FPNG` so you can fall back to a general purpose PNG reader. Also, the decompressor validates the compressed data during decompression and will immediately stop and return `FPNG_DECODE_NOT_FPNG` whenever any of the fpng constraints (implied by the fdEC marker's presence) are violated. You can use ```fpng_get_info()``` to quickly detect if a PNG file can be decoded using fpng.

```
namespace fpng {
  int fpng_get_info(const void* pImage, uint32_t image_size, uint32_t& width, uint32_t& height, uint32_t& channels_in_file);
  int fpng_decode_memory(const void* pImage, uint32_t image_size, std::vector<uint8_t>& out, uint32_t& width, uint32_t& height, uint32_t& channels_in_file, uint32_t desired_channels);
  int fpng_decode_file(const char* pFilename, std::vector<uint8_t>& out, uint32_t& width, uint32_t& height, uint32_t& channels_in_file, uint32_t desired_channels);
}
```

`pImage` and `image_size` point to the PNG file data.

`width`, `height`, `channels_in_file` will be set to the image's dimensions and number of channels, which will always be 3 or 4.

`desired_channels` must be 3 or 4. If the input PNG file is 32bpp and you request 24bpp, the alpha channel will be discarded. If the input is 24bpp and you request 32bpp, the alpha channel will be set to 0xFF.

The return code will be `fpng::FPNG_DECODE_SUCCESS` on success, `fpng::FPNG_DECODE_NOT_FPNG` if the PNG file should be decoded with a general purpose decoder, or one of the other error values.

### Utility Functions

For convenience some of the lib's internal functionality is exposed through these API's:

```
namespace fpng {
  bool fpng_cpu_supports_sse41();
  uint32_t fpng_crc32(const void* pData, size_t size, uint32_t prev_crc32 = FPNG_CRC32_INIT);
  uint32_t fpng_adler32(const void* pData, size_t size, uint32_t adler = FPNG_ADLER32_INIT);
}
```

## Notes

This version of FPNG always uses PNG filter #2 and is limited to only RLE matches (i.e. LZ matches with a match distance of either 3 or 4). It's around 5% weaker than the original release, which used LZRW1 parsing. (I'll eventually add back in the original parser as an option, but doing that will add more code/complexity to the project.)

Importantly, the fpng decoder can explictly/purposely only decode PNG files written by fpng, otherwise it returns fpng::FPNG_DECODE_NOT_FPNG (so you can fall back to a general purpose PNG decoder).

fpng's compressor places a special private ancillary chunk in its output files, which other PNG decompressors will ignore. The decompressor uses this chunk to determine if the file was written by fpng (enabling fast decompression). This chunk's definition is [here](https://github.com/richgel999/fpng/wiki/fdEC-PNG-chunk-definition).

In single pass mode (the default), fpng uses a set of precomputed Deflate dynamic Huffman tables. Here's [how to use the fpng_test tool to compute custom tables](https://github.com/richgel999/fpng/wiki/How-to-train-new-Huffman-tables-for-custom-content). 

Earlier versions of fpng (before 1.0.5) wrote valid PNG's that wuffs wouldn't accept. As far as I can tell this is a [bug in wuffs](https://github.com/google/wuffs/issues/66). I've added a workaround to fpng's encoder and re-trained its single pass Huffman tables, and I've also added the wuffs decoder to the png_test app.

lodepng v20210627 fetched 12/18/2021

stb_image_write.h v1.16 fetched 12/18/2021

qoi.h fetched 12/18/2021

## Low-level description

fpng's compressor uses a custom pixel-wise Deflate compressor which was optimized for simplicity over high ratios. The "parser" only supports RLE matches using a match distance of 3/4 bytes, all literals (except the PNG filter bytes) are output in groups of 3 or 4, all matches are multiples of 3/4 bytes, and it only utilizes a single dynamic Huffman block within a single PNG IDAT chunk. It utilizes 64-bit registers and exploits unaligned little endian reads/writes. (On big endian CPU's it'll use 32/64bpp byteswaps.)  

There are two compressor variants in this release: a faster single pass compressor that utilizes a set of precomputed Huffman tables, or a slightly better two pass compressor that results in smaller files (enabled by passing FPNG_ENCODE_SLOWER flag to the compressor). fpng will fall back to using uncompressed Deflate blocks if the image fails to compress.

The fast decompressor included in fpng.cpp can explictly only handle PNG files created by fpng. To detect these files, it looks for a PNG private ancillary chunk named "fdEC", which other readers will ignore because it's not marked as a "critical" PNG chunk. If this chunk isn't found, or the file doesn't conform to fpng's single IDAT and zlib constraints, the decompressor returns FPNG_DECODE_NOT_FPNG. The decompressor itself has numerous checks to ensure the PNG file was written by fpng (i.e. even if the fdEC chunk is present we don't blindly assume the Deflate data follows the right constraints).

The decompressor's memory usage is low relative to other PNG decompressors, because it doesn't need to make any temporary allocations to hold the decompressed zlib data. (This is one side benefit of always using LZ matches with a distance of only 3 or 4 bytes.) The only large allocation is the one used to hold the output image buffer, which it directly decompresses into. This property is useful on memory-constrained embedded platforms. It's possible for a fpng decompressor to only need to hold 2 scanlines in memory.

Passes over the input image and dynamic allocations are minimized, although it does use ```std::vector``` internally. The first scanline always uses filter #0, and the rest use filter #2 (previous scanline). It uses the fast "slice by 4" CRC-32 algorithm described by Brumme [here](https://create.stephan-brumme.com/crc32/). The original high-level PNG function (that code that writes the headers) was written by [Alex Evans](https://gist.github.com/908299).


## Fuzzing

fpng's encoder and decoder has been fuzzed to check for failures or crashes with random/corrupted input images and random image dimensions. The -e and -E options are used for this sort of fuzzing.

The fpng decoder's parser has been fuzzed to check for crashes with [zzuf](http://caca.zoy.org/wiki/zzuf). For more efficient decoder fuzzing (and more coverage), set `FPNG_DISABLE_DECODE_CRC32_CHECKS` to 1 in fpng.cpp before fuzzing. The -f option is used for fuzzing, like this:

```
zzuf -s 1:1000000 ./fpng_test -f fpng.png
```

## License for fpng.cpp/.h

See the [unlicense](https://unlicense.org/) 

At least in the US, no license is necessary, as this code is not Intellectual Property, and not copyrighted. It has been explictly and purposely placed into the Public Domain.
