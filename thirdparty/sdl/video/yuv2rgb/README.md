From: https://github.com/descampsa/yuv2rgb
# yuv2rgb
C library for fast image conversion between yuv420p and rgb24.

This is a simple library for optimized image conversion between YUV420p and rgb24.
It was done mainly as an exercise to learn to use sse intrinsics, so there may still be room for optimization.

For each conversion, a standard c optimized function and two sse function (with aligned and unaligned memory) are implemented.
The sse version requires only SSE2, which is available on any reasonably recent CPU.
The library also supports the three different YUV (YCrCb to be correct) color spaces that exist (see comments in code), and others can be added simply.

There is a simple test program, that convert a raw YUV file to rgb ppm format, and measure computation time.
Optionally, it also compares the result and computation time with the ffmpeg implementation (that uses MMX), and with the IPP functions.

To compile, simply do :

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

The test program only support raw YUV files for the YUV420 format, and ppm for the RGB24 format.
To generate a raw yuv file, you can use avconv:

    avconv -i example.jpg -c:v rawvideo -pix_fmt yuv420p example.yuv

To generate the rgb file, you can use the ImageMagick convert program:

    convert example.jpg example.ppm

Then, for YUV420 to RGB24 conversion, use the test program like that:

    ./test_yuv_rgb yuv2rgb image.yuv 4096 2160 image
  
The second and third parameters are image width and height (that are needed because not available in the raw YUV file), and fourth parameter is the output filename template (several output files will be generated, named for example output_sse.ppm, output_av.ppm, etc.)

Similarly, for RGB24 to YUV420 conversion:

    ./test_yuv_rgb yuv2rgb image.ppm image

On my computer, the test program on a 4K image give the following for yuv2rgb:

    Time will be measured in each configuration for 100 iterations...
    Processing time (std) : 2.630193 sec
    Processing time (sse2_unaligned) : 0.704394 sec
    Processing time (ffmpeg_unaligned) : 1.221432 sec
    Processing time (ipp_unaligned) : 0.636274 sec
    Processing time (sse2_aligned) : 0.606648 sec
    Processing time (ffmpeg_aligned) : 1.227100 sec
    Processing time (ipp_aligned) : 0.636951 sec

And for rgb2yuv:

    Time will be measured in each configuration for 100 iterations...
    Processing time (std) : 2.588675 sec
    Processing time (sse2_unaligned) : 0.676625 sec
    Processing time (ffmpeg_unaligned) : 3.385816 sec
    Processing time (ipp_unaligned) : 0.593890 sec
    Processing time (sse2_aligned) : 0.640630 sec
    Processing time (ffmpeg_aligned) : 3.397952 sec
    Processing time (ipp_aligned) : 0.579043 sec

configuration : gcc 4.9.2, swscale 3.0.0, IPP 9.0.1, intel i7-5500U
