/* stb_image_resize2 - v2.14 - public domain image resizing

   by Jeff Roberts (v2) and Jorge L Rodriguez
   http://github.com/nothings/stb

   Can be threaded with the extended API. SSE2, AVX, Neon and WASM SIMD support. Only
   scaling and translation is supported, no rotations or shears.

   COMPILING & LINKING
      In one C/C++ file that #includes this file, do this:
         #define STB_IMAGE_RESIZE_IMPLEMENTATION
      before the #include. That will create the implementation in that file.

   EASY API CALLS:
     Easy API downsamples w/Mitchell filter, upsamples w/cubic interpolation, clamps to edge.

     stbir_resize_uint8_srgb( input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                              output_pixels, output_w, output_h, output_stride_in_bytes,
                              pixel_layout_enum )

     stbir_resize_uint8_linear( input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                                output_pixels, output_w, output_h, output_stride_in_bytes,
                                pixel_layout_enum )

     stbir_resize_float_linear( input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                                output_pixels, output_w, output_h, output_stride_in_bytes,
                                pixel_layout_enum )

     If you pass NULL or zero for the output_pixels, we will allocate the output buffer
     for you and return it from the function (free with free() or STBIR_FREE).
     As a special case, XX_stride_in_bytes of 0 means packed continuously in memory.

   API LEVELS
      There are three levels of API - easy-to-use, medium-complexity and extended-complexity.

      See the "header file" section of the source for API documentation.

   ADDITIONAL DOCUMENTATION

      MEMORY ALLOCATION
         By default, we use malloc and free for memory allocation.  To override the
         memory allocation, before the implementation #include, add a:

            #define STBIR_MALLOC(size,user_data) ...
            #define STBIR_FREE(ptr,user_data)   ...

         Each resize makes exactly one call to malloc/free (unless you use the
         extended API where you can do one allocation for many resizes). Under
         address sanitizer, we do separate allocations to find overread/writes.

      PERFORMANCE
         This library was written with an emphasis on performance. When testing
         stb_image_resize with RGBA, the fastest mode is STBIR_4CHANNEL with
         STBIR_TYPE_UINT8 pixels and CLAMPed edges (which is what many other resize
         libs do by default). Also, make sure SIMD is turned on of course (default
         for 64-bit targets). Avoid WRAP edge mode if you want the fastest speed.

         This library also comes with profiling built-in. If you define STBIR_PROFILE,
         you can use the advanced API and get low-level profiling information by
         calling stbir_resize_extended_profile_info() or stbir_resize_split_profile_info()
         after a resize.

      SIMD
         Most of the routines have optimized SSE2, AVX, NEON and WASM versions.

         On Microsoft compilers, we automatically turn on SIMD for 64-bit x64 and
         ARM; for 32-bit x86 and ARM, you select SIMD mode by defining STBIR_SSE2 or
         STBIR_NEON. For AVX and AVX2, we auto-select it by detecting the /arch:AVX
         or /arch:AVX2 switches. You can also always manually turn SSE2, AVX or AVX2
         support on by defining STBIR_SSE2, STBIR_AVX or STBIR_AVX2.

         On Linux, SSE2 and Neon is on by default for 64-bit x64 or ARM64. For 32-bit,
         we select x86 SIMD mode by whether you have -msse2, -mavx or -mavx2 enabled
         on the command line. For 32-bit ARM, you must pass -mfpu=neon-vfpv4 for both
         clang and GCC, but GCC also requires an additional -mfp16-format=ieee to
         automatically enable NEON.

         On x86 platforms, you can also define STBIR_FP16C to turn on FP16C instructions
         for converting back and forth to half-floats. This is autoselected when we
         are using AVX2. Clang and GCC also require the -mf16c switch. ARM always uses
         the built-in half float hardware NEON instructions.

         You can also tell us to use multiply-add instructions with STBIR_USE_FMA.
         Because x86 doesn't always have fma, we turn it off by default to maintain
         determinism across all platforms. If you don't care about non-FMA determinism
         and are willing to restrict yourself to more recent x86 CPUs (around the AVX
         timeframe), then fma will give you around a 15% speedup.

         You can force off SIMD in all cases by defining STBIR_NO_SIMD. You can turn
         off AVX or AVX2 specifically with STBIR_NO_AVX or STBIR_NO_AVX2. AVX is 10%
         to 40% faster, and AVX2 is generally another 12%.

      ALPHA CHANNEL
         Most of the resizing functions provide the ability to control how the alpha
         channel of an image is processed.

         When alpha represents transparency, it is important that when combining
         colors with filtering, the pixels should not be treated equally; they
         should use a weighted average based on their alpha values. For example,
         if a pixel is 1% opaque bright green and another pixel is 99% opaque
         black and you average them, the average will be 50% opaque, but the
         unweighted average and will be a middling green color, while the weighted
         average will be nearly black. This means the unweighted version introduced
         green energy that didn't exist in the source image.

         (If you want to know why this makes sense, you can work out the math for
         the following: consider what happens if you alpha composite a source image
         over a fixed color and then average the output, vs. if you average the
         source image pixels and then composite that over the same fixed color.
         Only the weighted average produces the same result as the ground truth
         composite-then-average result.)

         Therefore, it is in general best to "alpha weight" the pixels when applying
         filters to them. This essentially means multiplying the colors by the alpha
         values before combining them, and then dividing by the alpha value at the
         end.

         The computer graphics industry introduced a technique called "premultiplied
         alpha" or "associated alpha" in which image colors are stored in image files
         already multiplied by their alpha. This saves some math when compositing,
         and also avoids the need to divide by the alpha at the end (which is quite
         inefficient). However, while premultiplied alpha is common in the movie CGI
         industry, it is not commonplace in other industries like videogames, and most
         consumer file formats are generally expected to contain not-premultiplied
         colors. For example, Photoshop saves PNG files "unpremultiplied", and web
         browsers like Chrome and Firefox expect PNG images to be unpremultiplied.

         Note that there are three possibilities that might describe your image
         and resize expectation:

             1. images are not premultiplied, alpha weighting is desired
             2. images are not premultiplied, alpha weighting is not desired
             3. images are premultiplied

         Both case #2 and case #3 require the exact same math: no alpha weighting
         should be applied or removed. Only case 1 requires extra math operations;
         the other two cases can be handled identically.

         stb_image_resize expects case #1 by default, applying alpha weighting to
         images, expecting the input images to be unpremultiplied. This is what the
         COLOR+ALPHA buffer types tell the resizer to do.

         When you use the pixel layouts STBIR_RGBA, STBIR_BGRA, STBIR_ARGB,
         STBIR_ABGR, STBIR_RX, or STBIR_XR you are telling us that the pixels are
         non-premultiplied. In these cases, the resizer will alpha weight the colors
         (effectively creating the premultiplied image), do the filtering, and then
         convert back to non-premult on exit.

         When you use the pixel layouts STBIR_RGBA_PM, STBIR_RGBA_PM, STBIR_RGBA_PM,
         STBIR_RGBA_PM, STBIR_RX_PM or STBIR_XR_PM, you are telling that the pixels
         ARE premultiplied. In this case, the resizer doesn't have to do the
         premultipling - it can filter directly on the input. This about twice as
         fast as the non-premultiplied case, so it's the right option if your data is
         already setup correctly.

         When you use the pixel layout STBIR_4CHANNEL or STBIR_2CHANNEL, you are
         telling us that there is no channel that represents transparency; it may be
         RGB and some unrelated fourth channel that has been stored in the alpha
         channel, but it is actually not alpha. No special processing will be
         performed.

         The difference between the generic 4 or 2 channel layouts, and the
         specialized _PM versions is with the _PM versions you are telling us that
         the data *is* alpha, just don't premultiply it. That's important when
         using SRGB pixel formats, we need to know where the alpha is, because
         it is converted linearly (rather than with the SRGB converters).

         Because alpha weighting produces the same effect as premultiplying, you
         even have the option with non-premultiplied inputs to let the resizer
         produce a premultiplied output. Because the intially computed alpha-weighted
         output image is effectively premultiplied, this is actually more performant
         than the normal path which un-premultiplies the output image as a final step.

         Finally, when converting both in and out of non-premulitplied space (for
         example, when using STBIR_RGBA), we go to somewhat heroic measures to
         ensure that areas with zero alpha value pixels get something reasonable
         in the RGB values. If you don't care about the RGB values of zero alpha
         pixels, you can call the stbir_set_non_pm_alpha_speed_over_quality()
         function - this runs a premultiplied resize about 25% faster. That said,
         when you really care about speed, using premultiplied pixels for both in
         and out (STBIR_RGBA_PM, etc) much faster than both of these premultiplied
         options.

      PIXEL LAYOUT CONVERSION
         The resizer can convert from some pixel layouts to others. When using the
         stbir_set_pixel_layouts(), you can, for example, specify STBIR_RGBA
         on input, and STBIR_ARGB on output, and it will re-organize the channels
         during the resize. Currently, you can only convert between two pixel
         layouts with the same number of channels.

      DETERMINISM
         We commit to being deterministic (from x64 to ARM to scalar to SIMD, etc).
         This requires compiling with fast-math off (using at least /fp:precise).
         Also, you must turn off fp-contracting (which turns mult+adds into fmas)!
         We attempt to do this with pragmas, but with Clang, you usually want to add
         -ffp-contract=off to the command line as well.

         For 32-bit x86, you must use SSE and SSE2 codegen for determinism. That is,
         if the scalar x87 unit gets used at all, we immediately lose determinism.
         On Microsoft Visual Studio 2008 and earlier, from what we can tell there is
         no way to be deterministic in 32-bit x86 (some x87 always leaks in, even
         with fp:strict). On 32-bit x86 GCC, determinism requires both -msse2 and
         -fpmath=sse.

         Note that we will not be deterministic with float data containing NaNs -
         the NaNs will propagate differently on different SIMD and platforms.

         If you turn on STBIR_USE_FMA, then we will be deterministic with other
         fma targets, but we will differ from non-fma targets (this is unavoidable,
         because a fma isn't simply an add with a mult - it also introduces a
         rounding difference compared to non-fma instruction sequences.

      FLOAT PIXEL FORMAT RANGE
         Any range of values can be used for the non-alpha float data that you pass
         in (0 to 1, -1 to 1, whatever). However, if you are inputting float values
         but *outputting* bytes or shorts, you must use a range of 0 to 1 so that we
         scale back properly. The alpha channel must also be 0 to 1 for any format
         that does premultiplication prior to resizing.

         Note also that with float output, using filters with negative lobes, the
         output filtered values might go slightly out of range. You can define
         STBIR_FLOAT_LOW_CLAMP and/or STBIR_FLOAT_HIGH_CLAMP to specify the range
         to clamp to on output, if that's important.

      MAX/MIN SCALE FACTORS
         The input pixel resolutions are in integers, and we do the internal pointer
         resolution in size_t sized integers. However, the scale ratio from input
         resolution to output resolution is calculated in float form. This means
         the effective possible scale ratio is limited to 24 bits (or 16 million
         to 1). As you get close to the size of the float resolution (again, 16
         million pixels wide or high), you might start seeing float inaccuracy
         issues in general in the pipeline. If you have to do extreme resizes,
         you can usually do this is multiple stages (using float intermediate
         buffers).

      FLIPPED IMAGES
         Stride is just the delta from one scanline to the next. This means you can
         use a negative stride to handle inverted images (point to the final
         scanline and use a negative stride). You can invert the input or output,
         using negative strides.

      DEFAULT FILTERS
         For functions which don't provide explicit control over what filters to
         use, you can change the compile-time defaults with:

            #define STBIR_DEFAULT_FILTER_UPSAMPLE     STBIR_FILTER_something
            #define STBIR_DEFAULT_FILTER_DOWNSAMPLE   STBIR_FILTER_something

         See stbir_filter in the header-file section for the list of filters.

      NEW FILTERS
         A number of 1D filter kernels are supplied. For a list of supported
         filters, see the stbir_filter enum. You can install your own filters by
         using the stbir_set_filter_callbacks function.

      PROGRESS
         For interactive use with slow resize operations, you can use the 
         scanline callbacks in the extended API. It would have to be a *very* large
         image resample to need progress though - we're very fast.

      CEIL and FLOOR
         In scalar mode, the only functions we use from math.h are ceilf and floorf,
         but if you have your own versions, you can define the STBIR_CEILF(v) and
         STBIR_FLOORF(v) macros and we'll use them instead. In SIMD, we just use
         our own versions.

      ASSERT
         Define STBIR_ASSERT(boolval) to override assert() and not use assert.h

     PORTING FROM VERSION 1
        The API has changed. You can continue to use the old version of stb_image_resize.h,
        which is available in the "deprecated/" directory.

        If you're using the old simple-to-use API, porting is straightforward.
        (For more advanced APIs, read the documentation.)

          stbir_resize_uint8():
            - call `stbir_resize_uint8_linear`, cast channel count to `stbir_pixel_layout`

          stbir_resize_float():
            - call `stbir_resize_float_linear`, cast channel count to `stbir_pixel_layout`

          stbir_resize_uint8_srgb():
            - function name is unchanged
            - cast channel count to `stbir_pixel_layout`
            - above is sufficient unless your image has alpha and it's not RGBA/BGRA
              - in that case, follow the below instructions for stbir_resize_uint8_srgb_edgemode

          stbir_resize_uint8_srgb_edgemode()
            - switch to the "medium complexity" API
            - stbir_resize(), very similar API but a few more parameters:
              - pixel_layout: cast channel count to `stbir_pixel_layout`
              - data_type:    STBIR_TYPE_UINT8_SRGB
              - edge:         unchanged (STBIR_EDGE_WRAP, etc.)
              - filter:       STBIR_FILTER_DEFAULT
            - which channel is alpha is specified in stbir_pixel_layout, see enum for details

      FUTURE TODOS
        *  For polyphase integral filters, we just memcpy the coeffs to dupe
           them, but we should indirect and use the same coeff memory.
        *  Add pixel layout conversions for sensible different channel counts
           (maybe, 1->3/4, 3->4, 4->1, 3->1).
         * For SIMD encode and decode scanline routines, do any pre-aligning
           for bad input/output buffer alignments and pitch?
         * For very wide scanlines, we should we do vertical strips to stay within
           L2 cache. Maybe do chunks of 1K pixels at a time. There would be
           some pixel reconversion, but probably dwarfed by things falling out
           of cache. Probably also something possible with alternating between
           scattering and gathering at high resize scales?
         * Should we have a multiple MIPs at the same time function (could keep
           more memory in cache during multiple resizes)?
         * Rewrite the coefficient generator to do many at once.
         * AVX-512 vertical kernels - worried about downclocking here.
         * Convert the reincludes to macros when we know they aren't changing.
         * Experiment with pivoting the horizontal and always using the
           vertical filters (which are faster, but perhaps not enough to overcome
           the pivot cost and the extra memory touches). Need to buffer the whole
           image so have to balance memory use.
         * Most of our code is internally function pointers, should we compile
           all the SIMD stuff always and dynamically dispatch?

   CONTRIBUTORS
      Jeff Roberts: 2.0 implementation, optimizations, SIMD
      Martins Mozeiko: NEON simd, WASM simd, clang and GCC whisperer
      Fabian Giesen: half float and srgb converters
      Sean Barrett: API design, optimizations
      Jorge L Rodriguez: Original 1.0 implementation
      Aras Pranckevicius: bugfixes
      Nathan Reed: warning fixes for 1.0

   REVISIONS
      2.14 (2025-05-09) fixed a bug using downsampling gather horizontal first, and 
                          scatter with vertical first.
      2.13 (2025-02-27) fixed a bug when using input callbacks, turned off simd for 
                          tiny-c, fixed some variables that should have been static,
                          fixes a bug when calculating temp memory with resizes that
                          exceed 2GB of temp memory (very large resizes).
      2.12 (2024-10-18) fix incorrect use of user_data with STBIR_FREE
      2.11 (2024-09-08) fix harmless asan warnings in 2-channel and 3-channel mode
                          with AVX-2, fix some weird scaling edge conditions with
                          point sample mode.
      2.10 (2024-07-27) fix the defines GCC and mingw for loop unroll control,
                          fix MSVC 32-bit arm half float routines.
      2.09 (2024-06-19) fix the defines for 32-bit ARM GCC builds (was selecting
                          hardware half floats).
      2.08 (2024-06-10) fix for RGB->BGR three channel flips and add SIMD (thanks
                          to Ryan Salsbury), fix for sub-rect resizes, use the
                          pragmas to control unrolling when they are available.
      2.07 (2024-05-24) fix for slow final split during threaded conversions of very 
                          wide scanlines when downsampling (caused by extra input 
                          converting), fix for wide scanline resamples with many 
                          splits (int overflow), fix GCC warning.
      2.06 (2024-02-10) fix for identical width/height 3x or more down-scaling 
                          undersampling a single row on rare resize ratios (about 1%).
      2.05 (2024-02-07) fix for 2 pixel to 1 pixel resizes with wrap (thanks Aras),
                        fix for output callback (thanks Julien Koenen).
      2.04 (2023-11-17) fix for rare AVX bug, shadowed symbol (thanks Nikola Smiljanic).
      2.03 (2023-11-01) ASAN and TSAN warnings fixed, minor tweaks.
      2.00 (2023-10-10) mostly new source: new api, optimizations, simd, vertical-first, etc
                          2x-5x faster without simd, 4x-12x faster with simd,
                          in some cases, 20x to 40x faster esp resizing large to very small.
      0.96 (2019-03-04) fixed warnings
      0.95 (2017-07-23) fixed warnings
      0.94 (2017-03-18) fixed warnings
      0.93 (2017-03-03) fixed bug with certain combinations of heights
      0.92 (2017-01-02) fix integer overflow on large (>2GB) images
      0.91 (2016-04-02) fix warnings; fix handling of subpixel regions
      0.90 (2014-09-17) first released version

   LICENSE
     See end of file for license information.
*/

#if !defined(STB_IMAGE_RESIZE_DO_HORIZONTALS) && !defined(STB_IMAGE_RESIZE_DO_VERTICALS) && !defined(STB_IMAGE_RESIZE_DO_CODERS)   // for internal re-includes

#ifndef STBIR_INCLUDE_STB_IMAGE_RESIZE2_H
#define STBIR_INCLUDE_STB_IMAGE_RESIZE2_H

#include <stddef.h>
#ifdef _MSC_VER
typedef unsigned char    stbir_uint8;
typedef unsigned short   stbir_uint16;
typedef unsigned int     stbir_uint32;
typedef unsigned __int64 stbir_uint64;
#else
#include <stdint.h>
typedef uint8_t  stbir_uint8;
typedef uint16_t stbir_uint16;
typedef uint32_t stbir_uint32;
typedef uint64_t stbir_uint64;
#endif

#ifndef STBIRDEF
#ifdef STB_IMAGE_RESIZE_STATIC
#define STBIRDEF static
#else
#ifdef __cplusplus
#define STBIRDEF extern "C"
#else
#define STBIRDEF extern
#endif
#endif
#endif

//////////////////////////////////////////////////////////////////////////////
////   start "header file" ///////////////////////////////////////////////////
//
// Easy-to-use API:
//
//     * stride is the offset between successive rows of image data
//        in memory, in bytes. specify 0 for packed continuously in memory
//     * colorspace is linear or sRGB as specified by function name
//     * Uses the default filters
//     * Uses edge mode clamped
//     * returned result is 1 for success or 0 in case of an error.


// stbir_pixel_layout specifies:
//   number of channels
//   order of channels
//   whether color is premultiplied by alpha
// for back compatibility, you can cast the old channel count to an stbir_pixel_layout
typedef enum
{
  STBIR_1CHANNEL = 1,
  STBIR_2CHANNEL = 2,
  STBIR_RGB      = 3,               // 3-chan, with order specified (for channel flipping)
  STBIR_BGR      = 0,               // 3-chan, with order specified (for channel flipping)
  STBIR_4CHANNEL = 5,

  STBIR_RGBA = 4,                   // alpha formats, where alpha is NOT premultiplied into color channels
  STBIR_BGRA = 6,
  STBIR_ARGB = 7,
  STBIR_ABGR = 8,
  STBIR_RA   = 9,
  STBIR_AR   = 10,

  STBIR_RGBA_PM = 11,               // alpha formats, where alpha is premultiplied into color channels
  STBIR_BGRA_PM = 12,
  STBIR_ARGB_PM = 13,
  STBIR_ABGR_PM = 14,
  STBIR_RA_PM   = 15,
  STBIR_AR_PM   = 16,

  STBIR_RGBA_NO_AW = 11,            // alpha formats, where NO alpha weighting is applied at all!
  STBIR_BGRA_NO_AW = 12,            //   these are just synonyms for the _PM flags (which also do
  STBIR_ARGB_NO_AW = 13,            //   no alpha weighting). These names just make it more clear
  STBIR_ABGR_NO_AW = 14,            //   for some folks).
  STBIR_RA_NO_AW   = 15,
  STBIR_AR_NO_AW   = 16,

} stbir_pixel_layout;

//===============================================================
//  Simple-complexity API
//
//    If output_pixels is NULL (0), then we will allocate the buffer and return it to you.
//--------------------------------

STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                        unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                        stbir_pixel_layout pixel_type );

STBIRDEF unsigned char * stbir_resize_uint8_linear( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                          unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                          stbir_pixel_layout pixel_type );

STBIRDEF float * stbir_resize_float_linear( const float *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                  float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                  stbir_pixel_layout pixel_type );
//===============================================================

//===============================================================
// Medium-complexity API
//
// This extends the easy-to-use API as follows:
//
//     * Can specify the datatype - U8, U8_SRGB, U16, FLOAT, HALF_FLOAT
//     * Edge wrap can selected explicitly
//     * Filter can be selected explicitly
//--------------------------------

typedef enum
{
  STBIR_EDGE_CLAMP   = 0,
  STBIR_EDGE_REFLECT = 1,
  STBIR_EDGE_WRAP    = 2,  // this edge mode is slower and uses more memory
  STBIR_EDGE_ZERO    = 3,
} stbir_edge;

typedef enum
{
  STBIR_FILTER_DEFAULT      = 0,  // use same filter type that easy-to-use API chooses
  STBIR_FILTER_BOX          = 1,  // A trapezoid w/1-pixel wide ramps, same result as box for integer scale ratios
  STBIR_FILTER_TRIANGLE     = 2,  // On upsampling, produces same results as bilinear texture filtering
  STBIR_FILTER_CUBICBSPLINE = 3,  // The cubic b-spline (aka Mitchell-Netrevalli with B=1,C=0), gaussian-esque
  STBIR_FILTER_CATMULLROM   = 4,  // An interpolating cubic spline
  STBIR_FILTER_MITCHELL     = 5,  // Mitchell-Netrevalli filter with B=1/3, C=1/3
  STBIR_FILTER_POINT_SAMPLE = 6,  // Simple point sampling
  STBIR_FILTER_OTHER        = 7,  // User callback specified
} stbir_filter;

typedef enum
{
  STBIR_TYPE_UINT8            = 0,
  STBIR_TYPE_UINT8_SRGB       = 1,
  STBIR_TYPE_UINT8_SRGB_ALPHA = 2,  // alpha channel, when present, should also be SRGB (this is very unusual)
  STBIR_TYPE_UINT16           = 3,
  STBIR_TYPE_FLOAT            = 4,
  STBIR_TYPE_HALF_FLOAT       = 5
} stbir_datatype;

// medium api
STBIRDEF void *  stbir_resize( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                     void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                               stbir_pixel_layout pixel_layout, stbir_datatype data_type,
                               stbir_edge edge, stbir_filter filter );
//===============================================================



//===============================================================
// Extended-complexity API
//
// This API exposes all resize functionality.
//
//     * Separate filter types for each axis
//     * Separate edge modes for each axis
//     * Separate input and output data types
//     * Can specify regions with subpixel correctness
//     * Can specify alpha flags
//     * Can specify a memory callback
//     * Can specify a callback data type for pixel input and output
//     * Can be threaded for a single resize
//     * Can be used to resize many frames without recalculating the sampler info
//
//  Use this API as follows:
//     1) Call the stbir_resize_init function on a local STBIR_RESIZE structure
//     2) Call any of the stbir_set functions
//     3) Optionally call stbir_build_samplers() if you are going to resample multiple times
//        with the same input and output dimensions (like resizing video frames)
//     4) Resample by calling stbir_resize_extended().
//     5) Call stbir_free_samplers() if you called stbir_build_samplers()
//--------------------------------


// Types:

// INPUT CALLBACK: this callback is used for input scanlines
typedef void const * stbir_input_callback( void * optional_output, void const * input_ptr, int num_pixels, int x, int y, void * context );

// OUTPUT CALLBACK: this callback is used for output scanlines
typedef void stbir_output_callback( void const * output_ptr, int num_pixels, int y, void * context );

// callbacks for user installed filters
typedef float stbir__kernel_callback( float x, float scale, void * user_data ); // centered at zero
typedef float stbir__support_callback( float scale, void * user_data );

// internal structure with precomputed scaling
typedef struct stbir__info stbir__info;

typedef struct STBIR_RESIZE  // use the stbir_resize_init and stbir_override functions to set these values for future compatibility
{
  void * user_data;
  void const * input_pixels;
  int input_w, input_h;
  double input_s0, input_t0, input_s1, input_t1;
  stbir_input_callback * input_cb;
  void * output_pixels;
  int output_w, output_h;
  int output_subx, output_suby, output_subw, output_subh;
  stbir_output_callback * output_cb;
  int input_stride_in_bytes;
  int output_stride_in_bytes;
  int splits;
  int fast_alpha;
  int needs_rebuild;
  int called_alloc;
  stbir_pixel_layout input_pixel_layout_public;
  stbir_pixel_layout output_pixel_layout_public;
  stbir_datatype input_data_type;
  stbir_datatype output_data_type;
  stbir_filter horizontal_filter, vertical_filter;
  stbir_edge horizontal_edge, vertical_edge;
  stbir__kernel_callback * horizontal_filter_kernel; stbir__support_callback * horizontal_filter_support;
  stbir__kernel_callback * vertical_filter_kernel; stbir__support_callback * vertical_filter_support;
  stbir__info * samplers;
} STBIR_RESIZE;

// extended complexity api


// First off, you must ALWAYS call stbir_resize_init on your resize structure before any of the other calls!
STBIRDEF void stbir_resize_init( STBIR_RESIZE * resize,
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes, // stride can be zero
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes, // stride can be zero
                                 stbir_pixel_layout pixel_layout, stbir_datatype data_type );

//===============================================================
// You can update these parameters any time after resize_init and there is no cost
//--------------------------------

STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type );
STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb );   // no callbacks by default
STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data );                                               // pass back STBIR_RESIZE* by default
STBIRDEF void stbir_set_buffer_ptrs( STBIR_RESIZE * resize, const void * input_pixels, int input_stride_in_bytes, void * output_pixels, int output_stride_in_bytes );

//===============================================================


//===============================================================
// If you call any of these functions, you will trigger a sampler rebuild!
//--------------------------------

STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout );  // sets new buffer layouts
STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge );       // CLAMP by default

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter ); // STBIR_DEFAULT_FILTER_UPSAMPLE/DOWNSAMPLE by default
STBIRDEF int stbir_set_filter_callbacks( STBIR_RESIZE * resize, stbir__kernel_callback * horizontal_filter, stbir__support_callback * horizontal_support, stbir__kernel_callback * vertical_filter, stbir__support_callback * vertical_support );

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh );        // sets both sub-regions (full regions by default)
STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 );    // sets input sub-region (full region by default)
STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh ); // sets output sub-region (full region by default)

// when inputting AND outputting non-premultiplied alpha pixels, we use a slower but higher quality technique
//   that fills the zero alpha pixel's RGB values with something plausible.  If you don't care about areas of
//   zero alpha, you can call this function to get about a 25% speed improvement for STBIR_RGBA to STBIR_RGBA
//   types of resizes.
STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality );
//===============================================================


//===============================================================
// You can call build_samplers to prebuild all the internal data we need to resample.
//   Then, if you call resize_extended many times with the same resize, you only pay the
//   cost once.
// If you do call build_samplers, you MUST call free_samplers eventually.
//--------------------------------

// This builds the samplers and does one allocation
STBIRDEF int stbir_build_samplers( STBIR_RESIZE * resize );

// You MUST call this, if you call stbir_build_samplers or stbir_build_samplers_with_splits
STBIRDEF void stbir_free_samplers( STBIR_RESIZE * resize );
//===============================================================


// And this is the main function to perform the resize synchronously on one thread.
STBIRDEF int stbir_resize_extended( STBIR_RESIZE * resize );


//===============================================================
// Use these functions for multithreading.
//   1) You call stbir_build_samplers_with_splits first on the main thread
//   2) Then stbir_resize_with_split on each thread
//   3) stbir_free_samplers when done on the main thread
//--------------------------------

// This will build samplers for threading.
//   You can pass in the number of threads you'd like to use (try_splits).
//   It returns the number of splits (threads) that you can call it with.
///  It might be less if the image resize can't be split up that many ways.

STBIRDEF int stbir_build_samplers_with_splits( STBIR_RESIZE * resize, int try_splits );

// This function does a split of the resizing (you call this fuction for each
// split, on multiple threads). A split is a piece of the output resize pixel space.

// Note that you MUST call stbir_build_samplers_with_splits before stbir_resize_extended_split!

// Usually, you will always call stbir_resize_split with split_start as the thread_index
//   and "1" for the split_count.
// But, if you have a weird situation where you MIGHT want 8 threads, but sometimes
//   only 4 threads, you can use 0,2,4,6 for the split_start's and use "2" for the
//   split_count each time to turn in into a 4 thread resize. (This is unusual).

STBIRDEF int stbir_resize_extended_split( STBIR_RESIZE * resize, int split_start, int split_count );
//===============================================================


//===============================================================
// Pixel Callbacks info:
//--------------------------------

//   The input callback is super flexible - it calls you with the input address
//   (based on the stride and base pointer), it gives you an optional_output
//   pointer that you can fill, or you can just return your own pointer into
//   your own data.
//
//   You can also do conversion from non-supported data types if necessary - in
//   this case, you ignore the input_ptr and just use the x and y parameters to
//   calculate your own input_ptr based on the size of each non-supported pixel.
//   (Something like the third example below.)
//
//   You can also install just an input or just an output callback by setting the
//   callback that you don't want to zero.
//
//     First example, progress: (getting a callback that you can monitor the progress):
//        void const * my_callback( void * optional_output, void const * input_ptr, int num_pixels, int x, int y, void * context )
//        {
//           percentage_done = y / input_height;
//           return input_ptr;  // use buffer from call
//        }
//
//     Next example, copying: (copy from some other buffer or stream):
//        void const * my_callback( void * optional_output, void const * input_ptr, int num_pixels, int x, int y, void * context )
//        {
//           CopyOrStreamData( optional_output, other_data_src, num_pixels * pixel_width_in_bytes );
//           return optional_output;  // return the optional buffer that we filled
//        }
//
//     Third example, input another buffer without copying: (zero-copy from other buffer):
//        void const * my_callback( void * optional_output, void const * input_ptr, int num_pixels, int x, int y, void * context )
//        {
//           void * pixels = ( (char*) other_image_base ) + ( y * other_image_stride ) + ( x * other_pixel_width_in_bytes );
//           return pixels;       // return pointer to your data without copying
//        }
//
//
//   The output callback is considerably simpler - it just calls you so that you can dump
//   out each scanline. You could even directly copy out to disk if you have a simple format
//   like TGA or BMP. You can also convert to other output types here if you want.
//
//   Simple example:
//        void const * my_output( void * output_ptr, int num_pixels, int y, void * context )
//        {
//           percentage_done = y / output_height;
//           fwrite( output_ptr, pixel_width_in_bytes, num_pixels, output_file );
//        }
//===============================================================




//===============================================================
// optional built-in profiling API
//--------------------------------

#ifdef STBIR_PROFILE

typedef struct STBIR_PROFILE_INFO
{
  stbir_uint64 total_clocks;

  // how many clocks spent (of total_clocks) in the various resize routines, along with a string description
  //    there are "resize_count" number of zones
  stbir_uint64 clocks[ 8 ];
  char const ** descriptions;

  // count of clocks and descriptions
  stbir_uint32 count;
} STBIR_PROFILE_INFO;

// use after calling stbir_resize_extended (or stbir_build_samplers or stbir_build_samplers_with_splits)
STBIRDEF void stbir_resize_build_profile_info( STBIR_PROFILE_INFO * out_info, STBIR_RESIZE const * resize );

// use after calling stbir_resize_extended
STBIRDEF void stbir_resize_extended_profile_info( STBIR_PROFILE_INFO * out_info, STBIR_RESIZE const * resize );

// use after calling stbir_resize_extended_split
STBIRDEF void stbir_resize_split_profile_info( STBIR_PROFILE_INFO * out_info, STBIR_RESIZE const * resize, int split_start, int split_num );

//===============================================================

#endif


////   end header file   /////////////////////////////////////////////////////
#endif // STBIR_INCLUDE_STB_IMAGE_RESIZE2_H

#if defined(STB_IMAGE_RESIZE_IMPLEMENTATION) || defined(STB_IMAGE_RESIZE2_IMPLEMENTATION)

#ifndef STBIR_ASSERT
#include <assert.h>
#define STBIR_ASSERT(x) assert(x)
#endif

#ifndef STBIR_MALLOC
#include <stdlib.h>
#define STBIR_MALLOC(size,user_data) ((void)(user_data), malloc(size))
#define STBIR_FREE(ptr,user_data)    ((void)(user_data), free(ptr))
// (we used the comma operator to evaluate user_data, to avoid "unused parameter" warnings)
#endif

#ifdef _MSC_VER

#define stbir__inline __forceinline

#else

#define stbir__inline __inline__

// Clang address sanitizer
#if defined(__has_feature)
  #if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer)
    #ifndef STBIR__SEPARATE_ALLOCATIONS
      #define STBIR__SEPARATE_ALLOCATIONS
    #endif
  #endif
#endif

#endif

// GCC and MSVC
#if defined(__SANITIZE_ADDRESS__)
  #ifndef STBIR__SEPARATE_ALLOCATIONS
    #define STBIR__SEPARATE_ALLOCATIONS
  #endif
#endif

// Always turn off automatic FMA use - use STBIR_USE_FMA if you want.
// Otherwise, this is a determinism disaster.
#ifndef STBIR_DONT_CHANGE_FP_CONTRACT  // override in case you don't want this behavior
#if defined(_MSC_VER) && !defined(__clang__)
#if _MSC_VER > 1200
#pragma fp_contract(off)
#endif
#elif defined(__GNUC__) &&  !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#else
#pragma STDC FP_CONTRACT OFF
#endif
#endif

#ifdef _MSC_VER
#define STBIR__UNUSED(v)  (void)(v)
#else
#define STBIR__UNUSED(v)  (void)sizeof(v)
#endif

#define STBIR__ARRAY_SIZE(a) (sizeof((a))/sizeof((a)[0]))


#ifndef STBIR_DEFAULT_FILTER_UPSAMPLE
#define STBIR_DEFAULT_FILTER_UPSAMPLE    STBIR_FILTER_CATMULLROM
#endif

#ifndef STBIR_DEFAULT_FILTER_DOWNSAMPLE
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE  STBIR_FILTER_MITCHELL
#endif


#ifndef STBIR__HEADER_FILENAME
#define STBIR__HEADER_FILENAME "stb_image_resize2.h"
#endif

// the internal pixel layout enums are in a different order, so we can easily do range comparisons of types
//   the public pixel layout is ordered in a way that if you cast num_channels (1-4) to the enum, you get something sensible
typedef enum
{
  STBIRI_1CHANNEL = 0,
  STBIRI_2CHANNEL = 1,
  STBIRI_RGB      = 2,
  STBIRI_BGR      = 3,
  STBIRI_4CHANNEL = 4,

  STBIRI_RGBA = 5,
  STBIRI_BGRA = 6,
  STBIRI_ARGB = 7,
  STBIRI_ABGR = 8,
  STBIRI_RA   = 9,
  STBIRI_AR   = 10,

  STBIRI_RGBA_PM = 11,
  STBIRI_BGRA_PM = 12,
  STBIRI_ARGB_PM = 13,
  STBIRI_ABGR_PM = 14,
  STBIRI_RA_PM   = 15,
  STBIRI_AR_PM   = 16,
} stbir_internal_pixel_layout;

// define the public pixel layouts to not compile inside the implementation (to avoid accidental use)
#define STBIR_BGR bad_dont_use_in_implementation
#define STBIR_1CHANNEL STBIR_BGR
#define STBIR_2CHANNEL STBIR_BGR
#define STBIR_RGB STBIR_BGR
#define STBIR_RGBA STBIR_BGR
#define STBIR_4CHANNEL STBIR_BGR
#define STBIR_BGRA STBIR_BGR
#define STBIR_ARGB STBIR_BGR
#define STBIR_ABGR STBIR_BGR
#define STBIR_RA STBIR_BGR
#define STBIR_AR STBIR_BGR
#define STBIR_RGBA_PM STBIR_BGR
#define STBIR_BGRA_PM STBIR_BGR
#define STBIR_ARGB_PM STBIR_BGR
#define STBIR_ABGR_PM STBIR_BGR
#define STBIR_RA_PM STBIR_BGR
#define STBIR_AR_PM STBIR_BGR

// must match stbir_datatype
static unsigned char stbir__type_size[] = {
  1,1,1,2,4,2 // STBIR_TYPE_UINT8,STBIR_TYPE_UINT8_SRGB,STBIR_TYPE_UINT8_SRGB_ALPHA,STBIR_TYPE_UINT16,STBIR_TYPE_FLOAT,STBIR_TYPE_HALF_FLOAT
};

// When gathering, the contributors are which source pixels contribute.
// When scattering, the contributors are which destination pixels are contributed to.
typedef struct
{
  int n0; // First contributing pixel
  int n1; // Last contributing pixel
} stbir__contributors;

typedef struct
{
  int lowest;    // First sample index for whole filter
  int highest;   // Last sample index for whole filter
  int widest;    // widest single set of samples for an output
} stbir__filter_extent_info;

typedef struct
{
  int n0; // First pixel of decode buffer to write to
  int n1; // Last pixel of decode that will be written to
  int pixel_offset_for_input;  // Pixel offset into input_scanline
} stbir__span;

typedef struct stbir__scale_info
{
  int input_full_size;
  int output_sub_size;
  float scale;
  float inv_scale;
  float pixel_shift; // starting shift in output pixel space (in pixels)
  int scale_is_rational;
  stbir_uint32 scale_numerator, scale_denominator;
} stbir__scale_info;

typedef struct
{
  stbir__contributors * contributors;
  float* coefficients;
  stbir__contributors * gather_prescatter_contributors;
  float * gather_prescatter_coefficients;
  stbir__scale_info scale_info;
  float support;
  stbir_filter filter_enum;
  stbir__kernel_callback * filter_kernel;
  stbir__support_callback * filter_support;
  stbir_edge edge;
  int coefficient_width;
  int filter_pixel_width;
  int filter_pixel_margin;
  int num_contributors;
  int contributors_size;
  int coefficients_size;
  stbir__filter_extent_info extent_info;
  int is_gather;  // 0 = scatter, 1 = gather with scale >= 1, 2 = gather with scale < 1
  int gather_prescatter_num_contributors;
  int gather_prescatter_coefficient_width;
  int gather_prescatter_contributors_size;
  int gather_prescatter_coefficients_size;
} stbir__sampler;

typedef struct
{
  stbir__contributors conservative;
  int edge_sizes[2];    // this can be less than filter_pixel_margin, if the filter and scaling falls off
  stbir__span spans[2]; // can be two spans, if doing input subrect with clamp mode WRAP
} stbir__extents;

typedef struct
{
#ifdef STBIR_PROFILE
  union
  {
    struct { stbir_uint64 total, looping, vertical, horizontal, decode, encode, alpha, unalpha; } named;
    stbir_uint64 array[8];
  } profile;
  stbir_uint64 * current_zone_excluded_ptr;
#endif
  float* decode_buffer;

  int ring_buffer_first_scanline;
  int ring_buffer_last_scanline;
  int ring_buffer_begin_index;    // first_scanline is at this index in the ring buffer
  int start_output_y, end_output_y;
  int start_input_y, end_input_y;  // used in scatter only

  #ifdef STBIR__SEPARATE_ALLOCATIONS
    float** ring_buffers; // one pointer for each ring buffer
  #else
    float* ring_buffer;  // one big buffer that we index into
  #endif

  float* vertical_buffer;

  char no_cache_straddle[64];
} stbir__per_split_info;

typedef float * stbir__decode_pixels_func( float * decode, int width_times_channels, void const * input );
typedef void stbir__alpha_weight_func( float * decode_buffer, int width_times_channels );
typedef void stbir__horizontal_gather_channels_func( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer,
  stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width );
typedef void stbir__alpha_unweight_func(float * encode_buffer, int width_times_channels );
typedef void stbir__encode_pixels_func( void * output, int width_times_channels, float const * encode );

struct stbir__info
{
#ifdef STBIR_PROFILE
  union
  {
    struct { stbir_uint64 total, build, alloc, horizontal, vertical, cleanup, pivot; } named;
    stbir_uint64 array[7];
  } profile;
  stbir_uint64 * current_zone_excluded_ptr;
#endif
  stbir__sampler horizontal;
  stbir__sampler vertical;

  void const * input_data;
  void * output_data;

  int input_stride_bytes;
  int output_stride_bytes;
  int ring_buffer_length_bytes;   // The length of an individual entry in the ring buffer. The total number of ring buffers is stbir__get_filter_pixel_width(filter)
  int ring_buffer_num_entries;    // Total number of entries in the ring buffer.

  stbir_datatype input_type;
  stbir_datatype output_type;

  stbir_input_callback * in_pixels_cb;
  void * user_data;
  stbir_output_callback * out_pixels_cb;

  stbir__extents scanline_extents;

  void * alloced_mem;
  stbir__per_split_info * split_info;  // by default 1, but there will be N of these allocated based on the thread init you did

  stbir__decode_pixels_func * decode_pixels;
  stbir__alpha_weight_func * alpha_weight;
  stbir__horizontal_gather_channels_func * horizontal_gather_channels;
  stbir__alpha_unweight_func * alpha_unweight;
  stbir__encode_pixels_func * encode_pixels;

  int alloc_ring_buffer_num_entries;    // Number of entries in the ring buffer that will be allocated
  int splits; // count of splits

  stbir_internal_pixel_layout input_pixel_layout_internal;
  stbir_internal_pixel_layout output_pixel_layout_internal;

  int input_color_and_type;
  int offset_x, offset_y; // offset within output_data
  int vertical_first;
  int channels;
  int effective_channels; // same as channels, except on RGBA/ARGB (7), or XA/AX (3)
  size_t alloced_total;
};


#define stbir__max_uint8_as_float             255.0f
#define stbir__max_uint16_as_float            65535.0f
#define stbir__max_uint8_as_float_inverted    3.9215689e-03f     // (1.0f/255.0f)
#define stbir__max_uint16_as_float_inverted   1.5259022e-05f     // (1.0f/65535.0f)
#define stbir__small_float ((float)1 / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20))

// min/max friendly
#define STBIR_CLAMP(x, xmin, xmax) for(;;) { \
  if ( (x) < (xmin) ) (x) = (xmin);     \
  if ( (x) > (xmax) ) (x) = (xmax);     \
  break;                                \
}

static stbir__inline int stbir__min(int a, int b)
{
  return a < b ? a : b;
}

static stbir__inline int stbir__max(int a, int b)
{
  return a > b ? a : b;
}

static float stbir__srgb_uchar_to_linear_float[256] = {
  0.000000f, 0.000304f, 0.000607f, 0.000911f, 0.001214f, 0.001518f, 0.001821f, 0.002125f, 0.002428f, 0.002732f, 0.003035f,
  0.003347f, 0.003677f, 0.004025f, 0.004391f, 0.004777f, 0.005182f, 0.005605f, 0.006049f, 0.006512f, 0.006995f, 0.007499f,
  0.008023f, 0.008568f, 0.009134f, 0.009721f, 0.010330f, 0.010960f, 0.011612f, 0.012286f, 0.012983f, 0.013702f, 0.014444f,
  0.015209f, 0.015996f, 0.016807f, 0.017642f, 0.018500f, 0.019382f, 0.020289f, 0.021219f, 0.022174f, 0.023153f, 0.024158f,
  0.025187f, 0.026241f, 0.027321f, 0.028426f, 0.029557f, 0.030713f, 0.031896f, 0.033105f, 0.034340f, 0.035601f, 0.036889f,
  0.038204f, 0.039546f, 0.040915f, 0.042311f, 0.043735f, 0.045186f, 0.046665f, 0.048172f, 0.049707f, 0.051269f, 0.052861f,
  0.054480f, 0.056128f, 0.057805f, 0.059511f, 0.061246f, 0.063010f, 0.064803f, 0.066626f, 0.068478f, 0.070360f, 0.072272f,
  0.074214f, 0.076185f, 0.078187f, 0.080220f, 0.082283f, 0.084376f, 0.086500f, 0.088656f, 0.090842f, 0.093059f, 0.095307f,
  0.097587f, 0.099899f, 0.102242f, 0.104616f, 0.107023f, 0.109462f, 0.111932f, 0.114435f, 0.116971f, 0.119538f, 0.122139f,
  0.124772f, 0.127438f, 0.130136f, 0.132868f, 0.135633f, 0.138432f, 0.141263f, 0.144128f, 0.147027f, 0.149960f, 0.152926f,
  0.155926f, 0.158961f, 0.162029f, 0.165132f, 0.168269f, 0.171441f, 0.174647f, 0.177888f, 0.181164f, 0.184475f, 0.187821f,
  0.191202f, 0.194618f, 0.198069f, 0.201556f, 0.205079f, 0.208637f, 0.212231f, 0.215861f, 0.219526f, 0.223228f, 0.226966f,
  0.230740f, 0.234551f, 0.238398f, 0.242281f, 0.246201f, 0.250158f, 0.254152f, 0.258183f, 0.262251f, 0.266356f, 0.270498f,
  0.274677f, 0.278894f, 0.283149f, 0.287441f, 0.291771f, 0.296138f, 0.300544f, 0.304987f, 0.309469f, 0.313989f, 0.318547f,
  0.323143f, 0.327778f, 0.332452f, 0.337164f, 0.341914f, 0.346704f, 0.351533f, 0.356400f, 0.361307f, 0.366253f, 0.371238f,
  0.376262f, 0.381326f, 0.386430f, 0.391573f, 0.396755f, 0.401978f, 0.407240f, 0.412543f, 0.417885f, 0.423268f, 0.428691f,
  0.434154f, 0.439657f, 0.445201f, 0.450786f, 0.456411f, 0.462077f, 0.467784f, 0.473532f, 0.479320f, 0.485150f, 0.491021f,
  0.496933f, 0.502887f, 0.508881f, 0.514918f, 0.520996f, 0.527115f, 0.533276f, 0.539480f, 0.545725f, 0.552011f, 0.558340f,
  0.564712f, 0.571125f, 0.577581f, 0.584078f, 0.590619f, 0.597202f, 0.603827f, 0.610496f, 0.617207f, 0.623960f, 0.630757f,
  0.637597f, 0.644480f, 0.651406f, 0.658375f, 0.665387f, 0.672443f, 0.679543f, 0.686685f, 0.693872f, 0.701102f, 0.708376f,
  0.715694f, 0.723055f, 0.730461f, 0.737911f, 0.745404f, 0.752942f, 0.760525f, 0.768151f, 0.775822f, 0.783538f, 0.791298f,
  0.799103f, 0.806952f, 0.814847f, 0.822786f, 0.830770f, 0.838799f, 0.846873f, 0.854993f, 0.863157f, 0.871367f, 0.879622f,
  0.887923f, 0.896269f, 0.904661f, 0.913099f, 0.921582f, 0.930111f, 0.938686f, 0.947307f, 0.955974f, 0.964686f, 0.973445f,
  0.982251f, 0.991102f, 1.0f
};

typedef union
{
  unsigned int u;
  float f;
} stbir__FP32;

// From https://gist.github.com/rygorous/2203834

static const stbir_uint32 fp32_to_srgb8_tab4[104] = {
  0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
  0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
  0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
  0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
  0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
  0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
  0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
  0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
  0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
  0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
  0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
  0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
  0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
};

static stbir__inline stbir_uint8 stbir__linear_to_srgb_uchar(float in)
{
  static const stbir__FP32 almostone = { 0x3f7fffff }; // 1-eps
  static const stbir__FP32 minval = { (127-13) << 23 };
  stbir_uint32 tab,bias,scale,t;
  stbir__FP32 f;

  // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
  // The tests are carefully written so that NaNs map to 0, same as in the reference
  // implementation.
  if (!(in > minval.f)) // written this way to catch NaNs
      return 0;
  if (in > almostone.f)
      return 255;

  // Do the table lookup and unpack bias, scale
  f.f = in;
  tab = fp32_to_srgb8_tab4[(f.u - minval.u) >> 20];
  bias = (tab >> 16) << 9;
  scale = tab & 0xffff;

  // Grab next-highest mantissa bits and perform linear interpolation
  t = (f.u >> 12) & 0xff;
  return (unsigned char) ((bias + scale*t) >> 16);
}

#ifndef STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT
#define STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT 32 // when downsampling and <= 32 scanlines of buffering, use gather. gather used down to 1/8th scaling for 25% win.
#endif

#ifndef STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS
#define STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS 4 // when threading, what is the minimum number of scanlines for a split?
#endif

#define STBIR_INPUT_CALLBACK_PADDING 3

#ifdef _M_IX86_FP
#if ( _M_IX86_FP >= 1 )
#ifndef STBIR_SSE
#define STBIR_SSE
#endif
#endif
#endif

#ifdef __TINYC__
  // tiny c has no intrinsics yet - this can become a version check if they add them
  #define STBIR_NO_SIMD
#endif

#if defined(_x86_64) || defined( __x86_64__ ) || defined( _M_X64 ) || defined(__x86_64) || defined(_M_AMD64) || defined(__SSE2__) || defined(STBIR_SSE) || defined(STBIR_SSE2)
  #ifndef STBIR_SSE2
    #define STBIR_SSE2
  #endif
  #if defined(__AVX__) || defined(STBIR_AVX2)
    #ifndef STBIR_AVX
      #ifndef STBIR_NO_AVX
        #define STBIR_AVX
      #endif
    #endif
  #endif
  #if defined(__AVX2__) || defined(STBIR_AVX2)
    #ifndef STBIR_NO_AVX2
      #ifndef STBIR_AVX2
        #define STBIR_AVX2
      #endif
      #if defined( _MSC_VER ) && !defined(__clang__)
        #ifndef STBIR_FP16C  // FP16C instructions are on all AVX2 cpus, so we can autoselect it here on microsoft - clang needs -m16c
          #define STBIR_FP16C
        #endif
      #endif
    #endif
  #endif
  #ifdef __F16C__
    #ifndef STBIR_FP16C  // turn on FP16C instructions if the define is set (for clang and gcc)
      #define STBIR_FP16C
    #endif
  #endif
#endif

#if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) || ((__ARM_NEON_FP & 4) != 0) || defined(__ARM_NEON__)
#ifndef STBIR_NEON
#define STBIR_NEON
#endif
#endif

#if defined(_M_ARM) || defined(__arm__)
#ifdef STBIR_USE_FMA
#undef STBIR_USE_FMA // no FMA for 32-bit arm on MSVC
#endif
#endif

#if defined(__wasm__) && defined(__wasm_simd128__)
#ifndef STBIR_WASM
#define STBIR_WASM
#endif
#endif

// restrict pointers for the output pointers, other loop and unroll control
#if defined( _MSC_VER ) && !defined(__clang__)
  #define STBIR_STREAMOUT_PTR( star ) star __restrict
  #define STBIR_NO_UNROLL( ptr ) __assume(ptr) // this oddly keeps msvc from unrolling a loop
  #if _MSC_VER >= 1900
    #define STBIR_NO_UNROLL_LOOP_START __pragma(loop( no_vector )) 
  #else
    #define STBIR_NO_UNROLL_LOOP_START 
  #endif
#elif defined( __clang__ )
  #define STBIR_STREAMOUT_PTR( star ) star __restrict__
  #define STBIR_NO_UNROLL( ptr ) __asm__ (""::"r"(ptr)) 
  #if ( __clang_major__ >= 4 ) || ( ( __clang_major__ >= 3 ) && ( __clang_minor__ >= 5 ) )
    #define STBIR_NO_UNROLL_LOOP_START _Pragma("clang loop unroll(disable)") _Pragma("clang loop vectorize(disable)")
  #else
    #define STBIR_NO_UNROLL_LOOP_START
  #endif 
#elif defined( __GNUC__ )
  #define STBIR_STREAMOUT_PTR( star ) star __restrict__
  #define STBIR_NO_UNROLL( ptr ) __asm__ (""::"r"(ptr))
  #if __GNUC__ >= 14
    #define STBIR_NO_UNROLL_LOOP_START _Pragma("GCC unroll 0") _Pragma("GCC novector")
  #else
    #define STBIR_NO_UNROLL_LOOP_START
  #endif
  #define STBIR_NO_UNROLL_LOOP_START_INF_FOR
#else
  #define STBIR_STREAMOUT_PTR( star ) star
  #define STBIR_NO_UNROLL( ptr )
  #define STBIR_NO_UNROLL_LOOP_START
#endif

#ifndef STBIR_NO_UNROLL_LOOP_START_INF_FOR
#define STBIR_NO_UNROLL_LOOP_START_INF_FOR STBIR_NO_UNROLL_LOOP_START
#endif

#ifdef STBIR_NO_SIMD // force simd off for whatever reason

// force simd off overrides everything else, so clear it all

#ifdef STBIR_SSE2
#undef STBIR_SSE2
#endif

#ifdef STBIR_AVX
#undef STBIR_AVX
#endif

#ifdef STBIR_NEON
#undef STBIR_NEON
#endif

#ifdef STBIR_AVX2
#undef STBIR_AVX2
#endif

#ifdef STBIR_FP16C
#undef STBIR_FP16C
#endif

#ifdef STBIR_WASM
#undef STBIR_WASM
#endif

#ifdef STBIR_SIMD
#undef STBIR_SIMD
#endif

#else // STBIR_SIMD

#ifdef STBIR_SSE2
  #include <emmintrin.h>

  #define stbir__simdf __m128
  #define stbir__simdi __m128i

  #define stbir_simdi_castf( reg ) _mm_castps_si128(reg)
  #define stbir_simdf_casti( reg ) _mm_castsi128_ps(reg)

  #define stbir__simdf_load( reg, ptr ) (reg) = _mm_loadu_ps( (float const*)(ptr) )
  #define stbir__simdi_load( reg, ptr ) (reg) = _mm_loadu_si128 ( (stbir__simdi const*)(ptr) )
  #define stbir__simdf_load1( out, ptr ) (out) = _mm_load_ss( (float const*)(ptr) )  // top values can be random (not denormal or nan for perf)
  #define stbir__simdi_load1( out, ptr ) (out) = _mm_castps_si128( _mm_load_ss( (float const*)(ptr) ))
  #define stbir__simdf_load1z( out, ptr ) (out) = _mm_load_ss( (float const*)(ptr) )  // top values must be zero
  #define stbir__simdf_frep4( fvar ) _mm_set_ps1( fvar )
  #define stbir__simdf_load1frep4( out, fvar ) (out) = _mm_set_ps1( fvar )
  #define stbir__simdf_load2( out, ptr ) (out) = _mm_castsi128_ps( _mm_loadl_epi64( (__m128i*)(ptr)) ) // top values can be random (not denormal or nan for perf)
  #define stbir__simdf_load2z( out, ptr ) (out) = _mm_castsi128_ps( _mm_loadl_epi64( (__m128i*)(ptr)) ) // top values must be zero
  #define stbir__simdf_load2hmerge( out, reg, ptr ) (out) = _mm_castpd_ps(_mm_loadh_pd( _mm_castps_pd(reg), (double*)(ptr) ))

  #define stbir__simdf_zeroP() _mm_setzero_ps()
  #define stbir__simdf_zero( reg ) (reg) = _mm_setzero_ps()

  #define stbir__simdf_store( ptr, reg )  _mm_storeu_ps( (float*)(ptr), reg )
  #define stbir__simdf_store1( ptr, reg ) _mm_store_ss( (float*)(ptr), reg )
  #define stbir__simdf_store2( ptr, reg ) _mm_storel_epi64( (__m128i*)(ptr), _mm_castps_si128(reg) )
  #define stbir__simdf_store2h( ptr, reg ) _mm_storeh_pd( (double*)(ptr), _mm_castps_pd(reg) )

  #define stbir__simdi_store( ptr, reg )  _mm_storeu_si128( (__m128i*)(ptr), reg )
  #define stbir__simdi_store1( ptr, reg ) _mm_store_ss( (float*)(ptr), _mm_castsi128_ps(reg) )
  #define stbir__simdi_store2( ptr, reg ) _mm_storel_epi64( (__m128i*)(ptr), (reg) )

  #define stbir__prefetch( ptr ) _mm_prefetch((char*)(ptr), _MM_HINT_T0 )

  #define stbir__simdi_expand_u8_to_u32(out0,out1,out2,out3,ireg) \
  { \
    stbir__simdi zero = _mm_setzero_si128(); \
    out2 = _mm_unpacklo_epi8( ireg, zero ); \
    out3 = _mm_unpackhi_epi8( ireg, zero ); \
    out0 = _mm_unpacklo_epi16( out2, zero ); \
    out1 = _mm_unpackhi_epi16( out2, zero ); \
    out2 = _mm_unpacklo_epi16( out3, zero ); \
    out3 = _mm_unpackhi_epi16( out3, zero ); \
  }

#define stbir__simdi_expand_u8_to_1u32(out,ireg) \
  { \
    stbir__simdi zero = _mm_setzero_si128(); \
    out = _mm_unpacklo_epi8( ireg, zero ); \
    out = _mm_unpacklo_epi16( out, zero ); \
  }

  #define stbir__simdi_expand_u16_to_u32(out0,out1,ireg) \
  { \
    stbir__simdi zero = _mm_setzero_si128(); \
    out0 = _mm_unpacklo_epi16( ireg, zero ); \
    out1 = _mm_unpackhi_epi16( ireg, zero ); \
  }

  #define stbir__simdf_convert_float_to_i32( i, f ) (i) = _mm_cvttps_epi32(f)
  #define stbir__simdf_convert_float_to_int( f ) _mm_cvtt_ss2si(f)
  #define stbir__simdf_convert_float_to_uint8( f ) ((unsigned char)_mm_cvtsi128_si32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(f,STBIR__CONSTF(STBIR_max_uint8_as_float)),_mm_setzero_ps()))))
  #define stbir__simdf_convert_float_to_short( f ) ((unsigned short)_mm_cvtsi128_si32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(f,STBIR__CONSTF(STBIR_max_uint16_as_float)),_mm_setzero_ps()))))

  #define stbir__simdi_to_int( i ) _mm_cvtsi128_si32(i)
  #define stbir__simdi_convert_i32_to_float(out, ireg) (out) = _mm_cvtepi32_ps( ireg )
  #define stbir__simdf_add( out, reg0, reg1 ) (out) = _mm_add_ps( reg0, reg1 )
  #define stbir__simdf_mult( out, reg0, reg1 ) (out) = _mm_mul_ps( reg0, reg1 )
  #define stbir__simdf_mult_mem( out, reg, ptr ) (out) = _mm_mul_ps( reg, _mm_loadu_ps( (float const*)(ptr) ) )
  #define stbir__simdf_mult1_mem( out, reg, ptr ) (out) = _mm_mul_ss( reg, _mm_load_ss( (float const*)(ptr) ) )
  #define stbir__simdf_add_mem( out, reg, ptr ) (out) = _mm_add_ps( reg, _mm_loadu_ps( (float const*)(ptr) ) )
  #define stbir__simdf_add1_mem( out, reg, ptr ) (out) = _mm_add_ss( reg, _mm_load_ss( (float const*)(ptr) ) )

  #ifdef STBIR_USE_FMA           // not on by default to maintain bit identical simd to non-simd
  #include <immintrin.h>
  #define stbir__simdf_madd( out, add, mul1, mul2 ) (out) = _mm_fmadd_ps( mul1, mul2, add )
  #define stbir__simdf_madd1( out, add, mul1, mul2 ) (out) = _mm_fmadd_ss( mul1, mul2, add )
  #define stbir__simdf_madd_mem( out, add, mul, ptr ) (out) = _mm_fmadd_ps( mul, _mm_loadu_ps( (float const*)(ptr) ), add )
  #define stbir__simdf_madd1_mem( out, add, mul, ptr ) (out) = _mm_fmadd_ss( mul, _mm_load_ss( (float const*)(ptr) ), add )
  #else
  #define stbir__simdf_madd( out, add, mul1, mul2 ) (out) = _mm_add_ps( add, _mm_mul_ps( mul1, mul2 ) )
  #define stbir__simdf_madd1( out, add, mul1, mul2 ) (out) = _mm_add_ss( add, _mm_mul_ss( mul1, mul2 ) )
  #define stbir__simdf_madd_mem( out, add, mul, ptr ) (out) = _mm_add_ps( add, _mm_mul_ps( mul, _mm_loadu_ps( (float const*)(ptr) ) ) )
  #define stbir__simdf_madd1_mem( out, add, mul, ptr ) (out) = _mm_add_ss( add, _mm_mul_ss( mul, _mm_load_ss( (float const*)(ptr) ) ) )
  #endif

  #define stbir__simdf_add1( out, reg0, reg1 ) (out) = _mm_add_ss( reg0, reg1 )
  #define stbir__simdf_mult1( out, reg0, reg1 ) (out) = _mm_mul_ss( reg0, reg1 )

  #define stbir__simdf_and( out, reg0, reg1 ) (out) = _mm_and_ps( reg0, reg1 )
  #define stbir__simdf_or( out, reg0, reg1 ) (out) = _mm_or_ps( reg0, reg1 )

  #define stbir__simdf_min( out, reg0, reg1 ) (out) = _mm_min_ps( reg0, reg1 )
  #define stbir__simdf_max( out, reg0, reg1 ) (out) = _mm_max_ps( reg0, reg1 )
  #define stbir__simdf_min1( out, reg0, reg1 ) (out) = _mm_min_ss( reg0, reg1 )
  #define stbir__simdf_max1( out, reg0, reg1 ) (out) = _mm_max_ss( reg0, reg1 )

  #define stbir__simdf_0123ABCDto3ABx( out, reg0, reg1 ) (out)=_mm_castsi128_ps( _mm_shuffle_epi32( _mm_castps_si128( _mm_shuffle_ps( reg1,reg0, (0<<0) + (1<<2) + (2<<4) + (3<<6) )), (3<<0) + (0<<2) + (1<<4) + (2<<6) ) )
  #define stbir__simdf_0123ABCDto23Ax( out, reg0, reg1 ) (out)=_mm_castsi128_ps( _mm_shuffle_epi32( _mm_castps_si128( _mm_shuffle_ps( reg1,reg0, (0<<0) + (1<<2) + (2<<4) + (3<<6) )), (2<<0) + (3<<2) + (0<<4) + (1<<6) ) )

  static const stbir__simdf STBIR_zeroones = { 0.0f,1.0f,0.0f,1.0f };
  static const stbir__simdf STBIR_onezeros = { 1.0f,0.0f,1.0f,0.0f };
  #define stbir__simdf_aaa1( out, alp, ones ) (out)=_mm_castsi128_ps( _mm_shuffle_epi32( _mm_castps_si128( _mm_movehl_ps( ones, alp ) ), (1<<0) + (1<<2) + (1<<4) + (2<<6) ) )
  #define stbir__simdf_1aaa( out, alp, ones ) (out)=_mm_castsi128_ps( _mm_shuffle_epi32( _mm_castps_si128( _mm_movelh_ps( ones, alp ) ), (0<<0) + (2<<2) + (2<<4) + (2<<6) ) )
  #define stbir__simdf_a1a1( out, alp, ones) (out) = _mm_or_ps( _mm_castsi128_ps( _mm_srli_epi64( _mm_castps_si128(alp), 32 ) ), STBIR_zeroones )
  #define stbir__simdf_1a1a( out, alp, ones) (out) = _mm_or_ps( _mm_castsi128_ps( _mm_slli_epi64( _mm_castps_si128(alp), 32 ) ), STBIR_onezeros )

  #define stbir__simdf_swiz( reg, one, two, three, four ) _mm_castsi128_ps( _mm_shuffle_epi32( _mm_castps_si128( reg ), (one<<0) + (two<<2) + (three<<4) + (four<<6) ) )

  #define stbir__simdi_and( out, reg0, reg1 ) (out) = _mm_and_si128( reg0, reg1 )
  #define stbir__simdi_or( out, reg0, reg1 ) (out) = _mm_or_si128( reg0, reg1 )
  #define stbir__simdi_16madd( out, reg0, reg1 ) (out) = _mm_madd_epi16( reg0, reg1 )

  #define stbir__simdf_pack_to_8bytes(out,aa,bb) \
  { \
    stbir__simdf af,bf; \
    stbir__simdi a,b; \
    af = _mm_min_ps( aa, STBIR_max_uint8_as_float ); \
    bf = _mm_min_ps( bb, STBIR_max_uint8_as_float ); \
    af = _mm_max_ps( af, _mm_setzero_ps() ); \
    bf = _mm_max_ps( bf, _mm_setzero_ps() ); \
    a = _mm_cvttps_epi32( af ); \
    b = _mm_cvttps_epi32( bf ); \
    a = _mm_packs_epi32( a, b ); \
    out = _mm_packus_epi16( a, a ); \
  }

  #define stbir__simdf_load4_transposed( o0, o1, o2, o3, ptr ) \
      stbir__simdf_load( o0, (ptr) );    \
      stbir__simdf_load( o1, (ptr)+4 );  \
      stbir__simdf_load( o2, (ptr)+8 );  \
      stbir__simdf_load( o3, (ptr)+12 ); \
      {                                  \
        __m128 tmp0, tmp1, tmp2, tmp3;   \
        tmp0 = _mm_unpacklo_ps(o0, o1);  \
        tmp2 = _mm_unpacklo_ps(o2, o3);  \
        tmp1 = _mm_unpackhi_ps(o0, o1);  \
        tmp3 = _mm_unpackhi_ps(o2, o3);  \
        o0 = _mm_movelh_ps(tmp0, tmp2);  \
        o1 = _mm_movehl_ps(tmp2, tmp0);  \
        o2 = _mm_movelh_ps(tmp1, tmp3);  \
        o3 = _mm_movehl_ps(tmp3, tmp1);  \
      }

  #define stbir__interleave_pack_and_store_16_u8( ptr, r0, r1, r2, r3 ) \
      r0 = _mm_packs_epi32( r0, r1 ); \
      r2 = _mm_packs_epi32( r2, r3 ); \
      r1 = _mm_unpacklo_epi16( r0, r2 ); \
      r3 = _mm_unpackhi_epi16( r0, r2 ); \
      r0 = _mm_unpacklo_epi16( r1, r3 ); \
      r2 = _mm_unpackhi_epi16( r1, r3 ); \
      r0 = _mm_packus_epi16( r0, r2 ); \
      stbir__simdi_store( ptr, r0 ); \

  #define stbir__simdi_32shr( out, reg, imm ) out = _mm_srli_epi32( reg, imm )

  #if defined(_MSC_VER) && !defined(__clang__)
    // msvc inits with 8 bytes
    #define STBIR__CONST_32_TO_8( v ) (char)(unsigned char)((v)&255),(char)(unsigned char)(((v)>>8)&255),(char)(unsigned char)(((v)>>16)&255),(char)(unsigned char)(((v)>>24)&255)
    #define STBIR__CONST_4_32i( v ) STBIR__CONST_32_TO_8( v ), STBIR__CONST_32_TO_8( v ), STBIR__CONST_32_TO_8( v ), STBIR__CONST_32_TO_8( v )
    #define STBIR__CONST_4d_32i( v0, v1, v2, v3 ) STBIR__CONST_32_TO_8( v0 ), STBIR__CONST_32_TO_8( v1 ), STBIR__CONST_32_TO_8( v2 ), STBIR__CONST_32_TO_8( v3 )
  #else
    // everything else inits with long long's
    #define STBIR__CONST_4_32i( v ) (long long)((((stbir_uint64)(stbir_uint32)(v))<<32)|((stbir_uint64)(stbir_uint32)(v))),(long long)((((stbir_uint64)(stbir_uint32)(v))<<32)|((stbir_uint64)(stbir_uint32)(v)))
    #define STBIR__CONST_4d_32i( v0, v1, v2, v3 ) (long long)((((stbir_uint64)(stbir_uint32)(v1))<<32)|((stbir_uint64)(stbir_uint32)(v0))),(long long)((((stbir_uint64)(stbir_uint32)(v3))<<32)|((stbir_uint64)(stbir_uint32)(v2)))
  #endif

  #define STBIR__SIMDF_CONST(var, x) stbir__simdf var = { x, x, x, x }
  #define STBIR__SIMDI_CONST(var, x) stbir__simdi var = { STBIR__CONST_4_32i(x) }
  #define STBIR__CONSTF(var) (var)
  #define STBIR__CONSTI(var) (var)

  #if defined(STBIR_AVX) || defined(__SSE4_1__)
    #include <smmintrin.h>
    #define stbir__simdf_pack_to_8words(out,reg0,reg1) out = _mm_packus_epi32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(reg0,STBIR__CONSTF(STBIR_max_uint16_as_float)),_mm_setzero_ps())), _mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(reg1,STBIR__CONSTF(STBIR_max_uint16_as_float)),_mm_setzero_ps())))
  #else
    static STBIR__SIMDI_CONST(stbir__s32_32768, 32768);
    static STBIR__SIMDI_CONST(stbir__s16_32768, ((32768<<16)|32768));

    #define stbir__simdf_pack_to_8words(out,reg0,reg1) \
      { \
        stbir__simdi tmp0,tmp1; \
        tmp0 = _mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(reg0,STBIR__CONSTF(STBIR_max_uint16_as_float)),_mm_setzero_ps())); \
        tmp1 = _mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(reg1,STBIR__CONSTF(STBIR_max_uint16_as_float)),_mm_setzero_ps())); \
        tmp0 = _mm_sub_epi32( tmp0, stbir__s32_32768 ); \
        tmp1 = _mm_sub_epi32( tmp1, stbir__s32_32768 ); \
        out = _mm_packs_epi32( tmp0, tmp1 ); \
        out = _mm_sub_epi16( out, stbir__s16_32768 ); \
      }

  #endif

  #define STBIR_SIMD

  // if we detect AVX, set the simd8 defines
  #ifdef STBIR_AVX
    #include <immintrin.h>
    #define STBIR_SIMD8
    #define stbir__simdf8 __m256
    #define stbir__simdi8 __m256i
    #define stbir__simdf8_load( out, ptr ) (out) = _mm256_loadu_ps( (float const *)(ptr) )
    #define stbir__simdi8_load( out, ptr ) (out) = _mm256_loadu_si256( (__m256i const *)(ptr) )
    #define stbir__simdf8_mult( out, a, b ) (out) = _mm256_mul_ps( (a), (b) )
    #define stbir__simdf8_store( ptr, out ) _mm256_storeu_ps( (float*)(ptr), out )
    #define stbir__simdi8_store( ptr, reg )  _mm256_storeu_si256( (__m256i*)(ptr), reg )
    #define stbir__simdf8_frep8( fval ) _mm256_set1_ps( fval )

    #define stbir__simdf8_min( out, reg0, reg1 ) (out) = _mm256_min_ps( reg0, reg1 )
    #define stbir__simdf8_max( out, reg0, reg1 ) (out) = _mm256_max_ps( reg0, reg1 )

    #define stbir__simdf8_add4halves( out, bot4, top8 ) (out) = _mm_add_ps( bot4, _mm256_extractf128_ps( top8, 1 ) )
    #define stbir__simdf8_mult_mem( out, reg, ptr ) (out) = _mm256_mul_ps( reg, _mm256_loadu_ps( (float const*)(ptr) ) )
    #define stbir__simdf8_add_mem( out, reg, ptr ) (out) = _mm256_add_ps( reg, _mm256_loadu_ps( (float const*)(ptr) ) )
    #define stbir__simdf8_add( out, a, b ) (out) = _mm256_add_ps( a, b )
    #define stbir__simdf8_load1b( out, ptr ) (out) = _mm256_broadcast_ss( ptr )
    #define stbir__simdf_load1rep4( out, ptr ) (out) = _mm_broadcast_ss( ptr )  // avx load instruction

    #define stbir__simdi8_convert_i32_to_float(out, ireg) (out) = _mm256_cvtepi32_ps( ireg )
    #define stbir__simdf8_convert_float_to_i32( i, f ) (i) = _mm256_cvttps_epi32(f)

    #define stbir__simdf8_bot4s( out, a, b ) (out) = _mm256_permute2f128_ps(a,b, (0<<0)+(2<<4) )
    #define stbir__simdf8_top4s( out, a, b ) (out) = _mm256_permute2f128_ps(a,b, (1<<0)+(3<<4) )

    #define stbir__simdf8_gettop4( reg ) _mm256_extractf128_ps(reg,1)

    #ifdef STBIR_AVX2

    #define stbir__simdi8_expand_u8_to_u32(out0,out1,ireg) \
    { \
      stbir__simdi8 a, zero  =_mm256_setzero_si256();\
      a = _mm256_permute4x64_epi64( _mm256_unpacklo_epi8( _mm256_permute4x64_epi64(_mm256_castsi128_si256(ireg),(0<<0)+(2<<2)+(1<<4)+(3<<6)), zero ),(0<<0)+(2<<2)+(1<<4)+(3<<6)); \
      out0 = _mm256_unpacklo_epi16( a, zero ); \
      out1 = _mm256_unpackhi_epi16( a, zero ); \
    }

    #define stbir__simdf8_pack_to_16bytes(out,aa,bb) \
    { \
      stbir__simdi8 t; \
      stbir__simdf8 af,bf; \
      stbir__simdi8 a,b; \
      af = _mm256_min_ps( aa, STBIR_max_uint8_as_floatX ); \
      bf = _mm256_min_ps( bb, STBIR_max_uint8_as_floatX ); \
      af = _mm256_max_ps( af, _mm256_setzero_ps() ); \
      bf = _mm256_max_ps( bf, _mm256_setzero_ps() ); \
      a = _mm256_cvttps_epi32( af ); \
      b = _mm256_cvttps_epi32( bf ); \
      t = _mm256_permute4x64_epi64( _mm256_packs_epi32( a, b ), (0<<0)+(2<<2)+(1<<4)+(3<<6) ); \
      out = _mm256_castsi256_si128( _mm256_permute4x64_epi64( _mm256_packus_epi16( t, t ), (0<<0)+(2<<2)+(1<<4)+(3<<6) ) ); \
    }

    #define stbir__simdi8_expand_u16_to_u32(out,ireg) out = _mm256_unpacklo_epi16( _mm256_permute4x64_epi64(_mm256_castsi128_si256(ireg),(0<<0)+(2<<2)+(1<<4)+(3<<6)), _mm256_setzero_si256() );

    #define stbir__simdf8_pack_to_16words(out,aa,bb) \
      { \
        stbir__simdf8 af,bf; \
        stbir__simdi8 a,b; \
        af = _mm256_min_ps( aa, STBIR_max_uint16_as_floatX ); \
        bf = _mm256_min_ps( bb, STBIR_max_uint16_as_floatX ); \
        af = _mm256_max_ps( af, _mm256_setzero_ps() ); \
        bf = _mm256_max_ps( bf, _mm256_setzero_ps() ); \
        a = _mm256_cvttps_epi32( af ); \
        b = _mm256_cvttps_epi32( bf ); \
        (out) = _mm256_permute4x64_epi64( _mm256_packus_epi32(a, b), (0<<0)+(2<<2)+(1<<4)+(3<<6) ); \
      }

    #else

    #define stbir__simdi8_expand_u8_to_u32(out0,out1,ireg) \
    { \
      stbir__simdi a,zero = _mm_setzero_si128(); \
      a = _mm_unpacklo_epi8( ireg, zero ); \
      out0 = _mm256_setr_m128i( _mm_unpacklo_epi16( a, zero ), _mm_unpackhi_epi16( a, zero ) ); \
      a = _mm_unpackhi_epi8( ireg, zero ); \
      out1 = _mm256_setr_m128i( _mm_unpacklo_epi16( a, zero ), _mm_unpackhi_epi16( a, zero ) ); \
    }

    #define stbir__simdf8_pack_to_16bytes(out,aa,bb) \
    { \
      stbir__simdi t; \
      stbir__simdf8 af,bf; \
      stbir__simdi8 a,b; \
      af = _mm256_min_ps( aa, STBIR_max_uint8_as_floatX ); \
      bf = _mm256_min_ps( bb, STBIR_max_uint8_as_floatX ); \
      af = _mm256_max_ps( af, _mm256_setzero_ps() ); \
      bf = _mm256_max_ps( bf, _mm256_setzero_ps() ); \
      a = _mm256_cvttps_epi32( af ); \
      b = _mm256_cvttps_epi32( bf ); \
      out = _mm_packs_epi32( _mm256_castsi256_si128(a), _mm256_extractf128_si256( a, 1 ) ); \
      out = _mm_packus_epi16( out, out ); \
      t = _mm_packs_epi32( _mm256_castsi256_si128(b), _mm256_extractf128_si256( b, 1 ) ); \
      t = _mm_packus_epi16( t, t ); \
      out = _mm_castps_si128( _mm_shuffle_ps( _mm_castsi128_ps(out), _mm_castsi128_ps(t), (0<<0)+(1<<2)+(0<<4)+(1<<6) ) ); \
    }

    #define stbir__simdi8_expand_u16_to_u32(out,ireg) \
    { \
      stbir__simdi a,b,zero = _mm_setzero_si128(); \
      a = _mm_unpacklo_epi16( ireg, zero ); \
      b = _mm_unpackhi_epi16( ireg, zero ); \
      out = _mm256_insertf128_si256( _mm256_castsi128_si256( a ), b, 1 ); \
    }

    #define stbir__simdf8_pack_to_16words(out,aa,bb) \
      { \
        stbir__simdi t0,t1; \
        stbir__simdf8 af,bf; \
        stbir__simdi8 a,b; \
        af = _mm256_min_ps( aa, STBIR_max_uint16_as_floatX ); \
        bf = _mm256_min_ps( bb, STBIR_max_uint16_as_floatX ); \
        af = _mm256_max_ps( af, _mm256_setzero_ps() ); \
        bf = _mm256_max_ps( bf, _mm256_setzero_ps() ); \
        a = _mm256_cvttps_epi32( af ); \
        b = _mm256_cvttps_epi32( bf ); \
        t0 = _mm_packus_epi32( _mm256_castsi256_si128(a), _mm256_extractf128_si256( a, 1 ) ); \
        t1 = _mm_packus_epi32( _mm256_castsi256_si128(b), _mm256_extractf128_si256( b, 1 ) ); \
        out = _mm256_setr_m128i( t0, t1 ); \
      }

    #endif

    static __m256i stbir_00001111 = { STBIR__CONST_4d_32i( 0, 0, 0, 0 ), STBIR__CONST_4d_32i( 1, 1, 1, 1 ) };
    #define stbir__simdf8_0123to00001111( out, in ) (out) = _mm256_permutevar_ps ( in, stbir_00001111 )

    static __m256i stbir_22223333 = { STBIR__CONST_4d_32i( 2, 2, 2, 2 ), STBIR__CONST_4d_32i( 3, 3, 3, 3 ) };
    #define stbir__simdf8_0123to22223333( out, in ) (out) = _mm256_permutevar_ps ( in, stbir_22223333 )

    #define stbir__simdf8_0123to2222( out, in ) (out) = stbir__simdf_swiz(_mm256_castps256_ps128(in), 2,2,2,2 )

    #define stbir__simdf8_load4b( out, ptr ) (out) = _mm256_broadcast_ps( (__m128 const *)(ptr) )

    static __m256i stbir_00112233 = { STBIR__CONST_4d_32i( 0, 0, 1, 1 ), STBIR__CONST_4d_32i( 2, 2, 3, 3 ) };
    #define stbir__simdf8_0123to00112233( out, in ) (out) = _mm256_permutevar_ps ( in, stbir_00112233 )
    #define stbir__simdf8_add4( out, a8, b ) (out) = _mm256_add_ps( a8,  _mm256_castps128_ps256( b ) )

    static __m256i stbir_load6 = { STBIR__CONST_4_32i( 0x80000000 ), STBIR__CONST_4d_32i(  0x80000000,  0x80000000, 0, 0 ) };
    #define stbir__simdf8_load6z( out, ptr ) (out) = _mm256_maskload_ps( ptr, stbir_load6 )

    #define stbir__simdf8_0123to00000000( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (0<<0)+(0<<2)+(0<<4)+(0<<6) )
    #define stbir__simdf8_0123to11111111( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (1<<0)+(1<<2)+(1<<4)+(1<<6) )
    #define stbir__simdf8_0123to22222222( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (2<<0)+(2<<2)+(2<<4)+(2<<6) )
    #define stbir__simdf8_0123to33333333( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (3<<0)+(3<<2)+(3<<4)+(3<<6) )
    #define stbir__simdf8_0123to21032103( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (2<<0)+(1<<2)+(0<<4)+(3<<6) )
    #define stbir__simdf8_0123to32103210( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (3<<0)+(2<<2)+(1<<4)+(0<<6) )
    #define stbir__simdf8_0123to12301230( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (1<<0)+(2<<2)+(3<<4)+(0<<6) )
    #define stbir__simdf8_0123to10321032( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (1<<0)+(0<<2)+(3<<4)+(2<<6) )
    #define stbir__simdf8_0123to30123012( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (3<<0)+(0<<2)+(1<<4)+(2<<6) )

    #define stbir__simdf8_0123to11331133( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (1<<0)+(1<<2)+(3<<4)+(3<<6) )
    #define stbir__simdf8_0123to00220022( out, in ) (out) =  _mm256_shuffle_ps ( in, in, (0<<0)+(0<<2)+(2<<4)+(2<<6) )

    #define stbir__simdf8_aaa1( out, alp, ones ) (out) = _mm256_blend_ps( alp, ones, (1<<0)+(1<<1)+(1<<2)+(0<<3)+(1<<4)+(1<<5)+(1<<6)+(0<<7)); (out)=_mm256_shuffle_ps( out,out, (3<<0) + (3<<2) + (3<<4) + (0<<6) )
    #define stbir__simdf8_1aaa( out, alp, ones ) (out) = _mm256_blend_ps( alp, ones, (0<<0)+(1<<1)+(1<<2)+(1<<3)+(0<<4)+(1<<5)+(1<<6)+(1<<7)); (out)=_mm256_shuffle_ps( out,out, (1<<0) + (0<<2) + (0<<4) + (0<<6) )
    #define stbir__simdf8_a1a1( out, alp, ones) (out) = _mm256_blend_ps( alp, ones, (1<<0)+(0<<1)+(1<<2)+(0<<3)+(1<<4)+(0<<5)+(1<<6)+(0<<7)); (out)=_mm256_shuffle_ps( out,out, (1<<0) + (0<<2) + (3<<4) + (2<<6) )
    #define stbir__simdf8_1a1a( out, alp, ones) (out) = _mm256_blend_ps( alp, ones, (0<<0)+(1<<1)+(0<<2)+(1<<3)+(0<<4)+(1<<5)+(0<<6)+(1<<7)); (out)=_mm256_shuffle_ps( out,out, (1<<0) + (0<<2) + (3<<4) + (2<<6) )

    #define stbir__simdf8_zero( reg ) (reg) = _mm256_setzero_ps()

    #ifdef STBIR_USE_FMA           // not on by default to maintain bit identical simd to non-simd
    #define stbir__simdf8_madd( out, add, mul1, mul2 ) (out) = _mm256_fmadd_ps( mul1, mul2, add )
    #define stbir__simdf8_madd_mem( out, add, mul, ptr ) (out) = _mm256_fmadd_ps( mul, _mm256_loadu_ps( (float const*)(ptr) ), add )
    #define stbir__simdf8_madd_mem4( out, add, mul, ptr )(out) = _mm256_fmadd_ps( _mm256_setr_m128( mul, _mm_setzero_ps() ), _mm256_setr_m128( _mm_loadu_ps( (float const*)(ptr) ), _mm_setzero_ps() ), add )
    #else
    #define stbir__simdf8_madd( out, add, mul1, mul2 ) (out) = _mm256_add_ps( add, _mm256_mul_ps( mul1, mul2 ) )
    #define stbir__simdf8_madd_mem( out, add, mul, ptr ) (out) = _mm256_add_ps( add, _mm256_mul_ps( mul, _mm256_loadu_ps( (float const*)(ptr) ) ) )
    #define stbir__simdf8_madd_mem4( out, add, mul, ptr )  (out) = _mm256_add_ps( add, _mm256_setr_m128( _mm_mul_ps( mul, _mm_loadu_ps( (float const*)(ptr) ) ), _mm_setzero_ps() ) )
    #endif
    #define stbir__if_simdf8_cast_to_simdf4( val ) _mm256_castps256_ps128( val )

  #endif

  #ifdef STBIR_FLOORF
  #undef STBIR_FLOORF
  #endif
  #define STBIR_FLOORF stbir_simd_floorf
  static stbir__inline float stbir_simd_floorf(float x)  // martins floorf
  {
    #if defined(STBIR_AVX) || defined(__SSE4_1__) || defined(STBIR_SSE41)
    __m128 t = _mm_set_ss(x);
    return _mm_cvtss_f32( _mm_floor_ss(t, t) );
    #else
    __m128 f = _mm_set_ss(x);
    __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(f));
    __m128 r = _mm_add_ss(t, _mm_and_ps(_mm_cmplt_ss(f, t), _mm_set_ss(-1.0f)));
    return _mm_cvtss_f32(r);
    #endif
  }

  #ifdef STBIR_CEILF
  #undef STBIR_CEILF
  #endif
  #define STBIR_CEILF stbir_simd_ceilf
  static stbir__inline float stbir_simd_ceilf(float x)  // martins ceilf
  {
    #if defined(STBIR_AVX) || defined(__SSE4_1__) || defined(STBIR_SSE41)
    __m128 t = _mm_set_ss(x);
    return _mm_cvtss_f32( _mm_ceil_ss(t, t) );
    #else
    __m128 f = _mm_set_ss(x);
    __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(f));
    __m128 r = _mm_add_ss(t, _mm_and_ps(_mm_cmplt_ss(t, f), _mm_set_ss(1.0f)));
    return _mm_cvtss_f32(r);
    #endif
  }

#elif defined(STBIR_NEON)

  #include <arm_neon.h>

  #define stbir__simdf float32x4_t
  #define stbir__simdi uint32x4_t

  #define stbir_simdi_castf( reg ) vreinterpretq_u32_f32(reg)
  #define stbir_simdf_casti( reg ) vreinterpretq_f32_u32(reg)

  #define stbir__simdf_load( reg, ptr ) (reg) = vld1q_f32( (float const*)(ptr) )
  #define stbir__simdi_load( reg, ptr ) (reg) = vld1q_u32( (uint32_t const*)(ptr) )
  #define stbir__simdf_load1( out, ptr ) (out) = vld1q_dup_f32( (float const*)(ptr) ) // top values can be random (not denormal or nan for perf)
  #define stbir__simdi_load1( out, ptr ) (out) = vld1q_dup_u32( (uint32_t const*)(ptr) )
  #define stbir__simdf_load1z( out, ptr ) (out) = vld1q_lane_f32( (float const*)(ptr), vdupq_n_f32(0), 0 ) // top values must be zero
  #define stbir__simdf_frep4( fvar ) vdupq_n_f32( fvar )
  #define stbir__simdf_load1frep4( out, fvar ) (out) = vdupq_n_f32( fvar )
  #define stbir__simdf_load2( out, ptr ) (out) = vcombine_f32( vld1_f32( (float const*)(ptr) ), vcreate_f32(0) ) // top values can be random (not denormal or nan for perf)
  #define stbir__simdf_load2z( out, ptr ) (out) = vcombine_f32( vld1_f32( (float const*)(ptr) ), vcreate_f32(0) )  // top values must be zero
  #define stbir__simdf_load2hmerge( out, reg, ptr ) (out) = vcombine_f32( vget_low_f32(reg), vld1_f32( (float const*)(ptr) ) )

  #define stbir__simdf_zeroP() vdupq_n_f32(0)
  #define stbir__simdf_zero( reg ) (reg) = vdupq_n_f32(0)

  #define stbir__simdf_store( ptr, reg )  vst1q_f32( (float*)(ptr), reg )
  #define stbir__simdf_store1( ptr, reg ) vst1q_lane_f32( (float*)(ptr), reg, 0)
  #define stbir__simdf_store2( ptr, reg ) vst1_f32( (float*)(ptr), vget_low_f32(reg) )
  #define stbir__simdf_store2h( ptr, reg ) vst1_f32( (float*)(ptr), vget_high_f32(reg) )

  #define stbir__simdi_store( ptr, reg )  vst1q_u32( (uint32_t*)(ptr), reg )
  #define stbir__simdi_store1( ptr, reg ) vst1q_lane_u32( (uint32_t*)(ptr), reg, 0 )
  #define stbir__simdi_store2( ptr, reg ) vst1_u32( (uint32_t*)(ptr), vget_low_u32(reg) )

  #define stbir__prefetch( ptr )

  #define stbir__simdi_expand_u8_to_u32(out0,out1,out2,out3,ireg) \
  { \
    uint16x8_t l = vmovl_u8( vget_low_u8 ( vreinterpretq_u8_u32(ireg) ) ); \
    uint16x8_t h = vmovl_u8( vget_high_u8( vreinterpretq_u8_u32(ireg) ) ); \
    out0 = vmovl_u16( vget_low_u16 ( l ) ); \
    out1 = vmovl_u16( vget_high_u16( l ) ); \
    out2 = vmovl_u16( vget_low_u16 ( h ) ); \
    out3 = vmovl_u16( vget_high_u16( h ) ); \
  }

  #define stbir__simdi_expand_u8_to_1u32(out,ireg) \
  { \
    uint16x8_t tmp = vmovl_u8( vget_low_u8( vreinterpretq_u8_u32(ireg) ) ); \
    out = vmovl_u16( vget_low_u16( tmp ) ); \
  }

  #define stbir__simdi_expand_u16_to_u32(out0,out1,ireg) \
  { \
    uint16x8_t tmp = vreinterpretq_u16_u32(ireg); \
    out0 = vmovl_u16( vget_low_u16 ( tmp ) ); \
    out1 = vmovl_u16( vget_high_u16( tmp ) ); \
  }

  #define stbir__simdf_convert_float_to_i32( i, f ) (i) = vreinterpretq_u32_s32( vcvtq_s32_f32(f) )
  #define stbir__simdf_convert_float_to_int( f ) vgetq_lane_s32(vcvtq_s32_f32(f), 0)
  #define stbir__simdi_to_int( i ) (int)vgetq_lane_u32(i, 0)
  #define stbir__simdf_convert_float_to_uint8( f ) ((unsigned char)vgetq_lane_s32(vcvtq_s32_f32(vmaxq_f32(vminq_f32(f,STBIR__CONSTF(STBIR_max_uint8_as_float)),vdupq_n_f32(0))), 0))
  #define stbir__simdf_convert_float_to_short( f ) ((unsigned short)vgetq_lane_s32(vcvtq_s32_f32(vmaxq_f32(vminq_f32(f,STBIR__CONSTF(STBIR_max_uint16_as_float)),vdupq_n_f32(0))), 0))
  #define stbir__simdi_convert_i32_to_float(out, ireg) (out) = vcvtq_f32_s32( vreinterpretq_s32_u32(ireg) )
  #define stbir__simdf_add( out, reg0, reg1 ) (out) = vaddq_f32( reg0, reg1 )
  #define stbir__simdf_mult( out, reg0, reg1 ) (out) = vmulq_f32( reg0, reg1 )
  #define stbir__simdf_mult_mem( out, reg, ptr ) (out) = vmulq_f32( reg, vld1q_f32( (float const*)(ptr) ) )
  #define stbir__simdf_mult1_mem( out, reg, ptr ) (out) = vmulq_f32( reg, vld1q_dup_f32( (float const*)(ptr) ) )
  #define stbir__simdf_add_mem( out, reg, ptr ) (out) = vaddq_f32( reg, vld1q_f32( (float const*)(ptr) ) )
  #define stbir__simdf_add1_mem( out, reg, ptr ) (out) = vaddq_f32( reg, vld1q_dup_f32( (float const*)(ptr) ) )

  #ifdef STBIR_USE_FMA           // not on by default to maintain bit identical simd to non-simd (and also x64 no madd to arm madd)
  #define stbir__simdf_madd( out, add, mul1, mul2 ) (out) = vfmaq_f32( add, mul1, mul2 )
  #define stbir__simdf_madd1( out, add, mul1, mul2 ) (out) = vfmaq_f32( add, mul1, mul2 )
  #define stbir__simdf_madd_mem( out, add, mul, ptr ) (out) = vfmaq_f32( add, mul, vld1q_f32( (float const*)(ptr) ) )
  #define stbir__simdf_madd1_mem( out, add, mul, ptr ) (out) = vfmaq_f32( add, mul, vld1q_dup_f32( (float const*)(ptr) ) )
  #else
  #define stbir__simdf_madd( out, add, mul1, mul2 ) (out) = vaddq_f32( add, vmulq_f32( mul1, mul2 ) )
  #define stbir__simdf_madd1( out, add, mul1, mul2 ) (out) = vaddq_f32( add, vmulq_f32( mul1, mul2 ) )
  #define stbir__simdf_madd_mem( out, add, mul, ptr ) (out) = vaddq_f32( add, vmulq_f32( mul, vld1q_f32( (float const*)(ptr) ) ) )
  #define stbir__simdf_madd1_mem( out, add, mul, ptr ) (out) = vaddq_f32( add, vmulq_f32( mul, vld1q_dup_f32( (float const*)(ptr) ) ) )
  #endif

  #define stbir__simdf_add1( out, reg0, reg1 ) (out) = vaddq_f32( reg0, reg1 )
  #define stbir__simdf_mult1( out, reg0, reg1 ) (out) = vmulq_f32( reg0, reg1 )

  #define stbir__simdf_and( out, reg0, reg1 ) (out) = vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32(reg0), vreinterpretq_u32_f32(reg1) ) )
  #define stbir__simdf_or( out, reg0, reg1 ) (out) = vreinterpretq_f32_u32( vorrq_u32( vreinterpretq_u32_f32(reg0), vreinterpretq_u32_f32(reg1) ) )

  #define stbir__simdf_min( out, reg0, reg1 ) (out) = vminq_f32( reg0, reg1 )
  #define stbir__simdf_max( out, reg0, reg1 ) (out) = vmaxq_f32( reg0, reg1 )
  #define stbir__simdf_min1( out, reg0, reg1 ) (out) = vminq_f32( reg0, reg1 )
  #define stbir__simdf_max1( out, reg0, reg1 ) (out) = vmaxq_f32( reg0, reg1 )

  #define stbir__simdf_0123ABCDto3ABx( out, reg0, reg1 ) (out) = vextq_f32( reg0, reg1, 3 )
  #define stbir__simdf_0123ABCDto23Ax( out, reg0, reg1 ) (out) = vextq_f32( reg0, reg1, 2 )

  #define stbir__simdf_a1a1( out, alp, ones ) (out) = vzipq_f32(vuzpq_f32(alp, alp).val[1], ones).val[0]
  #define stbir__simdf_1a1a( out, alp, ones ) (out) = vzipq_f32(ones, vuzpq_f32(alp, alp).val[0]).val[0]

  #if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ )

    #define stbir__simdf_aaa1( out, alp, ones ) (out) = vcopyq_laneq_f32(vdupq_n_f32(vgetq_lane_f32(alp, 3)), 3, ones, 3)
    #define stbir__simdf_1aaa( out, alp, ones ) (out) = vcopyq_laneq_f32(vdupq_n_f32(vgetq_lane_f32(alp, 0)), 0, ones, 0)

    #if defined( _MSC_VER ) && !defined(__clang__)
      #define stbir_make16(a,b,c,d) vcombine_u8( \
        vcreate_u8( (4*a+0) | ((4*a+1)<<8) | ((4*a+2)<<16) | ((4*a+3)<<24) | \
          ((stbir_uint64)(4*b+0)<<32) | ((stbir_uint64)(4*b+1)<<40) | ((stbir_uint64)(4*b+2)<<48) | ((stbir_uint64)(4*b+3)<<56)), \
        vcreate_u8( (4*c+0) | ((4*c+1)<<8) | ((4*c+2)<<16) | ((4*c+3)<<24) | \
          ((stbir_uint64)(4*d+0)<<32) | ((stbir_uint64)(4*d+1)<<40) | ((stbir_uint64)(4*d+2)<<48) | ((stbir_uint64)(4*d+3)<<56) ) )

      static stbir__inline uint8x16x2_t stbir_make16x2(float32x4_t rega,float32x4_t regb)
      {
        uint8x16x2_t r = { vreinterpretq_u8_f32(rega), vreinterpretq_u8_f32(regb) };
        return r;
      }
    #else
      #define stbir_make16(a,b,c,d) (uint8x16_t){4*a+0,4*a+1,4*a+2,4*a+3,4*b+0,4*b+1,4*b+2,4*b+3,4*c+0,4*c+1,4*c+2,4*c+3,4*d+0,4*d+1,4*d+2,4*d+3}
      #define stbir_make16x2(a,b) (uint8x16x2_t){{vreinterpretq_u8_f32(a),vreinterpretq_u8_f32(b)}}
    #endif

    #define stbir__simdf_swiz( reg, one, two, three, four ) vreinterpretq_f32_u8( vqtbl1q_u8( vreinterpretq_u8_f32(reg), stbir_make16(one, two, three, four) ) )
    #define stbir__simdf_swiz2( rega, regb, one, two, three, four ) vreinterpretq_f32_u8( vqtbl2q_u8( stbir_make16x2(rega,regb), stbir_make16(one, two, three, four) ) )

    #define stbir__simdi_16madd( out, reg0, reg1 ) \
    { \
      int16x8_t r0 = vreinterpretq_s16_u32(reg0); \
      int16x8_t r1 = vreinterpretq_s16_u32(reg1); \
      int32x4_t tmp0 = vmull_s16( vget_low_s16(r0), vget_low_s16(r1) ); \
      int32x4_t tmp1 = vmull_s16( vget_high_s16(r0), vget_high_s16(r1) ); \
      (out) = vreinterpretq_u32_s32( vpaddq_s32(tmp0, tmp1) ); \
    }

  #else

    #define stbir__simdf_aaa1( out, alp, ones ) (out) = vsetq_lane_f32(1.0f, vdupq_n_f32(vgetq_lane_f32(alp, 3)), 3)
    #define stbir__simdf_1aaa( out, alp, ones ) (out) = vsetq_lane_f32(1.0f, vdupq_n_f32(vgetq_lane_f32(alp, 0)), 0)

    #if defined( _MSC_VER ) && !defined(__clang__)
      static stbir__inline uint8x8x2_t stbir_make8x2(float32x4_t reg)
      {
        uint8x8x2_t r = { { vget_low_u8(vreinterpretq_u8_f32(reg)), vget_high_u8(vreinterpretq_u8_f32(reg)) } };
        return r;
      }
      #define stbir_make8(a,b) vcreate_u8( \
        (4*a+0) | ((4*a+1)<<8) | ((4*a+2)<<16) | ((4*a+3)<<24) | \
        ((stbir_uint64)(4*b+0)<<32) | ((stbir_uint64)(4*b+1)<<40) | ((stbir_uint64)(4*b+2)<<48) | ((stbir_uint64)(4*b+3)<<56) )
    #else
      #define stbir_make8x2(reg) (uint8x8x2_t){ { vget_low_u8(vreinterpretq_u8_f32(reg)), vget_high_u8(vreinterpretq_u8_f32(reg)) } }
      #define stbir_make8(a,b) (uint8x8_t){4*a+0,4*a+1,4*a+2,4*a+3,4*b+0,4*b+1,4*b+2,4*b+3}
    #endif

    #define stbir__simdf_swiz( reg, one, two, three, four ) vreinterpretq_f32_u8( vcombine_u8( \
        vtbl2_u8( stbir_make8x2( reg ), stbir_make8( one, two ) ), \
        vtbl2_u8( stbir_make8x2( reg ), stbir_make8( three, four ) ) ) )

    #define stbir__simdi_16madd( out, reg0, reg1 ) \
    { \
      int16x8_t r0 = vreinterpretq_s16_u32(reg0); \
      int16x8_t r1 = vreinterpretq_s16_u32(reg1); \
      int32x4_t tmp0 = vmull_s16( vget_low_s16(r0), vget_low_s16(r1) ); \
      int32x4_t tmp1 = vmull_s16( vget_high_s16(r0), vget_high_s16(r1) ); \
      int32x2_t out0 = vpadd_s32( vget_low_s32(tmp0), vget_high_s32(tmp0) ); \
      int32x2_t out1 = vpadd_s32( vget_low_s32(tmp1), vget_high_s32(tmp1) ); \
      (out) = vreinterpretq_u32_s32( vcombine_s32(out0, out1) ); \
    }

  #endif

  #define stbir__simdi_and( out, reg0, reg1 ) (out) = vandq_u32( reg0, reg1 )
  #define stbir__simdi_or( out, reg0, reg1 ) (out) = vorrq_u32( reg0, reg1 )

  #define stbir__simdf_pack_to_8bytes(out,aa,bb) \
  { \
    float32x4_t af = vmaxq_f32( vminq_f32(aa,STBIR__CONSTF(STBIR_max_uint8_as_float) ), vdupq_n_f32(0) ); \
    float32x4_t bf = vmaxq_f32( vminq_f32(bb,STBIR__CONSTF(STBIR_max_uint8_as_float) ), vdupq_n_f32(0) ); \
    int16x4_t ai = vqmovn_s32( vcvtq_s32_f32( af ) ); \
    int16x4_t bi = vqmovn_s32( vcvtq_s32_f32( bf ) ); \
    uint8x8_t out8 = vqmovun_s16( vcombine_s16(ai, bi) ); \
    out = vreinterpretq_u32_u8( vcombine_u8(out8, out8) ); \
  }

  #define stbir__simdf_pack_to_8words(out,aa,bb) \
  { \
    float32x4_t af = vmaxq_f32( vminq_f32(aa,STBIR__CONSTF(STBIR_max_uint16_as_float) ), vdupq_n_f32(0) ); \
    float32x4_t bf = vmaxq_f32( vminq_f32(bb,STBIR__CONSTF(STBIR_max_uint16_as_float) ), vdupq_n_f32(0) ); \
    int32x4_t ai = vcvtq_s32_f32( af ); \
    int32x4_t bi = vcvtq_s32_f32( bf ); \
    out = vreinterpretq_u32_u16( vcombine_u16(vqmovun_s32(ai), vqmovun_s32(bi)) ); \
  }

  #define stbir__interleave_pack_and_store_16_u8( ptr, r0, r1, r2, r3 ) \
  { \
    int16x4x2_t tmp0 = vzip_s16( vqmovn_s32(vreinterpretq_s32_u32(r0)), vqmovn_s32(vreinterpretq_s32_u32(r2)) ); \
    int16x4x2_t tmp1 = vzip_s16( vqmovn_s32(vreinterpretq_s32_u32(r1)), vqmovn_s32(vreinterpretq_s32_u32(r3)) ); \
    uint8x8x2_t out = \
    { { \
      vqmovun_s16( vcombine_s16(tmp0.val[0], tmp0.val[1]) ), \
      vqmovun_s16( vcombine_s16(tmp1.val[0], tmp1.val[1]) ), \
    } }; \
    vst2_u8(ptr, out); \
  }

  #define stbir__simdf_load4_transposed( o0, o1, o2, o3, ptr ) \
  { \
    float32x4x4_t tmp = vld4q_f32(ptr); \
    o0 = tmp.val[0]; \
    o1 = tmp.val[1]; \
    o2 = tmp.val[2]; \
    o3 = tmp.val[3]; \
  }

  #define stbir__simdi_32shr( out, reg, imm ) out = vshrq_n_u32( reg, imm )

  #if defined( _MSC_VER ) && !defined(__clang__)
    #define STBIR__SIMDF_CONST(var, x) __declspec(align(8)) float var[] = { x, x, x, x }
    #define STBIR__SIMDI_CONST(var, x) __declspec(align(8)) uint32_t var[] = { x, x, x, x }
    #define STBIR__CONSTF(var) (*(const float32x4_t*)var)
    #define STBIR__CONSTI(var) (*(const uint32x4_t*)var)
  #else
    #define STBIR__SIMDF_CONST(var, x) stbir__simdf var = { x, x, x, x }
    #define STBIR__SIMDI_CONST(var, x) stbir__simdi var = { x, x, x, x }
    #define STBIR__CONSTF(var) (var)
    #define STBIR__CONSTI(var) (var)
  #endif

  #ifdef STBIR_FLOORF
  #undef STBIR_FLOORF
  #endif
  #define STBIR_FLOORF stbir_simd_floorf
  static stbir__inline float stbir_simd_floorf(float x)
  {
    #if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ )
    return vget_lane_f32( vrndm_f32( vdup_n_f32(x) ), 0);
    #else
    float32x2_t f = vdup_n_f32(x);
    float32x2_t t = vcvt_f32_s32(vcvt_s32_f32(f));
    uint32x2_t a = vclt_f32(f, t);
    uint32x2_t b = vreinterpret_u32_f32(vdup_n_f32(-1.0f));
    float32x2_t r = vadd_f32(t, vreinterpret_f32_u32(vand_u32(a, b)));
    return vget_lane_f32(r, 0);
    #endif
  }

  #ifdef STBIR_CEILF
  #undef STBIR_CEILF
  #endif
  #define STBIR_CEILF stbir_simd_ceilf
  static stbir__inline float stbir_simd_ceilf(float x)
  {
    #if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ )
    return vget_lane_f32( vrndp_f32( vdup_n_f32(x) ), 0);
    #else
    float32x2_t f = vdup_n_f32(x);
    float32x2_t t = vcvt_f32_s32(vcvt_s32_f32(f));
    uint32x2_t a = vclt_f32(t, f);
    uint32x2_t b = vreinterpret_u32_f32(vdup_n_f32(1.0f));
    float32x2_t r = vadd_f32(t, vreinterpret_f32_u32(vand_u32(a, b)));
    return vget_lane_f32(r, 0);
    #endif
  }

  #define STBIR_SIMD

#elif defined(STBIR_WASM)

  #include <wasm_simd128.h>

  #define stbir__simdf v128_t
  #define stbir__simdi v128_t

  #define stbir_simdi_castf( reg ) (reg)
  #define stbir_simdf_casti( reg ) (reg)

  #define stbir__simdf_load( reg, ptr )             (reg) = wasm_v128_load( (void const*)(ptr) )
  #define stbir__simdi_load( reg, ptr )             (reg) = wasm_v128_load( (void const*)(ptr) )
  #define stbir__simdf_load1( out, ptr )            (out) = wasm_v128_load32_splat( (void const*)(ptr) ) // top values can be random (not denormal or nan for perf)
  #define stbir__simdi_load1( out, ptr )            (out) = wasm_v128_load32_splat( (void const*)(ptr) )
  #define stbir__simdf_load1z( out, ptr )           (out) = wasm_v128_load32_zero( (void const*)(ptr) ) // top values must be zero
  #define stbir__simdf_frep4( fvar )                wasm_f32x4_splat( fvar )
  #define stbir__simdf_load1frep4( out, fvar )      (out) = wasm_f32x4_splat( fvar )
  #define stbir__simdf_load2( out, ptr )            (out) = wasm_v128_load64_splat( (void const*)(ptr) ) // top values can be random (not denormal or nan for perf)
  #define stbir__simdf_load2z( out, ptr )           (out) = wasm_v128_load64_zero( (void const*)(ptr) ) // top values must be zero
  #define stbir__simdf_load2hmerge( out, reg, ptr ) (out) = wasm_v128_load64_lane( (void const*)(ptr), reg, 1 )

  #define stbir__simdf_zeroP() wasm_f32x4_const_splat(0)
  #define stbir__simdf_zero( reg ) (reg) = wasm_f32x4_const_splat(0)

  #define stbir__simdf_store( ptr, reg )   wasm_v128_store( (void*)(ptr), reg )
  #define stbir__simdf_store1( ptr, reg )  wasm_v128_store32_lane( (void*)(ptr), reg, 0 )
  #define stbir__simdf_store2( ptr, reg )  wasm_v128_store64_lane( (void*)(ptr), reg, 0 )
  #define stbir__simdf_store2h( ptr, reg ) wasm_v128_store64_lane( (void*)(ptr), reg, 1 )

  #define stbir__simdi_store( ptr, reg )  wasm_v128_store( (void*)(ptr), reg )
  #define stbir__simdi_store1( ptr, reg ) wasm_v128_store32_lane( (void*)(ptr), reg, 0 )
  #define stbir__simdi_store2( ptr, reg ) wasm_v128_store64_lane( (void*)(ptr), reg, 0 )

  #define stbir__prefetch( ptr )

  #define stbir__simdi_expand_u8_to_u32(out0,out1,out2,out3,ireg) \
  { \
    v128_t l = wasm_u16x8_extend_low_u8x16 ( ireg ); \
    v128_t h = wasm_u16x8_extend_high_u8x16( ireg ); \
    out0 = wasm_u32x4_extend_low_u16x8 ( l ); \
    out1 = wasm_u32x4_extend_high_u16x8( l ); \
    out2 = wasm_u32x4_extend_low_u16x8 ( h ); \
    out3 = wasm_u32x4_extend_high_u16x8( h ); \
  }

  #define stbir__simdi_expand_u8_to_1u32(out,ireg) \
  { \
    v128_t tmp = wasm_u16x8_extend_low_u8x16(ireg); \
    out = wasm_u32x4_extend_low_u16x8(tmp); \
  }

  #define stbir__simdi_expand_u16_to_u32(out0,out1,ireg) \
  { \
    out0 = wasm_u32x4_extend_low_u16x8 ( ireg ); \
    out1 = wasm_u32x4_extend_high_u16x8( ireg ); \
  }

  #define stbir__simdf_convert_float_to_i32( i, f )    (i) = wasm_i32x4_trunc_sat_f32x4(f)
  #define stbir__simdf_convert_float_to_int( f )       wasm_i32x4_extract_lane(wasm_i32x4_trunc_sat_f32x4(f), 0)
  #define stbir__simdi_to_int( i )                     wasm_i32x4_extract_lane(i, 0)
  #define stbir__simdf_convert_float_to_uint8( f )     ((unsigned char)wasm_i32x4_extract_lane(wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_max(wasm_f32x4_min(f,STBIR_max_uint8_as_float),wasm_f32x4_const_splat(0))), 0))
  #define stbir__simdf_convert_float_to_short( f )     ((unsigned short)wasm_i32x4_extract_lane(wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_max(wasm_f32x4_min(f,STBIR_max_uint16_as_float),wasm_f32x4_const_splat(0))), 0))
  #define stbir__simdi_convert_i32_to_float(out, ireg) (out) = wasm_f32x4_convert_i32x4(ireg)
  #define stbir__simdf_add( out, reg0, reg1 )          (out) = wasm_f32x4_add( reg0, reg1 )
  #define stbir__simdf_mult( out, reg0, reg1 )         (out) = wasm_f32x4_mul( reg0, reg1 )
  #define stbir__simdf_mult_mem( out, reg, ptr )       (out) = wasm_f32x4_mul( reg, wasm_v128_load( (void const*)(ptr) ) )
  #define stbir__simdf_mult1_mem( out, reg, ptr )      (out) = wasm_f32x4_mul( reg, wasm_v128_load32_splat( (void const*)(ptr) ) )
  #define stbir__simdf_add_mem( out, reg, ptr )        (out) = wasm_f32x4_add( reg, wasm_v128_load( (void const*)(ptr) ) )
  #define stbir__simdf_add1_mem( out, reg, ptr )       (out) = wasm_f32x4_add( reg, wasm_v128_load32_splat( (void const*)(ptr) ) )

  #define stbir__simdf_madd( out, add, mul1, mul2 )    (out) = wasm_f32x4_add( add, wasm_f32x4_mul( mul1, mul2 ) )
  #define stbir__simdf_madd1( out, add, mul1, mul2 )   (out) = wasm_f32x4_add( add, wasm_f32x4_mul( mul1, mul2 ) )
  #define stbir__simdf_madd_mem( out, add, mul, ptr )  (out) = wasm_f32x4_add( add, wasm_f32x4_mul( mul, wasm_v128_load( (void const*)(ptr) ) ) )
  #define stbir__simdf_madd1_mem( out, add, mul, ptr ) (out) = wasm_f32x4_add( add, wasm_f32x4_mul( mul, wasm_v128_load32_splat( (void const*)(ptr) ) ) )

  #define stbir__simdf_add1( out, reg0, reg1 )  (out) = wasm_f32x4_add( reg0, reg1 )
  #define stbir__simdf_mult1( out, reg0, reg1 ) (out) = wasm_f32x4_mul( reg0, reg1 )

  #define stbir__simdf_and( out, reg0, reg1 ) (out) = wasm_v128_and( reg0, reg1 )
  #define stbir__simdf_or( out, reg0, reg1 )  (out) = wasm_v128_or( reg0, reg1 )

  #define stbir__simdf_min( out, reg0, reg1 ) (out) = wasm_f32x4_min( reg0, reg1 )
  #define stbir__simdf_max( out, reg0, reg1 ) (out) = wasm_f32x4_max( reg0, reg1 )
  #define stbir__simdf_min1( out, reg0, reg1 ) (out) = wasm_f32x4_min( reg0, reg1 )
  #define stbir__simdf_max1( out, reg0, reg1 ) (out) = wasm_f32x4_max( reg0, reg1 )

  #define stbir__simdf_0123ABCDto3ABx( out, reg0, reg1 ) (out) = wasm_i32x4_shuffle( reg0, reg1, 3, 4, 5, -1 )
  #define stbir__simdf_0123ABCDto23Ax( out, reg0, reg1 ) (out) = wasm_i32x4_shuffle( reg0, reg1, 2, 3, 4, -1 )

  #define stbir__simdf_aaa1(out,alp,ones) (out) = wasm_i32x4_shuffle(alp, ones, 3, 3, 3, 4)
  #define stbir__simdf_1aaa(out,alp,ones) (out) = wasm_i32x4_shuffle(alp, ones, 4, 0, 0, 0)
  #define stbir__simdf_a1a1(out,alp,ones) (out) = wasm_i32x4_shuffle(alp, ones, 1, 4, 3, 4)
  #define stbir__simdf_1a1a(out,alp,ones) (out) = wasm_i32x4_shuffle(alp, ones, 4, 0, 4, 2)

  #define stbir__simdf_swiz( reg, one, two, three, four ) wasm_i32x4_shuffle(reg, reg, one, two, three, four)

  #define stbir__simdi_and( out, reg0, reg1 )    (out) = wasm_v128_and( reg0, reg1 )
  #define stbir__simdi_or( out, reg0, reg1 )     (out) = wasm_v128_or( reg0, reg1 )
  #define stbir__simdi_16madd( out, reg0, reg1 ) (out) = wasm_i32x4_dot_i16x8( reg0, reg1 )

  #define stbir__simdf_pack_to_8bytes(out,aa,bb) \
  { \
    v128_t af = wasm_f32x4_max( wasm_f32x4_min(aa, STBIR_max_uint8_as_float), wasm_f32x4_const_splat(0) ); \
    v128_t bf = wasm_f32x4_max( wasm_f32x4_min(bb, STBIR_max_uint8_as_float), wasm_f32x4_const_splat(0) ); \
    v128_t ai = wasm_i32x4_trunc_sat_f32x4( af ); \
    v128_t bi = wasm_i32x4_trunc_sat_f32x4( bf ); \
    v128_t out16 = wasm_i16x8_narrow_i32x4( ai, bi ); \
    out = wasm_u8x16_narrow_i16x8( out16, out16 ); \
  }

  #define stbir__simdf_pack_to_8words(out,aa,bb) \
  { \
    v128_t af = wasm_f32x4_max( wasm_f32x4_min(aa, STBIR_max_uint16_as_float), wasm_f32x4_const_splat(0)); \
    v128_t bf = wasm_f32x4_max( wasm_f32x4_min(bb, STBIR_max_uint16_as_float), wasm_f32x4_const_splat(0)); \
    v128_t ai = wasm_i32x4_trunc_sat_f32x4( af ); \
    v128_t bi = wasm_i32x4_trunc_sat_f32x4( bf ); \
    out = wasm_u16x8_narrow_i32x4( ai, bi ); \
  }

  #define stbir__interleave_pack_and_store_16_u8( ptr, r0, r1, r2, r3 ) \
  { \
    v128_t tmp0 = wasm_i16x8_narrow_i32x4(r0, r1); \
    v128_t tmp1 = wasm_i16x8_narrow_i32x4(r2, r3); \
    v128_t tmp = wasm_u8x16_narrow_i16x8(tmp0, tmp1); \
    tmp = wasm_i8x16_shuffle(tmp, tmp, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); \
    wasm_v128_store( (void*)(ptr), tmp); \
  }

  #define stbir__simdf_load4_transposed( o0, o1, o2, o3, ptr ) \
  { \
    v128_t t0 = wasm_v128_load( ptr    ); \
    v128_t t1 = wasm_v128_load( ptr+4  ); \
    v128_t t2 = wasm_v128_load( ptr+8  ); \
    v128_t t3 = wasm_v128_load( ptr+12 ); \
    v128_t s0 = wasm_i32x4_shuffle(t0, t1, 0, 4, 2, 6); \
    v128_t s1 = wasm_i32x4_shuffle(t0, t1, 1, 5, 3, 7); \
    v128_t s2 = wasm_i32x4_shuffle(t2, t3, 0, 4, 2, 6); \
    v128_t s3 = wasm_i32x4_shuffle(t2, t3, 1, 5, 3, 7); \
    o0 = wasm_i32x4_shuffle(s0, s2, 0, 1, 4, 5); \
    o1 = wasm_i32x4_shuffle(s1, s3, 0, 1, 4, 5); \
    o2 = wasm_i32x4_shuffle(s0, s2, 2, 3, 6, 7); \
    o3 = wasm_i32x4_shuffle(s1, s3, 2, 3, 6, 7); \
  }

  #define stbir__simdi_32shr( out, reg, imm ) out = wasm_u32x4_shr( reg, imm )

  typedef float stbir__f32x4 __attribute__((__vector_size__(16), __aligned__(16)));
  #define STBIR__SIMDF_CONST(var, x) stbir__simdf var = (v128_t)(stbir__f32x4){ x, x, x, x }
  #define STBIR__SIMDI_CONST(var, x) stbir__simdi var = { x, x, x, x }
  #define STBIR__CONSTF(var) (var)
  #define STBIR__CONSTI(var) (var)

  #ifdef STBIR_FLOORF
  #undef STBIR_FLOORF
  #endif
  #define STBIR_FLOORF stbir_simd_floorf
  static stbir__inline float stbir_simd_floorf(float x)
  {
    return wasm_f32x4_extract_lane( wasm_f32x4_floor( wasm_f32x4_splat(x) ), 0);
  }

  #ifdef STBIR_CEILF
  #undef STBIR_CEILF
  #endif
  #define STBIR_CEILF stbir_simd_ceilf
  static stbir__inline float stbir_simd_ceilf(float x)
  {
    return wasm_f32x4_extract_lane( wasm_f32x4_ceil( wasm_f32x4_splat(x) ), 0);
  }

  #define STBIR_SIMD

#endif  // SSE2/NEON/WASM

#endif // NO SIMD

#ifdef STBIR_SIMD8
  #define stbir__simdfX stbir__simdf8
  #define stbir__simdiX stbir__simdi8
  #define stbir__simdfX_load stbir__simdf8_load
  #define stbir__simdiX_load stbir__simdi8_load
  #define stbir__simdfX_mult stbir__simdf8_mult
  #define stbir__simdfX_add_mem stbir__simdf8_add_mem
  #define stbir__simdfX_madd_mem stbir__simdf8_madd_mem
  #define stbir__simdfX_store stbir__simdf8_store
  #define stbir__simdiX_store stbir__simdi8_store
  #define stbir__simdf_frepX  stbir__simdf8_frep8
  #define stbir__simdfX_madd stbir__simdf8_madd
  #define stbir__simdfX_min stbir__simdf8_min
  #define stbir__simdfX_max stbir__simdf8_max
  #define stbir__simdfX_aaa1 stbir__simdf8_aaa1
  #define stbir__simdfX_1aaa stbir__simdf8_1aaa
  #define stbir__simdfX_a1a1 stbir__simdf8_a1a1
  #define stbir__simdfX_1a1a stbir__simdf8_1a1a
  #define stbir__simdfX_convert_float_to_i32 stbir__simdf8_convert_float_to_i32
  #define stbir__simdfX_pack_to_words stbir__simdf8_pack_to_16words
  #define stbir__simdfX_zero stbir__simdf8_zero
  #define STBIR_onesX STBIR_ones8
  #define STBIR_max_uint8_as_floatX STBIR_max_uint8_as_float8
  #define STBIR_max_uint16_as_floatX STBIR_max_uint16_as_float8
  #define STBIR_simd_point5X STBIR_simd_point58
  #define stbir__simdfX_float_count 8
  #define stbir__simdfX_0123to1230 stbir__simdf8_0123to12301230
  #define stbir__simdfX_0123to2103 stbir__simdf8_0123to21032103
  static const stbir__simdf8 STBIR_max_uint16_as_float_inverted8 = { stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted };
  static const stbir__simdf8 STBIR_max_uint8_as_float_inverted8 = { stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted };
  static const stbir__simdf8 STBIR_ones8 = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
  static const stbir__simdf8 STBIR_simd_point58 = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
  static const stbir__simdf8 STBIR_max_uint8_as_float8 = { stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float, stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float };
  static const stbir__simdf8 STBIR_max_uint16_as_float8 = { stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float, stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float };
#else
  #define stbir__simdfX stbir__simdf
  #define stbir__simdiX stbir__simdi
  #define stbir__simdfX_load stbir__simdf_load
  #define stbir__simdiX_load stbir__simdi_load
  #define stbir__simdfX_mult stbir__simdf_mult
  #define stbir__simdfX_add_mem stbir__simdf_add_mem
  #define stbir__simdfX_madd_mem stbir__simdf_madd_mem
  #define stbir__simdfX_store stbir__simdf_store
  #define stbir__simdiX_store stbir__simdi_store
  #define stbir__simdf_frepX  stbir__simdf_frep4
  #define stbir__simdfX_madd stbir__simdf_madd
  #define stbir__simdfX_min stbir__simdf_min
  #define stbir__simdfX_max stbir__simdf_max
  #define stbir__simdfX_aaa1 stbir__simdf_aaa1
  #define stbir__simdfX_1aaa stbir__simdf_1aaa
  #define stbir__simdfX_a1a1 stbir__simdf_a1a1
  #define stbir__simdfX_1a1a stbir__simdf_1a1a
  #define stbir__simdfX_convert_float_to_i32 stbir__simdf_convert_float_to_i32
  #define stbir__simdfX_pack_to_words stbir__simdf_pack_to_8words
  #define stbir__simdfX_zero stbir__simdf_zero
  #define STBIR_onesX STBIR__CONSTF(STBIR_ones)
  #define STBIR_simd_point5X STBIR__CONSTF(STBIR_simd_point5)
  #define STBIR_max_uint8_as_floatX STBIR__CONSTF(STBIR_max_uint8_as_float)
  #define STBIR_max_uint16_as_floatX STBIR__CONSTF(STBIR_max_uint16_as_float)
  #define stbir__simdfX_float_count 4
  #define stbir__if_simdf8_cast_to_simdf4( val ) ( val )
  #define stbir__simdfX_0123to1230 stbir__simdf_0123to1230
  #define stbir__simdfX_0123to2103 stbir__simdf_0123to2103
#endif


#if defined(STBIR_NEON) && !defined(_M_ARM) && !defined(__arm__)

  #if defined( _MSC_VER ) && !defined(__clang__)
  typedef __int16 stbir__FP16;
  #else
  typedef float16_t stbir__FP16;
  #endif

#else // no NEON, or 32-bit ARM for MSVC

  typedef union stbir__FP16
  {
    unsigned short u;
  } stbir__FP16;

#endif

#if (!defined(STBIR_NEON) && !defined(STBIR_FP16C)) || (defined(STBIR_NEON) && defined(_M_ARM)) || (defined(STBIR_NEON) && defined(__arm__))

  // Fabian's half float routines, see: https://gist.github.com/rygorous/2156668

  static stbir__inline float stbir__half_to_float( stbir__FP16 h )
  {
    static const stbir__FP32 magic = { (254 - 15) << 23 };
    static const stbir__FP32 was_infnan = { (127 + 16) << 23 };
    stbir__FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    o.f *= magic.f;                 // exponent adjust
    if (o.f >= was_infnan.f)        // make sure Inf/NaN survive
      o.u |= 255 << 23;
    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
  }

  static stbir__inline stbir__FP16 stbir__float_to_half(float val)
  {
    stbir__FP32 f32infty = { 255 << 23 };
    stbir__FP32 f16max   = { (127 + 16) << 23 };
    stbir__FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
    unsigned int sign_mask = 0x80000000u;
    stbir__FP16 o = { 0 };
    stbir__FP32 f;
    unsigned int sign;

    f.f = val;
    sign = f.u & sign_mask;
    f.u ^= sign;

    if (f.u >= f16max.u) // result is Inf or NaN (all exponent bits set)
      o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
      if (f.u < (113 << 23)) // resulting FP16 is subnormal or zero
      {
        // use a magic value to align our 10 mantissa bits at the bottom of
        // the float. as long as FP addition is round-to-nearest-even this
        // just works.
        f.f += denorm_magic.f;
        // and one integer subtract of the bias later, we have our final float!
        o.u = (unsigned short) ( f.u - denorm_magic.u );
      }
      else
      {
        unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd
        // update exponent, rounding bias part 1
        f.u = f.u + ((15u - 127) << 23) + 0xfff;
        // rounding bias part 2
        f.u += mant_odd;
        // take the bits!
        o.u = (unsigned short) ( f.u >> 13 );
      }
    }

    o.u |= sign >> 16;
    return o;
  }

#endif


#if defined(STBIR_FP16C)

  #include <immintrin.h>

  static stbir__inline void stbir__half_to_float_SIMD(float * output, stbir__FP16 const * input)
  {
    _mm256_storeu_ps( (float*)output, _mm256_cvtph_ps( _mm_loadu_si128( (__m128i const* )input ) ) );
  }

  static stbir__inline void stbir__float_to_half_SIMD(stbir__FP16 * output, float const * input)
  {
    _mm_storeu_si128( (__m128i*)output, _mm256_cvtps_ph( _mm256_loadu_ps( input ), 0 ) );
  }

  static stbir__inline float stbir__half_to_float( stbir__FP16 h )
  {
    return _mm_cvtss_f32( _mm_cvtph_ps( _mm_cvtsi32_si128( (int)h.u ) ) );
  }

  static stbir__inline stbir__FP16 stbir__float_to_half( float f )
  {
    stbir__FP16 h;
    h.u = (unsigned short) _mm_cvtsi128_si32( _mm_cvtps_ph( _mm_set_ss( f ), 0 ) );
    return h;
  }

#elif defined(STBIR_SSE2)

  // Fabian's half float routines, see: https://gist.github.com/rygorous/2156668
  stbir__inline static void stbir__half_to_float_SIMD(float * output, void const * input)
  {
    static const STBIR__SIMDI_CONST(mask_nosign,      0x7fff);
    static const STBIR__SIMDI_CONST(smallest_normal,  0x0400);
    static const STBIR__SIMDI_CONST(infinity,         0x7c00);
    static const STBIR__SIMDI_CONST(expadjust_normal, (127 - 15) << 23);
    static const STBIR__SIMDI_CONST(magic_denorm,     113 << 23);

    __m128i i = _mm_loadu_si128 ( (__m128i const*)(input) );
    __m128i h = _mm_unpacklo_epi16 ( i, _mm_setzero_si128() );
    __m128i mnosign     = STBIR__CONSTI(mask_nosign);
    __m128i eadjust     = STBIR__CONSTI(expadjust_normal);
    __m128i smallest    = STBIR__CONSTI(smallest_normal);
    __m128i infty       = STBIR__CONSTI(infinity);
    __m128i expmant     = _mm_and_si128(mnosign, h);
    __m128i justsign    = _mm_xor_si128(h, expmant);
    __m128i b_notinfnan = _mm_cmpgt_epi32(infty, expmant);
    __m128i b_isdenorm  = _mm_cmpgt_epi32(smallest, expmant);
    __m128i shifted     = _mm_slli_epi32(expmant, 13);
    __m128i adj_infnan  = _mm_andnot_si128(b_notinfnan, eadjust);
    __m128i adjusted    = _mm_add_epi32(eadjust, shifted);
    __m128i den1        = _mm_add_epi32(shifted, STBIR__CONSTI(magic_denorm));
    __m128i adjusted2   = _mm_add_epi32(adjusted, adj_infnan);
    __m128  den2        = _mm_sub_ps(_mm_castsi128_ps(den1), *(const __m128 *)&magic_denorm);
    __m128  adjusted3   = _mm_and_ps(den2, _mm_castsi128_ps(b_isdenorm));
    __m128  adjusted4   = _mm_andnot_ps(_mm_castsi128_ps(b_isdenorm), _mm_castsi128_ps(adjusted2));
    __m128  adjusted5   = _mm_or_ps(adjusted3, adjusted4);
    __m128i sign        = _mm_slli_epi32(justsign, 16);
    __m128  final       = _mm_or_ps(adjusted5, _mm_castsi128_ps(sign));
    stbir__simdf_store( output + 0,  final );

    h = _mm_unpackhi_epi16 ( i, _mm_setzero_si128() );
    expmant     = _mm_and_si128(mnosign, h);
    justsign    = _mm_xor_si128(h, expmant);
    b_notinfnan = _mm_cmpgt_epi32(infty, expmant);
    b_isdenorm  = _mm_cmpgt_epi32(smallest, expmant);
    shifted     = _mm_slli_epi32(expmant, 13);
    adj_infnan  = _mm_andnot_si128(b_notinfnan, eadjust);
    adjusted    = _mm_add_epi32(eadjust, shifted);
    den1        = _mm_add_epi32(shifted, STBIR__CONSTI(magic_denorm));
    adjusted2   = _mm_add_epi32(adjusted, adj_infnan);
    den2        = _mm_sub_ps(_mm_castsi128_ps(den1), *(const __m128 *)&magic_denorm);
    adjusted3   = _mm_and_ps(den2, _mm_castsi128_ps(b_isdenorm));
    adjusted4   = _mm_andnot_ps(_mm_castsi128_ps(b_isdenorm), _mm_castsi128_ps(adjusted2));
    adjusted5   = _mm_or_ps(adjusted3, adjusted4);
    sign        = _mm_slli_epi32(justsign, 16);
    final       = _mm_or_ps(adjusted5, _mm_castsi128_ps(sign));
    stbir__simdf_store( output + 4,  final );

    // ~38 SSE2 ops for 8 values
  }

  // Fabian's round-to-nearest-even float to half
  // ~48 SSE2 ops for 8 output
  stbir__inline static void stbir__float_to_half_SIMD(void * output, float const * input)
  {
    static const STBIR__SIMDI_CONST(mask_sign,      0x80000000u);
    static const STBIR__SIMDI_CONST(c_f16max,       (127 + 16) << 23); // all FP32 values >=this round to +inf
    static const STBIR__SIMDI_CONST(c_nanbit,        0x200);
    static const STBIR__SIMDI_CONST(c_infty_as_fp16, 0x7c00);
    static const STBIR__SIMDI_CONST(c_min_normal,    (127 - 14) << 23); // smallest FP32 that yields a normalized FP16
    static const STBIR__SIMDI_CONST(c_subnorm_magic, ((127 - 15) + (23 - 10) + 1) << 23);
    static const STBIR__SIMDI_CONST(c_normal_bias,    0xfff - ((127 - 15) << 23)); // adjust exponent and add mantissa rounding

    __m128  f           =  _mm_loadu_ps(input);
    __m128  msign       = _mm_castsi128_ps(STBIR__CONSTI(mask_sign));
    __m128  justsign    = _mm_and_ps(msign, f);
    __m128  absf        = _mm_xor_ps(f, justsign);
    __m128i absf_int    = _mm_castps_si128(absf); // the cast is "free" (extra bypass latency, but no thruput hit)
    __m128i f16max      = STBIR__CONSTI(c_f16max);
    __m128  b_isnan     = _mm_cmpunord_ps(absf, absf); // is this a NaN?
    __m128i b_isregular = _mm_cmpgt_epi32(f16max, absf_int); // (sub)normalized or special?
    __m128i nanbit      = _mm_and_si128(_mm_castps_si128(b_isnan), STBIR__CONSTI(c_nanbit));
    __m128i inf_or_nan  = _mm_or_si128(nanbit, STBIR__CONSTI(c_infty_as_fp16)); // output for specials

    __m128i min_normal  = STBIR__CONSTI(c_min_normal);
    __m128i b_issub     = _mm_cmpgt_epi32(min_normal, absf_int);

    // "result is subnormal" path
    __m128  subnorm1    = _mm_add_ps(absf, _mm_castsi128_ps(STBIR__CONSTI(c_subnorm_magic))); // magic value to round output mantissa
    __m128i subnorm2    = _mm_sub_epi32(_mm_castps_si128(subnorm1), STBIR__CONSTI(c_subnorm_magic)); // subtract out bias

    // "result is normal" path
    __m128i mantoddbit  = _mm_slli_epi32(absf_int, 31 - 13); // shift bit 13 (mantissa LSB) to sign
    __m128i mantodd     = _mm_srai_epi32(mantoddbit, 31); // -1 if FP16 mantissa odd, else 0

    __m128i round1      = _mm_add_epi32(absf_int, STBIR__CONSTI(c_normal_bias));
    __m128i round2      = _mm_sub_epi32(round1, mantodd); // if mantissa LSB odd, bias towards rounding up (RTNE)
    __m128i normal      = _mm_srli_epi32(round2, 13); // rounded result

    // combine the two non-specials
    __m128i nonspecial  = _mm_or_si128(_mm_and_si128(subnorm2, b_issub), _mm_andnot_si128(b_issub, normal));

    // merge in specials as well
    __m128i joined      = _mm_or_si128(_mm_and_si128(nonspecial, b_isregular), _mm_andnot_si128(b_isregular, inf_or_nan));

    __m128i sign_shift  = _mm_srai_epi32(_mm_castps_si128(justsign), 16);
    __m128i final2, final= _mm_or_si128(joined, sign_shift);

    f           =  _mm_loadu_ps(input+4);
    justsign    = _mm_and_ps(msign, f);
    absf        = _mm_xor_ps(f, justsign);
    absf_int    = _mm_castps_si128(absf); // the cast is "free" (extra bypass latency, but no thruput hit)
    b_isnan     = _mm_cmpunord_ps(absf, absf); // is this a NaN?
    b_isregular = _mm_cmpgt_epi32(f16max, absf_int); // (sub)normalized or special?
    nanbit      = _mm_and_si128(_mm_castps_si128(b_isnan), c_nanbit);
    inf_or_nan  = _mm_or_si128(nanbit, STBIR__CONSTI(c_infty_as_fp16)); // output for specials

    b_issub     = _mm_cmpgt_epi32(min_normal, absf_int);

    // "result is subnormal" path
    subnorm1    = _mm_add_ps(absf, _mm_castsi128_ps(STBIR__CONSTI(c_subnorm_magic))); // magic value to round output mantissa
    subnorm2    = _mm_sub_epi32(_mm_castps_si128(subnorm1), STBIR__CONSTI(c_subnorm_magic)); // subtract out bias

    // "result is normal" path
    mantoddbit  = _mm_slli_epi32(absf_int, 31 - 13); // shift bit 13 (mantissa LSB) to sign
    mantodd     = _mm_srai_epi32(mantoddbit, 31); // -1 if FP16 mantissa odd, else 0

    round1      = _mm_add_epi32(absf_int, STBIR__CONSTI(c_normal_bias));
    round2      = _mm_sub_epi32(round1, mantodd); // if mantissa LSB odd, bias towards rounding up (RTNE)
    normal      = _mm_srli_epi32(round2, 13); // rounded result

    // combine the two non-specials
    nonspecial  = _mm_or_si128(_mm_and_si128(subnorm2, b_issub), _mm_andnot_si128(b_issub, normal));

    // merge in specials as well
    joined      = _mm_or_si128(_mm_and_si128(nonspecial, b_isregular), _mm_andnot_si128(b_isregular, inf_or_nan));

    sign_shift  = _mm_srai_epi32(_mm_castps_si128(justsign), 16);
    final2      = _mm_or_si128(joined, sign_shift);
    final       = _mm_packs_epi32(final, final2);
    stbir__simdi_store( output,final );
  }

#elif defined(STBIR_NEON) && defined(_MSC_VER) && defined(_M_ARM64) && !defined(__clang__) // 64-bit ARM on MSVC (not clang)

  static stbir__inline void stbir__half_to_float_SIMD(float * output, stbir__FP16 const * input)
  {
    float16x4_t in0 = vld1_f16(input + 0);
    float16x4_t in1 = vld1_f16(input + 4);
    vst1q_f32(output + 0, vcvt_f32_f16(in0));
    vst1q_f32(output + 4, vcvt_f32_f16(in1));
  }

  static stbir__inline void stbir__float_to_half_SIMD(stbir__FP16 * output, float const * input)
  {
    float16x4_t out0 = vcvt_f16_f32(vld1q_f32(input + 0));
    float16x4_t out1 = vcvt_f16_f32(vld1q_f32(input + 4));
    vst1_f16(output+0, out0);
    vst1_f16(output+4, out1);
  }

  static stbir__inline float stbir__half_to_float( stbir__FP16 h )
  {
    return vgetq_lane_f32(vcvt_f32_f16(vld1_dup_f16(&h)), 0);
  }

  static stbir__inline stbir__FP16 stbir__float_to_half( float f )
  {
    return vget_lane_f16(vcvt_f16_f32(vdupq_n_f32(f)), 0).n16_u16[0];
  }

#elif defined(STBIR_NEON) && ( defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) ) // 64-bit ARM

  static stbir__inline void stbir__half_to_float_SIMD(float * output, stbir__FP16 const * input)
  {
    float16x8_t in = vld1q_f16(input);
    vst1q_f32(output + 0, vcvt_f32_f16(vget_low_f16(in)));
    vst1q_f32(output + 4, vcvt_f32_f16(vget_high_f16(in)));
  }

  static stbir__inline void stbir__float_to_half_SIMD(stbir__FP16 * output, float const * input)
  {
    float16x4_t out0 = vcvt_f16_f32(vld1q_f32(input + 0));
    float16x4_t out1 = vcvt_f16_f32(vld1q_f32(input + 4));
    vst1q_f16(output, vcombine_f16(out0, out1));
  }

  static stbir__inline float stbir__half_to_float( stbir__FP16 h )
  {
    return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(h)), 0);
  }

  static stbir__inline stbir__FP16 stbir__float_to_half( float f )
  {
    return vget_lane_f16(vcvt_f16_f32(vdupq_n_f32(f)), 0);
  }

#elif defined(STBIR_WASM) || (defined(STBIR_NEON) && (defined(_MSC_VER) || defined(_M_ARM) || defined(__arm__))) // WASM or 32-bit ARM on MSVC/clang

  static stbir__inline void stbir__half_to_float_SIMD(float * output, stbir__FP16 const * input)
  {
    for (int i=0; i<8; i++)
    {
      output[i] = stbir__half_to_float(input[i]);
    }
  }
  static stbir__inline void stbir__float_to_half_SIMD(stbir__FP16 * output, float const * input)
  {
    for (int i=0; i<8; i++)
    {
      output[i] = stbir__float_to_half(input[i]);
    }
  }

#endif


#ifdef STBIR_SIMD

#define stbir__simdf_0123to3333( out, reg ) (out) = stbir__simdf_swiz( reg, 3,3,3,3 )
#define stbir__simdf_0123to2222( out, reg ) (out) = stbir__simdf_swiz( reg, 2,2,2,2 )
#define stbir__simdf_0123to1111( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,1,1 )
#define stbir__simdf_0123to0000( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,0 )
#define stbir__simdf_0123to0003( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,3 )
#define stbir__simdf_0123to0001( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,1 )
#define stbir__simdf_0123to1122( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,2,2 )
#define stbir__simdf_0123to2333( out, reg ) (out) = stbir__simdf_swiz( reg, 2,3,3,3 )
#define stbir__simdf_0123to0023( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,2,3 )
#define stbir__simdf_0123to1230( out, reg ) (out) = stbir__simdf_swiz( reg, 1,2,3,0 )
#define stbir__simdf_0123to2103( out, reg ) (out) = stbir__simdf_swiz( reg, 2,1,0,3 )
#define stbir__simdf_0123to3210( out, reg ) (out) = stbir__simdf_swiz( reg, 3,2,1,0 )
#define stbir__simdf_0123to2301( out, reg ) (out) = stbir__simdf_swiz( reg, 2,3,0,1 )
#define stbir__simdf_0123to3012( out, reg ) (out) = stbir__simdf_swiz( reg, 3,0,1,2 )
#define stbir__simdf_0123to0011( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,1,1 )
#define stbir__simdf_0123to1100( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,0,0 )
#define stbir__simdf_0123to2233( out, reg ) (out) = stbir__simdf_swiz( reg, 2,2,3,3 )
#define stbir__simdf_0123to1133( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,3,3 )
#define stbir__simdf_0123to0022( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,2,2 )
#define stbir__simdf_0123to1032( out, reg ) (out) = stbir__simdf_swiz( reg, 1,0,3,2 )

typedef union stbir__simdi_u32
{
  stbir_uint32 m128i_u32[4];
  int m128i_i32[4];
  stbir__simdi m128i_i128;
} stbir__simdi_u32;

static const int STBIR_mask[9] = { 0,0,0,-1,-1,-1,0,0,0 };

static const STBIR__SIMDF_CONST(STBIR_max_uint8_as_float,           stbir__max_uint8_as_float);
static const STBIR__SIMDF_CONST(STBIR_max_uint16_as_float,          stbir__max_uint16_as_float);
static const STBIR__SIMDF_CONST(STBIR_max_uint8_as_float_inverted,  stbir__max_uint8_as_float_inverted);
static const STBIR__SIMDF_CONST(STBIR_max_uint16_as_float_inverted, stbir__max_uint16_as_float_inverted);

static const STBIR__SIMDF_CONST(STBIR_simd_point5,   0.5f);
static const STBIR__SIMDF_CONST(STBIR_ones,          1.0f);
static const STBIR__SIMDI_CONST(STBIR_almost_zero,   (127 - 13) << 23);
static const STBIR__SIMDI_CONST(STBIR_almost_one,    0x3f7fffff);
static const STBIR__SIMDI_CONST(STBIR_mastissa_mask, 0xff);
static const STBIR__SIMDI_CONST(STBIR_topscale,      0x02000000);

//   Basically, in simd mode, we unroll the proper amount, and we don't want
//   the non-simd remnant loops to be unroll because they only run a few times
//   Adding this switch saves about 5K on clang which is Captain Unroll the 3rd.
#define STBIR_SIMD_STREAMOUT_PTR( star )  STBIR_STREAMOUT_PTR( star )
#define STBIR_SIMD_NO_UNROLL(ptr) STBIR_NO_UNROLL(ptr)
#define STBIR_SIMD_NO_UNROLL_LOOP_START STBIR_NO_UNROLL_LOOP_START
#define STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR STBIR_NO_UNROLL_LOOP_START_INF_FOR

#ifdef STBIR_MEMCPY
#undef STBIR_MEMCPY
#endif
#define STBIR_MEMCPY stbir_simd_memcpy

// override normal use of memcpy with much simpler copy (faster and smaller with our sized copies)
static void stbir_simd_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) d = (char*) dest;
  char STBIR_SIMD_STREAMOUT_PTR( * ) d_end = ((char*) dest) + bytes;
  ptrdiff_t ofs_to_src = (char*)src - (char*)dest;

  // check overlaps
  STBIR_ASSERT( ( ( d >= ( (char*)src) + bytes ) ) || ( ( d + bytes ) <= (char*)src ) );

  if ( bytes < (16*stbir__simdfX_float_count) )
  {
    if ( bytes < 16 )
    {
      if ( bytes )
      {
        STBIR_SIMD_NO_UNROLL_LOOP_START
        do
        {
          STBIR_SIMD_NO_UNROLL(d);
          d[ 0 ] = d[ ofs_to_src ];
          ++d;
        } while ( d < d_end );
      }
    }
    else
    {
      stbir__simdf x;
      // do one unaligned to get us aligned for the stream out below
      stbir__simdf_load( x, ( d + ofs_to_src ) );
      stbir__simdf_store( d, x );
      d = (char*)( ( ( (size_t)d ) + 16 ) & ~15 );

      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        STBIR_SIMD_NO_UNROLL(d);

        if ( d > ( d_end - 16 ) )
        {
          if ( d == d_end )
            return;
          d = d_end - 16;
        }

        stbir__simdf_load( x, ( d + ofs_to_src ) );
        stbir__simdf_store( d, x );
        d += 16;
      }
    }
  }
  else
  {
    stbir__simdfX x0,x1,x2,x3;

    // do one unaligned to get us aligned for the stream out below
    stbir__simdfX_load( x0, ( d + ofs_to_src ) +  0*stbir__simdfX_float_count );
    stbir__simdfX_load( x1, ( d + ofs_to_src ) +  4*stbir__simdfX_float_count );
    stbir__simdfX_load( x2, ( d + ofs_to_src ) +  8*stbir__simdfX_float_count );
    stbir__simdfX_load( x3, ( d + ofs_to_src ) + 12*stbir__simdfX_float_count );
    stbir__simdfX_store( d +  0*stbir__simdfX_float_count, x0 );
    stbir__simdfX_store( d +  4*stbir__simdfX_float_count, x1 );
    stbir__simdfX_store( d +  8*stbir__simdfX_float_count, x2 );
    stbir__simdfX_store( d + 12*stbir__simdfX_float_count, x3 );
    d = (char*)( ( ( (size_t)d ) + (16*stbir__simdfX_float_count) ) & ~((16*stbir__simdfX_float_count)-1) );

    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      STBIR_SIMD_NO_UNROLL(d);

      if ( d > ( d_end - (16*stbir__simdfX_float_count) ) )
      {
        if ( d == d_end )
          return;
        d = d_end - (16*stbir__simdfX_float_count);
      }

      stbir__simdfX_load( x0, ( d + ofs_to_src ) +  0*stbir__simdfX_float_count );
      stbir__simdfX_load( x1, ( d + ofs_to_src ) +  4*stbir__simdfX_float_count );
      stbir__simdfX_load( x2, ( d + ofs_to_src ) +  8*stbir__simdfX_float_count );
      stbir__simdfX_load( x3, ( d + ofs_to_src ) + 12*stbir__simdfX_float_count );
      stbir__simdfX_store( d +  0*stbir__simdfX_float_count, x0 );
      stbir__simdfX_store( d +  4*stbir__simdfX_float_count, x1 );
      stbir__simdfX_store( d +  8*stbir__simdfX_float_count, x2 );
      stbir__simdfX_store( d + 12*stbir__simdfX_float_count, x3 );
      d += (16*stbir__simdfX_float_count);
    }
  }
}

// memcpy that is specically intentionally overlapping (src is smaller then dest, so can be
//   a normal forward copy, bytes is divisible by 4 and bytes is greater than or equal to
//   the diff between dest and src)
static void stbir_overlapping_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) sd = (char*) src;
  char STBIR_SIMD_STREAMOUT_PTR( * ) s_end = ((char*) src) + bytes;
  ptrdiff_t ofs_to_dest = (char*)dest - (char*)src;

  if ( ofs_to_dest >= 16 ) // is the overlap more than 16 away?
  {
    char STBIR_SIMD_STREAMOUT_PTR( * ) s_end16 = ((char*) src) + (bytes&~15);
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do
    {
      stbir__simdf x;
      STBIR_SIMD_NO_UNROLL(sd);
      stbir__simdf_load( x, sd );
      stbir__simdf_store(  ( sd + ofs_to_dest ), x );
      sd += 16;
    } while ( sd < s_end16 );

    if ( sd == s_end )
      return;
  }

  do
  {
    STBIR_SIMD_NO_UNROLL(sd);
    *(int*)( sd + ofs_to_dest ) = *(int*) sd;
    sd += 4;
  } while ( sd < s_end );
}

#else // no SSE2

// when in scalar mode, we let unrolling happen, so this macro just does the __restrict
#define STBIR_SIMD_STREAMOUT_PTR( star ) STBIR_STREAMOUT_PTR( star )
#define STBIR_SIMD_NO_UNROLL(ptr)
#define STBIR_SIMD_NO_UNROLL_LOOP_START
#define STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR

#endif // SSE2


#ifdef STBIR_PROFILE

#ifndef STBIR_PROFILE_FUNC

#if defined(_x86_64) || defined( __x86_64__ ) || defined( _M_X64 ) || defined(__x86_64) || defined(__SSE2__) || defined(STBIR_SSE) || defined( _M_IX86_FP ) || defined(__i386) || defined( __i386__ ) || defined( _M_IX86 ) || defined( _X86_ )

#ifdef _MSC_VER

  STBIRDEF stbir_uint64 __rdtsc();
  #define STBIR_PROFILE_FUNC() __rdtsc()

#else // non msvc

  static stbir__inline stbir_uint64 STBIR_PROFILE_FUNC()
  {
    stbir_uint32 lo, hi;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi) );
    return ( ( (stbir_uint64) hi ) << 32 ) | ( (stbir_uint64) lo );
  }

#endif  // msvc

#elif defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) || defined(__ARM_NEON__)

#if defined( _MSC_VER ) && !defined(__clang__)

  #define STBIR_PROFILE_FUNC() _ReadStatusReg(ARM64_CNTVCT)

#else

  static stbir__inline stbir_uint64 STBIR_PROFILE_FUNC()
  {
    stbir_uint64 tsc;
    asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
    return tsc;
  }

#endif

#else // x64, arm

#error Unknown platform for profiling.

#endif  // x64, arm

#endif // STBIR_PROFILE_FUNC

#define STBIR_ONLY_PROFILE_GET_SPLIT_INFO ,stbir__per_split_info * split_info
#define STBIR_ONLY_PROFILE_SET_SPLIT_INFO ,split_info

#define STBIR_ONLY_PROFILE_BUILD_GET_INFO ,stbir__info * profile_info
#define STBIR_ONLY_PROFILE_BUILD_SET_INFO ,profile_info

// super light-weight micro profiler
#define STBIR_PROFILE_START_ll( info, wh ) { stbir_uint64 wh##thiszonetime = STBIR_PROFILE_FUNC(); stbir_uint64 * wh##save_parent_excluded_ptr = info->current_zone_excluded_ptr; stbir_uint64 wh##current_zone_excluded = 0; info->current_zone_excluded_ptr = &wh##current_zone_excluded;
#define STBIR_PROFILE_END_ll( info, wh ) wh##thiszonetime = STBIR_PROFILE_FUNC() - wh##thiszonetime; info->profile.named.wh += wh##thiszonetime - wh##current_zone_excluded; *wh##save_parent_excluded_ptr += wh##thiszonetime; info->current_zone_excluded_ptr = wh##save_parent_excluded_ptr; }
#define STBIR_PROFILE_FIRST_START_ll( info, wh ) { int i; info->current_zone_excluded_ptr = &info->profile.named.total; for(i=0;i<STBIR__ARRAY_SIZE(info->profile.array);i++) info->profile.array[i]=0; } STBIR_PROFILE_START_ll( info, wh );
#define STBIR_PROFILE_CLEAR_EXTRAS_ll( info, num ) { int extra; for(extra=1;extra<(num);extra++) { int i; for(i=0;i<STBIR__ARRAY_SIZE((info)->profile.array);i++) (info)[extra].profile.array[i]=0; } }

// for thread data
#define STBIR_PROFILE_START( wh ) STBIR_PROFILE_START_ll( split_info, wh )
#define STBIR_PROFILE_END( wh ) STBIR_PROFILE_END_ll( split_info, wh )
#define STBIR_PROFILE_FIRST_START( wh ) STBIR_PROFILE_FIRST_START_ll( split_info, wh )
#define STBIR_PROFILE_CLEAR_EXTRAS() STBIR_PROFILE_CLEAR_EXTRAS_ll( split_info, split_count )

// for build data
#define STBIR_PROFILE_BUILD_START( wh ) STBIR_PROFILE_START_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_END( wh ) STBIR_PROFILE_END_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_FIRST_START( wh ) STBIR_PROFILE_FIRST_START_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_CLEAR( info ) { int i; for(i=0;i<STBIR__ARRAY_SIZE(info->profile.array);i++) info->profile.array[i]=0; }

#else  // no profile

#define STBIR_ONLY_PROFILE_GET_SPLIT_INFO
#define STBIR_ONLY_PROFILE_SET_SPLIT_INFO

#define STBIR_ONLY_PROFILE_BUILD_GET_INFO
#define STBIR_ONLY_PROFILE_BUILD_SET_INFO

#define STBIR_PROFILE_START( wh )
#define STBIR_PROFILE_END( wh )
#define STBIR_PROFILE_FIRST_START( wh )
#define STBIR_PROFILE_CLEAR_EXTRAS( )

#define STBIR_PROFILE_BUILD_START( wh )
#define STBIR_PROFILE_BUILD_END( wh )
#define STBIR_PROFILE_BUILD_FIRST_START( wh )
#define STBIR_PROFILE_BUILD_CLEAR( info )

#endif  // stbir_profile

#ifndef STBIR_CEILF
#include <math.h>
#if _MSC_VER <= 1200 // support VC6 for Sean
#define STBIR_CEILF(x) ((float)ceil((float)(x)))
#define STBIR_FLOORF(x) ((float)floor((float)(x)))
#else
#define STBIR_CEILF(x) ceilf(x)
#define STBIR_FLOORF(x) floorf(x)
#endif
#endif

#ifndef STBIR_MEMCPY
// For memcpy
#include <string.h>
#define STBIR_MEMCPY( dest, src, len ) memcpy( dest, src, len )
#endif

#ifndef STBIR_SIMD

// memcpy that is specifically intentionally overlapping (src is smaller then dest, so can be
//   a normal forward copy, bytes is divisible by 4 and bytes is greater than or equal to
//   the diff between dest and src)
static void stbir_overlapping_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) sd = (char*) src;
  char STBIR_SIMD_STREAMOUT_PTR( * ) s_end = ((char*) src) + bytes;
  ptrdiff_t ofs_to_dest = (char*)dest - (char*)src;

  if ( ofs_to_dest >= 8 ) // is the overlap more than 8 away?
  {
    char STBIR_SIMD_STREAMOUT_PTR( * ) s_end8 = ((char*) src) + (bytes&~7);
    STBIR_NO_UNROLL_LOOP_START
    do
    {
      STBIR_NO_UNROLL(sd);
      *(stbir_uint64*)( sd + ofs_to_dest ) = *(stbir_uint64*) sd;
      sd += 8;
    } while ( sd < s_end8 );

    if ( sd == s_end )
      return;
  }

  STBIR_NO_UNROLL_LOOP_START
  do
  {
    STBIR_NO_UNROLL(sd);
    *(int*)( sd + ofs_to_dest ) = *(int*) sd;
    sd += 4;
  } while ( sd < s_end );
}

#endif

static float stbir__filter_trapezoid(float x, float scale, void * user_data)
{
  float halfscale = scale / 2;
  float t = 0.5f + halfscale;
  STBIR_ASSERT(scale <= 1);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x >= t)
    return 0.0f;
  else
  {
    float r = 0.5f - halfscale;
    if (x <= r)
      return 1.0f;
    else
      return (t - x) / scale;
  }
}

static float stbir__support_trapezoid(float scale, void * user_data)
{
  STBIR__UNUSED(user_data);
  return 0.5f + scale / 2.0f;
}

static float stbir__filter_triangle(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x <= 1.0f)
    return 1.0f - x;
  else
    return 0.0f;
}

static float stbir__filter_point(float x, float s, void * user_data)
{
  STBIR__UNUSED(x);
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  return 1.0f;
}

static float stbir__filter_cubic(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return (4.0f + x*x*(3.0f*x - 6.0f))/6.0f;
  else if (x < 2.0f)
    return (8.0f + x*(-12.0f + x*(6.0f - x)))/6.0f;

  return (0.0f);
}

static float stbir__filter_catmullrom(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return 1.0f - x*x*(2.5f - 1.5f*x);
  else if (x < 2.0f)
    return 2.0f - x*(4.0f + x*(0.5f*x - 2.5f));

  return (0.0f);
}

static float stbir__filter_mitchell(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return (16.0f + x*x*(21.0f * x - 36.0f))/18.0f;
  else if (x < 2.0f)
    return (32.0f + x*(-60.0f + x*(36.0f - 7.0f*x)))/18.0f;

  return (0.0f);
}

static float stbir__support_zeropoint5(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 0.5f;
}

static float stbir__support_one(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 1;
}

static float stbir__support_two(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 2;
}

// This is the maximum number of input samples that can affect an output sample
// with the given filter from the output pixel's perspective
static int stbir__get_filter_pixel_width(stbir__support_callback * support, float scale, void * user_data)
{
  STBIR_ASSERT(support != 0);

  if ( scale >= ( 1.0f-stbir__small_float ) ) // upscale
    return (int)STBIR_CEILF(support(1.0f/scale,user_data) * 2.0f);
  else
    return (int)STBIR_CEILF(support(scale,user_data) * 2.0f / scale);
}

// this is how many coefficents per run of the filter (which is different
//   from the filter_pixel_width depending on if we are scattering or gathering)
static int stbir__get_coefficient_width(stbir__sampler * samp, int is_gather, void * user_data)
{
  float scale = samp->scale_info.scale;
  stbir__support_callback * support = samp->filter_support;

  switch( is_gather )
  {
    case 1:
      return (int)STBIR_CEILF(support(1.0f / scale, user_data) * 2.0f);
    case 2:
      return (int)STBIR_CEILF(support(scale, user_data) * 2.0f / scale);
    case 0:
      return (int)STBIR_CEILF(support(scale, user_data) * 2.0f);
    default:
      STBIR_ASSERT( (is_gather >= 0 ) && (is_gather <= 2 ) );
      return 0;
  }
}

static int stbir__get_contributors(stbir__sampler * samp, int is_gather)
{
  if (is_gather)
      return samp->scale_info.output_sub_size;
  else
      return (samp->scale_info.input_full_size + samp->filter_pixel_margin * 2);
}

static int stbir__edge_zero_full( int n, int max )
{
  STBIR__UNUSED(n);
  STBIR__UNUSED(max);
  return 0; // NOTREACHED
}

static int stbir__edge_clamp_full( int n, int max )
{
  if (n < 0)
    return 0;

  if (n >= max)
    return max - 1;

  return n; // NOTREACHED
}

static int stbir__edge_reflect_full( int n, int max )
{
  if (n < 0)
  {
    if (n > -max)
      return -n;
    else
      return max - 1;
  }

  if (n >= max)
  {
    int max2 = max * 2;
    if (n >= max2)
      return 0;
    else
      return max2 - n - 1;
  }

  return n; // NOTREACHED
}

static int stbir__edge_wrap_full( int n, int max )
{
  if (n >= 0)
    return (n % max);
  else
  {
    int m = (-n) % max;

    if (m != 0)
      m = max - m;

    return (m);
  }
}

typedef int stbir__edge_wrap_func( int n, int max );
static stbir__edge_wrap_func * stbir__edge_wrap_slow[] =
{
  stbir__edge_clamp_full,    // STBIR_EDGE_CLAMP
  stbir__edge_reflect_full,  // STBIR_EDGE_REFLECT
  stbir__edge_wrap_full,     // STBIR_EDGE_WRAP
  stbir__edge_zero_full,     // STBIR_EDGE_ZERO
};

stbir__inline static int stbir__edge_wrap(stbir_edge edge, int n, int max)
{
  // avoid per-pixel switch
  if (n >= 0 && n < max)
      return n;
  return stbir__edge_wrap_slow[edge]( n, max );
}

#define STBIR__MERGE_RUNS_PIXEL_THRESHOLD 16

// get information on the extents of a sampler
static void stbir__get_extents( stbir__sampler * samp, stbir__extents * scanline_extents )
{
  int j, stop;
  int left_margin, right_margin;
  int min_n = 0x7fffffff, max_n = -0x7fffffff;
  int min_left = 0x7fffffff, max_left = -0x7fffffff;
  int min_right = 0x7fffffff, max_right = -0x7fffffff;
  stbir_edge edge = samp->edge;
  stbir__contributors* contributors = samp->contributors;
  int output_sub_size = samp->scale_info.output_sub_size;
  int input_full_size = samp->scale_info.input_full_size;
  int filter_pixel_margin = samp->filter_pixel_margin;

  STBIR_ASSERT( samp->is_gather );

  stop = output_sub_size;
  for (j = 0; j < stop; j++ )
  {
    STBIR_ASSERT( contributors[j].n1 >= contributors[j].n0 );
    if ( contributors[j].n0 < min_n )
    {
      min_n = contributors[j].n0;
      stop = j + filter_pixel_margin;  // if we find a new min, only scan another filter width
      if ( stop > output_sub_size ) stop = output_sub_size;
    }
  }

  stop = 0;
  for (j = output_sub_size - 1; j >= stop; j-- )
  {
    STBIR_ASSERT( contributors[j].n1 >= contributors[j].n0 );
    if ( contributors[j].n1 > max_n )
    {
      max_n = contributors[j].n1;
      stop = j - filter_pixel_margin;  // if we find a new max, only scan another filter width
      if (stop<0) stop = 0;
    }
  }

  STBIR_ASSERT( scanline_extents->conservative.n0 <= min_n );
  STBIR_ASSERT( scanline_extents->conservative.n1 >= max_n );

  // now calculate how much into the margins we really read
  left_margin = 0;
  if ( min_n < 0 )
  {
    left_margin = -min_n;
    min_n = 0;
  }

  right_margin = 0;
  if ( max_n >= input_full_size )
  {
    right_margin = max_n - input_full_size + 1;
    max_n = input_full_size - 1;
  }

  // index 1 is margin pixel extents (how many pixels we hang over the edge)
  scanline_extents->edge_sizes[0] = left_margin;
  scanline_extents->edge_sizes[1] = right_margin;

  // index 2 is pixels read from the input
  scanline_extents->spans[0].n0 = min_n;
  scanline_extents->spans[0].n1 = max_n;
  scanline_extents->spans[0].pixel_offset_for_input = min_n;

  // default to no other input range
  scanline_extents->spans[1].n0 = 0;
  scanline_extents->spans[1].n1 = -1;
  scanline_extents->spans[1].pixel_offset_for_input = 0;

  // don't have to do edge calc for zero clamp
  if ( edge == STBIR_EDGE_ZERO )
    return;

  // convert margin pixels to the pixels within the input (min and max)
  for( j = -left_margin ; j < 0 ; j++ )
  {
      int p = stbir__edge_wrap( edge, j, input_full_size );
      if ( p < min_left )
        min_left = p;
      if ( p > max_left )
        max_left = p;
  }

  for( j = input_full_size ; j < (input_full_size + right_margin) ; j++ )
  {
      int p = stbir__edge_wrap( edge, j, input_full_size );
      if ( p < min_right )
        min_right = p;
      if ( p > max_right )
        max_right = p;
  }

  // merge the left margin pixel region if it connects within 4 pixels of main pixel region
  if ( min_left != 0x7fffffff )
  {
    if ( ( ( min_left <= min_n ) && ( ( max_left  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= min_n ) ) ||
         ( ( min_n <= min_left ) && ( ( max_n  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= max_left ) ) )
    {
      scanline_extents->spans[0].n0 = min_n = stbir__min( min_n, min_left );
      scanline_extents->spans[0].n1 = max_n = stbir__max( max_n, max_left );
      scanline_extents->spans[0].pixel_offset_for_input = min_n;
      left_margin = 0;
    }
  }

  // merge the right margin pixel region if it connects within 4 pixels of main pixel region
  if ( min_right != 0x7fffffff )
  {
    if ( ( ( min_right <= min_n ) && ( ( max_right  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= min_n ) ) ||
         ( ( min_n <= min_right ) && ( ( max_n  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= max_right ) ) )
    {
      scanline_extents->spans[0].n0 = min_n = stbir__min( min_n, min_right );
      scanline_extents->spans[0].n1 = max_n = stbir__max( max_n, max_right );
      scanline_extents->spans[0].pixel_offset_for_input = min_n;
      right_margin = 0;
    }
  }

  STBIR_ASSERT( scanline_extents->conservative.n0 <= min_n );
  STBIR_ASSERT( scanline_extents->conservative.n1 >= max_n );

  // you get two ranges when you have the WRAP edge mode and you are doing just the a piece of the resize
  //   so you need to get a second run of pixels from the opposite side of the scanline (which you
  //   wouldn't need except for WRAP)


  // if we can't merge the min_left range, add it as a second range
  if ( ( left_margin ) && ( min_left != 0x7fffffff ) )
  {
    stbir__span * newspan = scanline_extents->spans + 1;
    STBIR_ASSERT( right_margin == 0 );
    if ( min_left < scanline_extents->spans[0].n0 )
    {
      scanline_extents->spans[1].pixel_offset_for_input = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n0 = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n1 = scanline_extents->spans[0].n1;
      --newspan;
    }
    newspan->pixel_offset_for_input = min_left;
    newspan->n0 = -left_margin;
    newspan->n1 = ( max_left - min_left ) - left_margin;
    scanline_extents->edge_sizes[0] = 0;  // don't need to copy the left margin, since we are directly decoding into the margin
  }
  // if we can't merge the min_left range, add it as a second range
  else  
  if ( ( right_margin ) && ( min_right != 0x7fffffff ) )
  {
    stbir__span * newspan = scanline_extents->spans + 1;
    if ( min_right < scanline_extents->spans[0].n0 )
    {
      scanline_extents->spans[1].pixel_offset_for_input = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n0 = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n1 = scanline_extents->spans[0].n1;
      --newspan;
    }
    newspan->pixel_offset_for_input = min_right;
    newspan->n0 = scanline_extents->spans[1].n1 + 1;
    newspan->n1 = scanline_extents->spans[1].n1 + 1 + ( max_right - min_right );
    scanline_extents->edge_sizes[1] = 0;  // don't need to copy the right margin, since we are directly decoding into the margin
  }

  // sort the spans into write output order
  if ( ( scanline_extents->spans[1].n1 > scanline_extents->spans[1].n0 ) && ( scanline_extents->spans[0].n0 > scanline_extents->spans[1].n0 ) )
  {
    stbir__span tspan = scanline_extents->spans[0];
    scanline_extents->spans[0] = scanline_extents->spans[1];
    scanline_extents->spans[1] = tspan;
  }
}

static void stbir__calculate_in_pixel_range( int * first_pixel, int * last_pixel, float out_pixel_center, float out_filter_radius, float inv_scale, float out_shift, int input_size, stbir_edge edge )
{
  int first, last;
  float out_pixel_influence_lowerbound = out_pixel_center - out_filter_radius;
  float out_pixel_influence_upperbound = out_pixel_center + out_filter_radius;

  float in_pixel_influence_lowerbound = (out_pixel_influence_lowerbound + out_shift) * inv_scale;
  float in_pixel_influence_upperbound = (out_pixel_influence_upperbound + out_shift) * inv_scale;

  first = (int)(STBIR_FLOORF(in_pixel_influence_lowerbound + 0.5f));
  last = (int)(STBIR_FLOORF(in_pixel_influence_upperbound - 0.5f));
  if ( last < first ) last = first; // point sample mode can span a value *right* at 0.5, and cause these to cross

  if ( edge == STBIR_EDGE_WRAP )
  {
    if ( first < -input_size )
      first = -input_size;
    if ( last >= (input_size*2))
      last = (input_size*2) - 1;
  }

  *first_pixel = first;
  *last_pixel = last;
}

static void stbir__calculate_coefficients_for_gather_upsample( float out_filter_radius, stbir__kernel_callback * kernel, stbir__scale_info * scale_info, int num_contributors, stbir__contributors* contributors, float* coefficient_group, int coefficient_width, stbir_edge edge, void * user_data )
{
  int n, end;
  float inv_scale = scale_info->inv_scale;
  float out_shift = scale_info->pixel_shift;
  int input_size  = scale_info->input_full_size;
  int numerator = scale_info->scale_numerator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < num_contributors ) );

  // Looping through out pixels
  end = num_contributors; if ( polyphase ) end = numerator;
  for (n = 0; n < end; n++)
  {
    int i;
    int last_non_zero;
    float out_pixel_center = (float)n + 0.5f;
    float in_center_of_out = (out_pixel_center + out_shift) * inv_scale;

    int in_first_pixel, in_last_pixel;

    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, out_pixel_center, out_filter_radius, inv_scale, out_shift, input_size, edge );

    // make sure we never generate a range larger than our precalculated coeff width
    //   this only happens in point sample mode, but it's a good safe thing to do anyway
    if ( ( in_last_pixel - in_first_pixel + 1 ) > coefficient_width )
      in_last_pixel = in_first_pixel + coefficient_width - 1;

    last_non_zero = -1;
    for (i = 0; i <= in_last_pixel - in_first_pixel; i++)
    {
      float in_pixel_center = (float)(i + in_first_pixel) + 0.5f;
      float coeff = kernel(in_center_of_out - in_pixel_center, inv_scale, user_data);

      // kill denormals
      if ( ( ( coeff < stbir__small_float ) && ( coeff > -stbir__small_float ) ) )
      {
        if ( i == 0 )  // if we're at the front, just eat zero contributors
        {
          STBIR_ASSERT ( ( in_last_pixel - in_first_pixel ) != 0 ); // there should be at least one contrib
          ++in_first_pixel;
          i--;
          continue;
        }
        coeff = 0;  // make sure is fully zero (should keep denormals away)
      }
      else
        last_non_zero = i;

      coefficient_group[i] = coeff;
    }

    in_last_pixel = last_non_zero+in_first_pixel; // kills trailing zeros
    contributors->n0 = in_first_pixel;
    contributors->n1 = in_last_pixel;

    STBIR_ASSERT(contributors->n1 >= contributors->n0);

    ++contributors;
    coefficient_group += coefficient_width;
  }
}

static void stbir__insert_coeff( stbir__contributors * contribs, float * coeffs, int new_pixel, float new_coeff, int max_width )
{
  if ( new_pixel <= contribs->n1 )  // before the end
  {
    if ( new_pixel < contribs->n0 ) // before the front?
    {
      if ( ( contribs->n1 - new_pixel + 1 ) <= max_width )
      { 
        int j, o = contribs->n0 - new_pixel;
        for ( j = contribs->n1 - contribs->n0 ; j <= 0 ; j-- )
          coeffs[ j + o ] = coeffs[ j ];
        for ( j = 1 ; j < o ; j-- )
          coeffs[ j ] = coeffs[ 0 ];
        coeffs[ 0 ] = new_coeff;
        contribs->n0 = new_pixel;
      }
    }
    else
    {
      coeffs[ new_pixel - contribs->n0 ] += new_coeff;
    }
  }
  else
  {
    if ( ( new_pixel - contribs->n0 + 1 ) <= max_width )
    {
      int j, e = new_pixel - contribs->n0;
      for( j = ( contribs->n1 - contribs->n0 ) + 1 ; j < e ; j++ ) // clear in-betweens coeffs if there are any
        coeffs[j] = 0;

      coeffs[ e ] = new_coeff;
      contribs->n1 = new_pixel;
    }
  }
}

static void stbir__calculate_out_pixel_range( int * first_pixel, int * last_pixel, float in_pixel_center, float in_pixels_radius, float scale, float out_shift, int out_size )
{
  float in_pixel_influence_lowerbound = in_pixel_center - in_pixels_radius;
  float in_pixel_influence_upperbound = in_pixel_center + in_pixels_radius;
  float out_pixel_influence_lowerbound = in_pixel_influence_lowerbound * scale - out_shift;
  float out_pixel_influence_upperbound = in_pixel_influence_upperbound * scale - out_shift;
  int out_first_pixel = (int)(STBIR_FLOORF(out_pixel_influence_lowerbound + 0.5f));
  int out_last_pixel = (int)(STBIR_FLOORF(out_pixel_influence_upperbound - 0.5f));

  if ( out_first_pixel < 0 )
    out_first_pixel = 0;
  if ( out_last_pixel >= out_size )
    out_last_pixel = out_size - 1;
  *first_pixel = out_first_pixel;
  *last_pixel = out_last_pixel;
}

static void stbir__calculate_coefficients_for_gather_downsample( int start, int end, float in_pixels_radius, stbir__kernel_callback * kernel, stbir__scale_info * scale_info, int coefficient_width, int num_contributors, stbir__contributors * contributors, float * coefficient_group, void * user_data )
{
  int in_pixel;
  int i;
  int first_out_inited = -1;
  float scale = scale_info->scale;
  float out_shift = scale_info->pixel_shift;
  int out_size = scale_info->output_sub_size;
  int numerator = scale_info->scale_numerator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < out_size ) );

  STBIR__UNUSED(num_contributors);

  // Loop through the input pixels
  for (in_pixel = start; in_pixel < end; in_pixel++)
  {
    float in_pixel_center = (float)in_pixel + 0.5f;
    float out_center_of_in = in_pixel_center * scale - out_shift;
    int out_first_pixel, out_last_pixel;

    stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, in_pixel_center, in_pixels_radius, scale, out_shift, out_size );

    if ( out_first_pixel > out_last_pixel )
      continue;

    // clamp or exit if we are using polyphase filtering, and the limit is up
    if ( polyphase )
    {
      // when polyphase, you only have to do coeffs up to the numerator count
      if ( out_first_pixel == numerator )
        break;

      // don't do any extra work, clamp last pixel at numerator too
      if ( out_last_pixel >= numerator )
        out_last_pixel = numerator - 1;
    }

    for (i = 0; i <= out_last_pixel - out_first_pixel; i++)
    {
      float out_pixel_center = (float)(i + out_first_pixel) + 0.5f;
      float x = out_pixel_center - out_center_of_in;
      float coeff = kernel(x, scale, user_data) * scale;

      // kill the coeff if it's too small (avoid denormals)
      if ( ( ( coeff < stbir__small_float ) && ( coeff > -stbir__small_float ) ) )
        coeff = 0.0f;

      {
        int out = i + out_first_pixel;
        float * coeffs = coefficient_group + out * coefficient_width;
        stbir__contributors * contribs = contributors + out;

        // is this the first time this output pixel has been seen?  Init it.
        if ( out > first_out_inited )
        {
          STBIR_ASSERT( out == ( first_out_inited + 1 ) ); // ensure we have only advanced one at time
          first_out_inited = out;
          contribs->n0 = in_pixel;
          contribs->n1 = in_pixel;
          coeffs[0]  = coeff;
        }
        else
        {
          // insert on end (always in order)
          if ( coeffs[0] == 0.0f )  // if the first coefficent is zero, then zap it for this coeffs
          {
            STBIR_ASSERT( ( in_pixel - contribs->n0 ) == 1 ); // ensure that when we zap, we're at the 2nd pos
            contribs->n0 = in_pixel;
          }
          contribs->n1 = in_pixel;
          STBIR_ASSERT( ( in_pixel - contribs->n0 ) < coefficient_width );
          coeffs[in_pixel - contribs->n0]  = coeff;
        }
      }
    }
  }
}

#ifdef STBIR_RENORMALIZE_IN_FLOAT
#define STBIR_RENORM_TYPE float
#else
#define STBIR_RENORM_TYPE double
#endif

static void stbir__cleanup_gathered_coefficients( stbir_edge edge, stbir__filter_extent_info* filter_info, stbir__scale_info * scale_info, int num_contributors, stbir__contributors* contributors, float * coefficient_group, int coefficient_width )
{
  int input_size = scale_info->input_full_size;
  int input_last_n1 = input_size - 1;
  int n, end;
  int lowest = 0x7fffffff;
  int highest = -0x7fffffff;
  int widest = -1;
  int numerator = scale_info->scale_numerator;
  int denominator = scale_info->scale_denominator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < num_contributors ) );
  float * coeffs;
  stbir__contributors * contribs;

  // weight all the coeffs for each sample
  coeffs = coefficient_group;
  contribs = contributors;
  end = num_contributors; if ( polyphase ) end = numerator;
  for (n = 0; n < end; n++)
  {
    int i;
    STBIR_RENORM_TYPE filter_scale, total_filter = 0;
    int e;

    // add all contribs
    e = contribs->n1 - contribs->n0;
    for( i = 0 ; i <= e ; i++ )
    {
      total_filter += (STBIR_RENORM_TYPE) coeffs[i];
      STBIR_ASSERT( ( coeffs[i] >= -2.0f ) && ( coeffs[i] <= 2.0f )  ); // check for wonky weights
    }

    // rescale
    if ( ( total_filter < stbir__small_float ) && ( total_filter > -stbir__small_float ) )
    {
      // all coeffs are extremely small, just zero it
      contribs->n1 = contribs->n0;
      coeffs[0] = 0.0f;
    }
    else
    {
      // if the total isn't 1.0, rescale everything
      if ( ( total_filter < (1.0f-stbir__small_float) ) || ( total_filter > (1.0f+stbir__small_float) ) )
      {
        filter_scale = ((STBIR_RENORM_TYPE)1.0) / total_filter;

        // scale them all
        for (i = 0; i <= e; i++)
          coeffs[i] = (float) ( coeffs[i] * filter_scale );
      }
    }
    ++contribs;
    coeffs += coefficient_width;
  }

  // if we have a rational for the scale, we can exploit the polyphaseness to not calculate
  //   most of the coefficients, so we copy them here
  if ( polyphase )
  {
    stbir__contributors * prev_contribs = contributors;
    stbir__contributors * cur_contribs = contributors + numerator;

    for( n = numerator ; n < num_contributors ; n++ )
    {
      cur_contribs->n0 = prev_contribs->n0 + denominator;
      cur_contribs->n1 = prev_contribs->n1 + denominator;
      ++cur_contribs;
      ++prev_contribs;
    }
    stbir_overlapping_memcpy( coefficient_group + numerator * coefficient_width, coefficient_group, ( num_contributors - numerator ) * coefficient_width * sizeof( coeffs[ 0 ] ) );
  }

  coeffs = coefficient_group;
  contribs = contributors;

  for (n = 0; n < num_contributors; n++)
  {
    int i;

    // in zero edge mode, just remove out of bounds contribs completely (since their weights are accounted for now)
    if ( edge == STBIR_EDGE_ZERO )
    {
      // shrink the right side if necessary
      if ( contribs->n1 > input_last_n1 )
        contribs->n1 = input_last_n1;

      // shrink the left side
      if ( contribs->n0 < 0 )
      {
        int j, left, skips = 0;

        skips = -contribs->n0;
        contribs->n0 = 0;

        // now move down the weights
        left = contribs->n1 - contribs->n0 + 1;
        if ( left > 0 )
        {
          for( j = 0 ; j < left ; j++ )
            coeffs[ j ] = coeffs[ j + skips ];
        }
      }
    }
    else if ( ( edge == STBIR_EDGE_CLAMP ) || ( edge == STBIR_EDGE_REFLECT ) )
    {
      // for clamp and reflect, calculate the true inbounds position (based on edge type) and just add that to the existing weight

      // right hand side first
      if ( contribs->n1 > input_last_n1 )
      {
        int start = contribs->n0;
        int endi = contribs->n1;
        contribs->n1 = input_last_n1;
        for( i = input_size; i <= endi; i++ )
          stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( i, input_size ), coeffs[i-start], coefficient_width );
      }

      // now check left hand edge
      if ( contribs->n0 < 0 )
      {
        int save_n0;
        float save_n0_coeff;
        float * c = coeffs - ( contribs->n0 + 1 );

        // reinsert the coeffs with it reflected or clamped (insert accumulates, if the coeffs exist)
        for( i = -1 ; i > contribs->n0 ; i-- )
          stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( i, input_size ), *c--, coefficient_width );
        save_n0 = contribs->n0;
        save_n0_coeff = c[0]; // save it, since we didn't do the final one (i==n0), because there might be too many coeffs to hold (before we resize)!

        // now slide all the coeffs down (since we have accumulated them in the positive contribs) and reset the first contrib
        contribs->n0 = 0;
        for(i = 0 ; i <= contribs->n1 ; i++ )
          coeffs[i] = coeffs[i-save_n0];

        // now that we have shrunk down the contribs, we insert the first one safely
        stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( save_n0, input_size ), save_n0_coeff, coefficient_width );
      }
    }

    if ( contribs->n0 <= contribs->n1 )
    {
      int diff = contribs->n1 - contribs->n0 + 1;
      while ( diff && ( coeffs[ diff-1 ] == 0.0f ) )
        --diff;

      contribs->n1 = contribs->n0 + diff - 1;

      if ( contribs->n0 <= contribs->n1 )
      {
        if ( contribs->n0 < lowest )
          lowest = contribs->n0;
        if ( contribs->n1 > highest )
          highest = contribs->n1;
        if ( diff > widest )
          widest = diff;
      }

      // re-zero out unused coefficients (if any)
      for( i = diff ; i < coefficient_width ; i++ )
        coeffs[i] = 0.0f;
    }

    ++contribs;
    coeffs += coefficient_width;
  }
  filter_info->lowest = lowest;
  filter_info->highest = highest;
  filter_info->widest = widest;
}

#undef STBIR_RENORM_TYPE 

static int stbir__pack_coefficients( int num_contributors, stbir__contributors* contributors, float * coefficents, int coefficient_width, int widest, int row0, int row1 ) 
{
  #define STBIR_MOVE_1( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint32*)(dest))[0] = ((stbir_uint32*)(src))[0]; }
  #define STBIR_MOVE_2( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint64*)(dest))[0] = ((stbir_uint64*)(src))[0]; }
  #ifdef STBIR_SIMD
  #define STBIR_MOVE_4( dest, src ) { stbir__simdf t; STBIR_NO_UNROLL(dest); stbir__simdf_load( t, src ); stbir__simdf_store( dest, t ); }
  #else
  #define STBIR_MOVE_4( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint64*)(dest))[0] = ((stbir_uint64*)(src))[0]; ((stbir_uint64*)(dest))[1] = ((stbir_uint64*)(src))[1]; }
  #endif

  int row_end = row1 + 1;
  STBIR__UNUSED( row0 ); // only used in an assert

  if ( coefficient_width != widest )
  {
    float * pc = coefficents;
    float * coeffs = coefficents;
    float * pc_end = coefficents + num_contributors * widest;
    switch( widest )
    {
      case 1:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_1( pc, coeffs );
          ++pc;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 2:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_2( pc, coeffs );
          pc += 2;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 3:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_2( pc, coeffs );
          STBIR_MOVE_1( pc+2, coeffs+2 );
          pc += 3;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 4:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          pc += 4;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 5:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_1( pc+4, coeffs+4 );
          pc += 5;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 6:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_2( pc+4, coeffs+4 );
          pc += 6;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 7:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_2( pc+4, coeffs+4 );
          STBIR_MOVE_1( pc+6, coeffs+6 );
          pc += 7;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 8:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          pc += 8;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 9:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_1( pc+8, coeffs+8 );
          pc += 9;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 10:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_2( pc+8, coeffs+8 );
          pc += 10;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 11:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_2( pc+8, coeffs+8 );
          STBIR_MOVE_1( pc+10, coeffs+10 );
          pc += 11;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 12:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_4( pc+8, coeffs+8 );
          pc += 12;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      default:
        STBIR_NO_UNROLL_LOOP_START
        do {
          float * copy_end = pc + widest - 4;
          float * c = coeffs;
          do {
            STBIR_NO_UNROLL( pc );
            STBIR_MOVE_4( pc, c );
            pc += 4;
            c += 4;
          } while ( pc <= copy_end );
          copy_end += 4;
          STBIR_NO_UNROLL_LOOP_START
          while ( pc < copy_end )
          {
            STBIR_MOVE_1( pc, c );
            ++pc; ++c;
          }
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
    }
  }

  // some horizontal routines read one float off the end (which is then masked off), so put in a sentinal so we don't read an snan or denormal
  coefficents[ widest * num_contributors ] = 8888.0f;

  // the minimum we might read for unrolled filters widths is 12. So, we need to
  //   make sure we never read outside the decode buffer, by possibly moving
  //   the sample area back into the scanline, and putting zeros weights first.
  // we start on the right edge and check until we're well past the possible
  //   clip area (2*widest).
  {
    stbir__contributors * contribs = contributors + num_contributors - 1;
    float * coeffs = coefficents + widest * ( num_contributors - 1 );

    // go until no chance of clipping (this is usually less than 8 lops)
    while ( ( contribs >= contributors ) && ( ( contribs->n0 + widest*2 ) >= row_end ) )
    {
      // might we clip??
      if ( ( contribs->n0 + widest ) > row_end )
      {
        int stop_range = widest;

        // if range is larger than 12, it will be handled by generic loops that can terminate on the exact length
        //   of this contrib n1, instead of a fixed widest amount - so calculate this
        if ( widest > 12 )
        {
          int mod;

          // how far will be read in the n_coeff loop (which depends on the widest count mod4);
          mod = widest & 3;
          stop_range = ( ( ( contribs->n1 - contribs->n0 + 1 ) - mod + 3 ) & ~3 ) + mod;

          // the n_coeff loops do a minimum amount of coeffs, so factor that in!
          if ( stop_range < ( 8 + mod ) ) stop_range = 8 + mod;
        }

        // now see if we still clip with the refined range
        if ( ( contribs->n0 + stop_range ) > row_end )
        {
          int new_n0 = row_end - stop_range;
          int num = contribs->n1 - contribs->n0 + 1;
          int backup = contribs->n0 - new_n0;
          float * from_co = coeffs + num - 1;
          float * to_co = from_co + backup;

          STBIR_ASSERT( ( new_n0 >= row0 ) && ( new_n0 < contribs->n0 ) );

          // move the coeffs over
          while( num )
          {
            *to_co-- = *from_co--;
            --num;
          }
          // zero new positions
          while ( to_co >= coeffs )
            *to_co-- = 0;
          // set new start point
          contribs->n0 = new_n0;
          if ( widest > 12 )
          {
            int mod;

            // how far will be read in the n_coeff loop (which depends on the widest count mod4);
            mod = widest & 3;
            stop_range = ( ( ( contribs->n1 - contribs->n0 + 1 ) - mod + 3 ) & ~3 ) + mod;

            // the n_coeff loops do a minimum amount of coeffs, so factor that in!
            if ( stop_range < ( 8 + mod ) ) stop_range = 8 + mod;
          }
        }
      }
      --contribs;
      coeffs -= widest;
    }
  }

  return widest;
  #undef STBIR_MOVE_1
  #undef STBIR_MOVE_2
  #undef STBIR_MOVE_4
}

static void stbir__calculate_filters( stbir__sampler * samp, stbir__sampler * other_axis_for_pivot, void * user_data STBIR_ONLY_PROFILE_BUILD_GET_INFO )
{
  int n;
  float scale = samp->scale_info.scale;
  stbir__kernel_callback * kernel = samp->filter_kernel;
  stbir__support_callback * support = samp->filter_support;
  float inv_scale = samp->scale_info.inv_scale;
  int input_full_size = samp->scale_info.input_full_size;
  int gather_num_contributors = samp->num_contributors;
  stbir__contributors* gather_contributors = samp->contributors;
  float * gather_coeffs = samp->coefficients;
  int gather_coefficient_width = samp->coefficient_width;

  switch ( samp->is_gather )
  {
    case 1: // gather upsample
    {
      float out_pixels_radius = support(inv_scale,user_data) * scale;

      stbir__calculate_coefficients_for_gather_upsample( out_pixels_radius, kernel, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width, samp->edge, user_data );

      STBIR_PROFILE_BUILD_START( cleanup );
      stbir__cleanup_gathered_coefficients( samp->edge, &samp->extent_info, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width );
      STBIR_PROFILE_BUILD_END( cleanup );
    }
    break;

    case 0: // scatter downsample (only on vertical)
    case 2: // gather downsample
    {
      float in_pixels_radius = support(scale,user_data) * inv_scale;
      int filter_pixel_margin = samp->filter_pixel_margin;
      int input_end = input_full_size + filter_pixel_margin;

      // if this is a scatter, we do a downsample gather to get the coeffs, and then pivot after
      if ( !samp->is_gather )
      {
        // check if we are using the same gather downsample on the horizontal as this vertical,
        //   if so, then we don't have to generate them, we can just pivot from the horizontal.
        if ( other_axis_for_pivot )
        {
          gather_contributors = other_axis_for_pivot->contributors;
          gather_coeffs = other_axis_for_pivot->coefficients;
          gather_coefficient_width = other_axis_for_pivot->coefficient_width;
          gather_num_contributors = other_axis_for_pivot->num_contributors;
          samp->extent_info.lowest = other_axis_for_pivot->extent_info.lowest;
          samp->extent_info.highest = other_axis_for_pivot->extent_info.highest;
          samp->extent_info.widest = other_axis_for_pivot->extent_info.widest;
          goto jump_right_to_pivot;
        }

        gather_contributors = samp->gather_prescatter_contributors;
        gather_coeffs = samp->gather_prescatter_coefficients;
        gather_coefficient_width = samp->gather_prescatter_coefficient_width;
        gather_num_contributors = samp->gather_prescatter_num_contributors;
      }

      stbir__calculate_coefficients_for_gather_downsample( -filter_pixel_margin, input_end, in_pixels_radius, kernel, &samp->scale_info, gather_coefficient_width, gather_num_contributors, gather_contributors, gather_coeffs, user_data );

      STBIR_PROFILE_BUILD_START( cleanup );
      stbir__cleanup_gathered_coefficients( samp->edge, &samp->extent_info, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width );
      STBIR_PROFILE_BUILD_END( cleanup );

      if ( !samp->is_gather )
      {
        // if this is a scatter (vertical only), then we need to pivot the coeffs
        stbir__contributors * scatter_contributors;
        int highest_set;

        jump_right_to_pivot:

        STBIR_PROFILE_BUILD_START( pivot );

        highest_set = (-filter_pixel_margin) - 1;
        for (n = 0; n < gather_num_contributors; n++)
        {
          int k;
          int gn0 = gather_contributors->n0, gn1 = gather_contributors->n1;
          int scatter_coefficient_width = samp->coefficient_width;
          float * scatter_coeffs = samp->coefficients + ( gn0 + filter_pixel_margin ) * scatter_coefficient_width;
          float * g_coeffs = gather_coeffs;
          scatter_contributors = samp->contributors + ( gn0 + filter_pixel_margin );

          for (k = gn0 ; k <= gn1 ; k++ )
          {
            float gc = *g_coeffs++;
            
            // skip zero and denormals - must skip zeros to avoid adding coeffs beyond scatter_coefficient_width
            //   (which happens when pivoting from horizontal, which might have dummy zeros)
            if ( ( ( gc >= stbir__small_float ) || ( gc <= -stbir__small_float ) ) )
            {
              if ( ( k > highest_set ) || ( scatter_contributors->n0 > scatter_contributors->n1 ) )
              {
                {
                  // if we are skipping over several contributors, we need to clear the skipped ones
                  stbir__contributors * clear_contributors = samp->contributors + ( highest_set + filter_pixel_margin + 1);
                  while ( clear_contributors < scatter_contributors )
                  {
                    clear_contributors->n0 = 0;
                    clear_contributors->n1 = -1;
                    ++clear_contributors;
                  }
                }
                scatter_contributors->n0 = n;
                scatter_contributors->n1 = n;
                scatter_coeffs[0]  = gc;
                highest_set = k;
              }
              else
              {
                stbir__insert_coeff( scatter_contributors, scatter_coeffs, n, gc, scatter_coefficient_width );
              }
              STBIR_ASSERT( ( scatter_contributors->n1 - scatter_contributors->n0 + 1 ) <= scatter_coefficient_width );
            }
            ++scatter_contributors;
            scatter_coeffs += scatter_coefficient_width;
          }

          ++gather_contributors;
          gather_coeffs += gather_coefficient_width;
        }

        // now clear any unset contribs
        {
          stbir__contributors * clear_contributors = samp->contributors + ( highest_set + filter_pixel_margin + 1);
          stbir__contributors * end_contributors = samp->contributors + samp->num_contributors;
          while ( clear_contributors < end_contributors )
          {
            clear_contributors->n0 = 0;
            clear_contributors->n1 = -1;
            ++clear_contributors;
          }
        }

        STBIR_PROFILE_BUILD_END( pivot );
      }
    }
    break;
  }
}


//========================================================================================================
// scanline decoders and encoders

#define stbir__coder_min_num 1
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix BGRA
#define stbir__decode_swizzle
#define stbir__decode_order0  2
#define stbir__decode_order1  1
#define stbir__decode_order2  0
#define stbir__decode_order3  3
#define stbir__encode_order0  2
#define stbir__encode_order1  1
#define stbir__encode_order2  0
#define stbir__encode_order3  3
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix ARGB
#define stbir__decode_swizzle
#define stbir__decode_order0  1
#define stbir__decode_order1  2
#define stbir__decode_order2  3
#define stbir__decode_order3  0
#define stbir__encode_order0  3
#define stbir__encode_order1  0
#define stbir__encode_order2  1
#define stbir__encode_order3  2
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix ABGR
#define stbir__decode_swizzle
#define stbir__decode_order0  3
#define stbir__decode_order1  2
#define stbir__decode_order2  1
#define stbir__decode_order3  0
#define stbir__encode_order0  3
#define stbir__encode_order1  2
#define stbir__encode_order2  1
#define stbir__encode_order3  0
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix AR
#define stbir__decode_swizzle
#define stbir__decode_order0  1
#define stbir__decode_order1  0
#define stbir__decode_order2  3
#define stbir__decode_order3  2
#define stbir__encode_order0  1
#define stbir__encode_order1  0
#define stbir__encode_order2  3
#define stbir__encode_order3  2
#define stbir__coder_min_num 2
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME


// fancy alpha means we expand to keep both premultipied and non-premultiplied color channels
static void stbir__fancy_alpha_weight_4ch( float * out_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) out = out_buffer;
  float const * end_decode = out_buffer + ( width_times_channels / 4 ) * 7;  // decode buffer aligned to end of out_buffer
  float STBIR_STREAMOUT_PTR(*) decode = (float*)end_decode - width_times_channels;

  // fancy alpha is stored internally as R G B A Rpm Gpm Bpm

  #ifdef STBIR_SIMD

  #ifdef STBIR_SIMD8
  decode += 16;
  STBIR_NO_UNROLL_LOOP_START
  while ( decode <= end_decode )
  {
    stbir__simdf8 d0,d1,a0,a1,p0,p1;
    STBIR_NO_UNROLL(decode);
    stbir__simdf8_load( d0, decode-16 );
    stbir__simdf8_load( d1, decode-16+8 );
    stbir__simdf8_0123to33333333( a0, d0 );
    stbir__simdf8_0123to33333333( a1, d1 );
    stbir__simdf8_mult( p0, a0, d0 );
    stbir__simdf8_mult( p1, a1, d1 );
    stbir__simdf8_bot4s( a0, d0, p0 );
    stbir__simdf8_bot4s( a1, d1, p1 );
    stbir__simdf8_top4s( d0, d0, p0 );
    stbir__simdf8_top4s( d1, d1, p1 );
    stbir__simdf8_store ( out, a0 );
    stbir__simdf8_store ( out+7, d0 );
    stbir__simdf8_store ( out+14, a1 );
    stbir__simdf8_store ( out+21, d1 );
    decode += 16;
    out += 28;
  }
  decode -= 16;
  #else
  decode += 8;
  STBIR_NO_UNROLL_LOOP_START
  while ( decode <= end_decode )
  {
    stbir__simdf d0,a0,d1,a1,p0,p1;
    STBIR_NO_UNROLL(decode);
    stbir__simdf_load( d0, decode-8 );
    stbir__simdf_load( d1, decode-8+4 );
    stbir__simdf_0123to3333( a0, d0 );
    stbir__simdf_0123to3333( a1, d1 );
    stbir__simdf_mult( p0, a0, d0 );
    stbir__simdf_mult( p1, a1, d1 );
    stbir__simdf_store ( out, d0 );
    stbir__simdf_store ( out+4, p0 );
    stbir__simdf_store ( out+7, d1 );
    stbir__simdf_store ( out+7+4, p1 );
    decode += 8;
    out += 14;
  }
  decode -= 8;
  #endif

  // might be one last odd pixel
  #ifdef STBIR_SIMD8
  STBIR_NO_UNROLL_LOOP_START
  while ( decode < end_decode )
  #else
  if ( decode < end_decode )
  #endif
  {
    stbir__simdf d,a,p;
    STBIR_NO_UNROLL(decode);
    stbir__simdf_load( d, decode );
    stbir__simdf_0123to3333( a, d );
    stbir__simdf_mult( p, a, d );
    stbir__simdf_store ( out, d );
    stbir__simdf_store ( out+4, p );
    decode += 4;
    out += 7;
  }

  #else

  while( decode < end_decode )
  {
    float r = decode[0], g = decode[1], b = decode[2], alpha = decode[3];
    out[0] = r;
    out[1] = g;
    out[2] = b;
    out[3] = alpha;
    out[4] = r * alpha;
    out[5] = g * alpha;
    out[6] = b * alpha;
    out += 7;
    decode += 4;
  }

  #endif
}

static void stbir__fancy_alpha_weight_2ch( float * out_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) out = out_buffer;
  float const * end_decode = out_buffer + ( width_times_channels / 2 ) * 3;
  float STBIR_STREAMOUT_PTR(*) decode = (float*)end_decode - width_times_channels;

  //  for fancy alpha, turns into: [X A Xpm][X A Xpm],etc

  #ifdef STBIR_SIMD

  decode += 8;
  if ( decode <= end_decode )
  {
    STBIR_NO_UNROLL_LOOP_START
    do {
      #ifdef STBIR_SIMD8
      stbir__simdf8 d0,a0,p0;
      STBIR_NO_UNROLL(decode);
      stbir__simdf8_load( d0, decode-8 );
      stbir__simdf8_0123to11331133( p0, d0 );
      stbir__simdf8_0123to00220022( a0, d0 );
      stbir__simdf8_mult( p0, p0, a0 );

      stbir__simdf_store2( out, stbir__if_simdf8_cast_to_simdf4( d0 ) );
      stbir__simdf_store( out+2, stbir__if_simdf8_cast_to_simdf4( p0 ) );
      stbir__simdf_store2h( out+3, stbir__if_simdf8_cast_to_simdf4( d0 ) );

      stbir__simdf_store2( out+6, stbir__simdf8_gettop4( d0 ) );
      stbir__simdf_store( out+8, stbir__simdf8_gettop4( p0 ) );
      stbir__simdf_store2h( out+9, stbir__simdf8_gettop4( d0 ) );
      #else
      stbir__simdf d0,a0,d1,a1,p0,p1;
      STBIR_NO_UNROLL(decode);
      stbir__simdf_load( d0, decode-8 );
      stbir__simdf_load( d1, decode-8+4 );
      stbir__simdf_0123to1133( p0, d0 );
      stbir__simdf_0123to1133( p1, d1 );
      stbir__simdf_0123to0022( a0, d0 );
      stbir__simdf_0123to0022( a1, d1 );
      stbir__simdf_mult( p0, p0, a0 );
      stbir__simdf_mult( p1, p1, a1 );

      stbir__simdf_store2( out, d0 );
      stbir__simdf_store( out+2, p0 );
      stbir__simdf_store2h( out+3, d0 );

      stbir__simdf_store2( out+6, d1 );
      stbir__simdf_store( out+8, p1 );
      stbir__simdf_store2h( out+9, d1 );
      #endif
      decode += 8;
      out += 12;
    } while ( decode <= end_decode );
  }
  decode -= 8;
  #endif

  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float x = decode[0], y = decode[1];
    STBIR_SIMD_NO_UNROLL(decode);
    out[0] = x;
    out[1] = y;
    out[2] = x * y;
    out += 3;
    decode += 2;
  }
}

static void stbir__fancy_alpha_unweight_4ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float STBIR_SIMD_STREAMOUT_PTR(*) input = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  // fancy RGBA is stored internally as R G B A Rpm Gpm Bpm

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float alpha = input[3];
#ifdef STBIR_SIMD
    stbir__simdf i,ia;
    STBIR_SIMD_NO_UNROLL(encode);
    if ( alpha < stbir__small_float )
    {
      stbir__simdf_load( i, input );
      stbir__simdf_store( encode, i );
    }
    else
    {
      stbir__simdf_load1frep4( ia, 1.0f / alpha );
      stbir__simdf_load( i, input+4 );
      stbir__simdf_mult( i, i, ia );
      stbir__simdf_store( encode, i );
      encode[3] = alpha;
    }
#else
    if ( alpha < stbir__small_float )
    {
      encode[0] = input[0];
      encode[1] = input[1];
      encode[2] = input[2];
    }
    else
    {
      float ialpha = 1.0f / alpha;
      encode[0] = input[4] * ialpha;
      encode[1] = input[5] * ialpha;
      encode[2] = input[6] * ialpha;
    }
    encode[3] = alpha;
#endif

    input += 7;
    encode += 4;
  } while ( encode < end_output );
}

//  format: [X A Xpm][X A Xpm] etc
static void stbir__fancy_alpha_unweight_2ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float STBIR_SIMD_STREAMOUT_PTR(*) input = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  do {
    float alpha = input[1];
    encode[0] = input[0];
    if ( alpha >= stbir__small_float )
      encode[0] = input[2] / alpha;
    encode[1] = alpha;

    input += 3;
    encode += 2;
  } while ( encode < end_output );
}

static void stbir__simple_alpha_weight_4ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

  #ifdef STBIR_SIMD
  {
    decode += 2 * stbir__simdfX_float_count;
    STBIR_NO_UNROLL_LOOP_START
    while ( decode <= end_decode )
    {
      stbir__simdfX d0,a0,d1,a1;
      STBIR_NO_UNROLL(decode);
      stbir__simdfX_load( d0, decode-2*stbir__simdfX_float_count );
      stbir__simdfX_load( d1, decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count );
      stbir__simdfX_aaa1( a0, d0, STBIR_onesX );
      stbir__simdfX_aaa1( a1, d1, STBIR_onesX );
      stbir__simdfX_mult( d0, d0, a0 );
      stbir__simdfX_mult( d1, d1, a1 );
      stbir__simdfX_store ( decode-2*stbir__simdfX_float_count, d0 );
      stbir__simdfX_store ( decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count, d1 );
      decode += 2 * stbir__simdfX_float_count;
    }
    decode -= 2 * stbir__simdfX_float_count;

    // few last pixels remnants
    #ifdef STBIR_SIMD8
    STBIR_NO_UNROLL_LOOP_START
    while ( decode < end_decode )
    #else
    if ( decode < end_decode )
    #endif
    {
      stbir__simdf d,a;
      stbir__simdf_load( d, decode );
      stbir__simdf_aaa1( a, d, STBIR__CONSTF(STBIR_ones) );
      stbir__simdf_mult( d, d, a );
      stbir__simdf_store ( decode, d );
      decode += 4;
    }
  }

  #else

  while( decode < end_decode )
  {
    float alpha = decode[3];
    decode[0] *= alpha;
    decode[1] *= alpha;
    decode[2] *= alpha;
    decode += 4;
  }

  #endif
}

static void stbir__simple_alpha_weight_2ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

  #ifdef STBIR_SIMD
  decode += 2 * stbir__simdfX_float_count;
  STBIR_NO_UNROLL_LOOP_START
  while ( decode <= end_decode )
  {
    stbir__simdfX d0,a0,d1,a1;
    STBIR_NO_UNROLL(decode);
    stbir__simdfX_load( d0, decode-2*stbir__simdfX_float_count );
    stbir__simdfX_load( d1, decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count );
    stbir__simdfX_a1a1( a0, d0, STBIR_onesX );
    stbir__simdfX_a1a1( a1, d1, STBIR_onesX );
    stbir__simdfX_mult( d0, d0, a0 );
    stbir__simdfX_mult( d1, d1, a1 );
    stbir__simdfX_store ( decode-2*stbir__simdfX_float_count, d0 );
    stbir__simdfX_store ( decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count, d1 );
    decode += 2 * stbir__simdfX_float_count;
  }
  decode -= 2 * stbir__simdfX_float_count;
  #endif

  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float alpha = decode[1];
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0] *= alpha;
    decode += 2;
  }
}

static void stbir__simple_alpha_unweight_4ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float alpha = encode[3];

#ifdef STBIR_SIMD
    stbir__simdf i,ia;
    STBIR_SIMD_NO_UNROLL(encode);
    if ( alpha >= stbir__small_float )
    {
      stbir__simdf_load1frep4( ia, 1.0f / alpha );
      stbir__simdf_load( i, encode );
      stbir__simdf_mult( i, i, ia );
      stbir__simdf_store( encode, i );
      encode[3] = alpha;
    }
#else
    if ( alpha >= stbir__small_float )
    {
      float ialpha = 1.0f / alpha;
      encode[0] *= ialpha;
      encode[1] *= ialpha;
      encode[2] *= ialpha;
    }
#endif
    encode += 4;
  } while ( encode < end_output );
}

static void stbir__simple_alpha_unweight_2ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  do {
    float alpha = encode[1];
    if ( alpha >= stbir__small_float )
      encode[0] /= alpha;
    encode += 2;
  } while ( encode < end_output );
}


// only used in RGB->BGR or BGR->RGB
static void stbir__simple_flip_3ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

#ifdef STBIR_SIMD
    #ifdef stbir__simdf_swiz2 // do we have two argument swizzles?
      end_decode -= 12; 
      STBIR_NO_UNROLL_LOOP_START
      while( decode <= end_decode )
      {
        // on arm64 8 instructions, no overlapping stores
        stbir__simdf a,b,c,na,nb;
        STBIR_SIMD_NO_UNROLL(decode);
        stbir__simdf_load( a, decode );
        stbir__simdf_load( b, decode+4 );
        stbir__simdf_load( c, decode+8 );

        na = stbir__simdf_swiz2( a, b, 2, 1, 0, 5 );   
        b  = stbir__simdf_swiz2( a, b, 4, 3, 6, 7 );   
        nb = stbir__simdf_swiz2( b, c, 0, 1, 4, 3 );   
        c  = stbir__simdf_swiz2( b, c, 2, 7, 6, 5 );   

        stbir__simdf_store( decode, na );
        stbir__simdf_store( decode+4, nb ); 
        stbir__simdf_store( decode+8, c );
        decode += 12;
      }
      end_decode += 12;
    #else
      end_decode -= 24;
      STBIR_NO_UNROLL_LOOP_START
      while( decode <= end_decode )
      {
        // 26 instructions on x64
        stbir__simdf a,b,c,d,e,f,g;
        float i21, i23;
        STBIR_SIMD_NO_UNROLL(decode);
        stbir__simdf_load( a, decode );
        stbir__simdf_load( b, decode+3 );
        stbir__simdf_load( c, decode+6 );
        stbir__simdf_load( d, decode+9 );
        stbir__simdf_load( e, decode+12 );
        stbir__simdf_load( f, decode+15 );
        stbir__simdf_load( g, decode+18 );

        a = stbir__simdf_swiz( a, 2, 1, 0, 3 );   
        b = stbir__simdf_swiz( b, 2, 1, 0, 3 );   
        c = stbir__simdf_swiz( c, 2, 1, 0, 3 );   
        d = stbir__simdf_swiz( d, 2, 1, 0, 3 );   
        e = stbir__simdf_swiz( e, 2, 1, 0, 3 );   
        f = stbir__simdf_swiz( f, 2, 1, 0, 3 );   
        g = stbir__simdf_swiz( g, 2, 1, 0, 3 );   

        // stores overlap, need to be in order, 
        stbir__simdf_store( decode,    a );
        i21 = decode[21];
        stbir__simdf_store( decode+3,  b ); 
        i23 = decode[23];
        stbir__simdf_store( decode+6,  c );
        stbir__simdf_store( decode+9,  d );
        stbir__simdf_store( decode+12, e );
        stbir__simdf_store( decode+15, f );
        stbir__simdf_store( decode+18, g );
        decode[21] = i23;
        decode[23] = i21;
        decode += 24;
      }
      end_decode += 24;
    #endif
#else
  end_decode -= 12;
  STBIR_NO_UNROLL_LOOP_START
  while( decode <= end_decode )
  {
    // 16 instructions
    float t0,t1,t2,t3;
    STBIR_NO_UNROLL(decode);
    t0 = decode[0]; t1 = decode[3]; t2 = decode[6]; t3 = decode[9];
    decode[0] = decode[2]; decode[3] = decode[5]; decode[6] = decode[8]; decode[9] = decode[11];
    decode[2] = t0; decode[5] = t1; decode[8] = t2; decode[11] = t3;
    decode += 12;
  }
  end_decode += 12;
#endif

  STBIR_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float t = decode[0];
    STBIR_NO_UNROLL(decode);
    decode[0] = decode[2];
    decode[2] = t;
    decode += 3;
  }
}



static void stbir__decode_scanline(stbir__info const * stbir_info, int n, float * output_buffer STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  int channels = stbir_info->channels;
  int effective_channels = stbir_info->effective_channels;
  int input_sample_in_bytes = stbir__type_size[stbir_info->input_type] * channels;
  stbir_edge edge_horizontal = stbir_info->horizontal.edge;
  stbir_edge edge_vertical = stbir_info->vertical.edge;
  int row = stbir__edge_wrap(edge_vertical, n, stbir_info->vertical.scale_info.input_full_size);
  const void* input_plane_data = ( (char *) stbir_info->input_data ) + (size_t)row * (size_t) stbir_info->input_stride_bytes;
  stbir__span const * spans = stbir_info->scanline_extents.spans;
  float * full_decode_buffer = output_buffer - stbir_info->scanline_extents.conservative.n0 * effective_channels;
  float * last_decoded = 0;

  // if we are on edge_zero, and we get in here with an out of bounds n, then the calculate filters has failed
  STBIR_ASSERT( !(edge_vertical == STBIR_EDGE_ZERO && (n < 0 || n >= stbir_info->vertical.scale_info.input_full_size)) );

  do
  {
    float * decode_buffer;
    void const * input_data;
    float * end_decode;
    int width_times_channels;
    int width;

    if ( spans->n1 < spans->n0 )
      break;

    width = spans->n1 + 1 - spans->n0;
    decode_buffer = full_decode_buffer + spans->n0 * effective_channels;
    end_decode = full_decode_buffer + ( spans->n1 + 1 ) * effective_channels;
    width_times_channels = width * channels;

    // read directly out of input plane by default
    input_data = ( (char*)input_plane_data ) + spans->pixel_offset_for_input * input_sample_in_bytes;

    // if we have an input callback, call it to get the input data
    if ( stbir_info->in_pixels_cb )
    {
      // call the callback with a temp buffer (that they can choose to use or not).  the temp is just right aligned memory in the decode_buffer itself
      input_data = stbir_info->in_pixels_cb( ( (char*) end_decode ) - ( width * input_sample_in_bytes ) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING, input_plane_data, width, spans->pixel_offset_for_input, row, stbir_info->user_data );
    }

    STBIR_PROFILE_START( decode );
    // convert the pixels info the float decode_buffer, (we index from end_decode, so that when channels<effective_channels, we are right justified in the buffer)
    last_decoded = stbir_info->decode_pixels( (float*)end_decode - width_times_channels, width_times_channels, input_data );
    STBIR_PROFILE_END( decode );

    if (stbir_info->alpha_weight)
    {
      STBIR_PROFILE_START( alpha );
      stbir_info->alpha_weight( decode_buffer, width_times_channels );
      STBIR_PROFILE_END( alpha );
    }

    ++spans;
  } while ( spans <= ( &stbir_info->scanline_extents.spans[1] ) );

  // handle the edge_wrap filter (all other types are handled back out at the calculate_filter stage)
  // basically the idea here is that if we have the whole scanline in memory, we don't redecode the
  //   wrapped edge pixels, and instead just memcpy them from the scanline into the edge positions
  if ( ( edge_horizontal == STBIR_EDGE_WRAP ) && ( stbir_info->scanline_extents.edge_sizes[0] | stbir_info->scanline_extents.edge_sizes[1] ) )
  {
    // this code only runs if we're in edge_wrap, and we're doing the entire scanline
    int e, start_x[2];
    int input_full_size = stbir_info->horizontal.scale_info.input_full_size;

    start_x[0] = -stbir_info->scanline_extents.edge_sizes[0];  // left edge start x
    start_x[1] =  input_full_size;                             // right edge

    for( e = 0; e < 2 ; e++ )
    {
      // do each margin
      int margin = stbir_info->scanline_extents.edge_sizes[e];
      if ( margin )
      {
        int x = start_x[e];
        float * marg = full_decode_buffer + x * effective_channels;
        float const * src = full_decode_buffer + stbir__edge_wrap(edge_horizontal, x, input_full_size) * effective_channels;
        STBIR_MEMCPY( marg, src, margin * effective_channels * sizeof(float) );
        if ( e == 1 ) last_decoded = marg + margin * effective_channels;
      }
    }
  }
  
  // some of the horizontal gathers read one float off the edge (which is masked out), but we force a zero here to make sure no NaNs leak in
  //   (we can't pre-zero it, because the input callback can use that area as padding)
  last_decoded[0] = 0.0f; 

  // we clear this extra float, because the final output pixel filter kernel might have used one less coeff than the max filter width
  //   when this happens, we do read that pixel from the input, so it too could be Nan, so just zero an extra one.
  //   this fits because each scanline is padded by three floats (STBIR_INPUT_CALLBACK_PADDING)
  last_decoded[1] = 0.0f;
}


//=================
// Do 1 channel horizontal routines

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()          \
    stbir__simdf tot,c;                \
    STBIR_SIMD_NO_UNROLL(decode);      \
    stbir__simdf_load1( c, hc );       \
    stbir__simdf_mult1_mem( tot, c, decode );

#define stbir__2_coeff_only()          \
    stbir__simdf tot,c,d;              \
    STBIR_SIMD_NO_UNROLL(decode);      \
    stbir__simdf_load2z( c, hc );      \
    stbir__simdf_load2( d, decode );   \
    stbir__simdf_mult( tot, c, d );    \
    stbir__simdf_0123to1230( c, tot ); \
    stbir__simdf_add1( tot, tot, c );

#define stbir__3_coeff_only()                  \
    stbir__simdf tot,c,t;                      \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc );                \
    stbir__simdf_mult_mem( tot, c, decode );   \
    stbir__simdf_0123to1230( c, tot );         \
    stbir__simdf_0123to2301( t, tot );         \
    stbir__simdf_add1( tot, tot, c );          \
    stbir__simdf_add1( tot, tot, t );

#define stbir__store_output_tiny()                \
    stbir__simdf_store1( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#define stbir__4_coeff_start()                 \
    stbir__simdf tot,c;                        \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc );                \
    stbir__simdf_mult_mem( tot, c, decode );   \

#define stbir__4_coeff_continue_from_4( ofs )  \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc + (ofs) );        \
    stbir__simdf_madd_mem( tot, tot, c, decode+(ofs) );

#define stbir__1_coeff_remnant( ofs )          \
    { stbir__simdf d;                          \
    stbir__simdf_load1z( c, hc + (ofs) );      \
    stbir__simdf_load1( d, decode + (ofs) );   \
    stbir__simdf_madd( tot, tot, d, c ); }

#define stbir__2_coeff_remnant( ofs )          \
    { stbir__simdf d;                          \
    stbir__simdf_load2z( c, hc+(ofs) );        \
    stbir__simdf_load2( d, decode+(ofs) );     \
    stbir__simdf_madd( tot, tot, d, c ); }

#define stbir__3_coeff_setup()                 \
    stbir__simdf mask;                         \
    stbir__simdf_load( mask, STBIR_mask + 3 );

#define stbir__3_coeff_remnant( ofs )                  \
    stbir__simdf_load( c, hc+(ofs) );                  \
    stbir__simdf_and( c, c, mask );                    \
    stbir__simdf_madd_mem( tot, tot, c, decode+(ofs) );

#define stbir__store_output()                     \
    stbir__simdf_0123to2301( c, tot );            \
    stbir__simdf_add( tot, tot, c );              \
    stbir__simdf_0123to1230( c, tot );            \
    stbir__simdf_add1( tot, tot, c );             \
    stbir__simdf_store1( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#else

#define stbir__1_coeff_only()  \
    float tot;                 \
    tot = decode[0]*hc[0];

#define stbir__2_coeff_only()  \
    float tot;                 \
    tot = decode[0] * hc[0];   \
    tot += decode[1] * hc[1];

#define stbir__3_coeff_only()  \
    float tot;                 \
    tot = decode[0] * hc[0];   \
    tot += decode[1] * hc[1];  \
    tot += decode[2] * hc[2];

#define stbir__store_output_tiny()                \
    output[0] = tot;                              \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#define stbir__4_coeff_start()  \
    float tot0,tot1,tot2,tot3;  \
    tot0 = decode[0] * hc[0];   \
    tot1 = decode[1] * hc[1];   \
    tot2 = decode[2] * hc[2];   \
    tot3 = decode[3] * hc[3];

#define stbir__4_coeff_continue_from_4( ofs )  \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];     \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];     \
    tot2 += decode[2+(ofs)] * hc[2+(ofs)];     \
    tot3 += decode[3+(ofs)] * hc[3+(ofs)];

#define stbir__1_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];

#define stbir__2_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];   \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];   \

#define stbir__3_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];   \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];   \
    tot2 += decode[2+(ofs)] * hc[2+(ofs)];

#define stbir__store_output()                     \
    output[0] = (tot0+tot2)+(tot1+tot3);          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#endif

#define STBIR__horizontal_channels 1
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME


//=================
// Do 2 channel horizontal routines

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()         \
    stbir__simdf tot,c,d;             \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load1z( c, hc );     \
    stbir__simdf_0123to0011( c, c );  \
    stbir__simdf_load2( d, decode );  \
    stbir__simdf_mult( tot, d, c );

#define stbir__2_coeff_only()         \
    stbir__simdf tot,c;               \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load2( c, hc );      \
    stbir__simdf_0123to0011( c, c );  \
    stbir__simdf_mult_mem( tot, c, decode );

#define stbir__3_coeff_only()                \
    stbir__simdf tot,c,cs,d;                 \
    STBIR_SIMD_NO_UNROLL(decode);            \
    stbir__simdf_load( cs, hc );             \
    stbir__simdf_0123to0011( c, cs );        \
    stbir__simdf_mult_mem( tot, c, decode ); \
    stbir__simdf_0123to2222( c, cs );        \
    stbir__simdf_load2z( d, decode+4 );      \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__store_output_tiny()                \
    stbir__simdf_0123to2301( c, tot );            \
    stbir__simdf_add( tot, tot, c );              \
    stbir__simdf_store2( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                    \
    stbir__simdf8 tot0,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                 \
    stbir__simdf8_load4b( cs, hc );               \
    stbir__simdf8_0123to00112233( c, cs );        \
    stbir__simdf8_mult_mem( tot0, c, decode );

#define stbir__4_coeff_continue_from_4( ofs )        \
    STBIR_SIMD_NO_UNROLL(decode);                    \
    stbir__simdf8_load4b( cs, hc + (ofs) );          \
    stbir__simdf8_0123to00112233( c, cs );           \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*2 );

#define stbir__1_coeff_remnant( ofs )                \
    { stbir__simdf t,d;                              \
    stbir__simdf_load1z( t, hc + (ofs) );            \
    stbir__simdf_load2( d, decode + (ofs) * 2 );     \
    stbir__simdf_0123to0011( t, t );                 \
    stbir__simdf_mult( t, t, d );                    \
    stbir__simdf8_add4( tot0, tot0, t ); }
 
#define stbir__2_coeff_remnant( ofs )                \
    { stbir__simdf t;                                \
    stbir__simdf_load2( t, hc + (ofs) );             \
    stbir__simdf_0123to0011( t, t );                 \
    stbir__simdf_mult_mem( t, t, decode+(ofs)*2 );   \
    stbir__simdf8_add4( tot0, tot0, t ); }

#define stbir__3_coeff_remnant( ofs )                \
    { stbir__simdf8 d;                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );          \
    stbir__simdf8_0123to00112233( c, cs );           \
    stbir__simdf8_load6z( d, decode+(ofs)*2 );       \
    stbir__simdf8_madd( tot0, tot0, c, d ); }

#define stbir__store_output()                     \
    { stbir__simdf t,d;                           \
    stbir__simdf8_add4halves( t, stbir__if_simdf8_cast_to_simdf4(tot0), tot0 );    \
    stbir__simdf_0123to2301( d, t );              \
    stbir__simdf_add( t, t, d );                  \
    stbir__simdf_store2( output, t );             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2; }

#else

#define stbir__4_coeff_start()                   \
    stbir__simdf tot0,tot1,c,cs;                 \
    STBIR_SIMD_NO_UNROLL(decode);                \
    stbir__simdf_load( cs, hc );                 \
    stbir__simdf_0123to0011( c, cs );            \
    stbir__simdf_mult_mem( tot0, c, decode );    \
    stbir__simdf_0123to2233( c, cs );            \
    stbir__simdf_mult_mem( tot1, c, decode+4 );

#define stbir__4_coeff_continue_from_4( ofs )                \
    STBIR_SIMD_NO_UNROLL(decode);                            \
    stbir__simdf_load( cs, hc + (ofs) );                     \
    stbir__simdf_0123to0011( c, cs );                        \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 );  \
    stbir__simdf_0123to2233( c, cs );                        \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*2+4 );

#define stbir__1_coeff_remnant( ofs )            \
    { stbir__simdf d;                            \
    stbir__simdf_load1z( cs, hc + (ofs) );       \
    stbir__simdf_0123to0011( c, cs );            \
    stbir__simdf_load2( d, decode + (ofs) * 2 ); \
    stbir__simdf_madd( tot0, tot0, d, c ); }

#define stbir__2_coeff_remnant( ofs )                      \
    stbir__simdf_load2( cs, hc + (ofs) );                  \
    stbir__simdf_0123to0011( c, cs );                      \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 );

#define stbir__3_coeff_remnant( ofs )                       \
    { stbir__simdf d;                                       \
    stbir__simdf_load( cs, hc + (ofs) );                    \
    stbir__simdf_0123to0011( c, cs );                       \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 ); \
    stbir__simdf_0123to2222( c, cs );                       \
    stbir__simdf_load2z( d, decode + (ofs) * 2 + 4 );       \
    stbir__simdf_madd( tot1, tot1, d, c ); }

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot1 );         \
    stbir__simdf_0123to2301( c, tot0 );           \
    stbir__simdf_add( tot0, tot0, c );            \
    stbir__simdf_store2( output, tot0 );          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#endif

#else

#define stbir__1_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;

#define stbir__2_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;        \
    c = hc[1];                 \
    tota += decode[2]*c;       \
    totb += decode[3]*c;

// this weird order of add matches the simd
#define stbir__3_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;        \
    c = hc[2];                 \
    tota += decode[4]*c;       \
    totb += decode[5]*c;       \
    c = hc[1];                 \
    tota += decode[2]*c;       \
    totb += decode[3]*c;

#define stbir__store_output_tiny()                \
    output[0] = tota;                             \
    output[1] = totb;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#define stbir__4_coeff_start()      \
    float tota0,tota1,tota2,tota3,totb0,totb1,totb2,totb3,c;  \
    c = hc[0];                      \
    tota0 = decode[0]*c;            \
    totb0 = decode[1]*c;            \
    c = hc[1];                      \
    tota1 = decode[2]*c;            \
    totb1 = decode[3]*c;            \
    c = hc[2];                      \
    tota2 = decode[4]*c;            \
    totb2 = decode[5]*c;            \
    c = hc[3];                      \
    tota3 = decode[6]*c;            \
    totb3 = decode[7]*c;

#define stbir__4_coeff_continue_from_4( ofs )  \
    c = hc[0+(ofs)];                           \
    tota0 += decode[0+(ofs)*2]*c;              \
    totb0 += decode[1+(ofs)*2]*c;              \
    c = hc[1+(ofs)];                           \
    tota1 += decode[2+(ofs)*2]*c;              \
    totb1 += decode[3+(ofs)*2]*c;              \
    c = hc[2+(ofs)];                           \
    tota2 += decode[4+(ofs)*2]*c;              \
    totb2 += decode[5+(ofs)*2]*c;              \
    c = hc[3+(ofs)];                           \
    tota3 += decode[6+(ofs)*2]*c;              \
    totb3 += decode[7+(ofs)*2]*c;

#define stbir__1_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;

#define stbir__2_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;    \
    c = hc[1+(ofs)];                   \
    tota1 += decode[2+(ofs)*2] * c;    \
    totb1 += decode[3+(ofs)*2] * c;

#define stbir__3_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;    \
    c = hc[1+(ofs)];                   \
    tota1 += decode[2+(ofs)*2] * c;    \
    totb1 += decode[3+(ofs)*2] * c;    \
    c = hc[2+(ofs)];                   \
    tota2 += decode[4+(ofs)*2] * c;    \
    totb2 += decode[5+(ofs)*2] * c;

#define stbir__store_output()                     \
    output[0] = (tota0+tota2)+(tota1+tota3);      \
    output[1] = (totb0+totb2)+(totb1+totb3);      \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#endif

#define STBIR__horizontal_channels 2
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME


//=================
// Do 3 channel horizontal routines

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()         \
    stbir__simdf tot,c,d;             \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load1z( c, hc );     \
    stbir__simdf_0123to0001( c, c );  \
    stbir__simdf_load( d, decode );   \
    stbir__simdf_mult( tot, d, c );

#define stbir__2_coeff_only()         \
    stbir__simdf tot,c,cs,d;          \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load2( cs, hc );     \
    stbir__simdf_0123to0000( c, cs ); \
    stbir__simdf_load( d, decode );   \
    stbir__simdf_mult( tot, d, c );   \
    stbir__simdf_0123to1111( c, cs ); \
    stbir__simdf_load( d, decode+3 ); \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__3_coeff_only()            \
    stbir__simdf tot,c,d,cs;             \
    STBIR_SIMD_NO_UNROLL(decode);        \
    stbir__simdf_load( cs, hc );         \
    stbir__simdf_0123to0000( c, cs );    \
    stbir__simdf_load( d, decode );      \
    stbir__simdf_mult( tot, d, c );      \
    stbir__simdf_0123to1111( c, cs );    \
    stbir__simdf_load( d, decode+3 );    \
    stbir__simdf_madd( tot, tot, d, c ); \
    stbir__simdf_0123to2222( c, cs );    \
    stbir__simdf_load( d, decode+6 );    \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__store_output_tiny()                \
    stbir__simdf_store2( output, tot );           \
    stbir__simdf_0123to2301( tot, tot );          \
    stbir__simdf_store1( output+2, tot );         \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#ifdef STBIR_SIMD8

// we're loading from the XXXYYY decode by -1 to get the XXXYYY into different halves of the AVX reg fyi
#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,tot1,c,cs; stbir__simdf t;  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode - 1 ); \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_mult_mem( tot1, c, decode+6 - 1 );

#define stbir__4_coeff_continue_from_4( ofs )      \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc + (ofs) );        \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 ); \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*3 + 6 - 1 );

#define stbir__1_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load1rep4( t, hc + (ofs) );                   \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*3 - 1 );

#define stbir__2_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) - 2 );                \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 );

 #define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                                \
    stbir__simdf8_load4b( cs, hc + (ofs) );                      \
    stbir__simdf8_0123to00001111( c, cs );                       \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 ); \
    stbir__simdf8_0123to2222( t, cs );                           \
    stbir__simdf8_madd_mem4( tot1, tot1, t, decode+(ofs)*3 + 6 - 1 );

#define stbir__store_output()                       \
    stbir__simdf8_add( tot0, tot0, tot1 );          \
    stbir__simdf_0123to1230( t, stbir__if_simdf8_cast_to_simdf4( tot0 ) ); \
    stbir__simdf8_add4halves( t, t, tot0 );         \
    horizontal_coefficients += coefficient_width;   \
    ++horizontal_contributors;                      \
    output += 3;                                    \
    if ( output < output_end )                      \
    {                                               \
      stbir__simdf_store( output-3, t );            \
      continue;                                     \
    }                                               \
    { stbir__simdf tt; stbir__simdf_0123to2301( tt, t ); \
    stbir__simdf_store2( output-3, t );             \
    stbir__simdf_store1( output+2-3, tt ); }        \
    break;


#else

#define stbir__4_coeff_start()                  \
    stbir__simdf tot0,tot1,tot2,c,cs;           \
    STBIR_SIMD_NO_UNROLL(decode);               \
    stbir__simdf_load( cs, hc );                \
    stbir__simdf_0123to0001( c, cs );           \
    stbir__simdf_mult_mem( tot0, c, decode );   \
    stbir__simdf_0123to1122( c, cs );           \
    stbir__simdf_mult_mem( tot1, c, decode+4 ); \
    stbir__simdf_0123to2333( c, cs );           \
    stbir__simdf_mult_mem( tot2, c, decode+8 );

#define stbir__4_coeff_continue_from_4( ofs )                 \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load( cs, hc + (ofs) );                      \
    stbir__simdf_0123to0001( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );   \
    stbir__simdf_0123to1122( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*3+4 ); \
    stbir__simdf_0123to2333( c, cs );                         \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*3+8 );

#define stbir__1_coeff_remnant( ofs )         \
    STBIR_SIMD_NO_UNROLL(decode);             \
    stbir__simdf_load1z( c, hc + (ofs) );     \
    stbir__simdf_0123to0001( c, c );          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );

#define stbir__2_coeff_remnant( ofs )                       \
    { stbir__simdf d;                                       \
    STBIR_SIMD_NO_UNROLL(decode);                           \
    stbir__simdf_load2z( cs, hc + (ofs) );                  \
    stbir__simdf_0123to0001( c, cs );                       \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 ); \
    stbir__simdf_0123to1122( c, cs );                       \
    stbir__simdf_load2z( d, decode+(ofs)*3+4 );             \
    stbir__simdf_madd( tot1, tot1, c, d ); }

#define stbir__3_coeff_remnant( ofs )                         \
    { stbir__simdf d;                                         \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load( cs, hc + (ofs) );                      \
    stbir__simdf_0123to0001( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );   \
    stbir__simdf_0123to1122( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*3+4 ); \
    stbir__simdf_0123to2222( c, cs );                         \
    stbir__simdf_load1z( d, decode+(ofs)*3+8 );               \
    stbir__simdf_madd( tot2, tot2, c, d );  }

#define stbir__store_output()                       \
    stbir__simdf_0123ABCDto3ABx( c, tot0, tot1 );   \
    stbir__simdf_0123ABCDto23Ax( cs, tot1, tot2 );  \
    stbir__simdf_0123to1230( tot2, tot2 );          \
    stbir__simdf_add( tot0, tot0, cs );             \
    stbir__simdf_add( c, c, tot2 );                 \
    stbir__simdf_add( tot0, tot0, c );              \
    horizontal_coefficients += coefficient_width;   \
    ++horizontal_contributors;                      \
    output += 3;                                    \
    if ( output < output_end )                      \
    {                                               \
      stbir__simdf_store( output-3, tot0 );         \
      continue;                                     \
    }                                               \
    stbir__simdf_0123to2301( tot1, tot0 );          \
    stbir__simdf_store2( output-3, tot0 );          \
    stbir__simdf_store1( output+2-3, tot1 );        \
    break;

#endif

#else

#define stbir__1_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;

#define stbir__2_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;        \
    c = hc[1];                 \
    tot0 += decode[3]*c;       \
    tot1 += decode[4]*c;       \
    tot2 += decode[5]*c;

#define stbir__3_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;        \
    c = hc[1];                 \
    tot0 += decode[3]*c;       \
    tot1 += decode[4]*c;       \
    tot2 += decode[5]*c;       \
    c = hc[2];                 \
    tot0 += decode[6]*c;       \
    tot1 += decode[7]*c;       \
    tot2 += decode[8]*c;

#define stbir__store_output_tiny()                \
    output[0] = tot0;                             \
    output[1] = tot1;                             \
    output[2] = tot2;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#define stbir__4_coeff_start()      \
    float tota0,tota1,tota2,totb0,totb1,totb2,totc0,totc1,totc2,totd0,totd1,totd2,c;  \
    c = hc[0];                      \
    tota0 = decode[0]*c;            \
    tota1 = decode[1]*c;            \
    tota2 = decode[2]*c;            \
    c = hc[1];                      \
    totb0 = decode[3]*c;            \
    totb1 = decode[4]*c;            \
    totb2 = decode[5]*c;            \
    c = hc[2];                      \
    totc0 = decode[6]*c;            \
    totc1 = decode[7]*c;            \
    totc2 = decode[8]*c;            \
    c = hc[3];                      \
    totd0 = decode[9]*c;            \
    totd1 = decode[10]*c;           \
    totd2 = decode[11]*c;

#define stbir__4_coeff_continue_from_4( ofs )  \
    c = hc[0+(ofs)];                           \
    tota0 += decode[0+(ofs)*3]*c;              \
    tota1 += decode[1+(ofs)*3]*c;              \
    tota2 += decode[2+(ofs)*3]*c;              \
    c = hc[1+(ofs)];                           \
    totb0 += decode[3+(ofs)*3]*c;              \
    totb1 += decode[4+(ofs)*3]*c;              \
    totb2 += decode[5+(ofs)*3]*c;              \
    c = hc[2+(ofs)];                           \
    totc0 += decode[6+(ofs)*3]*c;              \
    totc1 += decode[7+(ofs)*3]*c;              \
    totc2 += decode[8+(ofs)*3]*c;              \
    c = hc[3+(ofs)];                           \
    totd0 += decode[9+(ofs)*3]*c;              \
    totd1 += decode[10+(ofs)*3]*c;             \
    totd2 += decode[11+(ofs)*3]*c;

#define stbir__1_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;

#define stbir__2_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;      \
    c = hc[1+(ofs)];                   \
    totb0 += decode[3+(ofs)*3]*c;      \
    totb1 += decode[4+(ofs)*3]*c;      \
    totb2 += decode[5+(ofs)*3]*c;      \

#define stbir__3_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;      \
    c = hc[1+(ofs)];                   \
    totb0 += decode[3+(ofs)*3]*c;      \
    totb1 += decode[4+(ofs)*3]*c;      \
    totb2 += decode[5+(ofs)*3]*c;      \
    c = hc[2+(ofs)];                   \
    totc0 += decode[6+(ofs)*3]*c;      \
    totc1 += decode[7+(ofs)*3]*c;      \
    totc2 += decode[8+(ofs)*3]*c;

#define stbir__store_output()                     \
    output[0] = (tota0+totc0)+(totb0+totd0);      \
    output[1] = (tota1+totc1)+(totb1+totd1);      \
    output[2] = (tota2+totc2)+(totb2+totd2);      \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#endif

#define STBIR__horizontal_channels 3
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

//=================
// Do 4 channel horizontal routines

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()             \
    stbir__simdf tot,c;                   \
    STBIR_SIMD_NO_UNROLL(decode);         \
    stbir__simdf_load1( c, hc );          \
    stbir__simdf_0123to0000( c, c );      \
    stbir__simdf_mult_mem( tot, c, decode );

#define stbir__2_coeff_only()                       \
    stbir__simdf tot,c,cs;                          \
    STBIR_SIMD_NO_UNROLL(decode);                   \
    stbir__simdf_load2( cs, hc );                   \
    stbir__simdf_0123to0000( c, cs );               \
    stbir__simdf_mult_mem( tot, c, decode );        \
    stbir__simdf_0123to1111( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+4 );

#define stbir__3_coeff_only()                       \
    stbir__simdf tot,c,cs;                          \
    STBIR_SIMD_NO_UNROLL(decode);                   \
    stbir__simdf_load( cs, hc );                    \
    stbir__simdf_0123to0000( c, cs );               \
    stbir__simdf_mult_mem( tot, c, decode );        \
    stbir__simdf_0123to1111( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+4 ); \
    stbir__simdf_0123to2222( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+8 );

#define stbir__store_output_tiny()                \
    stbir__simdf_store( output, tot );            \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,c,cs; stbir__simdf t;  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode );     \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+8 );

#define stbir__4_coeff_continue_from_4( ofs )                  \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) );                    \
    stbir__simdf8_0123to00001111( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );

#define stbir__1_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load1rep4( t, hc + (ofs) );                   \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*4 );

#define stbir__2_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) - 2 );                \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );

 #define stbir__3_coeff_remnant( ofs )                         \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) );                    \
    stbir__simdf8_0123to00001111( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf8_0123to2222( t, cs );                         \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*4+8 );

#define stbir__store_output()                      \
    stbir__simdf8_add4halves( t, stbir__if_simdf8_cast_to_simdf4(tot0), tot0 );     \
    stbir__simdf_store( output, t );               \
    horizontal_coefficients += coefficient_width;  \
    ++horizontal_contributors;                     \
    output += 4;

#else

#define stbir__4_coeff_start()                        \
    stbir__simdf tot0,tot1,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                     \
    stbir__simdf_load( cs, hc );                      \
    stbir__simdf_0123to0000( c, cs );                 \
    stbir__simdf_mult_mem( tot0, c, decode );         \
    stbir__simdf_0123to1111( c, cs );                 \
    stbir__simdf_mult_mem( tot1, c, decode+4 );       \
    stbir__simdf_0123to2222( c, cs );                 \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+8 ); \
    stbir__simdf_0123to3333( c, cs );                 \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+12 );

#define stbir__4_coeff_continue_from_4( ofs )                  \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );    \
    stbir__simdf_0123to1111( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );  \
    stbir__simdf_0123to2222( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );  \
    stbir__simdf_0123to3333( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+12 );

#define stbir__1_coeff_remnant( ofs )                       \
    STBIR_SIMD_NO_UNROLL(decode);                           \
    stbir__simdf_load1( c, hc + (ofs) );                    \
    stbir__simdf_0123to0000( c, c );                        \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );

#define stbir__2_coeff_remnant( ofs )                         \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load2( cs, hc + (ofs) );                     \
    stbir__simdf_0123to0000( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf_0123to1111( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );

#define stbir__3_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );    \
    stbir__simdf_0123to1111( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );  \
    stbir__simdf_0123to2222( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#endif

#else

#define stbir__1_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;

#define stbir__2_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;               \
    c = hc[1];                        \
    p0 += decode[4] * c;              \
    p1 += decode[5] * c;              \
    p2 += decode[6] * c;              \
    p3 += decode[7] * c;

#define stbir__3_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;               \
    c = hc[1];                        \
    p0 += decode[4] * c;              \
    p1 += decode[5] * c;              \
    p2 += decode[6] * c;              \
    p3 += decode[7] * c;              \
    c = hc[2];                        \
    p0 += decode[8] * c;              \
    p1 += decode[9] * c;              \
    p2 += decode[10] * c;             \
    p3 += decode[11] * c;

#define stbir__store_output_tiny()                \
    output[0] = p0;                               \
    output[1] = p1;                               \
    output[2] = p2;                               \
    output[3] = p3;                               \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#define stbir__4_coeff_start()        \
    float x0,x1,x2,x3,y0,y1,y2,y3,c;  \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    x0 = decode[0] * c;               \
    x1 = decode[1] * c;               \
    x2 = decode[2] * c;               \
    x3 = decode[3] * c;               \
    c = hc[1];                        \
    y0 = decode[4] * c;               \
    y1 = decode[5] * c;               \
    y2 = decode[6] * c;               \
    y3 = decode[7] * c;               \
    c = hc[2];                        \
    x0 += decode[8] * c;              \
    x1 += decode[9] * c;              \
    x2 += decode[10] * c;             \
    x3 += decode[11] * c;             \
    c = hc[3];                        \
    y0 += decode[12] * c;             \
    y1 += decode[13] * c;             \
    y2 += decode[14] * c;             \
    y3 += decode[15] * c;

#define stbir__4_coeff_continue_from_4( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;      \
    c = hc[2+(ofs)];                  \
    x0 += decode[8+(ofs)*4] * c;      \
    x1 += decode[9+(ofs)*4] * c;      \
    x2 += decode[10+(ofs)*4] * c;     \
    x3 += decode[11+(ofs)*4] * c;     \
    c = hc[3+(ofs)];                  \
    y0 += decode[12+(ofs)*4] * c;     \
    y1 += decode[13+(ofs)*4] * c;     \
    y2 += decode[14+(ofs)*4] * c;     \
    y3 += decode[15+(ofs)*4] * c;

#define stbir__1_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;

#define stbir__2_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;

#define stbir__3_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;      \
    c = hc[2+(ofs)];                  \
    x0 += decode[8+(ofs)*4] * c;      \
    x1 += decode[9+(ofs)*4] * c;      \
    x2 += decode[10+(ofs)*4] * c;     \
    x3 += decode[11+(ofs)*4] * c;

#define stbir__store_output()                     \
    output[0] = x0 + y0;                          \
    output[1] = x1 + y1;                          \
    output[2] = x2 + y2;                          \
    output[3] = x3 + y3;                          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#endif

#define STBIR__horizontal_channels 4
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME



//=================
// Do 7 channel horizontal routines

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()                   \
    stbir__simdf tot0,tot1,c;                   \
    STBIR_SIMD_NO_UNROLL(decode);               \
    stbir__simdf_load1( c, hc );                \
    stbir__simdf_0123to0000( c, c );            \
    stbir__simdf_mult_mem( tot0, c, decode );   \
    stbir__simdf_mult_mem( tot1, c, decode+3 );

#define stbir__2_coeff_only()                         \
    stbir__simdf tot0,tot1,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                     \
    stbir__simdf_load2( cs, hc );                     \
    stbir__simdf_0123to0000( c, cs );                 \
    stbir__simdf_mult_mem( tot0, c, decode );         \
    stbir__simdf_mult_mem( tot1, c, decode+3 );       \
    stbir__simdf_0123to1111( c, cs );                 \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+7 ); \
    stbir__simdf_madd_mem( tot1, tot1, c,decode+10 );

#define stbir__3_coeff_only()                           \
    stbir__simdf tot0,tot1,c,cs;                        \
    STBIR_SIMD_NO_UNROLL(decode);                       \
    stbir__simdf_load( cs, hc );                        \
    stbir__simdf_0123to0000( c, cs );                   \
    stbir__simdf_mult_mem( tot0, c, decode );           \
    stbir__simdf_mult_mem( tot1, c, decode+3 );         \
    stbir__simdf_0123to1111( c, cs );                   \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+7 );   \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+10 );  \
    stbir__simdf_0123to2222( c, cs );                   \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+17 );

#define stbir__store_output_tiny()                \
    stbir__simdf_store( output+3, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,tot1,c,cs;                  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00000000( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode );     \
    stbir__simdf8_0123to11111111( c, cs );         \
    stbir__simdf8_mult_mem( tot1, c, decode+7 );   \
    stbir__simdf8_0123to22222222( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf8_0123to33333333( c, cs );         \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+21 );

#define stbir__4_coeff_continue_from_4( ofs )                   \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );                     \
    stbir__simdf8_0123to00000000( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_0123to11111111( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );  \
    stbir__simdf8_0123to22222222( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 ); \
    stbir__simdf8_0123to33333333( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+21 );

#define stbir__1_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load1b( c, hc + (ofs) );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );

#define stbir__2_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load1b( c, hc + (ofs) );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_load1b( c, hc + (ofs)+1 );                    \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );

#define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );                     \
    stbir__simdf8_0123to00000000( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_0123to11111111( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );  \
    stbir__simdf8_0123to22222222( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );

#define stbir__store_output()                     \
    stbir__simdf8_add( tot0, tot0, tot1 );        \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;                                  \
    if ( output < output_end )                    \
    {                                             \
      stbir__simdf8_store( output-7, tot0 );      \
      continue;                                   \
    }                                             \
    stbir__simdf_store( output-7+3, stbir__simdf_swiz(stbir__simdf8_gettop4(tot0),0,0,1,2) ); \
    stbir__simdf_store( output-7, stbir__if_simdf8_cast_to_simdf4(tot0) );           \
    break;

#else

#define stbir__4_coeff_start()                    \
    stbir__simdf tot0,tot1,tot2,tot3,c,cs;        \
    STBIR_SIMD_NO_UNROLL(decode);                 \
    stbir__simdf_load( cs, hc );                  \
    stbir__simdf_0123to0000( c, cs );             \
    stbir__simdf_mult_mem( tot0, c, decode );     \
    stbir__simdf_mult_mem( tot1, c, decode+3 );   \
    stbir__simdf_0123to1111( c, cs );             \
    stbir__simdf_mult_mem( tot2, c, decode+7 );   \
    stbir__simdf_mult_mem( tot3, c, decode+10 );  \
    stbir__simdf_0123to2222( c, cs );             \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+17 );  \
    stbir__simdf_0123to3333( c, cs );                   \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+21 );  \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+24 );

#define stbir__4_coeff_continue_from_4( ofs )                   \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load( cs, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );  \
    stbir__simdf_0123to2222( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+17 );  \
    stbir__simdf_0123to3333( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+21 );  \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+24 );

#define stbir__1_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load1( c, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, c );                            \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \

#define stbir__2_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load2( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );

#define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load( cs, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );  \
    stbir__simdf_0123to2222( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+17 );

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot2 );         \
    stbir__simdf_add( tot1, tot1, tot3 );         \
    stbir__simdf_store( output+3, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#endif

#else

#define stbir__1_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;

#define stbir__2_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;              \
    c = hc[1];                       \
    tot0 += decode[7]*c;             \
    tot1 += decode[8]*c;             \
    tot2 += decode[9]*c;             \
    tot3 += decode[10]*c;            \
    tot4 += decode[11]*c;            \
    tot5 += decode[12]*c;            \
    tot6 += decode[13]*c;            \

#define stbir__3_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;              \
    c = hc[1];                       \
    tot0 += decode[7]*c;             \
    tot1 += decode[8]*c;             \
    tot2 += decode[9]*c;             \
    tot3 += decode[10]*c;            \
    tot4 += decode[11]*c;            \
    tot5 += decode[12]*c;            \
    tot6 += decode[13]*c;            \
    c = hc[2];                       \
    tot0 += decode[14]*c;            \
    tot1 += decode[15]*c;            \
    tot2 += decode[16]*c;            \
    tot3 += decode[17]*c;            \
    tot4 += decode[18]*c;            \
    tot5 += decode[19]*c;            \
    tot6 += decode[20]*c;            \

#define stbir__store_output_tiny()                \
    output[0] = tot0;                             \
    output[1] = tot1;                             \
    output[2] = tot2;                             \
    output[3] = tot3;                             \
    output[4] = tot4;                             \
    output[5] = tot5;                             \
    output[6] = tot6;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#define stbir__4_coeff_start()    \
    float x0,x1,x2,x3,x4,x5,x6,y0,y1,y2,y3,y4,y5,y6,c; \
    STBIR_SIMD_NO_UNROLL(decode); \
    c = hc[0];                    \
    x0 = decode[0] * c;           \
    x1 = decode[1] * c;           \
    x2 = decode[2] * c;           \
    x3 = decode[3] * c;           \
    x4 = decode[4] * c;           \
    x5 = decode[5] * c;           \
    x6 = decode[6] * c;           \
    c = hc[1];                    \
    y0 = decode[7] * c;           \
    y1 = decode[8] * c;           \
    y2 = decode[9] * c;           \
    y3 = decode[10] * c;          \
    y4 = decode[11] * c;          \
    y5 = decode[12] * c;          \
    y6 = decode[13] * c;          \
    c = hc[2];                    \
    x0 += decode[14] * c;         \
    x1 += decode[15] * c;         \
    x2 += decode[16] * c;         \
    x3 += decode[17] * c;         \
    x4 += decode[18] * c;         \
    x5 += decode[19] * c;         \
    x6 += decode[20] * c;         \
    c = hc[3];                    \
    y0 += decode[21] * c;         \
    y1 += decode[22] * c;         \
    y2 += decode[23] * c;         \
    y3 += decode[24] * c;         \
    y4 += decode[25] * c;         \
    y5 += decode[26] * c;         \
    y6 += decode[27] * c;

#define stbir__4_coeff_continue_from_4( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \
    c = hc[2+(ofs)];               \
    x0 += decode[14+(ofs)*7] * c;  \
    x1 += decode[15+(ofs)*7] * c;  \
    x2 += decode[16+(ofs)*7] * c;  \
    x3 += decode[17+(ofs)*7] * c;  \
    x4 += decode[18+(ofs)*7] * c;  \
    x5 += decode[19+(ofs)*7] * c;  \
    x6 += decode[20+(ofs)*7] * c;  \
    c = hc[3+(ofs)];               \
    y0 += decode[21+(ofs)*7] * c;  \
    y1 += decode[22+(ofs)*7] * c;  \
    y2 += decode[23+(ofs)*7] * c;  \
    y3 += decode[24+(ofs)*7] * c;  \
    y4 += decode[25+(ofs)*7] * c;  \
    y5 += decode[26+(ofs)*7] * c;  \
    y6 += decode[27+(ofs)*7] * c;

#define stbir__1_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \

#define stbir__2_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \

#define stbir__3_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \
    c = hc[2+(ofs)];               \
    x0 += decode[14+(ofs)*7] * c;  \
    x1 += decode[15+(ofs)*7] * c;  \
    x2 += decode[16+(ofs)*7] * c;  \
    x3 += decode[17+(ofs)*7] * c;  \
    x4 += decode[18+(ofs)*7] * c;  \
    x5 += decode[19+(ofs)*7] * c;  \
    x6 += decode[20+(ofs)*7] * c;  \

#define stbir__store_output()                     \
    output[0] = x0 + y0;                          \
    output[1] = x1 + y1;                          \
    output[2] = x2 + y2;                          \
    output[3] = x3 + y3;                          \
    output[4] = x4 + y4;                          \
    output[5] = x5 + y5;                          \
    output[6] = x6 + y6;                          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#endif

#define STBIR__horizontal_channels 7
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME


// include all of the vertical resamplers (both scatter and gather versions)

#define STBIR__vertical_channels 1
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 1
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 2
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 2
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 3
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 3
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 4
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 4
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 5
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 5
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 6
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 6
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 7
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 7
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 8
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 8
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

typedef void STBIR_VERTICAL_GATHERFUNC( float * output, float const * coeffs, float const ** inputs, float const * input0_end );

static STBIR_VERTICAL_GATHERFUNC * stbir__vertical_gathers[ 8 ] =
{
  stbir__vertical_gather_with_1_coeffs,stbir__vertical_gather_with_2_coeffs,stbir__vertical_gather_with_3_coeffs,stbir__vertical_gather_with_4_coeffs,stbir__vertical_gather_with_5_coeffs,stbir__vertical_gather_with_6_coeffs,stbir__vertical_gather_with_7_coeffs,stbir__vertical_gather_with_8_coeffs
};

static STBIR_VERTICAL_GATHERFUNC * stbir__vertical_gathers_continues[ 8 ] =
{
  stbir__vertical_gather_with_1_coeffs_cont,stbir__vertical_gather_with_2_coeffs_cont,stbir__vertical_gather_with_3_coeffs_cont,stbir__vertical_gather_with_4_coeffs_cont,stbir__vertical_gather_with_5_coeffs_cont,stbir__vertical_gather_with_6_coeffs_cont,stbir__vertical_gather_with_7_coeffs_cont,stbir__vertical_gather_with_8_coeffs_cont
};

typedef void STBIR_VERTICAL_SCATTERFUNC( float ** outputs, float const * coeffs, float const * input, float const * input_end );

static STBIR_VERTICAL_SCATTERFUNC * stbir__vertical_scatter_sets[ 8 ] =
{
  stbir__vertical_scatter_with_1_coeffs,stbir__vertical_scatter_with_2_coeffs,stbir__vertical_scatter_with_3_coeffs,stbir__vertical_scatter_with_4_coeffs,stbir__vertical_scatter_with_5_coeffs,stbir__vertical_scatter_with_6_coeffs,stbir__vertical_scatter_with_7_coeffs,stbir__vertical_scatter_with_8_coeffs
};

static STBIR_VERTICAL_SCATTERFUNC * stbir__vertical_scatter_blends[ 8 ] =
{
  stbir__vertical_scatter_with_1_coeffs_cont,stbir__vertical_scatter_with_2_coeffs_cont,stbir__vertical_scatter_with_3_coeffs_cont,stbir__vertical_scatter_with_4_coeffs_cont,stbir__vertical_scatter_with_5_coeffs_cont,stbir__vertical_scatter_with_6_coeffs_cont,stbir__vertical_scatter_with_7_coeffs_cont,stbir__vertical_scatter_with_8_coeffs_cont
};


static void stbir__encode_scanline( stbir__info const * stbir_info, void *output_buffer_data, float * encode_buffer, int row  STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  int num_pixels = stbir_info->horizontal.scale_info.output_sub_size;
  int channels = stbir_info->channels;
  int width_times_channels = num_pixels * channels;
  void * output_buffer;

  // un-alpha weight if we need to
  if ( stbir_info->alpha_unweight )
  {
    STBIR_PROFILE_START( unalpha );
    stbir_info->alpha_unweight( encode_buffer, width_times_channels );
    STBIR_PROFILE_END( unalpha );
  }

  // write directly into output by default
  output_buffer = output_buffer_data;

  // if we have an output callback, we first convert the decode buffer in place (and then hand that to the callback)
  if ( stbir_info->out_pixels_cb )
    output_buffer = encode_buffer;

  STBIR_PROFILE_START( encode );
  // convert into the output buffer
  stbir_info->encode_pixels( output_buffer, width_times_channels, encode_buffer );
  STBIR_PROFILE_END( encode );

  // if we have an output callback, call it to send the data
  if ( stbir_info->out_pixels_cb )
    stbir_info->out_pixels_cb( output_buffer, num_pixels, row, stbir_info->user_data );
}


// Get the ring buffer pointer for an index
static float* stbir__get_ring_buffer_entry(stbir__info const * stbir_info, stbir__per_split_info const * split_info, int index )
{
  STBIR_ASSERT( index < stbir_info->ring_buffer_num_entries );

  #ifdef STBIR__SEPARATE_ALLOCATIONS
    return split_info->ring_buffers[ index ];
  #else
    return (float*) ( ( (char*) split_info->ring_buffer ) + ( index * stbir_info->ring_buffer_length_bytes ) );
  #endif
}

// Get the specified scan line from the ring buffer
static float* stbir__get_ring_buffer_scanline(stbir__info const * stbir_info, stbir__per_split_info const * split_info, int get_scanline)
{
  int ring_buffer_index = (split_info->ring_buffer_begin_index + (get_scanline - split_info->ring_buffer_first_scanline)) % stbir_info->ring_buffer_num_entries;
  return stbir__get_ring_buffer_entry( stbir_info, split_info, ring_buffer_index );
}

static void stbir__resample_horizontal_gather(stbir__info const * stbir_info, float* output_buffer, float const * input_buffer STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  float const * decode_buffer = input_buffer - ( stbir_info->scanline_extents.conservative.n0 * stbir_info->effective_channels );

  STBIR_PROFILE_START( horizontal );
  if ( ( stbir_info->horizontal.filter_enum == STBIR_FILTER_POINT_SAMPLE ) && ( stbir_info->horizontal.scale_info.scale == 1.0f ) )
    STBIR_MEMCPY( output_buffer, input_buffer, stbir_info->horizontal.scale_info.output_sub_size * sizeof( float ) * stbir_info->effective_channels );
  else
    stbir_info->horizontal_gather_channels( output_buffer, stbir_info->horizontal.scale_info.output_sub_size, decode_buffer, stbir_info->horizontal.contributors, stbir_info->horizontal.coefficients, stbir_info->horizontal.coefficient_width );
  STBIR_PROFILE_END( horizontal );
}

static void stbir__resample_vertical_gather(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n, int contrib_n0, int contrib_n1, float const * vertical_coefficients )
{
  float* encode_buffer = split_info->vertical_buffer;
  float* decode_buffer = split_info->decode_buffer;
  int vertical_first = stbir_info->vertical_first;
  int width = (vertical_first) ? ( stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1 ) : stbir_info->horizontal.scale_info.output_sub_size;
  int width_times_channels = stbir_info->effective_channels * width;

  STBIR_ASSERT( stbir_info->vertical.is_gather );

  // loop over the contributing scanlines and scale into the buffer
  STBIR_PROFILE_START( vertical );
  {
    int k = 0, total = contrib_n1 - contrib_n0 + 1;
    STBIR_ASSERT( total > 0 );
    do {
      float const * inputs[8];
      int i, cnt = total; if ( cnt > 8 ) cnt = 8;
      for( i = 0 ; i < cnt ; i++ )
        inputs[ i ] = stbir__get_ring_buffer_scanline(stbir_info, split_info, k+i+contrib_n0 );

      // call the N scanlines at a time function (up to 8 scanlines of blending at once)
      ((k==0)?stbir__vertical_gathers:stbir__vertical_gathers_continues)[cnt-1]( (vertical_first) ? decode_buffer : encode_buffer, vertical_coefficients + k, inputs, inputs[0] + width_times_channels );
      k += cnt;
      total -= cnt;
    } while ( total );
  }
  STBIR_PROFILE_END( vertical );

  if ( vertical_first )
  {
    // Now resample the gathered vertical data in the horizontal axis into the encode buffer
    decode_buffer[ width_times_channels ] = 0.0f; // clear two over for horizontals with a remnant of 3
    decode_buffer[ width_times_channels+1 ] = 0.0f; 
    stbir__resample_horizontal_gather(stbir_info, encode_buffer, decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
  }

  stbir__encode_scanline( stbir_info, ( (char *) stbir_info->output_data ) + ((size_t)n * (size_t)stbir_info->output_stride_bytes),
                          encode_buffer, n  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
}

static void stbir__decode_and_resample_for_vertical_gather_loop(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n)
{
  int ring_buffer_index;
  float* ring_buffer;

  // Decode the nth scanline from the source image into the decode buffer.
  stbir__decode_scanline( stbir_info, n, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  // update new end scanline
  split_info->ring_buffer_last_scanline = n;

  // get ring buffer
  ring_buffer_index = (split_info->ring_buffer_begin_index + (split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline)) % stbir_info->ring_buffer_num_entries;
  ring_buffer = stbir__get_ring_buffer_entry(stbir_info, split_info, ring_buffer_index);

  // Now resample it into the ring buffer.
  stbir__resample_horizontal_gather( stbir_info, ring_buffer, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  // Now it's sitting in the ring buffer ready to be used as source for the vertical sampling.
}

static void stbir__vertical_gather_loop( stbir__info const * stbir_info, stbir__per_split_info* split_info, int split_count )
{
  int y, start_output_y, end_output_y;
  stbir__contributors* vertical_contributors = stbir_info->vertical.contributors;
  float const * vertical_coefficients = stbir_info->vertical.coefficients;

  STBIR_ASSERT( stbir_info->vertical.is_gather );

  start_output_y = split_info->start_output_y;
  end_output_y = split_info[split_count-1].end_output_y;

  vertical_contributors += start_output_y;
  vertical_coefficients += start_output_y * stbir_info->vertical.coefficient_width;

  // initialize the ring buffer for gathering
  split_info->ring_buffer_begin_index = 0;
  split_info->ring_buffer_first_scanline = vertical_contributors->n0;
  split_info->ring_buffer_last_scanline = split_info->ring_buffer_first_scanline - 1; // means "empty"

  for (y = start_output_y; y < end_output_y; y++)
  {
    int in_first_scanline, in_last_scanline;

    in_first_scanline = vertical_contributors->n0;
    in_last_scanline = vertical_contributors->n1;

    // make sure the indexing hasn't broken
    STBIR_ASSERT( in_first_scanline >= split_info->ring_buffer_first_scanline );

    // Load in new scanlines
    while (in_last_scanline > split_info->ring_buffer_last_scanline)
    {
      STBIR_ASSERT( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) <= stbir_info->ring_buffer_num_entries );

      // make sure there was room in the ring buffer when we add new scanlines
      if ( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) == stbir_info->ring_buffer_num_entries )
      {
        split_info->ring_buffer_first_scanline++;
        split_info->ring_buffer_begin_index++;
      }

      if ( stbir_info->vertical_first )
      {
        float * ring_buffer = stbir__get_ring_buffer_scanline( stbir_info, split_info, ++split_info->ring_buffer_last_scanline );
        // Decode the nth scanline from the source image into the decode buffer.
        stbir__decode_scanline( stbir_info, split_info->ring_buffer_last_scanline, ring_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
      }
      else
      {
        stbir__decode_and_resample_for_vertical_gather_loop(stbir_info, split_info, split_info->ring_buffer_last_scanline + 1);
      }
    }

    // Now all buffers should be ready to write a row of vertical sampling, so do it.
    stbir__resample_vertical_gather(stbir_info, split_info, y, in_first_scanline, in_last_scanline, vertical_coefficients );

    ++vertical_contributors;
    vertical_coefficients += stbir_info->vertical.coefficient_width;
  }
}

#define STBIR__FLOAT_EMPTY_MARKER 3.0e+38F
#define STBIR__FLOAT_BUFFER_IS_EMPTY(ptr) ((ptr)[0]==STBIR__FLOAT_EMPTY_MARKER)

static void stbir__encode_first_scanline_from_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info)
{
  // evict a scanline out into the output buffer
  float* ring_buffer_entry = stbir__get_ring_buffer_entry(stbir_info, split_info, split_info->ring_buffer_begin_index );

  // dump the scanline out
  stbir__encode_scanline( stbir_info, ( (char *)stbir_info->output_data ) + ( (size_t)split_info->ring_buffer_first_scanline * (size_t)stbir_info->output_stride_bytes ), ring_buffer_entry, split_info->ring_buffer_first_scanline  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  // mark it as empty
  ring_buffer_entry[ 0 ] = STBIR__FLOAT_EMPTY_MARKER;

  // advance the first scanline
  split_info->ring_buffer_first_scanline++;
  if ( ++split_info->ring_buffer_begin_index == stbir_info->ring_buffer_num_entries )
    split_info->ring_buffer_begin_index = 0;
}

static void stbir__horizontal_resample_and_encode_first_scanline_from_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info)
{
  // evict a scanline out into the output buffer

  float* ring_buffer_entry = stbir__get_ring_buffer_entry(stbir_info, split_info, split_info->ring_buffer_begin_index );

  // Now resample it into the buffer.
  stbir__resample_horizontal_gather( stbir_info, split_info->vertical_buffer, ring_buffer_entry  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  // dump the scanline out
  stbir__encode_scanline( stbir_info, ( (char *)stbir_info->output_data ) + ( (size_t)split_info->ring_buffer_first_scanline * (size_t)stbir_info->output_stride_bytes ), split_info->vertical_buffer, split_info->ring_buffer_first_scanline  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  // mark it as empty
  ring_buffer_entry[ 0 ] = STBIR__FLOAT_EMPTY_MARKER;

  // advance the first scanline
  split_info->ring_buffer_first_scanline++;
  if ( ++split_info->ring_buffer_begin_index == stbir_info->ring_buffer_num_entries )
    split_info->ring_buffer_begin_index = 0;
}

static void stbir__resample_vertical_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n0, int n1, float const * vertical_coefficients, float const * vertical_buffer, float const * vertical_buffer_end )
{
  STBIR_ASSERT( !stbir_info->vertical.is_gather );

  STBIR_PROFILE_START( vertical );
  {
    int k = 0, total = n1 - n0 + 1;
    STBIR_ASSERT( total > 0 );
    do {
      float * outputs[8];
      int i, n = total; if ( n > 8 ) n = 8;
      for( i = 0 ; i < n ; i++ )
      {
        outputs[ i ] = stbir__get_ring_buffer_scanline(stbir_info, split_info, k+i+n0 );
        if ( ( i ) && ( STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[i] ) != STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[0] ) ) ) // make sure runs are of the same type
        {
          n = i;
          break;
        }
      }
      // call the scatter to N scanlines at a time function (up to 8 scanlines of scattering at once)
      ((STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[0] ))?stbir__vertical_scatter_sets:stbir__vertical_scatter_blends)[n-1]( outputs, vertical_coefficients + k, vertical_buffer, vertical_buffer_end );
      k += n;
      total -= n;
    } while ( total );
  }

  STBIR_PROFILE_END( vertical );
}

typedef void stbir__handle_scanline_for_scatter_func(stbir__info const * stbir_info, stbir__per_split_info* split_info);

static void stbir__vertical_scatter_loop( stbir__info const * stbir_info, stbir__per_split_info* split_info, int split_count )
{
  int y, start_output_y, end_output_y, start_input_y, end_input_y;
  stbir__contributors* vertical_contributors = stbir_info->vertical.contributors;
  float const * vertical_coefficients = stbir_info->vertical.coefficients;
  stbir__handle_scanline_for_scatter_func * handle_scanline_for_scatter;
  void * scanline_scatter_buffer;
  void * scanline_scatter_buffer_end;
  int on_first_input_y, last_input_y;
  int width = (stbir_info->vertical_first) ? ( stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1 ) : stbir_info->horizontal.scale_info.output_sub_size;
  int width_times_channels = stbir_info->effective_channels * width;

  STBIR_ASSERT( !stbir_info->vertical.is_gather );

  start_output_y = split_info->start_output_y;
  end_output_y = split_info[split_count-1].end_output_y;  // may do multiple split counts

  start_input_y = split_info->start_input_y;
  end_input_y = split_info[split_count-1].end_input_y;

  // adjust for starting offset start_input_y
  y = start_input_y + stbir_info->vertical.filter_pixel_margin;
  vertical_contributors += y ;
  vertical_coefficients += stbir_info->vertical.coefficient_width * y;

  if ( stbir_info->vertical_first )
  {
    handle_scanline_for_scatter = stbir__horizontal_resample_and_encode_first_scanline_from_scatter;
    scanline_scatter_buffer = split_info->decode_buffer;
    scanline_scatter_buffer_end = ( (char*) scanline_scatter_buffer ) + sizeof( float ) * stbir_info->effective_channels * (stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1);
  }
  else
  {
    handle_scanline_for_scatter = stbir__encode_first_scanline_from_scatter;
    scanline_scatter_buffer = split_info->vertical_buffer;
    scanline_scatter_buffer_end = ( (char*) scanline_scatter_buffer ) + sizeof( float ) * stbir_info->effective_channels * stbir_info->horizontal.scale_info.output_sub_size;
  }

  // initialize the ring buffer for scattering
  split_info->ring_buffer_first_scanline = start_output_y;
  split_info->ring_buffer_last_scanline = -1;
  split_info->ring_buffer_begin_index = -1;

  // mark all the buffers as empty to start
  for( y = 0 ; y < stbir_info->ring_buffer_num_entries ; y++ )
  {
    float * decode_buffer = stbir__get_ring_buffer_entry( stbir_info, split_info, y );
    decode_buffer[ width_times_channels ] = 0.0f; // clear two over for horizontals with a remnant of 3
    decode_buffer[ width_times_channels+1 ] = 0.0f; 
    decode_buffer[0] = STBIR__FLOAT_EMPTY_MARKER; // only used on scatter
  }

  // do the loop in input space
  on_first_input_y = 1; last_input_y = start_input_y;
  for (y = start_input_y ; y < end_input_y; y++)
  {
    int out_first_scanline, out_last_scanline;

    out_first_scanline = vertical_contributors->n0;
    out_last_scanline = vertical_contributors->n1;

    STBIR_ASSERT(out_last_scanline - out_first_scanline + 1 <= stbir_info->ring_buffer_num_entries);

    if ( ( out_last_scanline >= out_first_scanline ) && ( ( ( out_first_scanline >= start_output_y ) && ( out_first_scanline < end_output_y ) ) || ( ( out_last_scanline >= start_output_y ) && ( out_last_scanline < end_output_y ) ) ) )
    {
      float const * vc = vertical_coefficients;

      // keep track of the range actually seen for the next resize
      last_input_y = y;
      if ( ( on_first_input_y ) && ( y > start_input_y ) )
        split_info->start_input_y = y;
      on_first_input_y = 0;

      // clip the region
      if ( out_first_scanline < start_output_y )
      {
        vc += start_output_y - out_first_scanline;
        out_first_scanline = start_output_y;
      }

      if ( out_last_scanline >= end_output_y )
        out_last_scanline = end_output_y - 1;

      // if very first scanline, init the index
      if (split_info->ring_buffer_begin_index < 0)
        split_info->ring_buffer_begin_index = out_first_scanline - start_output_y;

      STBIR_ASSERT( split_info->ring_buffer_begin_index <= out_first_scanline );

      // Decode the nth scanline from the source image into the decode buffer.
      stbir__decode_scanline( stbir_info, y, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

      // When horizontal first, we resample horizontally into the vertical buffer before we scatter it out
      if ( !stbir_info->vertical_first )
        stbir__resample_horizontal_gather( stbir_info, split_info->vertical_buffer, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

      // Now it's sitting in the buffer ready to be distributed into the ring buffers.

      // evict from the ringbuffer, if we need are full
      if ( ( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) == stbir_info->ring_buffer_num_entries ) &&
           ( out_last_scanline > split_info->ring_buffer_last_scanline ) )
        handle_scanline_for_scatter( stbir_info, split_info );

      // Now the horizontal buffer is ready to write to all ring buffer rows, so do it.
      stbir__resample_vertical_scatter(stbir_info, split_info, out_first_scanline, out_last_scanline, vc, (float*)scanline_scatter_buffer, (float*)scanline_scatter_buffer_end );

      // update the end of the buffer
      if ( out_last_scanline > split_info->ring_buffer_last_scanline )
        split_info->ring_buffer_last_scanline = out_last_scanline;
    }
    ++vertical_contributors;
    vertical_coefficients += stbir_info->vertical.coefficient_width;
  }

  // now evict the scanlines that are left over in the ring buffer
  while ( split_info->ring_buffer_first_scanline < end_output_y )
    handle_scanline_for_scatter(stbir_info, split_info);

  // update the end_input_y if we do multiple resizes with the same data
  ++last_input_y;
  for( y = 0 ; y < split_count; y++ )
    if ( split_info[y].end_input_y > last_input_y )
      split_info[y].end_input_y = last_input_y;
}


static stbir__kernel_callback * stbir__builtin_kernels[] =   { 0, stbir__filter_trapezoid,  stbir__filter_triangle, stbir__filter_cubic, stbir__filter_catmullrom, stbir__filter_mitchell, stbir__filter_point };
static stbir__support_callback * stbir__builtin_supports[] = { 0, stbir__support_trapezoid, stbir__support_one,     stbir__support_two,  stbir__support_two,       stbir__support_two,     stbir__support_zeropoint5 };

static void stbir__set_sampler(stbir__sampler * samp, stbir_filter filter, stbir__kernel_callback * kernel, stbir__support_callback * support, stbir_edge edge, stbir__scale_info * scale_info, int always_gather, void * user_data )
{
  // set filter
  if (filter == 0)
  {
    filter = STBIR_DEFAULT_FILTER_DOWNSAMPLE; // default to downsample
    if (scale_info->scale >= ( 1.0f - stbir__small_float ) )
    {
      if ( (scale_info->scale <= ( 1.0f + stbir__small_float ) ) && ( STBIR_CEILF(scale_info->pixel_shift) == scale_info->pixel_shift ) )
        filter = STBIR_FILTER_POINT_SAMPLE;
      else
        filter = STBIR_DEFAULT_FILTER_UPSAMPLE;
    }
  }
  samp->filter_enum = filter;

  STBIR_ASSERT(samp->filter_enum != 0);
  STBIR_ASSERT((unsigned)samp->filter_enum < STBIR_FILTER_OTHER);
  samp->filter_kernel = stbir__builtin_kernels[ filter ];
  samp->filter_support = stbir__builtin_supports[ filter ];

  if ( kernel && support )
  {
    samp->filter_kernel = kernel;
    samp->filter_support = support;
    samp->filter_enum = STBIR_FILTER_OTHER;
  }

  samp->edge = edge;
  samp->filter_pixel_width  = stbir__get_filter_pixel_width (samp->filter_support, scale_info->scale, user_data );
  // Gather is always better, but in extreme downsamples, you have to most or all of the data in memory
  //    For horizontal, we always have all the pixels, so we always use gather here (always_gather==1).
  //    For vertical, we use gather if scaling up (which means we will have samp->filter_pixel_width
  //    scanlines in memory at once).
  samp->is_gather = 0;
  if ( scale_info->scale >= ( 1.0f - stbir__small_float ) )
    samp->is_gather = 1;
  else if ( ( always_gather ) || ( samp->filter_pixel_width <= STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT ) )
    samp->is_gather = 2;

  // pre calculate stuff based on the above
  samp->coefficient_width = stbir__get_coefficient_width(samp, samp->is_gather, user_data);

  // filter_pixel_width is the conservative size in pixels of input that affect an output pixel.
  //   In rare cases (only with 2 pix to 1 pix with the default filters), it's possible that the 
  //   filter will extend before or after the scanline beyond just one extra entire copy of the 
  //   scanline (we would hit the edge twice). We don't let you do that, so we clamp the total 
  //   width to 3x the total of input pixel (once for the scanline, once for the left side 
  //   overhang, and once for the right side). We only do this for edge mode, since the other 
  //   modes can just re-edge clamp back in again.
  if ( edge == STBIR_EDGE_WRAP )
    if ( samp->filter_pixel_width > ( scale_info->input_full_size * 3 ) )
      samp->filter_pixel_width = scale_info->input_full_size * 3;

  // This is how much to expand buffers to account for filters seeking outside
  // the image boundaries.
  samp->filter_pixel_margin = samp->filter_pixel_width / 2;
  
  // filter_pixel_margin is the amount that this filter can overhang on just one side of either 
  //   end of the scanline (left or the right). Since we only allow you to overhang 1 scanline's 
  //   worth of pixels, we clamp this one side of overhang to the input scanline size. Again, 
  //   this clamping only happens in rare cases with the default filters (2 pix to 1 pix). 
  if ( edge == STBIR_EDGE_WRAP )
    if ( samp->filter_pixel_margin > scale_info->input_full_size )
      samp->filter_pixel_margin = scale_info->input_full_size;

  samp->num_contributors = stbir__get_contributors(samp, samp->is_gather);

  samp->contributors_size = samp->num_contributors * sizeof(stbir__contributors);
  samp->coefficients_size = samp->num_contributors * samp->coefficient_width * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra sizeof(float) is padding

  samp->gather_prescatter_contributors = 0;
  samp->gather_prescatter_coefficients = 0;
  if ( samp->is_gather == 0 )
  {
    samp->gather_prescatter_coefficient_width = samp->filter_pixel_width;
    samp->gather_prescatter_num_contributors  = stbir__get_contributors(samp, 2);
    samp->gather_prescatter_contributors_size = samp->gather_prescatter_num_contributors * sizeof(stbir__contributors);
    samp->gather_prescatter_coefficients_size = samp->gather_prescatter_num_contributors * samp->gather_prescatter_coefficient_width * sizeof(float);
  }
}

static void stbir__get_conservative_extents( stbir__sampler * samp, stbir__contributors * range, void * user_data )
{
  float scale = samp->scale_info.scale;
  float out_shift = samp->scale_info.pixel_shift;
  stbir__support_callback * support = samp->filter_support;
  int input_full_size = samp->scale_info.input_full_size;
  stbir_edge edge = samp->edge;
  float inv_scale = samp->scale_info.inv_scale;

  STBIR_ASSERT( samp->is_gather != 0 );

  if ( samp->is_gather == 1 )
  {
    int in_first_pixel, in_last_pixel;
    float out_filter_radius = support(inv_scale, user_data) * scale;

    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, 0.5, out_filter_radius, inv_scale, out_shift, input_full_size, edge );
    range->n0 = in_first_pixel;
    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, ( (float)(samp->scale_info.output_sub_size-1) ) + 0.5f, out_filter_radius, inv_scale, out_shift, input_full_size, edge );
    range->n1 = in_last_pixel;
  }
  else if ( samp->is_gather == 2 ) // downsample gather, refine
  {
    float in_pixels_radius = support(scale, user_data) * inv_scale;
    int filter_pixel_margin = samp->filter_pixel_margin;
    int output_sub_size = samp->scale_info.output_sub_size;
    int input_end;
    int n;
    int in_first_pixel, in_last_pixel;

    // get a conservative area of the input range
    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, 0, 0, inv_scale, out_shift, input_full_size, edge );
    range->n0 = in_first_pixel;
    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, (float)output_sub_size, 0, inv_scale, out_shift, input_full_size, edge );
    range->n1 = in_last_pixel;

    // now go through the margin to the start of area to find bottom
    n = range->n0 + 1;
    input_end = -filter_pixel_margin;
    while( n >= input_end )
    {
      int out_first_pixel, out_last_pixel;
      stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, ((float)n)+0.5f, in_pixels_radius, scale, out_shift, output_sub_size );
      if ( out_first_pixel > out_last_pixel )
        break;

      if ( ( out_first_pixel < output_sub_size ) || ( out_last_pixel >= 0 ) )
        range->n0 = n;
      --n;
    }

    // now go through the end of the area through the margin to find top
    n = range->n1 - 1;
    input_end = n + 1 + filter_pixel_margin;
    while( n <= input_end )
    {
      int out_first_pixel, out_last_pixel;
      stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, ((float)n)+0.5f, in_pixels_radius, scale, out_shift, output_sub_size );
      if ( out_first_pixel > out_last_pixel )
        break;
      if ( ( out_first_pixel < output_sub_size ) || ( out_last_pixel >= 0 ) )
        range->n1 = n;
      ++n;
    }
  }

  if ( samp->edge == STBIR_EDGE_WRAP )
  {
    // if we are wrapping, and we are very close to the image size (so the edges might merge), just use the scanline up to the edge
    if ( ( range->n0 > 0 ) && ( range->n1 >= input_full_size ) )
    {
      int marg = range->n1 - input_full_size + 1;
      if ( ( marg + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= range->n0 )
        range->n0 = 0;
    }
    if ( ( range->n0 < 0 ) && ( range->n1 < (input_full_size-1) ) )
    {
      int marg = -range->n0;
      if ( ( input_full_size - marg - STBIR__MERGE_RUNS_PIXEL_THRESHOLD - 1 ) <= range->n1 )
        range->n1 = input_full_size - 1;
    }
  }
  else
  {
    // for non-edge-wrap modes, we never read over the edge, so clamp
    if ( range->n0 < 0 )
      range->n0 = 0;
    if ( range->n1 >= input_full_size )
      range->n1 = input_full_size - 1;
  }
}

static void stbir__get_split_info( stbir__per_split_info* split_info, int splits, int output_height, int vertical_pixel_margin, int input_full_height )
{
  int i, cur;
  int left = output_height;

  cur = 0;
  for( i = 0 ; i < splits ; i++ )
  {
    int each;
    split_info[i].start_output_y = cur;
    each = left / ( splits - i );
    split_info[i].end_output_y = cur + each;
    cur += each;
    left -= each;

    // scatter range (updated to minimum as you run it)
    split_info[i].start_input_y = -vertical_pixel_margin;
    split_info[i].end_input_y = input_full_height + vertical_pixel_margin;
  }
}

static void stbir__free_internal_mem( stbir__info *info )
{
  #define STBIR__FREE_AND_CLEAR( ptr ) { if ( ptr ) { void * p = (ptr); (ptr) = 0; STBIR_FREE( p, info->user_data); } }

  if ( info )
  {
  #ifndef STBIR__SEPARATE_ALLOCATIONS
    STBIR__FREE_AND_CLEAR( info->alloced_mem );
  #else
    int i,j;

    if ( ( info->vertical.gather_prescatter_contributors ) && ( (void*)info->vertical.gather_prescatter_contributors != (void*)info->split_info[0].decode_buffer ) )
    {
      STBIR__FREE_AND_CLEAR( info->vertical.gather_prescatter_coefficients );
      STBIR__FREE_AND_CLEAR( info->vertical.gather_prescatter_contributors );
    }
    for( i = 0 ; i < info->splits ; i++ )
    {
      for( j = 0 ; j < info->alloc_ring_buffer_num_entries ; j++ )
      {
        #ifdef STBIR_SIMD8
        if ( info->effective_channels == 3 )
          --info->split_info[i].ring_buffers[j]; // avx in 3 channel mode needs one float at the start of the buffer
        #endif
        STBIR__FREE_AND_CLEAR( info->split_info[i].ring_buffers[j] );
      }

      #ifdef STBIR_SIMD8
      if ( info->effective_channels == 3 )
        --info->split_info[i].decode_buffer; // avx in 3 channel mode needs one float at the start of the buffer
      #endif
      STBIR__FREE_AND_CLEAR( info->split_info[i].decode_buffer );
      STBIR__FREE_AND_CLEAR( info->split_info[i].ring_buffers );
      STBIR__FREE_AND_CLEAR( info->split_info[i].vertical_buffer );
    }
    STBIR__FREE_AND_CLEAR( info->split_info );
    if ( info->vertical.coefficients != info->horizontal.coefficients )
    {
      STBIR__FREE_AND_CLEAR( info->vertical.coefficients );
      STBIR__FREE_AND_CLEAR( info->vertical.contributors );
    }
    STBIR__FREE_AND_CLEAR( info->horizontal.coefficients );
    STBIR__FREE_AND_CLEAR( info->horizontal.contributors );
    STBIR__FREE_AND_CLEAR( info->alloced_mem );
    STBIR_FREE( info, info->user_data );
  #endif
  }

  #undef STBIR__FREE_AND_CLEAR
}

static int stbir__get_max_split( int splits, int height )
{
  int i;
  int max = 0;

  for( i = 0 ; i < splits ; i++ )
  {
    int each = height / ( splits - i );
    if ( each > max )
      max = each;
    height -= each;
  }
  return max;
}

static stbir__horizontal_gather_channels_func ** stbir__horizontal_gather_n_coeffs_funcs[8] =
{
  0, stbir__horizontal_gather_1_channels_with_n_coeffs_funcs, stbir__horizontal_gather_2_channels_with_n_coeffs_funcs, stbir__horizontal_gather_3_channels_with_n_coeffs_funcs, stbir__horizontal_gather_4_channels_with_n_coeffs_funcs, 0,0, stbir__horizontal_gather_7_channels_with_n_coeffs_funcs
};

static stbir__horizontal_gather_channels_func ** stbir__horizontal_gather_channels_funcs[8] =
{
  0, stbir__horizontal_gather_1_channels_funcs, stbir__horizontal_gather_2_channels_funcs, stbir__horizontal_gather_3_channels_funcs, stbir__horizontal_gather_4_channels_funcs, 0,0, stbir__horizontal_gather_7_channels_funcs
};

// there are six resize classifications: 0 == vertical scatter, 1 == vertical gather < 1x scale, 2 == vertical gather 1x-2x scale, 4 == vertical gather < 3x scale, 4 == vertical gather > 3x scale, 5 == <=4 pixel height, 6 == <=4 pixel wide column
#define STBIR_RESIZE_CLASSIFICATIONS 8

static float stbir__compute_weights[5][STBIR_RESIZE_CLASSIFICATIONS][4]=  // 5 = 0=1chan, 1=2chan, 2=3chan, 3=4chan, 4=7chan
{
  {
    { 1.00000f, 1.00000f, 0.31250f, 1.00000f },
    { 0.56250f, 0.59375f, 0.00000f, 0.96875f },
    { 1.00000f, 0.06250f, 0.00000f, 1.00000f },
    { 0.00000f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.03125f },
  }, {
    { 0.00000f, 0.84375f, 0.00000f, 0.03125f },
    { 0.09375f, 0.93750f, 0.00000f, 0.78125f },
    { 0.87500f, 0.21875f, 0.00000f, 0.96875f },
    { 0.09375f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.53125f },
  }, {
    { 0.00000f, 0.53125f, 0.00000f, 0.03125f },
    { 0.06250f, 0.96875f, 0.00000f, 0.53125f },
    { 0.87500f, 0.18750f, 0.00000f, 0.93750f },
    { 0.00000f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.56250f },
  }, {
    { 0.00000f, 0.50000f, 0.00000f, 0.71875f },
    { 0.06250f, 0.84375f, 0.00000f, 0.87500f },
    { 1.00000f, 0.50000f, 0.50000f, 0.96875f },
    { 1.00000f, 0.09375f, 0.31250f, 0.50000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 1.00000f, 0.03125f, 0.03125f, 0.53125f },
    { 0.18750f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.03125f, 0.18750f },
  }, {
    { 0.00000f, 0.59375f, 0.00000f, 0.96875f },
    { 0.06250f, 0.81250f, 0.06250f, 0.59375f },
    { 0.75000f, 0.43750f, 0.12500f, 0.96875f },
    { 0.87500f, 0.06250f, 0.18750f, 0.43750f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.15625f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.03125f, 0.34375f },
  }
};

// structure that allow us to query and override info for training the costs
typedef struct STBIR__V_FIRST_INFO
{
  double v_cost, h_cost;
  int control_v_first; // 0 = no control, 1 = force hori, 2 = force vert
  int v_first;
  int v_resize_classification;
  int is_gather;
} STBIR__V_FIRST_INFO;

#ifdef STBIR__V_FIRST_INFO_BUFFER
static STBIR__V_FIRST_INFO STBIR__V_FIRST_INFO_BUFFER = {0};
#define STBIR__V_FIRST_INFO_POINTER &STBIR__V_FIRST_INFO_BUFFER
#else
#define STBIR__V_FIRST_INFO_POINTER 0
#endif

// Figure out whether to scale along the horizontal or vertical first.
//   This only *super* important when you are scaling by a massively
//   different amount in the vertical vs the horizontal (for example, if
//   you are scaling by 2x in the width, and 0.5x in the height, then you
//   want to do the vertical scale first, because it's around 3x faster
//   in that order.
//
//   In more normal circumstances, this makes a 20-40% differences, so
//     it's good to get right, but not critical. The normal way that you
//     decide which direction goes first is just figuring out which
//     direction does more multiplies. But with modern CPUs with their
//     fancy caches and SIMD and high IPC abilities, so there's just a lot
//     more that goes into it.
//
//   My handwavy sort of solution is to have an app that does a whole
//     bunch of timing for both vertical and horizontal first modes,
//     and then another app that can read lots of these timing files
//     and try to search for the best weights to use. Dotimings.c
//     is the app that does a bunch of timings, and vf_train.c is the
//     app that solves for the best weights (and shows how well it
//     does currently).

static int stbir__should_do_vertical_first( float weights_table[STBIR_RESIZE_CLASSIFICATIONS][4], int horizontal_filter_pixel_width, float horizontal_scale, int horizontal_output_size, int vertical_filter_pixel_width, float vertical_scale, int vertical_output_size, int is_gather, STBIR__V_FIRST_INFO * info )
{
  double v_cost, h_cost;
  float * weights;
  int vertical_first;
  int v_classification;

  // categorize the resize into buckets
  if ( ( vertical_output_size <= 4 ) || ( horizontal_output_size <= 4 ) )
    v_classification = ( vertical_output_size < horizontal_output_size ) ? 6 : 7;
  else if ( vertical_scale <= 1.0f )
    v_classification = ( is_gather ) ? 1 : 0;
  else if ( vertical_scale <= 2.0f)
    v_classification = 2;
  else if ( vertical_scale <= 3.0f)
    v_classification = 3;
  else if ( vertical_scale <= 4.0f)
    v_classification = 5;
  else
    v_classification = 6;

  // use the right weights
  weights = weights_table[ v_classification ];

  // this is the costs when you don't take into account modern CPUs with high ipc and simd and caches - wish we had a better estimate
  h_cost = (float)horizontal_filter_pixel_width * weights[0] + horizontal_scale * (float)vertical_filter_pixel_width * weights[1];
  v_cost = (float)vertical_filter_pixel_width  * weights[2] + vertical_scale * (float)horizontal_filter_pixel_width * weights[3];

  // use computation estimate to decide vertical first or not
  vertical_first = ( v_cost <= h_cost ) ? 1 : 0;

  // save these, if requested
  if ( info )
  {
    info->h_cost = h_cost;
    info->v_cost = v_cost;
    info->v_resize_classification = v_classification;
    info->v_first = vertical_first;
    info->is_gather = is_gather;
  }

  // and this allows us to override everything for testing (see dotiming.c)
  if ( ( info ) && ( info->control_v_first ) )
    vertical_first = ( info->control_v_first == 2 ) ? 1 : 0;

  return vertical_first;
}

// layout lookups - must match stbir_internal_pixel_layout
static unsigned char stbir__pixel_channels[] = {
  1,2,3,3,4,   // 1ch, 2ch, rgb, bgr, 4ch
  4,4,4,4,2,2, // RGBA,BGRA,ARGB,ABGR,RA,AR
  4,4,4,4,2,2, // RGBA_PM,BGRA_PM,ARGB_PM,ABGR_PM,RA_PM,AR_PM
};

// the internal pixel layout enums are in a different order, so we can easily do range comparisons of types
//   the public pixel layout is ordered in a way that if you cast num_channels (1-4) to the enum, you get something sensible
static stbir_internal_pixel_layout stbir__pixel_layout_convert_public_to_internal[] = {
  STBIRI_BGR, STBIRI_1CHANNEL, STBIRI_2CHANNEL, STBIRI_RGB, STBIRI_RGBA,
  STBIRI_4CHANNEL, STBIRI_BGRA, STBIRI_ARGB, STBIRI_ABGR, STBIRI_RA, STBIRI_AR,
  STBIRI_RGBA_PM, STBIRI_BGRA_PM, STBIRI_ARGB_PM, STBIRI_ABGR_PM, STBIRI_RA_PM, STBIRI_AR_PM,
};

static stbir__info * stbir__alloc_internal_mem_and_build_samplers( stbir__sampler * horizontal, stbir__sampler * vertical, stbir__contributors * conservative, stbir_pixel_layout input_pixel_layout_public, stbir_pixel_layout output_pixel_layout_public, int splits, int new_x, int new_y, int fast_alpha, void * user_data STBIR_ONLY_PROFILE_BUILD_GET_INFO )
{
  static char stbir_channel_count_index[8]={ 9,0,1,2, 3,9,9,4 };

  stbir__info * info = 0;
  void * alloced = 0;
  size_t alloced_total = 0;
  int vertical_first;
  size_t decode_buffer_size, ring_buffer_length_bytes, ring_buffer_size, vertical_buffer_size;
  int alloc_ring_buffer_num_entries;

  int alpha_weighting_type = 0; // 0=none, 1=simple, 2=fancy
  int conservative_split_output_size = stbir__get_max_split( splits, vertical->scale_info.output_sub_size );
  stbir_internal_pixel_layout input_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ input_pixel_layout_public ];
  stbir_internal_pixel_layout output_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ output_pixel_layout_public ];
  int channels = stbir__pixel_channels[ input_pixel_layout ];
  int effective_channels = channels;

  // first figure out what type of alpha weighting to use (if any)
  if ( ( horizontal->filter_enum != STBIR_FILTER_POINT_SAMPLE ) || ( vertical->filter_enum != STBIR_FILTER_POINT_SAMPLE ) ) // no alpha weighting on point sampling
  {
    if ( ( input_pixel_layout >= STBIRI_RGBA ) && ( input_pixel_layout <= STBIRI_AR ) && ( output_pixel_layout >= STBIRI_RGBA ) && ( output_pixel_layout <= STBIRI_AR ) )
    {
      if ( fast_alpha )
      {
        alpha_weighting_type = 4;
      }
      else
      {
        static int fancy_alpha_effective_cnts[6] = { 7, 7, 7, 7, 3, 3 };
        alpha_weighting_type = 2;
        effective_channels = fancy_alpha_effective_cnts[ input_pixel_layout - STBIRI_RGBA ];
      }
    }
    else if ( ( input_pixel_layout >= STBIRI_RGBA_PM ) && ( input_pixel_layout <= STBIRI_AR_PM ) && ( output_pixel_layout >= STBIRI_RGBA ) && ( output_pixel_layout <= STBIRI_AR ) )
    {
      // input premult, output non-premult
      alpha_weighting_type = 3;
    }
    else if ( ( input_pixel_layout >= STBIRI_RGBA ) && ( input_pixel_layout <= STBIRI_AR ) && ( output_pixel_layout >= STBIRI_RGBA_PM ) && ( output_pixel_layout <= STBIRI_AR_PM ) )
    {
      // input non-premult, output premult
      alpha_weighting_type = 1;
    }
  }

  // channel in and out count must match currently
  if ( channels != stbir__pixel_channels[ output_pixel_layout ] )
    return 0;

  // get vertical first
  vertical_first = stbir__should_do_vertical_first( stbir__compute_weights[ (int)stbir_channel_count_index[ effective_channels ] ], horizontal->filter_pixel_width, horizontal->scale_info.scale, horizontal->scale_info.output_sub_size, vertical->filter_pixel_width, vertical->scale_info.scale, vertical->scale_info.output_sub_size, vertical->is_gather, STBIR__V_FIRST_INFO_POINTER );

  // sometimes read one float off in some of the unrolled loops (with a weight of zero coeff, so it doesn't have an effect)
  //   we use a few extra floats instead of just 1, so that input callback buffer can overlap with the decode buffer without
  //   the conversion routines overwriting the callback input data.
  decode_buffer_size = ( conservative->n1 - conservative->n0 + 1 ) * effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra floats for input callback stagger

#if defined( STBIR__SEPARATE_ALLOCATIONS ) && defined(STBIR_SIMD8)
  if ( effective_channels == 3 )
    decode_buffer_size += sizeof(float); // avx in 3 channel mode needs one float at the start of the buffer (only with separate allocations)
#endif

  ring_buffer_length_bytes = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra floats for padding

  // if we do vertical first, the ring buffer holds a whole decoded line
  if ( vertical_first )
    ring_buffer_length_bytes = ( decode_buffer_size + 15 ) & ~15;

  if ( ( ring_buffer_length_bytes & 4095 ) == 0 ) ring_buffer_length_bytes += 64*3; // avoid 4k alias

  // One extra entry because floating point precision problems sometimes cause an extra to be necessary.
  alloc_ring_buffer_num_entries = vertical->filter_pixel_width + 1;

  // we never need more ring buffer entries than the scanlines we're outputting when in scatter mode
  if ( ( !vertical->is_gather ) && ( alloc_ring_buffer_num_entries > conservative_split_output_size ) )
    alloc_ring_buffer_num_entries = conservative_split_output_size;

  ring_buffer_size = (size_t)alloc_ring_buffer_num_entries * (size_t)ring_buffer_length_bytes;

  // The vertical buffer is used differently, depending on whether we are scattering
  //   the vertical scanlines, or gathering them.
  //   If scattering, it's used at the temp buffer to accumulate each output.
  //   If gathering, it's just the output buffer.
  vertical_buffer_size = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float);  // extra float for padding

  // we make two passes through this loop, 1st to add everything up, 2nd to allocate and init
  for(;;)
  {
    int i;
    void * advance_mem = alloced;
    int copy_horizontal = 0;
    stbir__sampler * possibly_use_horizontal_for_pivot = 0;

#ifdef STBIR__SEPARATE_ALLOCATIONS
    #define STBIR__NEXT_PTR( ptr, size, ntype ) if ( alloced ) { void * p = STBIR_MALLOC( size, user_data); if ( p == 0 ) { stbir__free_internal_mem( info ); return 0; } (ptr) = (ntype*)p; }
#else
    #define STBIR__NEXT_PTR( ptr, size, ntype ) advance_mem = (void*) ( ( ((size_t)advance_mem) + 15 ) & ~15 ); if ( alloced ) ptr = (ntype*)advance_mem; advance_mem = ((char*)advance_mem) + (size);
#endif

    STBIR__NEXT_PTR( info, sizeof( stbir__info ), stbir__info );

    STBIR__NEXT_PTR( info->split_info, sizeof( stbir__per_split_info ) * splits, stbir__per_split_info );

    if ( info )
    {
      static stbir__alpha_weight_func * fancy_alpha_weights[6]  =    { stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_2ch,   stbir__fancy_alpha_weight_2ch };
      static stbir__alpha_unweight_func * fancy_alpha_unweights[6] = { stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_2ch, stbir__fancy_alpha_unweight_2ch };
      static stbir__alpha_weight_func * simple_alpha_weights[6] = { stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_2ch, stbir__simple_alpha_weight_2ch };
      static stbir__alpha_unweight_func * simple_alpha_unweights[6] = { stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_2ch, stbir__simple_alpha_unweight_2ch };

      // initialize info fields
      info->alloced_mem = alloced;
      info->alloced_total = alloced_total;

      info->channels = channels;
      info->effective_channels = effective_channels;

      info->offset_x = new_x;
      info->offset_y = new_y;
      info->alloc_ring_buffer_num_entries = (int)alloc_ring_buffer_num_entries;
      info->ring_buffer_num_entries = 0;
      info->ring_buffer_length_bytes = (int)ring_buffer_length_bytes;
      info->splits = splits;
      info->vertical_first = vertical_first;

      info->input_pixel_layout_internal = input_pixel_layout;
      info->output_pixel_layout_internal = output_pixel_layout;

      // setup alpha weight functions
      info->alpha_weight = 0;
      info->alpha_unweight = 0;

      // handle alpha weighting functions and overrides
      if ( alpha_weighting_type == 2 )
      {
        // high quality alpha multiplying on the way in, dividing on the way out
        info->alpha_weight = fancy_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
        info->alpha_unweight = fancy_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 4 )
      {
        // fast alpha multiplying on the way in, dividing on the way out
        info->alpha_weight = simple_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
        info->alpha_unweight = simple_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 1 )
      {
        // fast alpha on the way in, leave in premultiplied form on way out
        info->alpha_weight = simple_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 3 )
      {
        // incoming is premultiplied, fast alpha dividing on the way out - non-premultiplied output
        info->alpha_unweight = simple_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }

      // handle 3-chan color flipping, using the alpha weight path
      if ( ( ( input_pixel_layout == STBIRI_RGB ) && ( output_pixel_layout == STBIRI_BGR ) ) ||
           ( ( input_pixel_layout == STBIRI_BGR ) && ( output_pixel_layout == STBIRI_RGB ) ) )
      {
        // do the flipping on the smaller of the two ends
        if ( horizontal->scale_info.scale < 1.0f )
          info->alpha_unweight = stbir__simple_flip_3ch;
        else
          info->alpha_weight = stbir__simple_flip_3ch;
      }

    }

    // get all the per-split buffers
    for( i = 0 ; i < splits ; i++ )
    {
      STBIR__NEXT_PTR( info->split_info[i].decode_buffer, decode_buffer_size, float );

#ifdef STBIR__SEPARATE_ALLOCATIONS

      #ifdef STBIR_SIMD8
      if ( ( info ) && ( effective_channels == 3 ) )
        ++info->split_info[i].decode_buffer; // avx in 3 channel mode needs one float at the start of the buffer
      #endif

      STBIR__NEXT_PTR( info->split_info[i].ring_buffers, alloc_ring_buffer_num_entries * sizeof(float*), float* );
      {
        int j;
        for( j = 0 ; j < alloc_ring_buffer_num_entries ; j++ )
        {
          STBIR__NEXT_PTR( info->split_info[i].ring_buffers[j], ring_buffer_length_bytes, float );
          #ifdef STBIR_SIMD8
          if ( ( info ) && ( effective_channels == 3 ) )
            ++info->split_info[i].ring_buffers[j]; // avx in 3 channel mode needs one float at the start of the buffer
          #endif
        }
      }
#else
      STBIR__NEXT_PTR( info->split_info[i].ring_buffer, ring_buffer_size, float );
#endif
      STBIR__NEXT_PTR( info->split_info[i].vertical_buffer, vertical_buffer_size, float );
    }

    // alloc memory for to-be-pivoted coeffs (if necessary)
    if ( vertical->is_gather == 0 )
    {
      size_t both;
      size_t temp_mem_amt;

      // when in vertical scatter mode, we first build the coefficients in gather mode, and then pivot after,
      //   that means we need two buffers, so we try to use the decode buffer and ring buffer for this. if that
      //   is too small, we just allocate extra memory to use as this temp.

      both = (size_t)vertical->gather_prescatter_contributors_size + (size_t)vertical->gather_prescatter_coefficients_size;

#ifdef STBIR__SEPARATE_ALLOCATIONS
      temp_mem_amt = decode_buffer_size;

      #ifdef STBIR_SIMD8
      if ( effective_channels == 3 )
        --temp_mem_amt; // avx in 3 channel mode needs one float at the start of the buffer
      #endif
#else
      temp_mem_amt = (size_t)( decode_buffer_size + ring_buffer_size + vertical_buffer_size ) * (size_t)splits;
#endif
      if ( temp_mem_amt >= both )
      {
        if ( info )
        {
          vertical->gather_prescatter_contributors = (stbir__contributors*)info->split_info[0].decode_buffer;
          vertical->gather_prescatter_coefficients = (float*) ( ( (char*)info->split_info[0].decode_buffer ) + vertical->gather_prescatter_contributors_size );
        }
      }
      else
      {
        // ring+decode memory is too small, so allocate temp memory
        STBIR__NEXT_PTR( vertical->gather_prescatter_contributors, vertical->gather_prescatter_contributors_size, stbir__contributors );
        STBIR__NEXT_PTR( vertical->gather_prescatter_coefficients, vertical->gather_prescatter_coefficients_size, float );
      }
    }

    STBIR__NEXT_PTR( horizontal->contributors, horizontal->contributors_size, stbir__contributors );
    STBIR__NEXT_PTR( horizontal->coefficients, horizontal->coefficients_size, float );

    // are the two filters identical?? (happens a lot with mipmap generation)
    if ( ( horizontal->filter_kernel == vertical->filter_kernel ) && ( horizontal->filter_support == vertical->filter_support ) && ( horizontal->edge == vertical->edge ) && ( horizontal->scale_info.output_sub_size == vertical->scale_info.output_sub_size ) )
    {
      float diff_scale = horizontal->scale_info.scale - vertical->scale_info.scale;
      float diff_shift = horizontal->scale_info.pixel_shift - vertical->scale_info.pixel_shift;
      if ( diff_scale < 0.0f ) diff_scale = -diff_scale;
      if ( diff_shift < 0.0f ) diff_shift = -diff_shift;
      if ( ( diff_scale <= stbir__small_float ) && ( diff_shift <= stbir__small_float ) )
      {
        if ( horizontal->is_gather == vertical->is_gather )
        {
          copy_horizontal = 1;
          goto no_vert_alloc;
        }
        // everything matches, but vertical is scatter, horizontal is gather, use horizontal coeffs for vertical pivot coeffs
        possibly_use_horizontal_for_pivot = horizontal;
      }
    }

    STBIR__NEXT_PTR( vertical->contributors, vertical->contributors_size, stbir__contributors );
    STBIR__NEXT_PTR( vertical->coefficients, vertical->coefficients_size, float );

   no_vert_alloc:

    if ( info )
    {
      STBIR_PROFILE_BUILD_START( horizontal );

      stbir__calculate_filters( horizontal, 0, user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );

      // setup the horizontal gather functions
      // start with defaulting to the n_coeffs functions (specialized on channels and remnant leftover)
      info->horizontal_gather_channels = stbir__horizontal_gather_n_coeffs_funcs[ effective_channels ][ horizontal->extent_info.widest & 3 ];
      // but if the number of coeffs <= 12, use another set of special cases. <=12 coeffs is any enlarging resize, or shrinking resize down to about 1/3 size
      if ( horizontal->extent_info.widest <= 12 )
        info->horizontal_gather_channels = stbir__horizontal_gather_channels_funcs[ effective_channels ][ horizontal->extent_info.widest - 1 ];

      info->scanline_extents.conservative.n0 = conservative->n0;
      info->scanline_extents.conservative.n1 = conservative->n1;

      // get exact extents
      stbir__get_extents( horizontal, &info->scanline_extents );

      // pack the horizontal coeffs
      horizontal->coefficient_width = stbir__pack_coefficients(horizontal->num_contributors, horizontal->contributors, horizontal->coefficients, horizontal->coefficient_width, horizontal->extent_info.widest, info->scanline_extents.conservative.n0, info->scanline_extents.conservative.n1 );

      STBIR_MEMCPY( &info->horizontal, horizontal, sizeof( stbir__sampler ) );

      STBIR_PROFILE_BUILD_END( horizontal );

      if ( copy_horizontal )
      {
        STBIR_MEMCPY( &info->vertical, horizontal, sizeof( stbir__sampler ) );
      }
      else
      {
        STBIR_PROFILE_BUILD_START( vertical );

        stbir__calculate_filters( vertical, possibly_use_horizontal_for_pivot, user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );
        STBIR_MEMCPY( &info->vertical, vertical, sizeof( stbir__sampler ) );

        STBIR_PROFILE_BUILD_END( vertical );
      }

      // setup the vertical split ranges
      stbir__get_split_info( info->split_info, info->splits, info->vertical.scale_info.output_sub_size, info->vertical.filter_pixel_margin, info->vertical.scale_info.input_full_size );

      // now we know precisely how many entries we need
      info->ring_buffer_num_entries = info->vertical.extent_info.widest;

      // we never need more ring buffer entries than the scanlines we're outputting
      if ( ( !info->vertical.is_gather ) && ( info->ring_buffer_num_entries > conservative_split_output_size ) )
        info->ring_buffer_num_entries = conservative_split_output_size;
      STBIR_ASSERT( info->ring_buffer_num_entries <= info->alloc_ring_buffer_num_entries );
    }
    #undef STBIR__NEXT_PTR


    // is this the first time through loop?
    if ( info == 0 )
    {
      alloced_total = ( 15 + (size_t)advance_mem );
      alloced = STBIR_MALLOC( alloced_total, user_data );
      if ( alloced == 0 )
        return 0;
    }
    else
      return info;  // success
  }
}

static int stbir__perform_resize( stbir__info const * info, int split_start, int split_count )
{
  stbir__per_split_info * split_info = info->split_info + split_start;

  STBIR_PROFILE_CLEAR_EXTRAS();

  STBIR_PROFILE_FIRST_START( looping );
  if (info->vertical.is_gather)
    stbir__vertical_gather_loop( info, split_info, split_count );
  else
    stbir__vertical_scatter_loop( info, split_info, split_count );
  STBIR_PROFILE_END( looping );

  return 1;
}

static void stbir__update_info_from_resize( stbir__info * info, STBIR_RESIZE * resize )
{
  static stbir__decode_pixels_func * decode_simple[STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    /* 1ch-4ch */ stbir__decode_uint8_srgb, stbir__decode_uint8_srgb, 0, stbir__decode_float_linear, stbir__decode_half_float_linear,
  };

  static stbir__decode_pixels_func * decode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    { /* RGBA */ stbir__decode_uint8_srgb4_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    { /* BGRA */ stbir__decode_uint8_srgb4_linearalpha_BGRA, stbir__decode_uint8_srgb_BGRA, 0, stbir__decode_float_linear_BGRA, stbir__decode_half_float_linear_BGRA },
    { /* ARGB */ stbir__decode_uint8_srgb4_linearalpha_ARGB, stbir__decode_uint8_srgb_ARGB, 0, stbir__decode_float_linear_ARGB, stbir__decode_half_float_linear_ARGB },
    { /* ABGR */ stbir__decode_uint8_srgb4_linearalpha_ABGR, stbir__decode_uint8_srgb_ABGR, 0, stbir__decode_float_linear_ABGR, stbir__decode_half_float_linear_ABGR },
    { /* RA   */ stbir__decode_uint8_srgb2_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    { /* AR   */ stbir__decode_uint8_srgb2_linearalpha_AR,   stbir__decode_uint8_srgb_AR,   0, stbir__decode_float_linear_AR,   stbir__decode_half_float_linear_AR },
  };

  static stbir__decode_pixels_func * decode_simple_scaled_or_not[2][2]=
  {
    { stbir__decode_uint8_linear_scaled,  stbir__decode_uint8_linear }, { stbir__decode_uint16_linear_scaled, stbir__decode_uint16_linear },
  };

  static stbir__decode_pixels_func * decode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    { /* RGBA */ { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    { /* BGRA */ { stbir__decode_uint8_linear_scaled_BGRA,  stbir__decode_uint8_linear_BGRA }, { stbir__decode_uint16_linear_scaled_BGRA, stbir__decode_uint16_linear_BGRA } },
    { /* ARGB */ { stbir__decode_uint8_linear_scaled_ARGB,  stbir__decode_uint8_linear_ARGB }, { stbir__decode_uint16_linear_scaled_ARGB, stbir__decode_uint16_linear_ARGB } },
    { /* ABGR */ { stbir__decode_uint8_linear_scaled_ABGR,  stbir__decode_uint8_linear_ABGR }, { stbir__decode_uint16_linear_scaled_ABGR, stbir__decode_uint16_linear_ABGR } },
    { /* RA   */ { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    { /* AR   */ { stbir__decode_uint8_linear_scaled_AR,    stbir__decode_uint8_linear_AR },   { stbir__decode_uint16_linear_scaled_AR,   stbir__decode_uint16_linear_AR } }
  };

  static stbir__encode_pixels_func * encode_simple[STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    /* 1ch-4ch */ stbir__encode_uint8_srgb, stbir__encode_uint8_srgb, 0, stbir__encode_float_linear, stbir__encode_half_float_linear,
  };

  static stbir__encode_pixels_func * encode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    { /* RGBA */ stbir__encode_uint8_srgb4_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    { /* BGRA */ stbir__encode_uint8_srgb4_linearalpha_BGRA, stbir__encode_uint8_srgb_BGRA, 0, stbir__encode_float_linear_BGRA, stbir__encode_half_float_linear_BGRA },
    { /* ARGB */ stbir__encode_uint8_srgb4_linearalpha_ARGB, stbir__encode_uint8_srgb_ARGB, 0, stbir__encode_float_linear_ARGB, stbir__encode_half_float_linear_ARGB },
    { /* ABGR */ stbir__encode_uint8_srgb4_linearalpha_ABGR, stbir__encode_uint8_srgb_ABGR, 0, stbir__encode_float_linear_ABGR, stbir__encode_half_float_linear_ABGR },
    { /* RA   */ stbir__encode_uint8_srgb2_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    { /* AR   */ stbir__encode_uint8_srgb2_linearalpha_AR,   stbir__encode_uint8_srgb_AR,   0, stbir__encode_float_linear_AR,   stbir__encode_half_float_linear_AR }
  };

  static stbir__encode_pixels_func * encode_simple_scaled_or_not[2][2]=
  {
    { stbir__encode_uint8_linear_scaled,  stbir__encode_uint8_linear }, { stbir__encode_uint16_linear_scaled, stbir__encode_uint16_linear },
  };

  static stbir__encode_pixels_func * encode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    { /* RGBA */ { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    { /* BGRA */ { stbir__encode_uint8_linear_scaled_BGRA,  stbir__encode_uint8_linear_BGRA },  { stbir__encode_uint16_linear_scaled_BGRA, stbir__encode_uint16_linear_BGRA } },
    { /* ARGB */ { stbir__encode_uint8_linear_scaled_ARGB,  stbir__encode_uint8_linear_ARGB },  { stbir__encode_uint16_linear_scaled_ARGB, stbir__encode_uint16_linear_ARGB } },
    { /* ABGR */ { stbir__encode_uint8_linear_scaled_ABGR,  stbir__encode_uint8_linear_ABGR },  { stbir__encode_uint16_linear_scaled_ABGR, stbir__encode_uint16_linear_ABGR } },
    { /* RA   */ { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    { /* AR   */ { stbir__encode_uint8_linear_scaled_AR,    stbir__encode_uint8_linear_AR },    { stbir__encode_uint16_linear_scaled_AR,   stbir__encode_uint16_linear_AR } }
  };

  stbir__decode_pixels_func * decode_pixels = 0;
  stbir__encode_pixels_func * encode_pixels = 0;
  stbir_datatype input_type, output_type;

  input_type = resize->input_data_type;
  output_type = resize->output_data_type;
  info->input_data = resize->input_pixels;
  info->input_stride_bytes = resize->input_stride_in_bytes;
  info->output_stride_bytes = resize->output_stride_in_bytes;

  // if we're completely point sampling, then we can turn off SRGB
  if ( ( info->horizontal.filter_enum == STBIR_FILTER_POINT_SAMPLE ) && ( info->vertical.filter_enum == STBIR_FILTER_POINT_SAMPLE ) )
  {
    if ( ( ( input_type  == STBIR_TYPE_UINT8_SRGB ) || ( input_type  == STBIR_TYPE_UINT8_SRGB_ALPHA ) ) &&
         ( ( output_type == STBIR_TYPE_UINT8_SRGB ) || ( output_type == STBIR_TYPE_UINT8_SRGB_ALPHA ) ) )
    {
      input_type = STBIR_TYPE_UINT8;
      output_type = STBIR_TYPE_UINT8;
    }
  }

  // recalc the output and input strides
  if ( info->input_stride_bytes == 0 )
    info->input_stride_bytes = info->channels * info->horizontal.scale_info.input_full_size * stbir__type_size[input_type];

  if ( info->output_stride_bytes == 0 )
    info->output_stride_bytes = info->channels * info->horizontal.scale_info.output_sub_size * stbir__type_size[output_type];

  // calc offset
  info->output_data = ( (char*) resize->output_pixels ) + ( (size_t) info->offset_y * (size_t) resize->output_stride_in_bytes ) + ( info->offset_x * info->channels * stbir__type_size[output_type] );

  info->in_pixels_cb = resize->input_cb;
  info->user_data = resize->user_data;
  info->out_pixels_cb = resize->output_cb;

  // setup the input format converters
  if ( ( input_type == STBIR_TYPE_UINT8 ) || ( input_type == STBIR_TYPE_UINT16 ) )
  {
    int non_scaled = 0;

    // check if we can run unscaled - 0-255.0/0-65535.0 instead of 0-1.0 (which is a tiny bit faster when doing linear 8->8 or 16->16)
    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight )  ) // don't short circuit when alpha weighting (get everything to 0-1.0 as usual)
      if ( ( ( input_type == STBIR_TYPE_UINT8 ) && ( output_type == STBIR_TYPE_UINT8 ) ) || ( ( input_type == STBIR_TYPE_UINT16 ) && ( output_type == STBIR_TYPE_UINT16 ) ) )
        non_scaled = 1;

    if ( info->input_pixel_layout_internal <= STBIRI_4CHANNEL )
      decode_pixels = decode_simple_scaled_or_not[ input_type == STBIR_TYPE_UINT16 ][ non_scaled ];
    else
      decode_pixels = decode_alphas_scaled_or_not[ ( info->input_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ input_type == STBIR_TYPE_UINT16 ][ non_scaled ];
  }
  else
  {
    if ( info->input_pixel_layout_internal <= STBIRI_4CHANNEL )
      decode_pixels = decode_simple[ input_type - STBIR_TYPE_UINT8_SRGB ];
    else
      decode_pixels = decode_alphas[ ( info->input_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ input_type - STBIR_TYPE_UINT8_SRGB ];
  }

  // setup the output format converters
  if ( ( output_type == STBIR_TYPE_UINT8 ) || ( output_type == STBIR_TYPE_UINT16 ) )
  {
    int non_scaled = 0;

    // check if we can run unscaled - 0-255.0/0-65535.0 instead of 0-1.0 (which is a tiny bit faster when doing linear 8->8 or 16->16)
    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight ) ) // don't short circuit when alpha weighting (get everything to 0-1.0 as usual)
      if ( ( ( input_type == STBIR_TYPE_UINT8 ) && ( output_type == STBIR_TYPE_UINT8 ) ) || ( ( input_type == STBIR_TYPE_UINT16 ) && ( output_type == STBIR_TYPE_UINT16 ) ) )
        non_scaled = 1;

    if ( info->output_pixel_layout_internal <= STBIRI_4CHANNEL )
      encode_pixels = encode_simple_scaled_or_not[ output_type == STBIR_TYPE_UINT16 ][ non_scaled ];
    else
      encode_pixels = encode_alphas_scaled_or_not[ ( info->output_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ output_type == STBIR_TYPE_UINT16 ][ non_scaled ];
  }
  else
  {
    if ( info->output_pixel_layout_internal <= STBIRI_4CHANNEL )
      encode_pixels = encode_simple[ output_type - STBIR_TYPE_UINT8_SRGB ];
    else
      encode_pixels = encode_alphas[ ( info->output_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ output_type - STBIR_TYPE_UINT8_SRGB ];
  }

  info->input_type = input_type;
  info->output_type = output_type;
  info->decode_pixels = decode_pixels;
  info->encode_pixels = encode_pixels;
}

static void stbir__clip( int * outx, int * outsubw, int outw, double * u0, double * u1 )
{
  double per, adj;
  int over;

  // do left/top edge
  if ( *outx < 0 )
  {
    per = ( (double)*outx ) / ( (double)*outsubw ); // is negative
    adj = per * ( *u1 - *u0 );
    *u0 -= adj; // increases u0
    *outx = 0;
  }

  // do right/bot edge
  over = outw - ( *outx + *outsubw );
  if ( over < 0 )
  {
    per = ( (double)over ) / ( (double)*outsubw ); // is negative
    adj = per * ( *u1 - *u0 );
    *u1 += adj; // decrease u1
    *outsubw = outw - *outx;
  }
}

// converts a double to a rational that has less than one float bit of error (returns 0 if unable to do so)
static int stbir__double_to_rational(double f, stbir_uint32 limit, stbir_uint32 *numer, stbir_uint32 *denom, int limit_denom ) // limit_denom (1) or limit numer (0)
{
  double err;
  stbir_uint64 top, bot;
  stbir_uint64 numer_last = 0;
  stbir_uint64 denom_last = 1;
  stbir_uint64 numer_estimate = 1;
  stbir_uint64 denom_estimate = 0;

  // scale to past float error range
  top = (stbir_uint64)( f * (double)(1 << 25) );
  bot = 1 << 25;

  // keep refining, but usually stops in a few loops - usually 5 for bad cases
  for(;;)
  {
    stbir_uint64 est, temp;

    // hit limit, break out and do best full range estimate
    if ( ( ( limit_denom ) ? denom_estimate : numer_estimate ) >= limit )
      break;

    // is the current error less than 1 bit of a float? if so, we're done
    if ( denom_estimate )
    {
      err = ( (double)numer_estimate / (double)denom_estimate ) - f;
      if ( err < 0.0 ) err = -err;
      if ( err < ( 1.0 / (double)(1<<24) ) )
      {
        // yup, found it
        *numer = (stbir_uint32) numer_estimate;
        *denom = (stbir_uint32) denom_estimate;
        return 1;
      }
    }

    // no more refinement bits left? break out and do full range estimate
    if ( bot == 0 )
      break;

    // gcd the estimate bits
    est = top / bot;
    temp = top % bot;
    top = bot;
    bot = temp;

    // move remainders
    temp = est * denom_estimate + denom_last;
    denom_last = denom_estimate;
    denom_estimate = temp;

    // move remainders
    temp = est * numer_estimate + numer_last;
    numer_last = numer_estimate;
    numer_estimate = temp;
  }

  // we didn't fine anything good enough for float, use a full range estimate
  if ( limit_denom )
  {
    numer_estimate= (stbir_uint64)( f * (double)limit + 0.5 );
    denom_estimate = limit;
  }
  else
  {
    numer_estimate = limit;
    denom_estimate = (stbir_uint64)( ( (double)limit / f ) + 0.5 );
  }

  *numer = (stbir_uint32) numer_estimate;
  *denom = (stbir_uint32) denom_estimate;

  err = ( denom_estimate ) ? ( ( (double)(stbir_uint32)numer_estimate / (double)(stbir_uint32)denom_estimate ) - f ) : 1.0;
  if ( err < 0.0 ) err = -err;
  return ( err < ( 1.0 / (double)(1<<24) ) ) ? 1 : 0;
}

static int stbir__calculate_region_transform( stbir__scale_info * scale_info, int output_full_range, int * output_offset, int output_sub_range, int input_full_range, double input_s0, double input_s1 )
{
  double output_range, input_range, output_s, input_s, ratio, scale;

  input_s = input_s1 - input_s0;

  // null area
  if ( ( output_full_range == 0 ) || ( input_full_range == 0 ) ||
       ( output_sub_range == 0 ) || ( input_s <= stbir__small_float ) )
    return 0;

  // are either of the ranges completely out of bounds?
  if ( ( *output_offset >= output_full_range ) || ( ( *output_offset + output_sub_range ) <= 0 ) || ( input_s0 >= (1.0f-stbir__small_float) ) || ( input_s1 <= stbir__small_float ) )
    return 0;

  output_range = (double)output_full_range;
  input_range = (double)input_full_range;

  output_s = ( (double)output_sub_range) / output_range;

  // figure out the scaling to use
  ratio = output_s / input_s;

  // save scale before clipping
  scale = ( output_range / input_range ) * ratio;
  scale_info->scale = (float)scale;
  scale_info->inv_scale = (float)( 1.0 / scale );

  // clip output area to left/right output edges (and adjust input area)
  stbir__clip( output_offset, &output_sub_range, output_full_range, &input_s0, &input_s1 );

  // recalc input area
  input_s = input_s1 - input_s0;

  // after clipping do we have zero input area?
  if ( input_s <= stbir__small_float )
    return 0;

  // calculate and store the starting source offsets in output pixel space
  scale_info->pixel_shift = (float) ( input_s0 * ratio * output_range );

  scale_info->scale_is_rational = stbir__double_to_rational( scale, ( scale <= 1.0 ) ? output_full_range : input_full_range, &scale_info->scale_numerator, &scale_info->scale_denominator, ( scale >= 1.0 ) );

  scale_info->input_full_size = input_full_range;
  scale_info->output_sub_size = output_sub_range;

  return 1;
}


static void stbir__init_and_set_layout( STBIR_RESIZE * resize, stbir_pixel_layout pixel_layout, stbir_datatype data_type )
{
  resize->input_cb = 0;
  resize->output_cb = 0;
  resize->user_data = resize;
  resize->samplers = 0;
  resize->called_alloc = 0;
  resize->horizontal_filter = STBIR_FILTER_DEFAULT;
  resize->horizontal_filter_kernel = 0; resize->horizontal_filter_support = 0;
  resize->vertical_filter = STBIR_FILTER_DEFAULT;
  resize->vertical_filter_kernel = 0; resize->vertical_filter_support = 0;
  resize->horizontal_edge = STBIR_EDGE_CLAMP;
  resize->vertical_edge = STBIR_EDGE_CLAMP;
  resize->input_s0 = 0; resize->input_t0 = 0; resize->input_s1 = 1; resize->input_t1 = 1;
  resize->output_subx = 0; resize->output_suby = 0; resize->output_subw = resize->output_w; resize->output_subh = resize->output_h;
  resize->input_data_type = data_type;
  resize->output_data_type = data_type;
  resize->input_pixel_layout_public = pixel_layout;
  resize->output_pixel_layout_public = pixel_layout;
  resize->needs_rebuild = 1;
}

STBIRDEF void stbir_resize_init( STBIR_RESIZE * resize,
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes, // stride can be zero
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes, // stride can be zero
                                 stbir_pixel_layout pixel_layout, stbir_datatype data_type )
{
  resize->input_pixels = input_pixels;
  resize->input_w = input_w;
  resize->input_h = input_h;
  resize->input_stride_in_bytes = input_stride_in_bytes;
  resize->output_pixels = output_pixels;
  resize->output_w = output_w;
  resize->output_h = output_h;
  resize->output_stride_in_bytes = output_stride_in_bytes;
  resize->fast_alpha = 0;

  stbir__init_and_set_layout( resize, pixel_layout, data_type );
}

// You can update parameters any time after resize_init
STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type )  // by default, datatype from resize_init
{
  resize->input_data_type = input_type;
  resize->output_data_type = output_type;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    stbir__update_info_from_resize( resize->samplers, resize );
}

STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb )   // no callbacks by default
{
  resize->input_cb = input_cb;
  resize->output_cb = output_cb;

  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
  {
    resize->samplers->in_pixels_cb = input_cb;
    resize->samplers->out_pixels_cb = output_cb;
  }
}

STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data )                                     // pass back STBIR_RESIZE* by default
{
  resize->user_data = user_data;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    resize->samplers->user_data = user_data;
}

STBIRDEF void stbir_set_buffer_ptrs( STBIR_RESIZE * resize, const void * input_pixels, int input_stride_in_bytes, void * output_pixels, int output_stride_in_bytes )
{
  resize->input_pixels = input_pixels;
  resize->input_stride_in_bytes = input_stride_in_bytes;
  resize->output_pixels = output_pixels;
  resize->output_stride_in_bytes = output_stride_in_bytes;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    stbir__update_info_from_resize( resize->samplers, resize );
}


STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge )       // CLAMP by default
{
  resize->horizontal_edge = horizontal_edge;
  resize->vertical_edge = vertical_edge;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter ) // STBIR_DEFAULT_FILTER_UPSAMPLE/DOWNSAMPLE by default
{
  resize->horizontal_filter = horizontal_filter;
  resize->vertical_filter = vertical_filter;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_filter_callbacks( STBIR_RESIZE * resize, stbir__kernel_callback * horizontal_filter, stbir__support_callback * horizontal_support, stbir__kernel_callback * vertical_filter, stbir__support_callback * vertical_support )
{
  resize->horizontal_filter_kernel = horizontal_filter; resize->horizontal_filter_support = horizontal_support;
  resize->vertical_filter_kernel = vertical_filter; resize->vertical_filter_support = vertical_support;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout )   // sets new pixel layouts
{
  resize->input_pixel_layout_public = input_pixel_layout;
  resize->output_pixel_layout_public = output_pixel_layout;
  resize->needs_rebuild = 1;
  return 1;
}


STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality )   // sets alpha speed
{
  resize->fast_alpha = non_pma_alpha_speed_over_quality;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 )                 // sets input region (full region by default)
{
  resize->input_s0 = s0;
  resize->input_t0 = t0;
  resize->input_s1 = s1;
  resize->input_t1 = t1;
  resize->needs_rebuild = 1;

  // are we inbounds?
  if ( ( s1 < stbir__small_float ) || ( (s1-s0) < stbir__small_float ) ||
       ( t1 < stbir__small_float ) || ( (t1-t0) < stbir__small_float ) ||
       ( s0 > (1.0f-stbir__small_float) ) ||
       ( t0 > (1.0f-stbir__small_float) ) )
    return 0;

  return 1;
}

STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )          // sets input region (full region by default)
{
  resize->output_subx = subx;
  resize->output_suby = suby;
  resize->output_subw = subw;
  resize->output_subh = subh;
  resize->needs_rebuild = 1;

  // are we inbounds?
  if ( ( subx >= resize->output_w ) || ( ( subx + subw ) <= 0 ) || ( suby >= resize->output_h ) || ( ( suby + subh ) <= 0 ) || ( subw == 0 ) || ( subh == 0 ) )
    return 0;

  return 1;
}

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )                 // sets both regions (full regions by default)
{
  double s0, t0, s1, t1;

  s0 = ( (double)subx ) / ( (double)resize->output_w );
  t0 = ( (double)suby ) / ( (double)resize->output_h );
  s1 = ( (double)(subx+subw) ) / ( (double)resize->output_w );
  t1 = ( (double)(suby+subh) ) / ( (double)resize->output_h );

  resize->input_s0 = s0;
  resize->input_t0 = t0;
  resize->input_s1 = s1;
  resize->input_t1 = t1;
  resize->output_subx = subx;
  resize->output_suby = suby;
  resize->output_subw = subw;
  resize->output_subh = subh;
  resize->needs_rebuild = 1;

  // are we inbounds?
  if ( ( subx >= resize->output_w ) || ( ( subx + subw ) <= 0 ) || ( suby >= resize->output_h ) || ( ( suby + subh ) <= 0 ) || ( subw == 0 ) || ( subh == 0 ) )
    return 0;

  return 1;
}

static int stbir__perform_build( STBIR_RESIZE * resize, int splits )
{
  stbir__contributors conservative = { 0, 0 };
  stbir__sampler horizontal, vertical;
  int new_output_subx, new_output_suby;
  stbir__info * out_info;
  #ifdef STBIR_PROFILE
  stbir__info profile_infod;  // used to contain building profile info before everything is allocated
  stbir__info * profile_info = &profile_infod;
  #endif

  // have we already built the samplers?
  if ( resize->samplers )
    return 0;

  #define STBIR_RETURN_ERROR_AND_ASSERT( exp )  STBIR_ASSERT( !(exp) ); if (exp) return 0;
  STBIR_RETURN_ERROR_AND_ASSERT( (unsigned)resize->horizontal_filter >= STBIR_FILTER_OTHER)
  STBIR_RETURN_ERROR_AND_ASSERT( (unsigned)resize->vertical_filter >= STBIR_FILTER_OTHER)
  #undef STBIR_RETURN_ERROR_AND_ASSERT

  if ( splits <= 0 )
    return 0;

  STBIR_PROFILE_BUILD_FIRST_START( build );

  new_output_subx = resize->output_subx;
  new_output_suby = resize->output_suby;

  // do horizontal clip and scale calcs
  if ( !stbir__calculate_region_transform( &horizontal.scale_info, resize->output_w, &new_output_subx, resize->output_subw, resize->input_w, resize->input_s0, resize->input_s1 ) )
    return 0;

  // do vertical clip and scale calcs
  if ( !stbir__calculate_region_transform( &vertical.scale_info, resize->output_h, &new_output_suby, resize->output_subh, resize->input_h, resize->input_t0, resize->input_t1 ) )
    return 0;

  // if nothing to do, just return
  if ( ( horizontal.scale_info.output_sub_size == 0 ) || ( vertical.scale_info.output_sub_size == 0 ) )
    return 0;

  stbir__set_sampler(&horizontal, resize->horizontal_filter, resize->horizontal_filter_kernel, resize->horizontal_filter_support, resize->horizontal_edge, &horizontal.scale_info, 1, resize->user_data );
  stbir__get_conservative_extents( &horizontal, &conservative, resize->user_data );
  stbir__set_sampler(&vertical, resize->vertical_filter, resize->horizontal_filter_kernel, resize->vertical_filter_support, resize->vertical_edge, &vertical.scale_info, 0, resize->user_data );

  if ( ( vertical.scale_info.output_sub_size / splits ) < STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS ) // each split should be a minimum of 4 scanlines (handwavey choice)
  {
    splits = vertical.scale_info.output_sub_size / STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS;
    if ( splits == 0 ) splits = 1;
  }

  STBIR_PROFILE_BUILD_START( alloc );
  out_info = stbir__alloc_internal_mem_and_build_samplers( &horizontal, &vertical, &conservative, resize->input_pixel_layout_public, resize->output_pixel_layout_public, splits, new_output_subx, new_output_suby, resize->fast_alpha, resize->user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );
  STBIR_PROFILE_BUILD_END( alloc );
  STBIR_PROFILE_BUILD_END( build );

  if ( out_info )
  {
    resize->splits = splits;
    resize->samplers = out_info;
    resize->needs_rebuild = 0;
    #ifdef STBIR_PROFILE
      STBIR_MEMCPY( &out_info->profile, &profile_infod.profile, sizeof( out_info->profile ) );
    #endif

    // update anything that can be changed without recalcing samplers
    stbir__update_info_from_resize( out_info, resize );

    return splits;
  }

  return 0;
}

void stbir_free_samplers( STBIR_RESIZE * resize )
{
  if ( resize->samplers )
  {
    stbir__free_internal_mem( resize->samplers );
    resize->samplers = 0;
    resize->called_alloc = 0;
  }
}

STBIRDEF int stbir_build_samplers_with_splits( STBIR_RESIZE * resize, int splits )
{
  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
  {
    if ( resize->samplers )
      stbir_free_samplers( resize );

    resize->called_alloc = 1;
    return stbir__perform_build( resize, splits );
  }

  STBIR_PROFILE_BUILD_CLEAR( resize->samplers );

  return 1;
}

STBIRDEF int stbir_build_samplers( STBIR_RESIZE * resize )
{
  return stbir_build_samplers_with_splits( resize, 1 );
}

STBIRDEF int stbir_resize_extended( STBIR_RESIZE * resize )
{
  int result;

  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
  {
    int alloc_state = resize->called_alloc;  // remember allocated state

    if ( resize->samplers )
    {
      stbir__free_internal_mem( resize->samplers );
      resize->samplers = 0;
    }

    if ( !stbir_build_samplers( resize ) )
      return 0;

    resize->called_alloc = alloc_state;

    // if build_samplers succeeded (above), but there are no samplers set, then
    //   the area to stretch into was zero pixels, so don't do anything and return
    //   success
    if ( resize->samplers == 0 )
      return 1;
  }
  else
  {
    // didn't build anything - clear it
    STBIR_PROFILE_BUILD_CLEAR( resize->samplers );
  }

  // do resize
  result = stbir__perform_resize( resize->samplers, 0, resize->splits );

  // if we alloced, then free
  if ( !resize->called_alloc )
  {
    stbir_free_samplers( resize );
    resize->samplers = 0;
  }

  return result;
}

STBIRDEF int stbir_resize_extended_split( STBIR_RESIZE * resize, int split_start, int split_count )
{
  STBIR_ASSERT( resize->samplers );

  // if we're just doing the whole thing, call full
  if ( ( split_start == -1 ) || ( ( split_start == 0 ) && ( split_count == resize->splits ) ) )
    return stbir_resize_extended( resize );

  // you **must** build samplers first when using split resize
  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
    return 0;

  if ( ( split_start >= resize->splits ) || ( split_start < 0 ) || ( ( split_start + split_count ) > resize->splits ) || ( split_count <= 0 ) )
    return 0;

  // do resize
  return stbir__perform_resize( resize->samplers, split_start, split_count );
}

static int stbir__check_output_stuff( void ** ret_ptr, int * ret_pitch, void * output_pixels, int type_size, int output_w, int output_h, int output_stride_in_bytes, stbir_internal_pixel_layout pixel_layout )
{
  size_t size;
  int pitch;
  void * ptr;

  pitch = output_w * type_size * stbir__pixel_channels[ pixel_layout ];
  if ( pitch == 0 )
    return 0;

  if ( output_stride_in_bytes == 0 )
    output_stride_in_bytes = pitch;

  if ( output_stride_in_bytes < pitch )
    return 0;

  size = (size_t)output_stride_in_bytes * (size_t)output_h;
  if ( size == 0 )
    return 0;

  *ret_ptr = 0;
  *ret_pitch = output_stride_in_bytes;

  if ( output_pixels == 0 )
  {
    ptr = STBIR_MALLOC( size, 0 );
    if ( ptr == 0 )
      return 0;

    *ret_ptr = ptr;
    *ret_pitch = pitch;
  }

  return 1;
}


STBIRDEF unsigned char * stbir_resize_uint8_linear( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                          unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                          stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  unsigned char * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( unsigned char ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_UINT8 );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}

STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                        unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                        stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  unsigned char * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( unsigned char ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_UINT8_SRGB );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}


STBIRDEF float * stbir_resize_float_linear( const float *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                  float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                  stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  float * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( float ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_FLOAT );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}


STBIRDEF void * stbir_resize( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                    void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                              stbir_pixel_layout pixel_layout, stbir_datatype data_type,
                              stbir_edge edge, stbir_filter filter )
{
  STBIR_RESIZE resize;
  float * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, stbir__type_size[data_type], output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, output_stride_in_bytes,
                     pixel_layout, data_type );

  resize.horizontal_edge = edge;
  resize.vertical_edge = edge;
  resize.horizontal_filter = filter;
  resize.vertical_filter = filter;

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}

#ifdef STBIR_PROFILE

STBIRDEF void stbir_resize_build_profile_info( STBIR_PROFILE_INFO * info, STBIR_RESIZE const * resize )
{
  static char const * bdescriptions[6] = { "Building", "Allocating", "Horizontal sampler", "Vertical sampler", "Coefficient cleanup", "Coefficient piovot" } ;
  stbir__info* samp = resize->samplers;
  int i;

  typedef int testa[ (STBIR__ARRAY_SIZE( bdescriptions ) == (STBIR__ARRAY_SIZE( samp->profile.array )-1) )?1:-1];
  typedef int testb[ (sizeof( samp->profile.array ) == (sizeof(samp->profile.named)) )?1:-1];
  typedef int testc[ (sizeof( info->clocks ) >= (sizeof(samp->profile.named)) )?1:-1];

  for( i = 0 ; i < STBIR__ARRAY_SIZE( bdescriptions ) ; i++)
    info->clocks[i] = samp->profile.array[i+1];

  info->total_clocks = samp->profile.named.total;
  info->descriptions = bdescriptions;
  info->count = STBIR__ARRAY_SIZE( bdescriptions );
}

STBIRDEF void stbir_resize_split_profile_info( STBIR_PROFILE_INFO * info, STBIR_RESIZE const * resize, int split_start, int split_count )
{
  static char const * descriptions[7] = { "Looping", "Vertical sampling", "Horizontal sampling", "Scanline input", "Scanline output", "Alpha weighting", "Alpha unweighting" };
  stbir__per_split_info * split_info;
  int s, i;

  typedef int testa[ (STBIR__ARRAY_SIZE( descriptions ) == (STBIR__ARRAY_SIZE( split_info->profile.array )-1) )?1:-1];
  typedef int testb[ (sizeof( split_info->profile.array ) == (sizeof(split_info->profile.named)) )?1:-1];
  typedef int testc[ (sizeof( info->clocks ) >= (sizeof(split_info->profile.named)) )?1:-1];

  if ( split_start == -1 )
  {
    split_start = 0;
    split_count = resize->samplers->splits;
  }

  if ( ( split_start >= resize->splits ) || ( split_start < 0 ) || ( ( split_start + split_count ) > resize->splits ) || ( split_count <= 0 ) )
  {
    info->total_clocks = 0;
    info->descriptions = 0;
    info->count = 0;
    return;
  }

  split_info = resize->samplers->split_info + split_start;

  // sum up the profile from all the splits
  for( i = 0 ; i < STBIR__ARRAY_SIZE( descriptions ) ; i++ )
  {
    stbir_uint64 sum = 0;
    for( s = 0 ; s < split_count ; s++ )
      sum += split_info[s].profile.array[i+1];
    info->clocks[i] = sum;
  }

  info->total_clocks = split_info->profile.named.total;
  info->descriptions = descriptions;
  info->count = STBIR__ARRAY_SIZE( descriptions );
}

STBIRDEF void stbir_resize_extended_profile_info( STBIR_PROFILE_INFO * info, STBIR_RESIZE const * resize )
{
  stbir_resize_split_profile_info( info, resize, -1, 0 );
}

#endif // STBIR_PROFILE

#undef STBIR_BGR
#undef STBIR_1CHANNEL
#undef STBIR_2CHANNEL
#undef STBIR_RGB
#undef STBIR_RGBA
#undef STBIR_4CHANNEL
#undef STBIR_BGRA
#undef STBIR_ARGB
#undef STBIR_ABGR
#undef STBIR_RA
#undef STBIR_AR
#undef STBIR_RGBA_PM
#undef STBIR_BGRA_PM
#undef STBIR_ARGB_PM
#undef STBIR_ABGR_PM
#undef STBIR_RA_PM
#undef STBIR_AR_PM

#endif // STB_IMAGE_RESIZE_IMPLEMENTATION

#else  // STB_IMAGE_RESIZE_HORIZONTALS&STB_IMAGE_RESIZE_DO_VERTICALS

// we reinclude the header file to define all the horizontal functions
//   specializing each function for the number of coeffs is 20-40% faster *OVERALL*

// by including the header file again this way, we can still debug the functions

#define STBIR_strs_join2( start, mid, end ) start##mid##end
#define STBIR_strs_join1( start, mid, end ) STBIR_strs_join2( start, mid, end )

#define STBIR_strs_join24( start, mid1, mid2, end ) start##mid1##mid2##end
#define STBIR_strs_join14( start, mid1, mid2, end ) STBIR_strs_join24( start, mid1, mid2, end )

#ifdef STB_IMAGE_RESIZE_DO_CODERS

#ifdef stbir__decode_suffix
#define STBIR__CODER_NAME( name ) STBIR_strs_join1( name, _, stbir__decode_suffix )
#else
#define STBIR__CODER_NAME( name ) name
#endif

#ifdef stbir__decode_swizzle
#define stbir__decode_simdf8_flip(reg) STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( stbir__simdf8_0123to,stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3),stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3)(reg, reg)
#define stbir__decode_simdf4_flip(reg) STBIR_strs_join1( STBIR_strs_join1( stbir__simdf_0123to,stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3)(reg, reg)
#define stbir__encode_simdf8_unflip(reg) STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( stbir__simdf8_0123to,stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3),stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3)(reg, reg)
#define stbir__encode_simdf4_unflip(reg) STBIR_strs_join1( STBIR_strs_join1( stbir__simdf_0123to,stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3)(reg, reg)
#else
#define stbir__decode_order0 0
#define stbir__decode_order1 1
#define stbir__decode_order2 2
#define stbir__decode_order3 3
#define stbir__encode_order0 0
#define stbir__encode_order1 1
#define stbir__encode_order2 2
#define stbir__encode_order3 3
#define stbir__decode_simdf8_flip(reg)
#define stbir__decode_simdf4_flip(reg)
#define stbir__encode_simdf8_unflip(reg)
#define stbir__encode_simdf4_unflip(reg)
#endif

#ifdef STBIR_SIMD8
#define stbir__encode_simdfX_unflip  stbir__encode_simdf8_unflip
#else
#define stbir__encode_simdfX_unflip  stbir__encode_simdf4_unflip
#endif

static float * STBIR__CODER_NAME( stbir__decode_uint8_linear_scaled )( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const*)inputp;

  #ifdef STBIR_SIMD
  unsigned char const * end_input_m16 = input + width_times_channels - 16;
  if ( width_times_channels >= 16 )
  {
    decode_end -= 16;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      #ifdef STBIR_SIMD8
      stbir__simdi i; stbir__simdi8 o0,o1;
      stbir__simdf8 of0, of1;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi8_expand_u8_to_u32( o0, o1, i );
      stbir__simdi8_convert_i32_to_float( of0, o0 );
      stbir__simdi8_convert_i32_to_float( of1, o1 );
      stbir__simdf8_mult( of0, of0, STBIR_max_uint8_as_float_inverted8);
      stbir__simdf8_mult( of1, of1, STBIR_max_uint8_as_float_inverted8);
      stbir__decode_simdf8_flip( of0 );
      stbir__decode_simdf8_flip( of1 );
      stbir__simdf8_store( decode + 0, of0 );
      stbir__simdf8_store( decode + 8, of1 );
      #else
      stbir__simdi i, o0, o1, o2, o3;
      stbir__simdf of0, of1, of2, of3;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi_expand_u8_to_u32( o0,o1,o2,o3,i);
      stbir__simdi_convert_i32_to_float( of0, o0 );
      stbir__simdi_convert_i32_to_float( of1, o1 );
      stbir__simdi_convert_i32_to_float( of2, o2 );
      stbir__simdi_convert_i32_to_float( of3, o3 );
      stbir__simdf_mult( of0, of0, STBIR__CONSTF(STBIR_max_uint8_as_float_inverted) );
      stbir__simdf_mult( of1, of1, STBIR__CONSTF(STBIR_max_uint8_as_float_inverted) );
      stbir__simdf_mult( of2, of2, STBIR__CONSTF(STBIR_max_uint8_as_float_inverted) );
      stbir__simdf_mult( of3, of3, STBIR__CONSTF(STBIR_max_uint8_as_float_inverted) );
      stbir__decode_simdf4_flip( of0 );
      stbir__decode_simdf4_flip( of1 );
      stbir__decode_simdf4_flip( of2 );
      stbir__decode_simdf4_flip( of3 );
      stbir__simdf_store( decode + 0,  of0 );
      stbir__simdf_store( decode + 4,  of1 );
      stbir__simdf_store( decode + 8,  of2 );
      stbir__simdf_store( decode + 12, of3 );
      #endif
      decode += 16;
      input += 16;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 16 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m16;
    }
    return decode_end + 16;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = ((float)(input[stbir__decode_order0])) * stbir__max_uint8_as_float_inverted;
    decode[1-4] = ((float)(input[stbir__decode_order1])) * stbir__max_uint8_as_float_inverted;
    decode[2-4] = ((float)(input[stbir__decode_order2])) * stbir__max_uint8_as_float_inverted;
    decode[3-4] = ((float)(input[stbir__decode_order3])) * stbir__max_uint8_as_float_inverted;
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = ((float)(input[stbir__decode_order0])) * stbir__max_uint8_as_float_inverted;
    #if stbir__coder_min_num >= 2
    decode[1] = ((float)(input[stbir__decode_order1])) * stbir__max_uint8_as_float_inverted;
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = ((float)(input[stbir__decode_order2])) * stbir__max_uint8_as_float_inverted;
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif

  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_uint8_linear_scaled )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char *) outputp;
  unsigned char * end_output = ( (unsigned char *) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= stbir__simdfX_float_count*2 )
  {
    float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
    end_output -= stbir__simdfX_float_count*2;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      stbir__simdi i;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_madd_mem( e0, STBIR_simd_point5X, STBIR_max_uint8_as_floatX, encode );
      stbir__simdfX_madd_mem( e1, STBIR_simd_point5X, STBIR_max_uint8_as_floatX, encode+stbir__simdfX_float_count );
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      #ifdef STBIR_SIMD8
      stbir__simdf8_pack_to_16bytes( i, e0, e1 );
      stbir__simdi_store( output, i );
      #else
      stbir__simdf_pack_to_8bytes( i, e0, e1 );
      stbir__simdi_store2( output, i );
      #endif
      encode += stbir__simdfX_float_count*2;
      output += stbir__simdfX_float_count*2;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m8;
    }
    return;
  }

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    stbir__simdf e0;
    stbir__simdi i0;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_load( e0, encode );
    stbir__simdf_madd( e0, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint8_as_float), e0 );
    stbir__encode_simdf4_unflip( e0 );
    stbir__simdf_pack_to_8bytes( i0, e0, e0 );  // only use first 4
    *(int*)(output-4) = stbir__simdi_to_int( i0 );
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    stbir__simdf e0;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_madd1_mem( e0, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint8_as_float), encode+stbir__encode_order0 ); output[0] = stbir__simdf_convert_float_to_uint8( e0 );
    #if stbir__coder_min_num >= 2
    stbir__simdf_madd1_mem( e0, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint8_as_float), encode+stbir__encode_order1 ); output[1] = stbir__simdf_convert_float_to_uint8( e0 );
    #endif
    #if stbir__coder_min_num >= 3
    stbir__simdf_madd1_mem( e0, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint8_as_float), encode+stbir__encode_order2 ); output[2] = stbir__simdf_convert_float_to_uint8( e0 );
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif

  #else

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  while( output <= end_output )
  {
    float f;
    f = encode[stbir__encode_order0] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[0-4] = (unsigned char)f;
    f = encode[stbir__encode_order1] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[1-4] = (unsigned char)f;
    f = encode[stbir__encode_order2] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[2-4] = (unsigned char)f;
    f = encode[stbir__encode_order3] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[3-4] = (unsigned char)f;
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    float f;
    STBIR_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[0] = (unsigned char)f;
    #if stbir__coder_min_num >= 2
    f = encode[stbir__encode_order1] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[1] = (unsigned char)f;
    #endif
    #if stbir__coder_min_num >= 3
    f = encode[stbir__encode_order2] * stbir__max_uint8_as_float + 0.5f; STBIR_CLAMP(f, 0, 255); output[2] = (unsigned char)f;
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_uint8_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const*)inputp;

  #ifdef STBIR_SIMD
  unsigned char const * end_input_m16 = input + width_times_channels - 16;
  if ( width_times_channels >= 16 )
  {
    decode_end -= 16;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      #ifdef STBIR_SIMD8
      stbir__simdi i; stbir__simdi8 o0,o1;
      stbir__simdf8 of0, of1;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi8_expand_u8_to_u32( o0, o1, i );
      stbir__simdi8_convert_i32_to_float( of0, o0 );
      stbir__simdi8_convert_i32_to_float( of1, o1 );
      stbir__decode_simdf8_flip( of0 );
      stbir__decode_simdf8_flip( of1 );
      stbir__simdf8_store( decode + 0, of0 );
      stbir__simdf8_store( decode + 8, of1 );
      #else
      stbir__simdi i, o0, o1, o2, o3;
      stbir__simdf of0, of1, of2, of3;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi_expand_u8_to_u32( o0,o1,o2,o3,i);
      stbir__simdi_convert_i32_to_float( of0, o0 );
      stbir__simdi_convert_i32_to_float( of1, o1 );
      stbir__simdi_convert_i32_to_float( of2, o2 );
      stbir__simdi_convert_i32_to_float( of3, o3 );
      stbir__decode_simdf4_flip( of0 );
      stbir__decode_simdf4_flip( of1 );
      stbir__decode_simdf4_flip( of2 );
      stbir__decode_simdf4_flip( of3 );
      stbir__simdf_store( decode + 0,  of0 );
      stbir__simdf_store( decode + 4,  of1 );
      stbir__simdf_store( decode + 8,  of2 );
      stbir__simdf_store( decode + 12, of3 );
#endif
      decode += 16;
      input += 16;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 16 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m16;
    }
    return decode_end + 16;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = ((float)(input[stbir__decode_order0]));
    decode[1-4] = ((float)(input[stbir__decode_order1]));
    decode[2-4] = ((float)(input[stbir__decode_order2]));
    decode[3-4] = ((float)(input[stbir__decode_order3]));
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = ((float)(input[stbir__decode_order0]));
    #if stbir__coder_min_num >= 2
    decode[1] = ((float)(input[stbir__decode_order1]));
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = ((float)(input[stbir__decode_order2]));
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_uint8_linear )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char *) outputp;
  unsigned char * end_output = ( (unsigned char *) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= stbir__simdfX_float_count*2 )
  {
    float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
    end_output -= stbir__simdfX_float_count*2;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      stbir__simdi i;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_add_mem( e0, STBIR_simd_point5X, encode );
      stbir__simdfX_add_mem( e1, STBIR_simd_point5X, encode+stbir__simdfX_float_count );
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      #ifdef STBIR_SIMD8
      stbir__simdf8_pack_to_16bytes( i, e0, e1 );
      stbir__simdi_store( output, i );
      #else
      stbir__simdf_pack_to_8bytes( i, e0, e1 );
      stbir__simdi_store2( output, i );
      #endif
      encode += stbir__simdfX_float_count*2;
      output += stbir__simdfX_float_count*2;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m8;
    }
    return;
  }

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    stbir__simdf e0;
    stbir__simdi i0;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_load( e0, encode );
    stbir__simdf_add( e0, STBIR__CONSTF(STBIR_simd_point5), e0 );
    stbir__encode_simdf4_unflip( e0 );
    stbir__simdf_pack_to_8bytes( i0, e0, e0 );  // only use first 4
    *(int*)(output-4) = stbir__simdi_to_int( i0 );
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  #else

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  while( output <= end_output )
  {
    float f;
    f = encode[stbir__encode_order0] + 0.5f; STBIR_CLAMP(f, 0, 255); output[0-4] = (unsigned char)f;
    f = encode[stbir__encode_order1] + 0.5f; STBIR_CLAMP(f, 0, 255); output[1-4] = (unsigned char)f;
    f = encode[stbir__encode_order2] + 0.5f; STBIR_CLAMP(f, 0, 255); output[2-4] = (unsigned char)f;
    f = encode[stbir__encode_order3] + 0.5f; STBIR_CLAMP(f, 0, 255); output[3-4] = (unsigned char)f;
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    float f;
    STBIR_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] + 0.5f; STBIR_CLAMP(f, 0, 255); output[0] = (unsigned char)f;
    #if stbir__coder_min_num >= 2
    f = encode[stbir__encode_order1] + 0.5f; STBIR_CLAMP(f, 0, 255); output[1] = (unsigned char)f;
    #endif
    #if stbir__coder_min_num >= 3
    f = encode[stbir__encode_order2] + 0.5f; STBIR_CLAMP(f, 0, 255); output[2] = (unsigned char)f;
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_uint8_srgb)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const *)inputp;

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  while( decode <= decode_end )
  {
    decode[0-4] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order0 ] ];
    decode[1-4] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order1 ] ];
    decode[2-4] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order2 ] ];
    decode[3-4] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order3 ] ];
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order0 ] ];
    #if stbir__coder_min_num >= 2
    decode[1] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order1 ] ];
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = stbir__srgb_uchar_to_linear_float[ input[ stbir__decode_order2 ] ];
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;
}

#define stbir__min_max_shift20( i, f ) \
    stbir__simdf_max( f, f, stbir_simdf_casti(STBIR__CONSTI( STBIR_almost_zero )) ); \
    stbir__simdf_min( f, f, stbir_simdf_casti(STBIR__CONSTI( STBIR_almost_one  )) ); \
    stbir__simdi_32shr( i, stbir_simdi_castf( f ), 20 );

#define stbir__scale_and_convert( i, f ) \
    stbir__simdf_madd( f, STBIR__CONSTF( STBIR_simd_point5 ), STBIR__CONSTF( STBIR_max_uint8_as_float ), f ); \
    stbir__simdf_max( f, f, stbir__simdf_zeroP() ); \
    stbir__simdf_min( f, f, STBIR__CONSTF( STBIR_max_uint8_as_float ) ); \
    stbir__simdf_convert_float_to_i32( i, f );

#define stbir__linear_to_srgb_finish( i, f ) \
{ \
    stbir__simdi temp;  \
    stbir__simdi_32shr( temp, stbir_simdi_castf( f ), 12 ) ; \
    stbir__simdi_and( temp, temp, STBIR__CONSTI(STBIR_mastissa_mask) ); \
    stbir__simdi_or( temp, temp, STBIR__CONSTI(STBIR_topscale) ); \
    stbir__simdi_16madd( i, i, temp ); \
    stbir__simdi_32shr( i, i, 16 ); \
}

#define stbir__simdi_table_lookup2( v0,v1, table ) \
{ \
  stbir__simdi_u32 temp0,temp1; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
}

#define stbir__simdi_table_lookup3( v0,v1,v2, table ) \
{ \
  stbir__simdi_u32 temp0,temp1,temp2; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp2.m128i_i128 = v2; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  temp2.m128i_u32[0] = table[temp2.m128i_i32[0]]; temp2.m128i_u32[1] = table[temp2.m128i_i32[1]]; temp2.m128i_u32[2] = table[temp2.m128i_i32[2]]; temp2.m128i_u32[3] = table[temp2.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
  v2 = temp2.m128i_i128; \
}

#define stbir__simdi_table_lookup4( v0,v1,v2,v3, table ) \
{ \
  stbir__simdi_u32 temp0,temp1,temp2,temp3; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp2.m128i_i128 = v2; \
  temp3.m128i_i128 = v3; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  temp2.m128i_u32[0] = table[temp2.m128i_i32[0]]; temp2.m128i_u32[1] = table[temp2.m128i_i32[1]]; temp2.m128i_u32[2] = table[temp2.m128i_i32[2]]; temp2.m128i_u32[3] = table[temp2.m128i_i32[3]]; \
  temp3.m128i_u32[0] = table[temp3.m128i_i32[0]]; temp3.m128i_u32[1] = table[temp3.m128i_i32[1]]; temp3.m128i_u32[2] = table[temp3.m128i_i32[2]]; temp3.m128i_u32[3] = table[temp3.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
  v2 = temp2.m128i_i128; \
  v3 = temp3.m128i_i128; \
}

static void STBIR__CODER_NAME( stbir__encode_uint8_srgb )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char*) outputp;
  unsigned char * end_output = ( (unsigned char*) output ) + width_times_channels;

  #ifdef STBIR_SIMD

  if ( width_times_channels >= 16 )
  {
    float const * end_encode_m16 = encode + width_times_channels - 16;
    end_output -= 16;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdf f0, f1, f2, f3;
      stbir__simdi i0, i1, i2, i3;
      STBIR_SIMD_NO_UNROLL(encode);

      stbir__simdf_load4_transposed( f0, f1, f2, f3, encode );

      stbir__min_max_shift20( i0, f0 );
      stbir__min_max_shift20( i1, f1 );
      stbir__min_max_shift20( i2, f2 );
      stbir__min_max_shift20( i3, f3 );

      stbir__simdi_table_lookup4( i0, i1, i2, i3, ( fp32_to_srgb8_tab4 - (127-13)*8 ) );

      stbir__linear_to_srgb_finish( i0, f0 );
      stbir__linear_to_srgb_finish( i1, f1 );
      stbir__linear_to_srgb_finish( i2, f2 );
      stbir__linear_to_srgb_finish( i3, f3 );

      stbir__interleave_pack_and_store_16_u8( output,  STBIR_strs_join1(i, ,stbir__encode_order0), STBIR_strs_join1(i, ,stbir__encode_order1), STBIR_strs_join1(i, ,stbir__encode_order2), STBIR_strs_join1(i, ,stbir__encode_order3) );

      encode += 16;
      output += 16;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + 16 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m16;
    }
    return;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while ( output <= end_output )
  {
    STBIR_SIMD_NO_UNROLL(encode);

    output[0-4] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order0] );
    output[1-4] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order1] );
    output[2-4] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order2] );
    output[3-4] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order3] );

    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    STBIR_NO_UNROLL(encode);
    output[0] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order0] );
    #if stbir__coder_min_num >= 2
    output[1] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order1] );
    #endif
    #if stbir__coder_min_num >= 3
    output[2] = stbir__linear_to_srgb_uchar( encode[stbir__encode_order2] );
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
}

#if ( stbir__coder_min_num == 4 ) || ( ( stbir__coder_min_num == 1 ) && ( !defined(stbir__decode_swizzle) ) )

static float * STBIR__CODER_NAME(stbir__decode_uint8_srgb4_linearalpha)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const *)inputp;

  do {
    decode[0] = stbir__srgb_uchar_to_linear_float[ input[stbir__decode_order0] ];
    decode[1] = stbir__srgb_uchar_to_linear_float[ input[stbir__decode_order1] ];
    decode[2] = stbir__srgb_uchar_to_linear_float[ input[stbir__decode_order2] ];
    decode[3] = ( (float) input[stbir__decode_order3] ) * stbir__max_uint8_as_float_inverted;
    input += 4;
    decode += 4;
  } while( decode < decode_end );
  return decode_end;
}


static void STBIR__CODER_NAME( stbir__encode_uint8_srgb4_linearalpha )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char*) outputp;
  unsigned char * end_output = ( (unsigned char*) output ) + width_times_channels;

  #ifdef STBIR_SIMD

  if ( width_times_channels >= 16 )
  {
    float const * end_encode_m16 = encode + width_times_channels - 16;
    end_output -= 16;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdf f0, f1, f2, f3;
      stbir__simdi i0, i1, i2, i3;

      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdf_load4_transposed( f0, f1, f2, f3, encode );

      stbir__min_max_shift20( i0, f0 );
      stbir__min_max_shift20( i1, f1 );
      stbir__min_max_shift20( i2, f2 );
      stbir__scale_and_convert( i3, f3 );

      stbir__simdi_table_lookup3( i0, i1, i2, ( fp32_to_srgb8_tab4 - (127-13)*8 ) );

      stbir__linear_to_srgb_finish( i0, f0 );
      stbir__linear_to_srgb_finish( i1, f1 );
      stbir__linear_to_srgb_finish( i2, f2 );

      stbir__interleave_pack_and_store_16_u8( output,  STBIR_strs_join1(i, ,stbir__encode_order0), STBIR_strs_join1(i, ,stbir__encode_order1), STBIR_strs_join1(i, ,stbir__encode_order2), STBIR_strs_join1(i, ,stbir__encode_order3) );

      output += 16;
      encode += 16;

      if ( output <= end_output )
        continue;
      if ( output == ( end_output + 16 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m16;
    }
    return;
  }
  #endif

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float f;
    STBIR_SIMD_NO_UNROLL(encode);

    output[stbir__decode_order0] = stbir__linear_to_srgb_uchar( encode[0] );
    output[stbir__decode_order1] = stbir__linear_to_srgb_uchar( encode[1] );
    output[stbir__decode_order2] = stbir__linear_to_srgb_uchar( encode[2] );

    f = encode[3] * stbir__max_uint8_as_float + 0.5f;
    STBIR_CLAMP(f, 0, 255);
    output[stbir__decode_order3] = (unsigned char) f;

    output += 4;
    encode += 4;
  } while( output < end_output );
}

#endif

#if ( stbir__coder_min_num == 2 ) || ( ( stbir__coder_min_num == 1 ) && ( !defined(stbir__decode_swizzle) ) )

static float * STBIR__CODER_NAME(stbir__decode_uint8_srgb2_linearalpha)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const *)inputp;

  decode += 4;
  while( decode <= decode_end )
  {
    decode[0-4] = stbir__srgb_uchar_to_linear_float[ input[stbir__decode_order0] ];
    decode[1-4] = ( (float) input[stbir__decode_order1] ) * stbir__max_uint8_as_float_inverted;
    decode[2-4] = stbir__srgb_uchar_to_linear_float[ input[stbir__decode_order0+2] ];
    decode[3-4] = ( (float) input[stbir__decode_order1+2] ) * stbir__max_uint8_as_float_inverted;
    input += 4;
    decode += 4;
  }
  decode -= 4;
  if( decode < decode_end )
  {
    decode[0] = stbir__srgb_uchar_to_linear_float[ stbir__decode_order0 ];
    decode[1] = ( (float) input[stbir__decode_order1] ) * stbir__max_uint8_as_float_inverted;
  }
  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_uint8_srgb2_linearalpha )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char*) outputp;
  unsigned char * end_output = ( (unsigned char*) output ) + width_times_channels;

  #ifdef STBIR_SIMD

  if ( width_times_channels >= 16 )
  {
    float const * end_encode_m16 = encode + width_times_channels - 16;
    end_output -= 16;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdf f0, f1, f2, f3;
      stbir__simdi i0, i1, i2, i3;

      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdf_load4_transposed( f0, f1, f2, f3, encode );

      stbir__min_max_shift20( i0, f0 );
      stbir__scale_and_convert( i1, f1 );
      stbir__min_max_shift20( i2, f2 );
      stbir__scale_and_convert( i3, f3 );

      stbir__simdi_table_lookup2( i0, i2, ( fp32_to_srgb8_tab4 - (127-13)*8 ) );

      stbir__linear_to_srgb_finish( i0, f0 );
      stbir__linear_to_srgb_finish( i2, f2 );

      stbir__interleave_pack_and_store_16_u8( output,  STBIR_strs_join1(i, ,stbir__encode_order0), STBIR_strs_join1(i, ,stbir__encode_order1), STBIR_strs_join1(i, ,stbir__encode_order2), STBIR_strs_join1(i, ,stbir__encode_order3) );

      output += 16;
      encode += 16;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + 16 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m16;
    }
    return;
  }
  #endif

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float f;
    STBIR_SIMD_NO_UNROLL(encode);

    output[stbir__decode_order0] = stbir__linear_to_srgb_uchar( encode[0] );

    f = encode[1] * stbir__max_uint8_as_float + 0.5f;
    STBIR_CLAMP(f, 0, 255);
    output[stbir__decode_order1] = (unsigned char) f;

    output += 2;
    encode += 2;
  } while( output < end_output );
}

#endif

static float * STBIR__CODER_NAME(stbir__decode_uint16_linear_scaled)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned short const * input = (unsigned short const *)inputp;

  #ifdef STBIR_SIMD
  unsigned short const * end_input_m8 = input + width_times_channels - 8;
  if ( width_times_channels >= 8 )
  {
    decode_end -= 8;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      #ifdef STBIR_SIMD8
      stbir__simdi i; stbir__simdi8 o;
      stbir__simdf8 of;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi8_expand_u16_to_u32( o, i );
      stbir__simdi8_convert_i32_to_float( of, o );
      stbir__simdf8_mult( of, of, STBIR_max_uint16_as_float_inverted8);
      stbir__decode_simdf8_flip( of );
      stbir__simdf8_store( decode + 0, of );
      #else
      stbir__simdi i, o0, o1;
      stbir__simdf of0, of1;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi_expand_u16_to_u32( o0,o1,i );
      stbir__simdi_convert_i32_to_float( of0, o0 );
      stbir__simdi_convert_i32_to_float( of1, o1 );
      stbir__simdf_mult( of0, of0, STBIR__CONSTF(STBIR_max_uint16_as_float_inverted) );
      stbir__simdf_mult( of1, of1, STBIR__CONSTF(STBIR_max_uint16_as_float_inverted));
      stbir__decode_simdf4_flip( of0 );
      stbir__decode_simdf4_flip( of1 );
      stbir__simdf_store( decode + 0,  of0 );
      stbir__simdf_store( decode + 4,  of1 );
      #endif
      decode += 8;
      input += 8;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 8 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m8;
    }
    return decode_end + 8;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = ((float)(input[stbir__decode_order0])) * stbir__max_uint16_as_float_inverted;
    decode[1-4] = ((float)(input[stbir__decode_order1])) * stbir__max_uint16_as_float_inverted;
    decode[2-4] = ((float)(input[stbir__decode_order2])) * stbir__max_uint16_as_float_inverted;
    decode[3-4] = ((float)(input[stbir__decode_order3])) * stbir__max_uint16_as_float_inverted;
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = ((float)(input[stbir__decode_order0])) * stbir__max_uint16_as_float_inverted;
    #if stbir__coder_min_num >= 2
    decode[1] = ((float)(input[stbir__decode_order1])) * stbir__max_uint16_as_float_inverted;
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = ((float)(input[stbir__decode_order2])) * stbir__max_uint16_as_float_inverted;
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;
}


static void STBIR__CODER_NAME(stbir__encode_uint16_linear_scaled)( void * outputp, int width_times_channels, float const * encode )
{
  unsigned short STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned short*) outputp;
  unsigned short * end_output = ( (unsigned short*) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  {
    if ( width_times_channels >= stbir__simdfX_float_count*2 )
    {
      float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
      end_output -= stbir__simdfX_float_count*2;
      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        stbir__simdfX e0, e1;
        stbir__simdiX i;
        STBIR_SIMD_NO_UNROLL(encode);
        stbir__simdfX_madd_mem( e0, STBIR_simd_point5X, STBIR_max_uint16_as_floatX, encode );
        stbir__simdfX_madd_mem( e1, STBIR_simd_point5X, STBIR_max_uint16_as_floatX, encode+stbir__simdfX_float_count );
        stbir__encode_simdfX_unflip( e0 );
        stbir__encode_simdfX_unflip( e1 );
        stbir__simdfX_pack_to_words( i, e0, e1 );
        stbir__simdiX_store( output, i );
        encode += stbir__simdfX_float_count*2;
        output += stbir__simdfX_float_count*2;
        if ( output <= end_output )
          continue;
        if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
          break;
        output = end_output;     // backup and do last couple
        encode = end_encode_m8;
      }
      return;
    }
  }

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    stbir__simdf e;
    stbir__simdi i;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_load( e, encode );
    stbir__simdf_madd( e, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint16_as_float), e );
    stbir__encode_simdf4_unflip( e );
    stbir__simdf_pack_to_8words( i, e, e );  // only use first 4
    stbir__simdi_store2( output-4, i );
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    stbir__simdf e;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_madd1_mem( e, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint16_as_float), encode+stbir__encode_order0 ); output[0] = stbir__simdf_convert_float_to_short( e );
    #if stbir__coder_min_num >= 2
    stbir__simdf_madd1_mem( e, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint16_as_float), encode+stbir__encode_order1 ); output[1] = stbir__simdf_convert_float_to_short( e );
    #endif
    #if stbir__coder_min_num >= 3
    stbir__simdf_madd1_mem( e, STBIR__CONSTF(STBIR_simd_point5), STBIR__CONSTF(STBIR_max_uint16_as_float), encode+stbir__encode_order2 ); output[2] = stbir__simdf_convert_float_to_short( e );
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif

  #else

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    float f;
    STBIR_SIMD_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[0-4] = (unsigned short)f;
    f = encode[stbir__encode_order1] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[1-4] = (unsigned short)f;
    f = encode[stbir__encode_order2] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[2-4] = (unsigned short)f;
    f = encode[stbir__encode_order3] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[3-4] = (unsigned short)f;
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    float f;
    STBIR_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[0] = (unsigned short)f;
    #if stbir__coder_min_num >= 2
    f = encode[stbir__encode_order1] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[1] = (unsigned short)f;
    #endif
    #if stbir__coder_min_num >= 3
    f = encode[stbir__encode_order2] * stbir__max_uint16_as_float + 0.5f; STBIR_CLAMP(f, 0, 65535); output[2] = (unsigned short)f;
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_uint16_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned short const * input = (unsigned short const *)inputp;

  #ifdef STBIR_SIMD
  unsigned short const * end_input_m8 = input + width_times_channels - 8;
  if ( width_times_channels >= 8 )
  {
    decode_end -= 8;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      #ifdef STBIR_SIMD8
      stbir__simdi i; stbir__simdi8 o;
      stbir__simdf8 of;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi8_expand_u16_to_u32( o, i );
      stbir__simdi8_convert_i32_to_float( of, o );
      stbir__decode_simdf8_flip( of );
      stbir__simdf8_store( decode + 0, of );
      #else
      stbir__simdi i, o0, o1;
      stbir__simdf of0, of1;
      STBIR_NO_UNROLL(decode);
      stbir__simdi_load( i, input );
      stbir__simdi_expand_u16_to_u32( o0, o1, i );
      stbir__simdi_convert_i32_to_float( of0, o0 );
      stbir__simdi_convert_i32_to_float( of1, o1 );
      stbir__decode_simdf4_flip( of0 );
      stbir__decode_simdf4_flip( of1 );
      stbir__simdf_store( decode + 0,  of0 );
      stbir__simdf_store( decode + 4,  of1 );
      #endif
      decode += 8;
      input += 8;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 8 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m8;
    }
    return decode_end + 8;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = ((float)(input[stbir__decode_order0]));
    decode[1-4] = ((float)(input[stbir__decode_order1]));
    decode[2-4] = ((float)(input[stbir__decode_order2]));
    decode[3-4] = ((float)(input[stbir__decode_order3]));
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = ((float)(input[stbir__decode_order0]));
    #if stbir__coder_min_num >= 2
    decode[1] = ((float)(input[stbir__decode_order1]));
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = ((float)(input[stbir__decode_order2]));
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;
}

static void STBIR__CODER_NAME(stbir__encode_uint16_linear)( void * outputp, int width_times_channels, float const * encode )
{
  unsigned short STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned short*) outputp;
  unsigned short * end_output = ( (unsigned short*) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  {
    if ( width_times_channels >= stbir__simdfX_float_count*2 )
    {
      float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
      end_output -= stbir__simdfX_float_count*2;
      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        stbir__simdfX e0, e1;
        stbir__simdiX i;
        STBIR_SIMD_NO_UNROLL(encode);
        stbir__simdfX_add_mem( e0, STBIR_simd_point5X, encode );
        stbir__simdfX_add_mem( e1, STBIR_simd_point5X, encode+stbir__simdfX_float_count );
        stbir__encode_simdfX_unflip( e0 );
        stbir__encode_simdfX_unflip( e1 );
        stbir__simdfX_pack_to_words( i, e0, e1 );
        stbir__simdiX_store( output, i );
        encode += stbir__simdfX_float_count*2;
        output += stbir__simdfX_float_count*2;
        if ( output <= end_output )
          continue;
        if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
          break;
        output = end_output; // backup and do last couple
        encode = end_encode_m8;
      }
      return;
    }
  }

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    stbir__simdf e;
    stbir__simdi i;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_load( e, encode );
    stbir__simdf_add( e, STBIR__CONSTF(STBIR_simd_point5), e );
    stbir__encode_simdf4_unflip( e );
    stbir__simdf_pack_to_8words( i, e, e );  // only use first 4
    stbir__simdi_store2( output-4, i );
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  #else

  // try to do blocks of 4 when you can
  #if  stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    float f;
    STBIR_SIMD_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[0-4] = (unsigned short)f;
    f = encode[stbir__encode_order1] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[1-4] = (unsigned short)f;
    f = encode[stbir__encode_order2] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[2-4] = (unsigned short)f;
    f = encode[stbir__encode_order3] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[3-4] = (unsigned short)f;
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    float f;
    STBIR_NO_UNROLL(encode);
    f = encode[stbir__encode_order0] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[0] = (unsigned short)f;
    #if stbir__coder_min_num >= 2
    f = encode[stbir__encode_order1] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[1] = (unsigned short)f;
    #endif
    #if stbir__coder_min_num >= 3
    f = encode[stbir__encode_order2] + 0.5f; STBIR_CLAMP(f, 0, 65535); output[2] = (unsigned short)f;
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_half_float_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  stbir__FP16 const * input = (stbir__FP16 const *)inputp;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= 8 )
  {
    stbir__FP16 const * end_input_m8 = input + width_times_channels - 8;
    decode_end -= 8;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      STBIR_NO_UNROLL(decode);

      stbir__half_to_float_SIMD( decode, input );
      #ifdef stbir__decode_swizzle
      #ifdef STBIR_SIMD8
      {
        stbir__simdf8 of;
        stbir__simdf8_load( of, decode );
        stbir__decode_simdf8_flip( of );
        stbir__simdf8_store( decode, of );
      }
      #else
      {
        stbir__simdf of0,of1;
        stbir__simdf_load( of0, decode );
        stbir__simdf_load( of1, decode+4 );
        stbir__decode_simdf4_flip( of0 );
        stbir__decode_simdf4_flip( of1 );
        stbir__simdf_store( decode, of0 );
        stbir__simdf_store( decode+4, of1 );
      }
      #endif
      #endif
      decode += 8;
      input += 8;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 8 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m8;
    }
    return decode_end + 8;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = stbir__half_to_float(input[stbir__decode_order0]);
    decode[1-4] = stbir__half_to_float(input[stbir__decode_order1]);
    decode[2-4] = stbir__half_to_float(input[stbir__decode_order2]);
    decode[3-4] = stbir__half_to_float(input[stbir__decode_order3]);
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = stbir__half_to_float(input[stbir__decode_order0]);
    #if stbir__coder_min_num >= 2
    decode[1] = stbir__half_to_float(input[stbir__decode_order1]);
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = stbir__half_to_float(input[stbir__decode_order2]);
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_half_float_linear )( void * outputp, int width_times_channels, float const * encode )
{
  stbir__FP16 STBIR_SIMD_STREAMOUT_PTR( * ) output = (stbir__FP16*) outputp;
  stbir__FP16 * end_output = ( (stbir__FP16*) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= 8 )
  {
    float const * end_encode_m8 = encode + width_times_channels - 8;
    end_output -= 8;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      STBIR_SIMD_NO_UNROLL(encode);
      #ifdef stbir__decode_swizzle
      #ifdef STBIR_SIMD8
      {
        stbir__simdf8 of;
        stbir__simdf8_load( of, encode );
        stbir__encode_simdf8_unflip( of );
        stbir__float_to_half_SIMD( output, (float*)&of );
      }
      #else
      {
        stbir__simdf of[2];
        stbir__simdf_load( of[0], encode );
        stbir__simdf_load( of[1], encode+4 );
        stbir__encode_simdf4_unflip( of[0] );
        stbir__encode_simdf4_unflip( of[1] );
        stbir__float_to_half_SIMD( output, (float*)of );
      }
      #endif
      #else
      stbir__float_to_half_SIMD( output, encode );
      #endif
      encode += 8;
      output += 8;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + 8 ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m8;
    }
    return;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    STBIR_SIMD_NO_UNROLL(output);
    output[0-4] = stbir__float_to_half(encode[stbir__encode_order0]);
    output[1-4] = stbir__float_to_half(encode[stbir__encode_order1]);
    output[2-4] = stbir__float_to_half(encode[stbir__encode_order2]);
    output[3-4] = stbir__float_to_half(encode[stbir__encode_order3]);
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    STBIR_NO_UNROLL(output);
    output[0] = stbir__float_to_half(encode[stbir__encode_order0]);
    #if stbir__coder_min_num >= 2
    output[1] = stbir__float_to_half(encode[stbir__encode_order1]);
    #endif
    #if stbir__coder_min_num >= 3
    output[2] = stbir__float_to_half(encode[stbir__encode_order2]);
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_float_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  #ifdef stbir__decode_swizzle
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  float const * input = (float const *)inputp;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= 16 )
  {
    float const * end_input_m16 = input + width_times_channels - 16;
    decode_end -= 16;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      STBIR_NO_UNROLL(decode);
      #ifdef stbir__decode_swizzle
      #ifdef STBIR_SIMD8
      {
        stbir__simdf8 of0,of1;
        stbir__simdf8_load( of0, input );
        stbir__simdf8_load( of1, input+8 );
        stbir__decode_simdf8_flip( of0 );
        stbir__decode_simdf8_flip( of1 );
        stbir__simdf8_store( decode, of0 );
        stbir__simdf8_store( decode+8, of1 );
      }
      #else
      {
        stbir__simdf of0,of1,of2,of3;
        stbir__simdf_load( of0, input );
        stbir__simdf_load( of1, input+4 );
        stbir__simdf_load( of2, input+8 );
        stbir__simdf_load( of3, input+12 );
        stbir__decode_simdf4_flip( of0 );
        stbir__decode_simdf4_flip( of1 );
        stbir__decode_simdf4_flip( of2 );
        stbir__decode_simdf4_flip( of3 );
        stbir__simdf_store( decode, of0 );
        stbir__simdf_store( decode+4, of1 );
        stbir__simdf_store( decode+8, of2 );
        stbir__simdf_store( decode+12, of3 );
      }
      #endif
      #endif
      decode += 16;
      input += 16;
      if ( decode <= decode_end )
        continue;
      if ( decode == ( decode_end + 16 ) )
        break;
      decode = decode_end; // backup and do last couple
      input = end_input_m16;
    }
    return decode_end + 16;
  }
  #endif

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  decode += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode <= decode_end )
  {
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0-4] = input[stbir__decode_order0];
    decode[1-4] = input[stbir__decode_order1];
    decode[2-4] = input[stbir__decode_order2];
    decode[3-4] = input[stbir__decode_order3];
    decode += 4;
    input += 4;
  }
  decode -= 4;
  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( decode < decode_end )
  {
    STBIR_NO_UNROLL(decode);
    decode[0] = input[stbir__decode_order0];
    #if stbir__coder_min_num >= 2
    decode[1] = input[stbir__decode_order1];
    #endif
    #if stbir__coder_min_num >= 3
    decode[2] = input[stbir__decode_order2];
    #endif
    decode += stbir__coder_min_num;
    input += stbir__coder_min_num;
  }
  #endif
  return decode_end;

  #else

  if ( (void*)decodep != inputp )
    STBIR_MEMCPY( decodep, inputp, width_times_channels * sizeof( float ) );

  return decodep + width_times_channels;

  #endif
}

static void STBIR__CODER_NAME( stbir__encode_float_linear )( void * outputp, int width_times_channels, float const * encode )
{
  #if !defined( STBIR_FLOAT_HIGH_CLAMP ) && !defined(STBIR_FLOAT_LO_CLAMP) && !defined(stbir__decode_swizzle)

  if ( (void*)outputp != (void*) encode )
    STBIR_MEMCPY( outputp, encode, width_times_channels * sizeof( float ) );

  #else

  float STBIR_SIMD_STREAMOUT_PTR( * ) output = (float*) outputp;
  float * end_output = ( (float*) output ) + width_times_channels;

  #ifdef STBIR_FLOAT_HIGH_CLAMP
  #define stbir_scalar_hi_clamp( v ) if ( v > STBIR_FLOAT_HIGH_CLAMP ) v = STBIR_FLOAT_HIGH_CLAMP;
  #else
  #define stbir_scalar_hi_clamp( v )
  #endif
  #ifdef STBIR_FLOAT_LOW_CLAMP
  #define stbir_scalar_lo_clamp( v ) if ( v < STBIR_FLOAT_LOW_CLAMP ) v = STBIR_FLOAT_LOW_CLAMP;
  #else
  #define stbir_scalar_lo_clamp( v )
  #endif

  #ifdef STBIR_SIMD

  #ifdef STBIR_FLOAT_HIGH_CLAMP
  const stbir__simdfX high_clamp = stbir__simdf_frepX(STBIR_FLOAT_HIGH_CLAMP);
  #endif
  #ifdef STBIR_FLOAT_LOW_CLAMP
  const stbir__simdfX low_clamp = stbir__simdf_frepX(STBIR_FLOAT_LOW_CLAMP);
  #endif

  if ( width_times_channels >= ( stbir__simdfX_float_count * 2 ) )
  {
    float const * end_encode_m8 = encode + width_times_channels - ( stbir__simdfX_float_count * 2 );
    end_output -= ( stbir__simdfX_float_count * 2 );
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_load( e0, encode );
      stbir__simdfX_load( e1, encode+stbir__simdfX_float_count );
#ifdef STBIR_FLOAT_HIGH_CLAMP
      stbir__simdfX_min( e0, e0, high_clamp );
      stbir__simdfX_min( e1, e1, high_clamp );
#endif
#ifdef STBIR_FLOAT_LOW_CLAMP
      stbir__simdfX_max( e0, e0, low_clamp );
      stbir__simdfX_max( e1, e1, low_clamp );
#endif
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      stbir__simdfX_store( output, e0 );
      stbir__simdfX_store( output+stbir__simdfX_float_count, e1 );
      encode += stbir__simdfX_float_count * 2;
      output += stbir__simdfX_float_count * 2;
      if ( output < end_output )
        continue;
      if ( output == ( end_output + ( stbir__simdfX_float_count * 2 ) ) )
        break;
      output = end_output; // backup and do last couple
      encode = end_encode_m8;
    }
    return;
  }

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    stbir__simdf e0;
    STBIR_NO_UNROLL(encode);
    stbir__simdf_load( e0, encode );
#ifdef STBIR_FLOAT_HIGH_CLAMP
    stbir__simdf_min( e0, e0, high_clamp );
#endif
#ifdef STBIR_FLOAT_LOW_CLAMP
    stbir__simdf_max( e0, e0, low_clamp );
#endif
    stbir__encode_simdf4_unflip( e0 );
    stbir__simdf_store( output-4, e0 );
    output += 4;
    encode += 4;
  }
  output -= 4;
  #endif

  #else

  // try to do blocks of 4 when you can
  #if stbir__coder_min_num != 3 // doesn't divide cleanly by four
  output += 4;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( output <= end_output )
  {
    float e;
    STBIR_SIMD_NO_UNROLL(encode);
    e = encode[ stbir__encode_order0 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[0-4] = e;
    e = encode[ stbir__encode_order1 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[1-4] = e;
    e = encode[ stbir__encode_order2 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[2-4] = e;
    e = encode[ stbir__encode_order3 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[3-4] = e;
    output += 4;
    encode += 4;
  }
  output -= 4;

  #endif

  #endif

  // do the remnants
  #if stbir__coder_min_num < 4
  STBIR_NO_UNROLL_LOOP_START
  while( output < end_output )
  {
    float e;
    STBIR_NO_UNROLL(encode);
    e = encode[ stbir__encode_order0 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[0] = e;
    #if stbir__coder_min_num >= 2
    e = encode[ stbir__encode_order1 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[1] = e;
    #endif
    #if stbir__coder_min_num >= 3
    e = encode[ stbir__encode_order2 ]; stbir_scalar_hi_clamp( e ); stbir_scalar_lo_clamp( e ); output[2] = e;
    #endif
    output += stbir__coder_min_num;
    encode += stbir__coder_min_num;
  }
  #endif

  #endif
}

#undef stbir__decode_suffix
#undef stbir__decode_simdf8_flip
#undef stbir__decode_simdf4_flip
#undef stbir__decode_order0
#undef stbir__decode_order1
#undef stbir__decode_order2
#undef stbir__decode_order3
#undef stbir__encode_order0
#undef stbir__encode_order1
#undef stbir__encode_order2
#undef stbir__encode_order3
#undef stbir__encode_simdf8_unflip
#undef stbir__encode_simdf4_unflip
#undef stbir__encode_simdfX_unflip
#undef STBIR__CODER_NAME
#undef stbir__coder_min_num
#undef stbir__decode_swizzle
#undef stbir_scalar_hi_clamp
#undef stbir_scalar_lo_clamp
#undef STB_IMAGE_RESIZE_DO_CODERS

#elif defined( STB_IMAGE_RESIZE_DO_VERTICALS)

#ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#define STBIR_chans( start, end ) STBIR_strs_join14(start,STBIR__vertical_channels,end,_cont)
#else
#define STBIR_chans( start, end ) STBIR_strs_join1(start,STBIR__vertical_channels,end)
#endif

#if STBIR__vertical_channels >= 1
#define stbIF0( code ) code
#else
#define stbIF0( code )
#endif
#if STBIR__vertical_channels >= 2
#define stbIF1( code ) code
#else
#define stbIF1( code )
#endif
#if STBIR__vertical_channels >= 3
#define stbIF2( code ) code
#else
#define stbIF2( code )
#endif
#if STBIR__vertical_channels >= 4
#define stbIF3( code ) code
#else
#define stbIF3( code )
#endif
#if STBIR__vertical_channels >= 5
#define stbIF4( code ) code
#else
#define stbIF4( code )
#endif
#if STBIR__vertical_channels >= 6
#define stbIF5( code ) code
#else
#define stbIF5( code )
#endif
#if STBIR__vertical_channels >= 7
#define stbIF6( code ) code
#else
#define stbIF6( code )
#endif
#if STBIR__vertical_channels >= 8
#define stbIF7( code ) code
#else
#define stbIF7( code )
#endif

static void STBIR_chans( stbir__vertical_scatter_with_,_coeffs)( float ** outputs, float const * vertical_coefficients, float const * input, float const * input_end )
{
  stbIF0( float STBIR_SIMD_STREAMOUT_PTR( * ) output0 = outputs[0]; float c0s = vertical_coefficients[0]; )
  stbIF1( float STBIR_SIMD_STREAMOUT_PTR( * ) output1 = outputs[1]; float c1s = vertical_coefficients[1]; )
  stbIF2( float STBIR_SIMD_STREAMOUT_PTR( * ) output2 = outputs[2]; float c2s = vertical_coefficients[2]; )
  stbIF3( float STBIR_SIMD_STREAMOUT_PTR( * ) output3 = outputs[3]; float c3s = vertical_coefficients[3]; )
  stbIF4( float STBIR_SIMD_STREAMOUT_PTR( * ) output4 = outputs[4]; float c4s = vertical_coefficients[4]; )
  stbIF5( float STBIR_SIMD_STREAMOUT_PTR( * ) output5 = outputs[5]; float c5s = vertical_coefficients[5]; )
  stbIF6( float STBIR_SIMD_STREAMOUT_PTR( * ) output6 = outputs[6]; float c6s = vertical_coefficients[6]; )
  stbIF7( float STBIR_SIMD_STREAMOUT_PTR( * ) output7 = outputs[7]; float c7s = vertical_coefficients[7]; )

  #ifdef STBIR_SIMD
  {
    stbIF0(stbir__simdfX c0 = stbir__simdf_frepX( c0s ); )
    stbIF1(stbir__simdfX c1 = stbir__simdf_frepX( c1s ); )
    stbIF2(stbir__simdfX c2 = stbir__simdf_frepX( c2s ); )
    stbIF3(stbir__simdfX c3 = stbir__simdf_frepX( c3s ); )
    stbIF4(stbir__simdfX c4 = stbir__simdf_frepX( c4s ); )
    stbIF5(stbir__simdfX c5 = stbir__simdf_frepX( c5s ); )
    stbIF6(stbir__simdfX c6 = stbir__simdf_frepX( c6s ); )
    stbIF7(stbir__simdfX c7 = stbir__simdf_frepX( c7s ); )
    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input_end - (char*) input ) >= (16*stbir__simdfX_float_count) )
    {
      stbir__simdfX o0, o1, o2, o3, r0, r1, r2, r3;
      STBIR_SIMD_NO_UNROLL(output0);

      stbir__simdfX_load( r0, input );               stbir__simdfX_load( r1, input+stbir__simdfX_float_count );     stbir__simdfX_load( r2, input+(2*stbir__simdfX_float_count) );      stbir__simdfX_load( r3, input+(3*stbir__simdfX_float_count) );

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdfX_load( o0, output0 );     stbir__simdfX_load( o1, output0+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output0+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c0 );  stbir__simdfX_madd( o1, o1, r1, c0 );  stbir__simdfX_madd( o2, o2, r2, c0 );   stbir__simdfX_madd( o3, o3, r3, c0 );
              stbir__simdfX_store( output0, o0 );    stbir__simdfX_store( output0+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output0+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output0+(3*stbir__simdfX_float_count), o3 ); )
      stbIF1( stbir__simdfX_load( o0, output1 );     stbir__simdfX_load( o1, output1+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output1+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output1+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c1 );  stbir__simdfX_madd( o1, o1, r1, c1 );  stbir__simdfX_madd( o2, o2, r2, c1 );   stbir__simdfX_madd( o3, o3, r3, c1 );
              stbir__simdfX_store( output1, o0 );    stbir__simdfX_store( output1+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output1+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output1+(3*stbir__simdfX_float_count), o3 ); )
      stbIF2( stbir__simdfX_load( o0, output2 );     stbir__simdfX_load( o1, output2+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output2+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output2+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c2 );  stbir__simdfX_madd( o1, o1, r1, c2 );  stbir__simdfX_madd( o2, o2, r2, c2 );   stbir__simdfX_madd( o3, o3, r3, c2 );
              stbir__simdfX_store( output2, o0 );    stbir__simdfX_store( output2+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output2+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output2+(3*stbir__simdfX_float_count), o3 ); )
      stbIF3( stbir__simdfX_load( o0, output3 );     stbir__simdfX_load( o1, output3+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output3+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output3+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c3 );  stbir__simdfX_madd( o1, o1, r1, c3 );  stbir__simdfX_madd( o2, o2, r2, c3 );   stbir__simdfX_madd( o3, o3, r3, c3 );
              stbir__simdfX_store( output3, o0 );    stbir__simdfX_store( output3+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output3+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output3+(3*stbir__simdfX_float_count), o3 ); )
      stbIF4( stbir__simdfX_load( o0, output4 );     stbir__simdfX_load( o1, output4+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output4+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output4+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c4 );  stbir__simdfX_madd( o1, o1, r1, c4 );  stbir__simdfX_madd( o2, o2, r2, c4 );   stbir__simdfX_madd( o3, o3, r3, c4 );
              stbir__simdfX_store( output4, o0 );    stbir__simdfX_store( output4+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output4+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output4+(3*stbir__simdfX_float_count), o3 ); )
      stbIF5( stbir__simdfX_load( o0, output5 );     stbir__simdfX_load( o1, output5+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output5+(2*stbir__simdfX_float_count));    stbir__simdfX_load( o3, output5+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c5 );  stbir__simdfX_madd( o1, o1, r1, c5 );  stbir__simdfX_madd( o2, o2, r2, c5 );   stbir__simdfX_madd( o3, o3, r3, c5 );
              stbir__simdfX_store( output5, o0 );    stbir__simdfX_store( output5+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output5+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output5+(3*stbir__simdfX_float_count), o3 ); )
      stbIF6( stbir__simdfX_load( o0, output6 );     stbir__simdfX_load( o1, output6+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output6+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output6+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c6 );  stbir__simdfX_madd( o1, o1, r1, c6 );  stbir__simdfX_madd( o2, o2, r2, c6 );   stbir__simdfX_madd( o3, o3, r3, c6 );
              stbir__simdfX_store( output6, o0 );    stbir__simdfX_store( output6+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output6+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output6+(3*stbir__simdfX_float_count), o3 ); )
      stbIF7( stbir__simdfX_load( o0, output7 );     stbir__simdfX_load( o1, output7+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output7+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output7+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c7 );  stbir__simdfX_madd( o1, o1, r1, c7 );  stbir__simdfX_madd( o2, o2, r2, c7 );   stbir__simdfX_madd( o3, o3, r3, c7 );
              stbir__simdfX_store( output7, o0 );    stbir__simdfX_store( output7+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output7+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output7+(3*stbir__simdfX_float_count), o3 ); )
      #else
      stbIF0( stbir__simdfX_mult( o0, r0, c0 );      stbir__simdfX_mult( o1, r1, c0 );      stbir__simdfX_mult( o2, r2, c0 );       stbir__simdfX_mult( o3, r3, c0 );
              stbir__simdfX_store( output0, o0 );    stbir__simdfX_store( output0+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output0+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output0+(3*stbir__simdfX_float_count), o3 ); )
      stbIF1( stbir__simdfX_mult( o0, r0, c1 );      stbir__simdfX_mult( o1, r1, c1 );      stbir__simdfX_mult( o2, r2, c1 );       stbir__simdfX_mult( o3, r3, c1 );
              stbir__simdfX_store( output1, o0 );    stbir__simdfX_store( output1+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output1+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output1+(3*stbir__simdfX_float_count), o3 ); )
      stbIF2( stbir__simdfX_mult( o0, r0, c2 );      stbir__simdfX_mult( o1, r1, c2 );      stbir__simdfX_mult( o2, r2, c2 );       stbir__simdfX_mult( o3, r3, c2 );
              stbir__simdfX_store( output2, o0 );    stbir__simdfX_store( output2+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output2+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output2+(3*stbir__simdfX_float_count), o3 ); )
      stbIF3( stbir__simdfX_mult( o0, r0, c3 );      stbir__simdfX_mult( o1, r1, c3 );      stbir__simdfX_mult( o2, r2, c3 );       stbir__simdfX_mult( o3, r3, c3 );
              stbir__simdfX_store( output3, o0 );    stbir__simdfX_store( output3+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output3+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output3+(3*stbir__simdfX_float_count), o3 ); )
      stbIF4( stbir__simdfX_mult( o0, r0, c4 );      stbir__simdfX_mult( o1, r1, c4 );      stbir__simdfX_mult( o2, r2, c4 );       stbir__simdfX_mult( o3, r3, c4 );
              stbir__simdfX_store( output4, o0 );    stbir__simdfX_store( output4+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output4+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output4+(3*stbir__simdfX_float_count), o3 ); )
      stbIF5( stbir__simdfX_mult( o0, r0, c5 );      stbir__simdfX_mult( o1, r1, c5 );      stbir__simdfX_mult( o2, r2, c5 );       stbir__simdfX_mult( o3, r3, c5 );
              stbir__simdfX_store( output5, o0 );    stbir__simdfX_store( output5+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output5+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output5+(3*stbir__simdfX_float_count), o3 ); )
      stbIF6( stbir__simdfX_mult( o0, r0, c6 );      stbir__simdfX_mult( o1, r1, c6 );      stbir__simdfX_mult( o2, r2, c6 );       stbir__simdfX_mult( o3, r3, c6 );
              stbir__simdfX_store( output6, o0 );    stbir__simdfX_store( output6+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output6+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output6+(3*stbir__simdfX_float_count), o3 ); )
      stbIF7( stbir__simdfX_mult( o0, r0, c7 );      stbir__simdfX_mult( o1, r1, c7 );      stbir__simdfX_mult( o2, r2, c7 );       stbir__simdfX_mult( o3, r3, c7 );
              stbir__simdfX_store( output7, o0 );    stbir__simdfX_store( output7+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output7+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output7+(3*stbir__simdfX_float_count), o3 ); )
      #endif

      input += (4*stbir__simdfX_float_count);
      stbIF0( output0 += (4*stbir__simdfX_float_count); ) stbIF1( output1 += (4*stbir__simdfX_float_count); ) stbIF2( output2 += (4*stbir__simdfX_float_count); ) stbIF3( output3 += (4*stbir__simdfX_float_count); ) stbIF4( output4 += (4*stbir__simdfX_float_count); ) stbIF5( output5 += (4*stbir__simdfX_float_count); ) stbIF6( output6 += (4*stbir__simdfX_float_count); ) stbIF7( output7 += (4*stbir__simdfX_float_count); )
    }
    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input_end - (char*) input ) >= 16 )
    {
      stbir__simdf o0, r0;
      STBIR_SIMD_NO_UNROLL(output0);

      stbir__simdf_load( r0, input );

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdf_load( o0, output0 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) );  stbir__simdf_store( output0, o0 ); )
      stbIF1( stbir__simdf_load( o0, output1 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) );  stbir__simdf_store( output1, o0 ); )
      stbIF2( stbir__simdf_load( o0, output2 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) );  stbir__simdf_store( output2, o0 ); )
      stbIF3( stbir__simdf_load( o0, output3 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) );  stbir__simdf_store( output3, o0 ); )
      stbIF4( stbir__simdf_load( o0, output4 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) );  stbir__simdf_store( output4, o0 ); )
      stbIF5( stbir__simdf_load( o0, output5 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) );  stbir__simdf_store( output5, o0 ); )
      stbIF6( stbir__simdf_load( o0, output6 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) );  stbir__simdf_store( output6, o0 ); )
      stbIF7( stbir__simdf_load( o0, output7 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) );  stbir__simdf_store( output7, o0 ); )
      #else
      stbIF0( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) );   stbir__simdf_store( output0, o0 ); )
      stbIF1( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) );   stbir__simdf_store( output1, o0 ); )
      stbIF2( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) );   stbir__simdf_store( output2, o0 ); )
      stbIF3( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) );   stbir__simdf_store( output3, o0 ); )
      stbIF4( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) );   stbir__simdf_store( output4, o0 ); )
      stbIF5( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) );   stbir__simdf_store( output5, o0 ); )
      stbIF6( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) );   stbir__simdf_store( output6, o0 ); )
      stbIF7( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) );   stbir__simdf_store( output7, o0 ); )
      #endif

      input += 4;
      stbIF0( output0 += 4; ) stbIF1( output1 += 4; ) stbIF2( output2 += 4; ) stbIF3( output3 += 4; ) stbIF4( output4 += 4; ) stbIF5( output5 += 4; ) stbIF6( output6 += 4; ) stbIF7( output7 += 4; )
    }
  }
  #else
  STBIR_NO_UNROLL_LOOP_START
  while ( ( (char*)input_end - (char*) input ) >= 16 )
  {
    float r0, r1, r2, r3;
    STBIR_NO_UNROLL(input);

    r0 = input[0], r1 = input[1], r2 = input[2], r3 = input[3];

    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( output0[0] += ( r0 * c0s ); output0[1] += ( r1 * c0s ); output0[2] += ( r2 * c0s ); output0[3] += ( r3 * c0s ); )
    stbIF1( output1[0] += ( r0 * c1s ); output1[1] += ( r1 * c1s ); output1[2] += ( r2 * c1s ); output1[3] += ( r3 * c1s ); )
    stbIF2( output2[0] += ( r0 * c2s ); output2[1] += ( r1 * c2s ); output2[2] += ( r2 * c2s ); output2[3] += ( r3 * c2s ); )
    stbIF3( output3[0] += ( r0 * c3s ); output3[1] += ( r1 * c3s ); output3[2] += ( r2 * c3s ); output3[3] += ( r3 * c3s ); )
    stbIF4( output4[0] += ( r0 * c4s ); output4[1] += ( r1 * c4s ); output4[2] += ( r2 * c4s ); output4[3] += ( r3 * c4s ); )
    stbIF5( output5[0] += ( r0 * c5s ); output5[1] += ( r1 * c5s ); output5[2] += ( r2 * c5s ); output5[3] += ( r3 * c5s ); )
    stbIF6( output6[0] += ( r0 * c6s ); output6[1] += ( r1 * c6s ); output6[2] += ( r2 * c6s ); output6[3] += ( r3 * c6s ); )
    stbIF7( output7[0] += ( r0 * c7s ); output7[1] += ( r1 * c7s ); output7[2] += ( r2 * c7s ); output7[3] += ( r3 * c7s ); )
    #else
    stbIF0( output0[0]  = ( r0 * c0s ); output0[1]  = ( r1 * c0s ); output0[2]  = ( r2 * c0s ); output0[3]  = ( r3 * c0s ); )
    stbIF1( output1[0]  = ( r0 * c1s ); output1[1]  = ( r1 * c1s ); output1[2]  = ( r2 * c1s ); output1[3]  = ( r3 * c1s ); )
    stbIF2( output2[0]  = ( r0 * c2s ); output2[1]  = ( r1 * c2s ); output2[2]  = ( r2 * c2s ); output2[3]  = ( r3 * c2s ); )
    stbIF3( output3[0]  = ( r0 * c3s ); output3[1]  = ( r1 * c3s ); output3[2]  = ( r2 * c3s ); output3[3]  = ( r3 * c3s ); )
    stbIF4( output4[0]  = ( r0 * c4s ); output4[1]  = ( r1 * c4s ); output4[2]  = ( r2 * c4s ); output4[3]  = ( r3 * c4s ); )
    stbIF5( output5[0]  = ( r0 * c5s ); output5[1]  = ( r1 * c5s ); output5[2]  = ( r2 * c5s ); output5[3]  = ( r3 * c5s ); )
    stbIF6( output6[0]  = ( r0 * c6s ); output6[1]  = ( r1 * c6s ); output6[2]  = ( r2 * c6s ); output6[3]  = ( r3 * c6s ); )
    stbIF7( output7[0]  = ( r0 * c7s ); output7[1]  = ( r1 * c7s ); output7[2]  = ( r2 * c7s ); output7[3]  = ( r3 * c7s ); )
    #endif

    input += 4;
    stbIF0( output0 += 4; ) stbIF1( output1 += 4; ) stbIF2( output2 += 4; ) stbIF3( output3 += 4; ) stbIF4( output4 += 4; ) stbIF5( output5 += 4; ) stbIF6( output6 += 4; ) stbIF7( output7 += 4; )
  }
  #endif
  STBIR_NO_UNROLL_LOOP_START
  while ( input < input_end )
  {
    float r = input[0];
    STBIR_NO_UNROLL(output0);

    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( output0[0] += ( r * c0s ); )
    stbIF1( output1[0] += ( r * c1s ); )
    stbIF2( output2[0] += ( r * c2s ); )
    stbIF3( output3[0] += ( r * c3s ); )
    stbIF4( output4[0] += ( r * c4s ); )
    stbIF5( output5[0] += ( r * c5s ); )
    stbIF6( output6[0] += ( r * c6s ); )
    stbIF7( output7[0] += ( r * c7s ); )
    #else
    stbIF0( output0[0]  = ( r * c0s ); )
    stbIF1( output1[0]  = ( r * c1s ); )
    stbIF2( output2[0]  = ( r * c2s ); )
    stbIF3( output3[0]  = ( r * c3s ); )
    stbIF4( output4[0]  = ( r * c4s ); )
    stbIF5( output5[0]  = ( r * c5s ); )
    stbIF6( output6[0]  = ( r * c6s ); )
    stbIF7( output7[0]  = ( r * c7s ); )
    #endif

    ++input;
    stbIF0( ++output0; ) stbIF1( ++output1; ) stbIF2( ++output2; ) stbIF3( ++output3; ) stbIF4( ++output4; ) stbIF5( ++output5; ) stbIF6( ++output6; ) stbIF7( ++output7; )
  }
}

static void STBIR_chans( stbir__vertical_gather_with_,_coeffs)( float * outputp, float const * vertical_coefficients, float const ** inputs, float const * input0_end )
{
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = outputp;

  stbIF0( float const * input0 = inputs[0]; float c0s = vertical_coefficients[0]; )
  stbIF1( float const * input1 = inputs[1]; float c1s = vertical_coefficients[1]; )
  stbIF2( float const * input2 = inputs[2]; float c2s = vertical_coefficients[2]; )
  stbIF3( float const * input3 = inputs[3]; float c3s = vertical_coefficients[3]; )
  stbIF4( float const * input4 = inputs[4]; float c4s = vertical_coefficients[4]; )
  stbIF5( float const * input5 = inputs[5]; float c5s = vertical_coefficients[5]; )
  stbIF6( float const * input6 = inputs[6]; float c6s = vertical_coefficients[6]; )
  stbIF7( float const * input7 = inputs[7]; float c7s = vertical_coefficients[7]; )

#if ( STBIR__vertical_channels == 1 ) && !defined(STB_IMAGE_RESIZE_VERTICAL_CONTINUE)
  // check single channel one weight
  if ( ( c0s >= (1.0f-0.000001f) ) && ( c0s <= (1.0f+0.000001f) ) )
  {
    STBIR_MEMCPY( output, input0, (char*)input0_end - (char*)input0 );
    return;
  }
#endif

  #ifdef STBIR_SIMD
  {
    stbIF0(stbir__simdfX c0 = stbir__simdf_frepX( c0s ); )
    stbIF1(stbir__simdfX c1 = stbir__simdf_frepX( c1s ); )
    stbIF2(stbir__simdfX c2 = stbir__simdf_frepX( c2s ); )
    stbIF3(stbir__simdfX c3 = stbir__simdf_frepX( c3s ); )
    stbIF4(stbir__simdfX c4 = stbir__simdf_frepX( c4s ); )
    stbIF5(stbir__simdfX c5 = stbir__simdf_frepX( c5s ); )
    stbIF6(stbir__simdfX c6 = stbir__simdf_frepX( c6s ); )
    stbIF7(stbir__simdfX c7 = stbir__simdf_frepX( c7s ); )

    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input0_end - (char*) input0 ) >= (16*stbir__simdfX_float_count) )
    {
      stbir__simdfX o0, o1, o2, o3, r0, r1, r2, r3;
      STBIR_SIMD_NO_UNROLL(output);

      // prefetch four loop iterations ahead (doesn't affect much for small resizes, but helps with big ones)
      stbIF0( stbir__prefetch( input0 + (16*stbir__simdfX_float_count) ); )
      stbIF1( stbir__prefetch( input1 + (16*stbir__simdfX_float_count) ); )
      stbIF2( stbir__prefetch( input2 + (16*stbir__simdfX_float_count) ); )
      stbIF3( stbir__prefetch( input3 + (16*stbir__simdfX_float_count) ); )
      stbIF4( stbir__prefetch( input4 + (16*stbir__simdfX_float_count) ); )
      stbIF5( stbir__prefetch( input5 + (16*stbir__simdfX_float_count) ); )
      stbIF6( stbir__prefetch( input6 + (16*stbir__simdfX_float_count) ); )
      stbIF7( stbir__prefetch( input7 + (16*stbir__simdfX_float_count) ); )

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdfX_load( o0, output );      stbir__simdfX_load( o1, output+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( o3, output+(3*stbir__simdfX_float_count) );
              stbir__simdfX_load( r0, input0 );      stbir__simdfX_load( r1, input0+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input0+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c0 );  stbir__simdfX_madd( o1, o1, r1, c0 );                         stbir__simdfX_madd( o2, o2, r2, c0 );                             stbir__simdfX_madd( o3, o3, r3, c0 ); )
      #else
      stbIF0( stbir__simdfX_load( r0, input0 );      stbir__simdfX_load( r1, input0+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input0+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_mult( o0, r0, c0 );      stbir__simdfX_mult( o1, r1, c0 );                             stbir__simdfX_mult( o2, r2, c0 );                                 stbir__simdfX_mult( o3, r3, c0 );  )
      #endif

      stbIF1( stbir__simdfX_load( r0, input1 );      stbir__simdfX_load( r1, input1+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input1+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input1+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c1 );  stbir__simdfX_madd( o1, o1, r1, c1 );                         stbir__simdfX_madd( o2, o2, r2, c1 );                             stbir__simdfX_madd( o3, o3, r3, c1 ); )
      stbIF2( stbir__simdfX_load( r0, input2 );      stbir__simdfX_load( r1, input2+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input2+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input2+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c2 );  stbir__simdfX_madd( o1, o1, r1, c2 );                         stbir__simdfX_madd( o2, o2, r2, c2 );                             stbir__simdfX_madd( o3, o3, r3, c2 ); )
      stbIF3( stbir__simdfX_load( r0, input3 );      stbir__simdfX_load( r1, input3+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input3+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input3+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c3 );  stbir__simdfX_madd( o1, o1, r1, c3 );                         stbir__simdfX_madd( o2, o2, r2, c3 );                             stbir__simdfX_madd( o3, o3, r3, c3 ); )
      stbIF4( stbir__simdfX_load( r0, input4 );      stbir__simdfX_load( r1, input4+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input4+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input4+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c4 );  stbir__simdfX_madd( o1, o1, r1, c4 );                         stbir__simdfX_madd( o2, o2, r2, c4 );                             stbir__simdfX_madd( o3, o3, r3, c4 ); )
      stbIF5( stbir__simdfX_load( r0, input5 );      stbir__simdfX_load( r1, input5+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input5+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input5+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c5 );  stbir__simdfX_madd( o1, o1, r1, c5 );                         stbir__simdfX_madd( o2, o2, r2, c5 );                             stbir__simdfX_madd( o3, o3, r3, c5 ); )
      stbIF6( stbir__simdfX_load( r0, input6 );      stbir__simdfX_load( r1, input6+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input6+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input6+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c6 );  stbir__simdfX_madd( o1, o1, r1, c6 );                         stbir__simdfX_madd( o2, o2, r2, c6 );                             stbir__simdfX_madd( o3, o3, r3, c6 ); )
      stbIF7( stbir__simdfX_load( r0, input7 );      stbir__simdfX_load( r1, input7+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input7+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input7+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c7 );  stbir__simdfX_madd( o1, o1, r1, c7 );                         stbir__simdfX_madd( o2, o2, r2, c7 );                             stbir__simdfX_madd( o3, o3, r3, c7 ); )

      stbir__simdfX_store( output, o0 );             stbir__simdfX_store( output+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output+(2*stbir__simdfX_float_count), o2 );  stbir__simdfX_store( output+(3*stbir__simdfX_float_count), o3 );
      output += (4*stbir__simdfX_float_count);
      stbIF0( input0 += (4*stbir__simdfX_float_count); ) stbIF1( input1 += (4*stbir__simdfX_float_count); ) stbIF2( input2 += (4*stbir__simdfX_float_count); ) stbIF3( input3 += (4*stbir__simdfX_float_count); ) stbIF4( input4 += (4*stbir__simdfX_float_count); ) stbIF5( input5 += (4*stbir__simdfX_float_count); ) stbIF6( input6 += (4*stbir__simdfX_float_count); ) stbIF7( input7 += (4*stbir__simdfX_float_count); )
    }

    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input0_end - (char*) input0 ) >= 16 )
    {
      stbir__simdf o0, r0;
      STBIR_SIMD_NO_UNROLL(output);

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdf_load( o0, output );   stbir__simdf_load( r0, input0 ); stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) ); )
      #else
      stbIF0( stbir__simdf_load( r0, input0 );  stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) ); )
      #endif
      stbIF1( stbir__simdf_load( r0, input1 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) ); )
      stbIF2( stbir__simdf_load( r0, input2 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) ); )
      stbIF3( stbir__simdf_load( r0, input3 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) ); )
      stbIF4( stbir__simdf_load( r0, input4 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) ); )
      stbIF5( stbir__simdf_load( r0, input5 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) ); )
      stbIF6( stbir__simdf_load( r0, input6 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) ); )
      stbIF7( stbir__simdf_load( r0, input7 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) ); )

      stbir__simdf_store( output, o0 );
      output += 4;
      stbIF0( input0 += 4; ) stbIF1( input1 += 4; ) stbIF2( input2 += 4; ) stbIF3( input3 += 4; ) stbIF4( input4 += 4; ) stbIF5( input5 += 4; ) stbIF6( input6 += 4; ) stbIF7( input7 += 4; )
    }
  }
  #else
  STBIR_NO_UNROLL_LOOP_START
  while ( ( (char*)input0_end - (char*) input0 ) >= 16 )
  {
    float o0, o1, o2, o3;
    STBIR_NO_UNROLL(output);
    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( o0 = output[0] + input0[0] * c0s; o1 = output[1] + input0[1] * c0s; o2 = output[2] + input0[2] * c0s; o3 = output[3] + input0[3] * c0s; )
    #else
    stbIF0( o0  = input0[0] * c0s; o1  = input0[1] * c0s; o2  = input0[2] * c0s; o3  = input0[3] * c0s; )
    #endif
    stbIF1( o0 += input1[0] * c1s; o1 += input1[1] * c1s; o2 += input1[2] * c1s; o3 += input1[3] * c1s; )
    stbIF2( o0 += input2[0] * c2s; o1 += input2[1] * c2s; o2 += input2[2] * c2s; o3 += input2[3] * c2s; )
    stbIF3( o0 += input3[0] * c3s; o1 += input3[1] * c3s; o2 += input3[2] * c3s; o3 += input3[3] * c3s; )
    stbIF4( o0 += input4[0] * c4s; o1 += input4[1] * c4s; o2 += input4[2] * c4s; o3 += input4[3] * c4s; )
    stbIF5( o0 += input5[0] * c5s; o1 += input5[1] * c5s; o2 += input5[2] * c5s; o3 += input5[3] * c5s; )
    stbIF6( o0 += input6[0] * c6s; o1 += input6[1] * c6s; o2 += input6[2] * c6s; o3 += input6[3] * c6s; )
    stbIF7( o0 += input7[0] * c7s; o1 += input7[1] * c7s; o2 += input7[2] * c7s; o3 += input7[3] * c7s; )
    output[0] = o0; output[1] = o1; output[2] = o2; output[3] = o3;
    output += 4;
    stbIF0( input0 += 4; ) stbIF1( input1 += 4; ) stbIF2( input2 += 4; ) stbIF3( input3 += 4; ) stbIF4( input4 += 4; ) stbIF5( input5 += 4; ) stbIF6( input6 += 4; ) stbIF7( input7 += 4; )
  }
  #endif
  STBIR_NO_UNROLL_LOOP_START
  while ( input0 < input0_end )
  {
    float o0;
    STBIR_NO_UNROLL(output);
    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( o0 = output[0] + input0[0] * c0s; )
    #else
    stbIF0( o0  = input0[0] * c0s; )
    #endif
    stbIF1( o0 += input1[0] * c1s; )
    stbIF2( o0 += input2[0] * c2s; )
    stbIF3( o0 += input3[0] * c3s; )
    stbIF4( o0 += input4[0] * c4s; )
    stbIF5( o0 += input5[0] * c5s; )
    stbIF6( o0 += input6[0] * c6s; )
    stbIF7( o0 += input7[0] * c7s; )
    output[0] = o0;
    ++output;
    stbIF0( ++input0; ) stbIF1( ++input1; ) stbIF2( ++input2; ) stbIF3( ++input3; ) stbIF4( ++input4; ) stbIF5( ++input5; ) stbIF6( ++input6; ) stbIF7( ++input7; )
  }
}

#undef stbIF0
#undef stbIF1
#undef stbIF2
#undef stbIF3
#undef stbIF4
#undef stbIF5
#undef stbIF6
#undef stbIF7
#undef STB_IMAGE_RESIZE_DO_VERTICALS
#undef STBIR__vertical_channels
#undef STB_IMAGE_RESIZE_DO_HORIZONTALS
#undef STBIR_strs_join24
#undef STBIR_strs_join14
#undef STBIR_chans
#ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#undef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#endif

#else // !STB_IMAGE_RESIZE_DO_VERTICALS

#define STBIR_chans( start, end ) STBIR_strs_join1(start,STBIR__horizontal_channels,end)

#ifndef stbir__2_coeff_only
#define stbir__2_coeff_only()             \
    stbir__1_coeff_only();                \
    stbir__1_coeff_remnant(1);
#endif

#ifndef stbir__2_coeff_remnant
#define stbir__2_coeff_remnant( ofs )     \
    stbir__1_coeff_remnant(ofs);          \
    stbir__1_coeff_remnant((ofs)+1);
#endif

#ifndef stbir__3_coeff_only
#define stbir__3_coeff_only()             \
    stbir__2_coeff_only();                \
    stbir__1_coeff_remnant(2);
#endif

#ifndef stbir__3_coeff_remnant
#define stbir__3_coeff_remnant( ofs )     \
    stbir__2_coeff_remnant(ofs);          \
    stbir__1_coeff_remnant((ofs)+2);
#endif

#ifndef stbir__3_coeff_setup
#define stbir__3_coeff_setup()
#endif

#ifndef stbir__4_coeff_start
#define stbir__4_coeff_start()            \
    stbir__2_coeff_only();                \
    stbir__2_coeff_remnant(2);
#endif

#ifndef stbir__4_coeff_continue_from_4
#define stbir__4_coeff_continue_from_4( ofs )     \
    stbir__2_coeff_remnant(ofs);                  \
    stbir__2_coeff_remnant((ofs)+2);
#endif

#ifndef stbir__store_output_tiny
#define stbir__store_output_tiny stbir__store_output
#endif

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_1_coeff)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__1_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_2_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__2_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_3_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__3_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_4_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_5_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__1_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_6_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__2_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_7_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    stbir__3_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_8_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_9_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__1_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_10_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__2_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_11_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__3_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_12_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__4_coeff_continue_from_4(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod0 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 4 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod1 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 5 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__1_coeff_remnant( 4 );
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod2 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 6 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__2_coeff_remnant( 4 );

    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod3 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 7 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__3_coeff_remnant( 4 );

    stbir__store_output();
  } while ( output < output_end );
}

static stbir__horizontal_gather_channels_func * STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_funcs)[4]=
{
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod0),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod1),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod2),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod3),
};

static stbir__horizontal_gather_channels_func * STBIR_chans(stbir__horizontal_gather_,_channels_funcs)[12]=
{
  STBIR_chans(stbir__horizontal_gather_,_channels_with_1_coeff),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_2_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_3_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_4_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_5_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_6_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_7_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_8_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_9_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_10_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_11_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_12_coeffs),
};

#undef STBIR__horizontal_channels
#undef STB_IMAGE_RESIZE_DO_HORIZONTALS
#undef stbir__1_coeff_only
#undef stbir__1_coeff_remnant
#undef stbir__2_coeff_only
#undef stbir__2_coeff_remnant
#undef stbir__3_coeff_only
#undef stbir__3_coeff_remnant
#undef stbir__3_coeff_setup
#undef stbir__4_coeff_start
#undef stbir__4_coeff_continue_from_4
#undef stbir__store_output
#undef stbir__store_output_tiny
#undef STBIR_chans

#endif  // HORIZONALS

#undef STBIR_strs_join2
#undef STBIR_strs_join1

#endif // STB_IMAGE_RESIZE_DO_HORIZONTALS/VERTICALS/CODERS

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/
