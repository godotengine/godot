#ifndef YUV_RGB_H_
#define YUV_RGB_H_

// Copyright 2016 Adrien Descamps
// Distributed under BSD 3-Clause License

// Provide optimized functions to convert images from 8bits yuv420 to rgb24 format

// There are a few slightly different variations of the YCbCr color space with different parameters that
// change the conversion matrix.
// The three most common YCbCr color space, defined by BT.601, BT.709 and JPEG standard are implemented here.
// See the respective standards for details
// The matrix values used are derived from http://www.equasys.de/colorconversion.html

// YUV420 is stored as three separate channels, with U and V (Cb and Cr) subsampled by a 2 factor
// For conversion from yuv to rgb, no interpolation is done, and the same UV value are used for 4 rgb pixels. This
// is suboptimal for image quality, but by far the fastest method.

// For all methods, width and height should be even, if not, the last row/column of the result image won't be affected.
// For sse methods, if the width if not divisable by 32, the last (width%32) pixels of each line won't be affected.

/*#include <stdint.h>*/

// yuv to rgb, standard c implementation
#include "yuv_rgb_std.h"

// yuv to rgb, sse2 implementation
#include "yuv_rgb_sse.h"

// yuv to rgb, lsx implementation
#include "yuv_rgb_lsx.h"

#endif /* YUV_RGB_H_ */
