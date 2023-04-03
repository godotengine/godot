/**************************************************************************
 *
 * Copyright 2009-2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#include <math.h>
#include <float.h>

#include "util/detect.h"
#include "util/u_memory.h"
#include "util/format/u_format_tests.h"


/*
 * Helper macros to create the packed bytes for longer words.
 */

#define PACKED_1x8(x)          {x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_2x8(x, y)       {x, y, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_3x8(x, y, z)    {x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_4x8(x, y, z, w) {x, y, z, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_8x8(a, b, c, d, e, f, g, h) {a, b, c, d, e, f, g, h, 0, 0, 0, 0, 0, 0, 0, 0}

#if UTIL_ARCH_BIG_ENDIAN
#define PACKED_1x16(x)          {(x) >> 8, (x) & 0xff,          0,        0,          0,        0,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_2x16(x, y)       {(x) >> 8, (x) & 0xff, (y) >> 8, (y) & 0xff,          0,        0,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_3x16(x, y, z)    {(x) >> 8, (x) & 0xff, (y) >> 8, (y) & 0xff, (z) >> 8, (z) & 0xff,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_4x16(x, y, z, w) {(x) >> 8, (x) & 0xff, (y) >> 8, (y) & 0xff, (z) >> 8, (z) & 0xff, (w) >> 8, (w) & 0xff, 0, 0, 0, 0, 0, 0, 0, 0}
#else
#define PACKED_1x16(x)          {(x) & 0xff, (x) >> 8,          0,        0,          0,        0,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_2x16(x, y)       {(x) & 0xff, (x) >> 8, (y) & 0xff, (y) >> 8,          0,        0,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_3x16(x, y, z)    {(x) & 0xff, (x) >> 8, (y) & 0xff, (y) >> 8, (z) & 0xff, (z) >> 8,          0,        0, 0, 0, 0, 0, 0, 0, 0, 0}
#define PACKED_4x16(x, y, z, w) {(x) & 0xff, (x) >> 8, (y) & 0xff, (y) >> 8, (z) & 0xff, (z) >> 8, (w) & 0xff, (w) >> 8, 0, 0, 0, 0, 0, 0, 0, 0}
#endif

#if UTIL_ARCH_BIG_ENDIAN
#define PACKED_1x32(x)          {(x) >> 24, ((x) >> 16) & 0xff, ((x) >> 8) & 0xff, (x) & 0xff,          0,                 0,                  0,         0,          0,                 0,                  0,         0,          0,                 0,                  0,         0}
#define PACKED_2x32(x, y)       {(x) >> 24, ((x) >> 16) & 0xff, ((x) >> 8) & 0xff, (x) & 0xff, (y) >> 24, ((y) >> 16) & 0xff, ((y) >> 8) & 0xff, (y) & 0xff,          0,                 0,                  0,         0,          0,                 0,                  0,         0}
#define PACKED_3x32(x, y, z)    {(x) >> 24, ((x) >> 16) & 0xff, ((x) >> 8) & 0xff, (x) & 0xff, (y) >> 24, ((y) >> 16) & 0xff, ((y) >> 8) & 0xff, (y) & 0xff, (z) >> 24, ((z) >> 16) & 0xff, ((z) >> 8) & 0xff, (z) & 0xff,          0,                 0,                  0,         0}
#define PACKED_4x32(x, y, z, w) {(x) >> 24, ((x) >> 16) & 0xff, ((x) >> 8) & 0xff, (x) & 0xff, (y) >> 24, ((y) >> 16) & 0xff, ((y) >> 8) & 0xff, (y) & 0xff, (z) >> 24, ((z) >> 16) & 0xff, ((z) >> 8) & 0xff, (z) & 0xff, (w) >> 24, ((w) >> 16) & 0xff, ((w) >> 8) & 0xff, (w) & 0xff}
#else
#define PACKED_1x32(x)          {(x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff, (x) >> 24,          0,                 0,                  0,         0,          0,                 0,                  0,         0,          0,                 0,                  0,         0}
#define PACKED_2x32(x, y)       {(x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff, (x) >> 24, (y) & 0xff, ((y) >> 8) & 0xff, ((y) >> 16) & 0xff, (y) >> 24,          0,                 0,                  0,         0,          0,                 0,                  0,         0}
#define PACKED_3x32(x, y, z)    {(x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff, (x) >> 24, (y) & 0xff, ((y) >> 8) & 0xff, ((y) >> 16) & 0xff, (y) >> 24, (z) & 0xff, ((z) >> 8) & 0xff, ((z) >> 16) & 0xff, (z) >> 24,          0,                 0,                  0,         0}
#define PACKED_4x32(x, y, z, w) {(x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff, (x) >> 24, (y) & 0xff, ((y) >> 8) & 0xff, ((y) >> 16) & 0xff, (y) >> 24, (z) & 0xff, ((z) >> 8) & 0xff, ((z) >> 16) & 0xff, (z) >> 24, (w) & 0xff, ((w) >> 8) & 0xff, ((w) >> 16) & 0xff, (w) >> 24}
#endif

#define UNPACKED_1x1(r, g, b, a) \
      {{{r, g, b, a}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}

#define UNPACKED_2x1(r0, g0, b0, a0, r1, g1, b1, a1) \
      {{{r0, g0, b0, a0}, {r1, g1, b1, a1}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{ 0,  0,  0,  0}, { 0,  0,  0,  0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{ 0,  0,  0,  0}, { 0,  0,  0,  0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{ 0,  0,  0,  0}, { 0,  0,  0,  0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}


/**
 * Test cases.
 *
 * These were manually entered. We could generate these
 *
 * To keep this to a we cover only the corner cases, which should produce
 * good enough coverage since that pixel format transformations are afine for
 * non SRGB formats.
 */
const struct util_format_test_case
util_format_test_cases[] =
{

   /*
    * 32-bit rendertarget formats
    */

   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_UNORM, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8X8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000003ff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000ffc00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x3ff00000), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xc0000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R10G10B10X2_UNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_UNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x000003ff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_UNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x000ffc00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_UNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x3ff00000), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_UNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x3fffffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000003ff), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000ffc00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x3ff00000), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xc0000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B10G10R10A2_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   /*
    * 16-bit rendertarget formats
    */

   {PIPE_FORMAT_B5G5R5X1_UNORM, PACKED_1x16(0x7fff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G5R5X1_UNORM, PACKED_1x16(0x7fff), PACKED_1x16(0x001f), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B5G5R5X1_UNORM, PACKED_1x16(0x7fff), PACKED_1x16(0x03e0), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G5R5X1_UNORM, PACKED_1x16(0x7fff), PACKED_1x16(0x7c00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G5R5X1_UNORM, PACKED_1x16(0x7fff), PACKED_1x16(0x7fff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x001f), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x03e0), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x7c00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x8000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G5R5A1_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_X1B5G5R5_UNORM, PACKED_1x16(0xfffe), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X1B5G5R5_UNORM, PACKED_1x16(0xfffe), PACKED_1x16(0x003e), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_X1B5G5R5_UNORM, PACKED_1x16(0xfffe), PACKED_1x16(0x07c0), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_X1B5G5R5_UNORM, PACKED_1x16(0xfffe), PACKED_1x16(0xf800), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X1B5G5R5_UNORM, PACKED_1x16(0xfffe), PACKED_1x16(0xfffe), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x003e), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x07c0), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xf800), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0001), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_A1B5G5R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B4G4R4X4_UNORM, PACKED_1x16(0x0fff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B4G4R4X4_UNORM, PACKED_1x16(0x0fff), PACKED_1x16(0x000f), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B4G4R4X4_UNORM, PACKED_1x16(0x0fff), PACKED_1x16(0x00f0), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B4G4R4X4_UNORM, PACKED_1x16(0x0fff), PACKED_1x16(0x0f00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B4G4R4X4_UNORM, PACKED_1x16(0x0fff), PACKED_1x16(0x0fff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x000f), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x00f0), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0f00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xf000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B4G4R4A4_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B5G6R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G6R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x001f), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B5G6R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x07e0), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G6R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xf800), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B5G6R5_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   /*
    * Luminance/intensity/alpha formats
    */

   {PIPE_FORMAT_L8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_I8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_I8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_L4A4_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_L4A4_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x0f), UNPACKED_1x1(1.0, 1.0, 1.0, 0.0)},
   {PIPE_FORMAT_L4A4_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xf0), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L4A4_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_L8A8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_L8A8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0x00), UNPACKED_1x1(1.0, 1.0, 1.0, 0.0)},
   {PIPE_FORMAT_L8A8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L8A8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_L16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   /*
    * SRGB formats
    */

   {PIPE_FORMAT_L8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0xbc), UNPACKED_1x1(0.502886458033, 0.502886458033, 0.502886458033, 1.0)},
   {PIPE_FORMAT_L8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0xbc), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8_SRGB, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R8G8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xbc, 0xbc), UNPACKED_1x1(0.502886458033, 0.502886458033, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 0.0, 1.0)},

   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xbc, 0x00), UNPACKED_1x1(0.502886458033, 0.502886458033, 0.502886458033, 0.0)},
   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0x00), UNPACKED_1x1(1.0, 1.0, 1.0, 0.0)},
   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0xcc), UNPACKED_1x1(0.0, 0.0, 0.0, 0.8)},
   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_L8A8_SRGB, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xbc, 0x00, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0xbc), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SRGB, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xbc, 0x00, 0x00, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xcc), UNPACKED_1x1(0.0, 0.0, 0.0, 0.8)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xbc, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xcc), UNPACKED_1x1(0.0, 0.0, 0.0, 0.8)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8A8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xbc, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_B8G8R8X8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0x00), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xcc, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.8)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xbc), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_A8R8G8B8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xbc), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_X8R8G8B8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xcc, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.8)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xbc), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_A8B8G8R8_SRGB, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xbc, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.502886458033, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xbc, 0x00), UNPACKED_1x1(0.0, 0.502886458033, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xbc), UNPACKED_1x1(0.502886458033, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_X8B8G8R8_SRGB, PACKED_4x8(0x00, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   /*
    * Mixed-signed formats
    */

   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00000000), UNPACKED_1x1( 0.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x0000007f), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00000081), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00007f00), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00008100), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8SG8SB8UX8U_NORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00ff0000), UNPACKED_1x1( 0.0,  0.0, 1.0, 1.0)},

   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000001ff), UNPACKED_1x1( 1.0,  0.0,  0.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000201), UNPACKED_1x1(-1.0,  0.0,  0.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x0007fc00), UNPACKED_1x1( 0.0,  1.0,  0.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00080400), UNPACKED_1x1( 0.0, -1.0,  0.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x1ff00000), UNPACKED_1x1( 0.0,  0.0,  1.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x20100000), UNPACKED_1x1( 0.0,  0.0, -1.0, 0.0)},
   {PIPE_FORMAT_R10SG10SB10SA2U_NORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xc0000000), UNPACKED_1x1( 0.0,  0.0,  0.0, 1.0)},

   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1( 0.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0x000f), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0x0011), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0x01e0), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0x0220), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R5SG5SB6U_NORM, PACKED_1x16(0xffff), PACKED_1x16(0xfc00), UNPACKED_1x1( 0.0,  0.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8G8Bx_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1( 0.0,  0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R8G8Bx_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x7f, 0x00), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8Bx_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x81, 0x00), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8Bx_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x7f), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8Bx_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x81), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},

   /*
    * Depth-stencil formats
    */

   {PIPE_FORMAT_S8_UINT, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_S8_UINT, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(0.0, 255.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z32_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z32_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z32_FLOAT, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z32_FLOAT, PACKED_1x32(0xffffffff), PACKED_1x32(0x3f800000), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z24_UNORM_S8_UINT, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z24_UNORM_S8_UINT, PACKED_1x32(0xffffffff), PACKED_1x32(0x00ffffff), UNPACKED_1x1(1.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z24_UNORM_S8_UINT, PACKED_1x32(0xffffffff), PACKED_1x32(0xff000000), UNPACKED_1x1(0.0, 255.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z24_UNORM_S8_UINT, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 255.0, 0.0, 0.0)},

   {PIPE_FORMAT_S8_UINT_Z24_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_S8_UINT_Z24_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffff00), UNPACKED_1x1(1.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_S8_UINT_Z24_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x000000ff), UNPACKED_1x1(0.0, 255.0, 0.0, 0.0)},
   {PIPE_FORMAT_S8_UINT_Z24_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 255.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z24X8_UNORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z24X8_UNORM, PACKED_1x32(0x00ffffff), PACKED_1x32(0x00ffffff), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},

   {PIPE_FORMAT_X8Z24_UNORM, PACKED_1x32(0xffffff00), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_X8Z24_UNORM, PACKED_1x32(0xffffff00), PACKED_1x32(0xffffff00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},

   {PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, PACKED_2x32(0xffffffff, 0x000000ff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1( 0.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, PACKED_2x32(0xffffffff, 0x000000ff), PACKED_2x32(0x3f800000, 0x00000000), UNPACKED_1x1( 1.0,   0.0, 0.0, 0.0)},
   {PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, PACKED_2x32(0xffffffff, 0x000000ff), PACKED_2x32(0x00000000, 0x000000ff), UNPACKED_1x1( 0.0, 255.0, 0.0, 0.0)},

   /*
    * YUV formats
    */

   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_2x1(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_2x1(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_2x1(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_B8G8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_2x1(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_2x1(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_2x1(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_2x1(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_G8R8_G8B8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_2x1(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)},

   /*
    * TODO: Exercise the UV channels as well.
    */
   {PIPE_FORMAT_UYVY, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x80, 0x10, 0x80, 0x10), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_UYVY, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x80, 0xeb, 0x80, 0x10), UNPACKED_2x1(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_UYVY, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x80, 0x10, 0x80, 0xeb), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_YUYV, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x10, 0x80, 0x10, 0x80), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_YUYV, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xeb, 0x80, 0x10, 0x80), UNPACKED_2x1(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_YUYV, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x10, 0x80, 0xeb, 0x80), UNPACKED_2x1(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)},

   /*
    * Compressed formats
    */

   {
      PIPE_FORMAT_DXT1_RGB,
      PACKED_8x8(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff),
      PACKED_8x8(0xf2, 0xd7, 0xb0, 0x20, 0xae, 0x2c, 0x6f, 0x97),
      {
         {
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0},
            {0x5d/255.0, 0x62/255.0, 0x89/255.0, 0xff/255.0},
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0},
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0}
         },
         {
            {0xd6/255.0, 0xff/255.0, 0x94/255.0, 0xff/255.0},
            {0x5d/255.0, 0x62/255.0, 0x89/255.0, 0xff/255.0},
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0},
            {0xd6/255.0, 0xff/255.0, 0x94/255.0, 0xff/255.0}
         },
         {
            {0x5d/255.0, 0x62/255.0, 0x89/255.0, 0xff/255.0},
            {0x5d/255.0, 0x62/255.0, 0x89/255.0, 0xff/255.0},
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0},
            {0x21/255.0, 0x14/255.0, 0x84/255.0, 0xff/255.0}
         },
         {
            {0x5d/255.0, 0x62/255.0, 0x89/255.0, 0xff/255.0},
            {0x21/255.0, 0x14/255.0, 0x84/255.0, 0xff/255.0},
            {0x21/255.0, 0x14/255.0, 0x84/255.0, 0xff/255.0},
            {0x99/255.0, 0xb0/255.0, 0x8e/255.0, 0xff/255.0}
         }
      }
   },
   {
      PIPE_FORMAT_DXT1_RGBA,
      PACKED_8x8(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff),
      PACKED_8x8(0xff, 0x2f, 0xa4, 0x72, 0xeb, 0xb2, 0xbd, 0xbe),
      {
         {
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0},
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0}
         },
         {
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0},
            {0x29/255.0, 0xff/255.0, 0xff/255.0, 0xff/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0}
         },
         {
            {0x73/255.0, 0x55/255.0, 0x21/255.0, 0xff/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0}
         },
         {
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x00/255.0, 0x00/255.0, 0x00/255.0, 0x00/255.0},
            {0x4e/255.0, 0xaa/255.0, 0x90/255.0, 0xff/255.0}
         }
      }
   },
   {
      PIPE_FORMAT_DXT3_RGBA,
      {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
      {0xe7, 0x4a, 0x8f, 0x96, 0x5b, 0xc1, 0x1c, 0x84, 0xf6, 0x8f, 0xab, 0x32, 0x2a, 0x9a, 0x95, 0x5a},
      {
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x77/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xee/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xaa/255.0},
            {0x8c/255.0, 0xff/255.0, 0xb5/255.0, 0x44/255.0}
         },
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xff/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x88/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x66/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x99/255.0}
         },
         {
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0xbb/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x55/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x11/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xcc/255.0}
         },
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xcc/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x11/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x44/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x88/255.0}
         }
      }
   },
   {
      PIPE_FORMAT_DXT5_RGBA,
      {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
      {0xf8, 0x11, 0xc5, 0x0c, 0x9a, 0x73, 0xb4, 0x9c, 0xf6, 0x8f, 0xab, 0x32, 0x2a, 0x9a, 0x95, 0x5a},
      {
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x74/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xf8/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xb6/255.0},
            {0x8c/255.0, 0xff/255.0, 0xb5/255.0, 0x53/255.0}
         },
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xf8/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x95/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x53/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x95/255.0}
         },
         {
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0xb6/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x53/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x11/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xd7/255.0}
         },
         {
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0xb6/255.0},
            {0x6d/255.0, 0xc6/255.0, 0x96/255.0, 0x11/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x32/255.0},
            {0x31/255.0, 0x55/255.0, 0x5a/255.0, 0x95/255.0}
         }
      }
   },


   /*
    * Standard 8-bit integer formats
    */

   {PIPE_FORMAT_R8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R8G8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0xff), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8_UNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R8G8B8_UNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_UNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_UNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_UNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_UNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8A8_UNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R8_USCALED, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(  0.0,   0.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8_USCALED, PACKED_1x8(0xff), PACKED_1x8(0xff), UNPACKED_1x1(255.0,   0.0,   0.0, 1.0)},

   {PIPE_FORMAT_R8G8_USCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(  0.0,   0.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8_USCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0x00), UNPACKED_1x1(255.0,   0.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8_USCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0xff), UNPACKED_1x1(  0.0, 255.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8_USCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0xff, 0xff), UNPACKED_1x1(255.0, 255.0,   0.0, 1.0)},

   {PIPE_FORMAT_R8G8B8_USCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x00), UNPACKED_1x1(  0.0,   0.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_USCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0x00, 0x00), UNPACKED_1x1(255.0,   0.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_USCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0xff, 0x00), UNPACKED_1x1(  0.0, 255.0,   0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_USCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0xff), UNPACKED_1x1(  0.0,   0.0, 255.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_USCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0xff, 0xff, 0xff), UNPACKED_1x1(255.0, 255.0, 255.0, 1.0)},

   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(  0.0,   0.0,   0.0,   0.0)},
   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0x00, 0x00, 0x00), UNPACKED_1x1(255.0,   0.0,   0.0,   0.0)},
   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0xff, 0x00, 0x00), UNPACKED_1x1(  0.0, 255.0,   0.0,   0.0)},
   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0xff, 0x00), UNPACKED_1x1(  0.0,   0.0, 255.0,   0.0)},
   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0xff), UNPACKED_1x1(  0.0,   0.0,   0.0, 255.0)},
   {PIPE_FORMAT_R8G8B8A8_USCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0xff, 0xff, 0xff, 0xff), UNPACKED_1x1(255.0, 255.0, 255.0, 255.0)},

   {PIPE_FORMAT_R8_SNORM, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8_SNORM, PACKED_1x8(0xff), PACKED_1x8(0x7f), UNPACKED_1x1( 1.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8_SNORM, PACKED_1x8(0xff), PACKED_1x8(0x81), UNPACKED_1x1(-1.0,  0.0,  0.0,  1.0)},

   {PIPE_FORMAT_R8G8_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x7f, 0x00), UNPACKED_1x1( 1.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x81, 0x00), UNPACKED_1x1(-1.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x7f), UNPACKED_1x1( 0.0,  1.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8_SNORM, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x81), UNPACKED_1x1( 0.0, -1.0,  0.0,  1.0)},

   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x00), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x7f, 0x00, 0x00), UNPACKED_1x1( 1.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x81, 0x00, 0x00), UNPACKED_1x1(-1.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x7f, 0x00), UNPACKED_1x1( 0.0,  1.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x81, 0x00), UNPACKED_1x1( 0.0, -1.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x7f), UNPACKED_1x1( 0.0,  0.0,  1.0,  1.0)},
   {PIPE_FORMAT_R8G8B8_SNORM, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x81), UNPACKED_1x1( 0.0,  0.0, -1.0,  1.0)},

   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1( 0.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x7f, 0x00, 0x00, 0x00), UNPACKED_1x1( 1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x81, 0x00, 0x00, 0x00), UNPACKED_1x1(-1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x7f, 0x00, 0x00), UNPACKED_1x1( 0.0,  1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x81, 0x00, 0x00), UNPACKED_1x1( 0.0, -1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x7f, 0x00), UNPACKED_1x1( 0.0,  0.0,  1.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x81, 0x00), UNPACKED_1x1( 0.0,  0.0, -1.0,  0.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x7f), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R8G8B8A8_SNORM, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x81), UNPACKED_1x1( 0.0,  0.0,  0.0, -1.0)},

   {PIPE_FORMAT_R8_SSCALED, PACKED_1x8(0xff), PACKED_1x8(0x00), UNPACKED_1x1(   0.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8_SSCALED, PACKED_1x8(0xff), PACKED_1x8(0x7f), UNPACKED_1x1( 127.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8_SSCALED, PACKED_1x8(0xff), PACKED_1x8(0x80), UNPACKED_1x1(-128.0,    0.0,    0.0, 1.0)},

   {PIPE_FORMAT_R8G8_SSCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x00), UNPACKED_1x1(   0.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SSCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x7f, 0x00), UNPACKED_1x1( 127.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SSCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x80, 0x00), UNPACKED_1x1(-128.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SSCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x7f), UNPACKED_1x1(   0.0,  127.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8_SSCALED, PACKED_2x8(0xff, 0xff), PACKED_2x8(0x00, 0x80), UNPACKED_1x1(   0.0, -128.0,    0.0, 1.0)},

   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x00), UNPACKED_1x1(   0.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x7f, 0x00, 0x00), UNPACKED_1x1( 127.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x80, 0x00, 0x00), UNPACKED_1x1(-128.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x7f, 0x00), UNPACKED_1x1(   0.0,  127.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x80, 0x00), UNPACKED_1x1(   0.0, -128.0,    0.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x7f), UNPACKED_1x1(   0.0,    0.0,  127.0, 1.0)},
   {PIPE_FORMAT_R8G8B8_SSCALED, PACKED_3x8(0xff, 0xff, 0xff), PACKED_3x8(0x00, 0x00, 0x80), UNPACKED_1x1(   0.0,    0.0, -128.0, 1.0)},

   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x00), UNPACKED_1x1(   0.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x7f, 0x00, 0x00, 0x00), UNPACKED_1x1( 127.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x80, 0x00, 0x00, 0x00), UNPACKED_1x1(-128.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x7f, 0x00, 0x00), UNPACKED_1x1(   0.0,  127.0,    0.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x80, 0x00, 0x00), UNPACKED_1x1(   0.0, -128.0,    0.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x7f, 0x00), UNPACKED_1x1(   0.0,    0.0,  127.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x80, 0x00), UNPACKED_1x1(   0.0,    0.0, -128.0,    0.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x7f), UNPACKED_1x1(   0.0,    0.0,    0.0,  127.0)},
   {PIPE_FORMAT_R8G8B8A8_SSCALED, PACKED_4x8(0xff, 0xff, 0xff, 0xff), PACKED_4x8(0x00, 0x00, 0x00, 0x80), UNPACKED_1x1(   0.0,    0.0,    0.0, -128.0)},

   /*
    * Standard 16-bit integer formats
    */

   {PIPE_FORMAT_R16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_UNORM, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R16G16_UNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_UNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0xffff, 0x0000), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_UNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0xffff), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_UNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0xffff, 0xffff), UNPACKED_1x1(1.0, 1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R16G16B16_UNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_UNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0xffff, 0x0000, 0x0000), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_UNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0xffff, 0x0000), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_UNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0xffff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_UNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0xffff, 0xffff, 0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0xffff, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0xffff, 0x0000, 0x0000), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0xffff, 0x0000), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0xffff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16A16_UNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R16_USCALED, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(    0.0,     0.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16_USCALED, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(65535.0,     0.0,     0.0,   1.0)},

   {PIPE_FORMAT_R16G16_USCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x0000), UNPACKED_1x1(    0.0,     0.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16_USCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0xffff, 0x0000), UNPACKED_1x1(65535.0,     0.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16_USCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0xffff), UNPACKED_1x1(    0.0, 65535.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16_USCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0xffff, 0xffff), UNPACKED_1x1(65535.0, 65535.0,     0.0,   1.0)},

   {PIPE_FORMAT_R16G16B16_USCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x0000), UNPACKED_1x1(    0.0,     0.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_USCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0xffff, 0x0000, 0x0000), UNPACKED_1x1(65535.0,     0.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_USCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0xffff, 0x0000), UNPACKED_1x1(    0.0, 65535.0,     0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_USCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0xffff), UNPACKED_1x1(    0.0,     0.0, 65535.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_USCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0xffff, 0xffff, 0xffff), UNPACKED_1x1(65535.0, 65535.0, 65535.0,   1.0)},

   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(    0.0,     0.0,     0.0,     0.0)},
   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0xffff, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(65535.0,     0.0,     0.0,     0.0)},
   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0xffff, 0x0000, 0x0000), UNPACKED_1x1(    0.0, 65535.0,     0.0,     0.0)},
   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0xffff, 0x0000), UNPACKED_1x1(    0.0,     0.0, 65535.0,     0.0)},
   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0xffff), UNPACKED_1x1(    0.0,     0.0,     0.0, 65535.0)},
   {PIPE_FORMAT_R16G16B16A16_USCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), UNPACKED_1x1(65535.0, 65535.0, 65535.0, 65535.0)},

   {PIPE_FORMAT_R16_SNORM, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16_SNORM, PACKED_1x16(0xffff), PACKED_1x16(0x7fff), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16_SNORM, PACKED_1x16(0xffff), PACKED_1x16(0x8001), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},

   {PIPE_FORMAT_R16G16_SNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x0000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16_SNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x7fff, 0x0000), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16_SNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x8001, 0x0000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16_SNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x7fff), UNPACKED_1x1(   0.0,    1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16_SNORM, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x8001), UNPACKED_1x1(   0.0,   -1.0,    0.0,    1.0)},

   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x0000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x7fff, 0x0000, 0x0000), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x8001, 0x0000, 0x0000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x7fff, 0x0000), UNPACKED_1x1(   0.0,    1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x8001, 0x0000), UNPACKED_1x1(   0.0,   -1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x7fff), UNPACKED_1x1(   0.0,    0.0,    1.0,    1.0)},
   {PIPE_FORMAT_R16G16B16_SNORM, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x8001), UNPACKED_1x1(   0.0,    0.0,   -1.0,    1.0)},

   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(   0.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x7fff, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(   1.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x8001, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x7fff, 0x0000, 0x0000), UNPACKED_1x1(   0.0,    1.0,    0.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x8001, 0x0000, 0x0000), UNPACKED_1x1(   0.0,   -1.0,    0.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x7fff, 0x0000), UNPACKED_1x1(   0.0,    0.0,    1.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x8001, 0x0000), UNPACKED_1x1(   0.0,    0.0,   -1.0,    0.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x7fff), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R16G16B16A16_SNORM, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x8001), UNPACKED_1x1(   0.0,    0.0,    0.0,   -1.0)},

   {PIPE_FORMAT_R16_SSCALED, PACKED_1x16(0xffff), PACKED_1x16(0x0000), UNPACKED_1x1(     0.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16_SSCALED, PACKED_1x16(0xffff), PACKED_1x16(0x7fff), UNPACKED_1x1( 32767.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16_SSCALED, PACKED_1x16(0xffff), PACKED_1x16(0x8000), UNPACKED_1x1(-32768.0,      0.0,      0.0,   1.0)},

   {PIPE_FORMAT_R16G16_SSCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x0000), UNPACKED_1x1(     0.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16_SSCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x7fff, 0x0000), UNPACKED_1x1( 32767.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16_SSCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x8000, 0x0000), UNPACKED_1x1(-32768.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16_SSCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x7fff), UNPACKED_1x1(     0.0,  32767.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16_SSCALED, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x8000), UNPACKED_1x1(     0.0, -32768.0,      0.0,   1.0)},

   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x0000), UNPACKED_1x1(     0.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x7fff, 0x0000, 0x0000), UNPACKED_1x1( 32767.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x8000, 0x0000, 0x0000), UNPACKED_1x1(-32768.0,      0.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x7fff, 0x0000), UNPACKED_1x1(     0.0,  32767.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x8000, 0x0000), UNPACKED_1x1(     0.0, -32768.0,      0.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x7fff), UNPACKED_1x1(     0.0,      0.0,  32767.0,   1.0)},
   {PIPE_FORMAT_R16G16B16_SSCALED, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x8000), UNPACKED_1x1(     0.0,      0.0, -32768.0,   1.0)},

   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(     0.0,      0.0,      0.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x7fff, 0x0000, 0x0000, 0x0000), UNPACKED_1x1( 32767.0,      0.0,      0.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x8000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(-32768.0,      0.0,      0.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x7fff, 0x0000, 0x0000), UNPACKED_1x1(     0.0,  32767.0,      0.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x8000, 0x0000, 0x0000), UNPACKED_1x1(     0.0, -32768.0,      0.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x7fff, 0x0000), UNPACKED_1x1(     0.0,      0.0,  32767.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x8000, 0x0000), UNPACKED_1x1(     0.0,      0.0, -32768.0,      0.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x7fff), UNPACKED_1x1(     0.0,      0.0,      0.0,  32767.0)},
   {PIPE_FORMAT_R16G16B16A16_SSCALED, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x8000), UNPACKED_1x1(     0.0,      0.0,      0.0, -32768.0)},

   /*
    * Standard 32-bit integer formats
    *
    * NOTE: We can't accurately represent integers larger than +/-0x1000000
    * with single precision floats, so that's as far as we test.
    */

   {PIPE_FORMAT_R32_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32_UNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0xffffffff), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32_UNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_UNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0xffffffff, 0x00000000), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_UNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0xffffffff), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_UNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0xffffffff, 0xffffffff), UNPACKED_1x1(1.0, 1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32B32_UNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_UNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0xffffffff, 0x00000000, 0x00000000), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_UNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0xffffffff, 0x00000000), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_UNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0xffffffff), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_UNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(0.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0xffffffff, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(1.0, 0.0, 0.0, 0.0)},
   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0xffffffff, 0x00000000, 0x00000000), UNPACKED_1x1(0.0, 1.0, 0.0, 0.0)},
   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0xffffffff, 0x00000000), UNPACKED_1x1(0.0, 0.0, 1.0, 0.0)},
   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0xffffffff), UNPACKED_1x1(0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32A32_UNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), UNPACKED_1x1(1.0, 1.0, 1.0, 1.0)},

   {PIPE_FORMAT_R32_USCALED, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(       0.0,        0.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32_USCALED, PACKED_1x32(0xffffffff), PACKED_1x32(0x01000000), UNPACKED_1x1(16777216.0,        0.0,        0.0,   1.0)},

   {PIPE_FORMAT_R32G32_USCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1(       0.0,        0.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32_USCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x01000000, 0x00000000), UNPACKED_1x1(16777216.0,        0.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32_USCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x01000000), UNPACKED_1x1(       0.0, 16777216.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32_USCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x01000000, 0x01000000), UNPACKED_1x1(16777216.0, 16777216.0,        0.0,   1.0)},

   {PIPE_FORMAT_R32G32B32_USCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(       0.0,        0.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_USCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x01000000, 0x00000000, 0x00000000), UNPACKED_1x1(16777216.0,        0.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_USCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x01000000, 0x00000000), UNPACKED_1x1(       0.0, 16777216.0,        0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_USCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x01000000), UNPACKED_1x1(       0.0,        0.0, 16777216.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_USCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x01000000, 0x01000000, 0x01000000), UNPACKED_1x1(16777216.0, 16777216.0, 16777216.0,   1.0)},

   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(       0.0,        0.0,        0.0,        0.0)},
   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x01000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(16777216.0,        0.0,        0.0,        0.0)},
   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x01000000, 0x00000000, 0x00000000), UNPACKED_1x1(       0.0, 16777216.0,        0.0,        0.0)},
   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x01000000, 0x00000000), UNPACKED_1x1(       0.0,        0.0, 16777216.0,        0.0)},
   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x01000000), UNPACKED_1x1(       0.0,        0.0,        0.0, 16777216.0)},
   {PIPE_FORMAT_R32G32B32A32_USCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x01000000, 0x01000000, 0x01000000, 0x01000000), UNPACKED_1x1(16777216.0, 16777216.0, 16777216.0, 16777216.0)},

   {PIPE_FORMAT_R32_SNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32_SNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x7fffffff), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32_SNORM, PACKED_1x32(0xffffffff), PACKED_1x32(0x80000001), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},

   {PIPE_FORMAT_R32G32_SNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32_SNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x7fffffff, 0x00000000), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32_SNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x80000001, 0x00000000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32_SNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x7fffffff), UNPACKED_1x1(   0.0,    1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32_SNORM, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x80000001), UNPACKED_1x1(   0.0,   -1.0,    0.0,    1.0)},

   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x7fffffff, 0x00000000, 0x00000000), UNPACKED_1x1(   1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x80000001, 0x00000000, 0x00000000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x7fffffff, 0x00000000), UNPACKED_1x1(   0.0,    1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x80000001, 0x00000000), UNPACKED_1x1(   0.0,   -1.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x7fffffff), UNPACKED_1x1(   0.0,    0.0,    1.0,    1.0)},
   {PIPE_FORMAT_R32G32B32_SNORM, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x80000001), UNPACKED_1x1(   0.0,    0.0,   -1.0,    1.0)},

   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(   0.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x7fffffff, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(   1.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x80000001, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(  -1.0,    0.0,    0.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x7fffffff, 0x00000000, 0x00000000), UNPACKED_1x1(   0.0,    1.0,    0.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x80000001, 0x00000000, 0x00000000), UNPACKED_1x1(   0.0,   -1.0,    0.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x7fffffff, 0x00000000), UNPACKED_1x1(   0.0,    0.0,    1.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x80000001, 0x00000000), UNPACKED_1x1(   0.0,    0.0,   -1.0,    0.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x7fffffff), UNPACKED_1x1(   0.0,    0.0,    0.0,    1.0)},
   {PIPE_FORMAT_R32G32B32A32_SNORM, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x80000001), UNPACKED_1x1(   0.0,    0.0,    0.0,   -1.0)},

   {PIPE_FORMAT_R32_SSCALED, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(        0.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32_SSCALED, PACKED_1x32(0xffffffff), PACKED_1x32(0x01000000), UNPACKED_1x1( 16777216.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32_SSCALED, PACKED_1x32(0xffffffff), PACKED_1x32(0xff000000), UNPACKED_1x1(-16777216.0,         0.0,         0.0,   1.0)},

   {PIPE_FORMAT_R32G32_SSCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1(        0.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32_SSCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x01000000, 0x00000000), UNPACKED_1x1( 16777216.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32_SSCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0xff000000, 0x00000000), UNPACKED_1x1(-16777216.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32_SSCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x01000000), UNPACKED_1x1(        0.0,  16777216.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32_SSCALED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0xff000000), UNPACKED_1x1(        0.0, -16777216.0,         0.0,   1.0)},

   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(        0.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x01000000, 0x00000000, 0x00000000), UNPACKED_1x1( 16777216.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0xff000000, 0x00000000, 0x00000000), UNPACKED_1x1(-16777216.0,         0.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x01000000, 0x00000000), UNPACKED_1x1(        0.0,  16777216.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0xff000000, 0x00000000), UNPACKED_1x1(        0.0, -16777216.0,         0.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x01000000), UNPACKED_1x1(        0.0,         0.0,  16777216.0,   1.0)},
   {PIPE_FORMAT_R32G32B32_SSCALED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0xff000000), UNPACKED_1x1(        0.0,         0.0, -16777216.0,   1.0)},

   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(        0.0,         0.0,         0.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x01000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 16777216.0,         0.0,         0.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0xff000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(-16777216.0,         0.0,         0.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x01000000, 0x00000000, 0x00000000), UNPACKED_1x1(        0.0,  16777216.0,         0.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0xff000000, 0x00000000, 0x00000000), UNPACKED_1x1(        0.0, -16777216.0,         0.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x01000000, 0x00000000), UNPACKED_1x1(        0.0,         0.0,  16777216.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0xff000000, 0x00000000), UNPACKED_1x1(        0.0,         0.0, -16777216.0,         0.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x01000000), UNPACKED_1x1(        0.0,         0.0,         0.0,  16777216.0)},
   {PIPE_FORMAT_R32G32B32A32_SSCALED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0xff000000), UNPACKED_1x1(        0.0,         0.0,         0.0, -16777216.0)},

   /*
    * Standard 32-bit float formats
    */

   {PIPE_FORMAT_R32_FLOAT, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(  0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32_FLOAT, PACKED_1x32(0xffffffff), PACKED_1x32(0x3f800000), UNPACKED_1x1(  1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32_FLOAT, PACKED_1x32(0xffffffff), PACKED_1x32(0xbf800000), UNPACKED_1x1( -1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x3f800000, 0x00000000), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0xbf800000, 0x00000000), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x3f800000), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0xbf800000), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FLOAT, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x3f800000, 0x3f800000), UNPACKED_1x1( 1.0,  1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x3f800000, 0x00000000, 0x00000000), UNPACKED_1x1( 1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0xbf800000, 0x00000000, 0x00000000), UNPACKED_1x1(-1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x3f800000, 0x00000000), UNPACKED_1x1( 0.0,  1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0xbf800000, 0x00000000), UNPACKED_1x1( 0.0, -1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x3f800000), UNPACKED_1x1( 0.0,  0.0,  1.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0xbf800000), UNPACKED_1x1( 0.0,  0.0, -1.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FLOAT, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x3f800000, 0x3f800000, 0x3f800000), UNPACKED_1x1( 1.0,  1.0,  1.0, 1.0)},

   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x3f800000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0xbf800000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(-1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x3f800000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0xbf800000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0, -1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x3f800000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  1.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0xbf800000, 0x00000000), UNPACKED_1x1( 0.0,  0.0, -1.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x3f800000), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0xbf800000), UNPACKED_1x1( 0.0,  0.0,  0.0, -1.0)},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000), UNPACKED_1x1( 1.0,  1.0,  1.0,  1.0)},

   /*
    * Half float formats
    */

   /* Minimum positive normal */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x0400), UNPACKED_1x1( 6.10352E-5, 0.0, 0.0, 1.0)},

   /* XXX: Now that we disable denormals this test cases fails, except on
    * IvyBridge processors which have intrinsics dedicated to half-float
    * packing/unpacking. */
#if 0
   /* Max denormal */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x03FF), UNPACKED_1x1( 6.09756E-5, 0.0, 0.0, 1.0)},
#endif

   /* This fails with _mesa_float_to_float16_rtz, but passes with _mesa_float_to_float16_rtne. */
#if 0
   /* Minimum positive denormal */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x0001), UNPACKED_1x1( 5.96046E-8, 0.0, 0.0, 1.0)},
#endif

   /* Min representable value */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0xfbff), UNPACKED_1x1(   -65504.0, 0.0, 0.0, 1.0)},

   /* Max representable value */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x7bff), UNPACKED_1x1(    65504.0, 0.0, 0.0, 1.0)},

#if !DETECT_CC_MSVC

   /* NaNs */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x7c01), UNPACKED_1x1(        NAN, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0xfc01), UNPACKED_1x1(       -NAN, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x7fff), UNPACKED_1x1(        NAN, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0xffff), UNPACKED_1x1(       -NAN, 0.0, 0.0, 1.0)},

   /* Inf */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x7c00), UNPACKED_1x1(        INFINITY, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0xfc00), UNPACKED_1x1(       -INFINITY, 0.0, 0.0, 1.0)},

#endif

   /* Zero, ignore sign */
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0x7fff), PACKED_1x16(0x8000), UNPACKED_1x1( -0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0x7fff), PACKED_1x16(0x0000), UNPACKED_1x1(  0.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0x3c00), UNPACKED_1x1(  1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16_FLOAT, PACKED_1x16(0xffff), PACKED_1x16(0xbc00), UNPACKED_1x1( -1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x0000), UNPACKED_1x1( 0.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x3c00, 0x0000), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0xbc00, 0x0000), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0x3c00), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x0000, 0xbc00), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R16G16_FLOAT, PACKED_2x16(0xffff, 0xffff), PACKED_2x16(0x3c00, 0x3c00), UNPACKED_1x1( 1.0,  1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x0000), UNPACKED_1x1( 0.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x3c00, 0x0000, 0x0000), UNPACKED_1x1( 1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0xbc00, 0x0000, 0x0000), UNPACKED_1x1(-1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x3c00, 0x0000), UNPACKED_1x1( 0.0,  1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0xbc00, 0x0000), UNPACKED_1x1( 0.0, -1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0x3c00), UNPACKED_1x1( 0.0,  0.0,  1.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x0000, 0x0000, 0xbc00), UNPACKED_1x1( 0.0,  0.0, -1.0, 1.0)},
   {PIPE_FORMAT_R16G16B16_FLOAT, PACKED_3x16(0xffff, 0xffff, 0xffff), PACKED_3x16(0x3c00, 0x3c00, 0x3c00), UNPACKED_1x1( 1.0,  1.0,  1.0, 1.0)},

   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x0000), UNPACKED_1x1( 0.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x3c00, 0x0000, 0x0000, 0x0000), UNPACKED_1x1( 1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0xbc00, 0x0000, 0x0000, 0x0000), UNPACKED_1x1(-1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x3c00, 0x0000, 0x0000), UNPACKED_1x1( 0.0,  1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0xbc00, 0x0000, 0x0000), UNPACKED_1x1( 0.0, -1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x3c00, 0x0000), UNPACKED_1x1( 0.0,  0.0,  1.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0xbc00, 0x0000), UNPACKED_1x1( 0.0,  0.0, -1.0,  0.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0x3c00), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x0000, 0x0000, 0x0000, 0xbc00), UNPACKED_1x1( 0.0,  0.0,  0.0, -1.0)},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, PACKED_4x16(0xffff, 0xffff, 0xffff, 0xffff), PACKED_4x16(0x3c00, 0x3c00, 0x3c00, 0x3c00), UNPACKED_1x1( 1.0,  1.0,  1.0,  1.0)},

   /*
    * 32-bit fixed point formats
    */

   {PIPE_FORMAT_R32_FIXED, PACKED_1x32(0xffffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(  0.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32_FIXED, PACKED_1x32(0xffffffff), PACKED_1x32(0x00010000), UNPACKED_1x1(  1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32_FIXED, PACKED_1x32(0xffffffff), PACKED_1x32(0xffff0000), UNPACKED_1x1( -1.0, 0.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00010000, 0x00000000), UNPACKED_1x1( 1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0xffff0000, 0x00000000), UNPACKED_1x1(-1.0,  0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0x00010000), UNPACKED_1x1( 0.0,  1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00000000, 0xffff0000), UNPACKED_1x1( 0.0, -1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R32G32_FIXED, PACKED_2x32(0xffffffff, 0xffffffff), PACKED_2x32(0x00010000, 0x00010000), UNPACKED_1x1( 1.0,  1.0, 0.0, 1.0)},

   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00010000, 0x00000000, 0x00000000), UNPACKED_1x1( 1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0xffff0000, 0x00000000, 0x00000000), UNPACKED_1x1(-1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00010000, 0x00000000), UNPACKED_1x1( 0.0,  1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0xffff0000, 0x00000000), UNPACKED_1x1( 0.0, -1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0x00010000), UNPACKED_1x1( 0.0,  0.0,  1.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00000000, 0x00000000, 0xffff0000), UNPACKED_1x1( 0.0,  0.0, -1.0, 1.0)},
   {PIPE_FORMAT_R32G32B32_FIXED, PACKED_3x32(0xffffffff, 0xffffffff, 0xffffffff), PACKED_3x32(0x00010000, 0x00010000, 0x00010000), UNPACKED_1x1( 1.0,  1.0,  1.0, 1.0)},

   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00010000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1( 1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0xffff0000, 0x00000000, 0x00000000, 0x00000000), UNPACKED_1x1(-1.0,  0.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00010000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0,  1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0xffff0000, 0x00000000, 0x00000000), UNPACKED_1x1( 0.0, -1.0,  0.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00010000, 0x00000000), UNPACKED_1x1( 0.0,  0.0,  1.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0xffff0000, 0x00000000), UNPACKED_1x1( 0.0,  0.0, -1.0,  0.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0x00010000), UNPACKED_1x1( 0.0,  0.0,  0.0,  1.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00000000, 0x00000000, 0x00000000, 0xffff0000), UNPACKED_1x1( 0.0,  0.0,  0.0, -1.0)},
   {PIPE_FORMAT_R32G32B32A32_FIXED, PACKED_4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), PACKED_4x32(0x00010000, 0x00010000, 0x00010000, 0x00010000), UNPACKED_1x1( 1.0,  1.0,  1.0,  1.0)},

   /*
    * D3D9 specific vertex formats
    */

   {PIPE_FORMAT_R10G10B10X2_USCALED, PACKED_1x32(0x3fffffff), PACKED_1x32(0x00000000), UNPACKED_1x1(   0.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_USCALED, PACKED_1x32(0x3fffffff), PACKED_1x32(0x000003ff), UNPACKED_1x1(1023.0,    0.0,    0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_USCALED, PACKED_1x32(0x3fffffff), PACKED_1x32(0x000ffc00), UNPACKED_1x1(   0.0, 1023.0,    0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_USCALED, PACKED_1x32(0x3fffffff), PACKED_1x32(0x3ff00000), UNPACKED_1x1(   0.0,    0.0, 1023.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_USCALED, PACKED_1x32(0x3fffffff), PACKED_1x32(0x3fffffff), UNPACKED_1x1(1023.0, 1023.0, 1023.0, 1.0)},

   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x00000000), UNPACKED_1x1( 0.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x000001ff), UNPACKED_1x1( 1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x00000201), UNPACKED_1x1(-1.0,  0.0,  0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x0007fc00), UNPACKED_1x1( 0.0,  1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x00080400), UNPACKED_1x1( 0.0, -1.0,  0.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x1ff00000), UNPACKED_1x1( 0.0,  0.0,  1.0, 1.0)},
   {PIPE_FORMAT_R10G10B10X2_SNORM, PACKED_1x32(0x3fffffff), PACKED_1x32(0x20100000), UNPACKED_1x1( 0.0,  0.0, -1.0, 1.0)},

   /*
    * Special formats that not fit anywhere else
    */
   {PIPE_FORMAT_R3G3B2_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x07), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
   {PIPE_FORMAT_R3G3B2_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x38), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_R3G3B2_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xc0), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},

   {PIPE_FORMAT_B2G3R3_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x03), UNPACKED_1x1(0.0, 0.0, 1.0, 1.0)},
   {PIPE_FORMAT_B2G3R3_UNORM, PACKED_1x8(0xff), PACKED_1x8(0x1c), UNPACKED_1x1(0.0, 1.0, 0.0, 1.0)},
   {PIPE_FORMAT_B2G3R3_UNORM, PACKED_1x8(0xff), PACKED_1x8(0xe0), UNPACKED_1x1(1.0, 0.0, 0.0, 1.0)},
};


const unsigned util_format_nr_test_cases = ARRAY_SIZE(util_format_test_cases);
