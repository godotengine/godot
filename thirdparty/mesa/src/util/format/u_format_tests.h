/**************************************************************************
 *
 * Copyright 2010 VMware, Inc.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/


#ifndef U_FORMAT_TESTS_H_
#define U_FORMAT_TESTS_H_


#include "pipe/p_compiler.h"
#include "util/format/u_formats.h"


#define UTIL_FORMAT_MAX_PACKED_BYTES 32  // R64G64B64A64_FLOAT
#define UTIL_FORMAT_MAX_UNPACKED_WIDTH 12  // ASTC 12x12
#define UTIL_FORMAT_MAX_UNPACKED_HEIGHT 12


/**
 * A (packed, unpacked) color pair.
 */
struct util_format_test_case
{
   enum pipe_format format;

   /**
    * Mask of the bits that actually meaningful data. Used to mask out the
    * "X" channels.
    */
   uint8_t mask[UTIL_FORMAT_MAX_PACKED_BYTES];

   uint8_t packed[UTIL_FORMAT_MAX_PACKED_BYTES];

   /**
    * RGBA.
    */
   double unpacked[UTIL_FORMAT_MAX_UNPACKED_HEIGHT][UTIL_FORMAT_MAX_UNPACKED_WIDTH][4];
};


extern const struct util_format_test_case
util_format_test_cases[];


extern const unsigned util_format_nr_test_cases;


#endif /* U_FORMAT_TESTS_H_ */
