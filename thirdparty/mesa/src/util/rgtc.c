/*
 * Copyright (C) 2011 Red Hat Inc.
 *
 * block compression parts are:
 * Copyright (C) 2004  Roland Scheidegger   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author:
 *    Dave Airlie
 */

#include <inttypes.h>
#include "macros.h"

#include "rgtc.h"

#define RGTC_DEBUG 0

#define TAG(x) util_format_unsigned_##x

#define TYPE unsigned char
#define T_MIN 0
#define T_MAX 0xff

#include "texcompress_rgtc_tmp.h"

#undef TAG
#undef TYPE
#undef T_MIN
#undef T_MAX

#define TAG(x) util_format_signed_##x
#define TYPE signed char
#define T_MIN (signed char)-128
#define T_MAX (signed char)127

#include "texcompress_rgtc_tmp.h"

#undef TAG
#undef TYPE
#undef T_MIN
#undef T_MAX

