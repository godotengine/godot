/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/x86/filter_x86.h"

DECLARE_ALIGNED(16, const short, vp8_bilinear_filters_x86_4[8][8]) =
{
    { 128, 128, 128, 128,   0,   0,   0,   0 },
    { 112, 112, 112, 112,  16,  16,  16,  16 },
    {  96,  96,  96,  96,  32,  32,  32,  32 },
    {  80,  80,  80,  80,  48,  48,  48,  48 },
    {  64,  64,  64,  64,  64,  64,  64,  64 },
    {  48,  48,  48,  48,  80,  80,  80,  80 },
    {  32,  32,  32,  32,  96,  96,  96,  96 },
    {  16,  16,  16,  16, 112, 112, 112, 112 }
};

DECLARE_ALIGNED(16, const short, vp8_bilinear_filters_x86_8[8][16]) =
{
    { 128, 128, 128, 128, 128, 128, 128, 128,   0,   0,   0,   0,   0,   0,   0,   0 },
    { 112, 112, 112, 112, 112, 112, 112, 112,  16,  16,  16,  16,  16,  16,  16,  16 },
    {  96,  96,  96,  96,  96,  96,  96,  96,  32,  32,  32,  32,  32,  32,  32,  32 },
    {  80,  80,  80,  80,  80,  80,  80,  80,  48,  48,  48,  48,  48,  48,  48,  48 },
    {  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64 },
    {  48,  48,  48,  48,  48,  48,  48,  48,  80,  80,  80,  80,  80,  80,  80,  80 },
    {  32,  32,  32,  32,  32,  32,  32,  32,  96,  96,  96,  96,  96,  96,  96,  96 },
    {  16,  16,  16,  16,  16,  16,  16,  16, 112, 112, 112, 112, 112, 112, 112, 112 }
};
