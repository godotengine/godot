// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_write.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lib/jxl/alpha.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_write.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::NearestInt;
using hwy::HWY_NAMESPACE::Or;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::RebindToSigned;
using hwy::HWY_NAMESPACE::RebindToUnsigned;
using hwy::HWY_NAMESPACE::ShiftLeftSame;
using hwy::HWY_NAMESPACE::ShiftRightSame;
using hwy::HWY_NAMESPACE::VFromD;

// 32x32 blue noise dithering pattern from
// https://momentsingraphics.de/BlueNoise.html#Downloads scaled to have
// an average of 0 and be fully contained in (0.49219 to -0.49219).
// Rows are padded to 48 (32 + 16) to allow SIMD to wrap around horizontally
const float kDither[48 * 32] = {
    -0.26057, 0.32619, 0.21039, -0.03281, -0.10616, 0.16792, 0.43042, -0.48061,
    -0.00965, -0.31075, 0.24899, -0.35322, -0.02509, -0.25285, 0.02895, 0.10230,
    -0.28373, -0.00193, 0.23355, 0.43428, -0.23741, 0.18336, -0.31847, -0.11002,
    -0.36094, 0.26057, -0.19108, -0.29531, 0.40726, -0.09458, 0.11002, -0.48833,
    -0.26057, 0.32619, 0.21039, -0.03281, -0.10616, 0.16792, 0.43042, -0.48061,
    -0.00965, -0.31075, 0.24899, -0.35322, -0.02509, -0.25285, 0.02895, 0.10230,
    0.16020, -0.35708, -0.18336, 0.36094, -0.28373, -0.34550, -0.20267, 0.07914,
    0.35708, -0.41498, 0.47675, -0.21811, -0.12546, 0.44200, -0.41884, -0.17178,
    0.39954, 0.33778, -0.33778, 0.04053, -0.46517, 0.27215, -0.16792, 0.39182,
    0.20653, -0.43814, -0.02895, 0.17950, -0.41498, 0.01737, 0.24899, 0.49219,
    0.16020, -0.35708, -0.18336, 0.36094, -0.28373, -0.34550, -0.20267, 0.07914,
    0.35708, -0.41498, 0.47675, -0.21811, -0.12546, 0.44200, -0.41884, -0.17178,
    -0.00965, 0.08300, 0.41112, -0.46903, 0.04053, 0.47289, 0.26057, -0.05983,
    -0.13704, 0.14862, 0.03281, 0.29531, -0.45744, 0.22583, 0.14862, -0.09072,
    -0.37638, 0.19881, -0.14476, 0.14476, -0.09072, 0.48447, -0.39954, 0.06369,
    -0.05983, -0.26829, 0.43428, -0.12546, 0.28759, -0.22969, -0.32619, -0.15248,
    -0.00965, 0.08300, 0.41112, -0.46903, 0.04053, 0.47289, 0.26057, -0.05983,
    -0.13704, 0.14862, 0.03281, 0.29531, -0.45744, 0.22583, 0.14862, -0.09072,
    -0.42270, 0.23741, -0.23355, -0.11774, 0.18722, 0.11388, -0.43814, -0.24899,
    0.41884, 0.21039, -0.28373, -0.06756, 0.07914, 0.36480, -0.31075, 0.30303,
    -0.03281, 0.07142, -0.42656, 0.38024, -0.27987, 0.00579, 0.12546, -0.22197,
    0.29917, 0.36866, 0.13704, -0.47289, 0.09072, 0.35708, -0.04825, 0.38796,
    -0.42270, 0.23741, -0.23355, -0.11774, 0.18722, 0.11388, -0.43814, -0.24899,
    0.41884, 0.21039, -0.28373, -0.06756, 0.07914, 0.36480, -0.31075, 0.30303,
    -0.28759, -0.07142, 0.44200, 0.27601, -0.38024, -0.16020, -0.01737, 0.30303,
    -0.33006, -0.40340, -0.16792, 0.40726, -0.36480, -0.00579, -0.19108, 0.41498,
    -0.26443, 0.46903, -0.21811, 0.28759, -0.04053, 0.22197, 0.34550, -0.44972,
    -0.14476, -0.34164, 0.04053, -0.19494, 0.45358, -0.37252, 0.21425, 0.05597,
    -0.28759, -0.07142, 0.44200, 0.27601, -0.38024, -0.16020, -0.01737, 0.30303,
    -0.33006, -0.40340, -0.16792, 0.40726, -0.36480, -0.00579, -0.19108, 0.41498,
    0.31075, 0.14090, -0.33778, 0.00579, 0.34550, -0.29917, 0.38796, 0.13704,
    0.05983, -0.10230, 0.34164, 0.10616, -0.23741, 0.19494, -0.47675, 0.04439,
    -0.39568, 0.24127, 0.10616, -0.49219, -0.17950, -0.36094, -0.30303, 0.45744,
    -0.01351, 0.24513, -0.39182, -0.07528, 0.18722, -0.26057, -0.11002, -0.45358,
    0.31075, 0.14090, -0.33778, 0.00579, 0.34550, -0.29917, 0.38796, 0.13704,
    0.05983, -0.10230, 0.34164, 0.10616, -0.23741, 0.19494, -0.47675, 0.04439,
    0.46903, -0.17178, -0.41112, 0.07528, -0.09458, 0.21811, -0.20267, -0.48833,
    0.44972, 0.00965, 0.24127, -0.42656, 0.48447, -0.11774, 0.26443, 0.14090,
    -0.15634, -0.07142, -0.32233, 0.36094, 0.42270, 0.19108, 0.07142, -0.11002,
    0.15634, 0.38024, -0.28759, 0.27987, -0.00193, 0.33006, 0.11388, -0.21039,
    0.46903, -0.17178, -0.41112, 0.07528, -0.09458, 0.21811, -0.20267, -0.48833,
    0.44972, 0.00965, 0.24127, -0.42656, 0.48447, -0.11774, 0.26443, 0.14090,
    0.02123, 0.17950, 0.38024, -0.24127, -0.44586, 0.48833, -0.03667, 0.26829,
    -0.36866, -0.22583, 0.17178, -0.30689, 0.29145, -0.04825, -0.35322, 0.43042,
    0.34936, 0.00193, 0.16792, -0.12932, 0.03667, -0.06756, 0.31847, -0.40726,
    -0.24513, 0.09458, -0.17564, 0.47675, -0.43042, -0.32233, 0.40340, 0.26057,
    0.02123, 0.17950, 0.38024, -0.24127, -0.44586, 0.48833, -0.03667, 0.26829,
    -0.36866, -0.22583, 0.17178, -0.30689, 0.29145, -0.04825, -0.35322, 0.43042,
    -0.47675, -0.12160, -0.04825, 0.28759, 0.10230, 0.15634, -0.14862, -0.27601,
    0.36094, -0.12932, -0.05983, -0.45358, -0.17950, 0.01737, 0.09458, -0.29145,
    -0.22969, -0.43428, 0.45744, -0.38796, -0.27601, -0.21039, -0.46131, 0.22969,
    0.41112, -0.05211, -0.48061, 0.16406, 0.05211, -0.14862, -0.03281, -0.36866,
    -0.47675, -0.12160, -0.04825, 0.28759, 0.10230, 0.15634, -0.14862, -0.27601,
    0.36094, -0.12932, -0.05983, -0.45358, -0.17950, 0.01737, 0.09458, -0.29145,
    -0.27215, 0.34164, -0.31075, 0.42656, -0.38410, -0.32619, 0.02895, 0.19881,
    0.08300, 0.42270, 0.31461, 0.13318, 0.45744, 0.37638, -0.40726, 0.31847,
    -0.08686, 0.21425, 0.29917, 0.07914, 0.26829, 0.13704, 0.48447, -0.15248,
    0.02509, -0.34936, 0.34936, -0.10230, 0.42656, -0.23741, 0.22583, 0.09072,
    -0.27215, 0.34164, -0.31075, 0.42656, -0.38410, -0.32619, 0.02895, 0.19881,
    0.08300, 0.42270, 0.31461, 0.13318, 0.45744, 0.37638, -0.40726, 0.31847,
    0.44972, 0.20267, 0.04825, -0.21425, 0.24513, -0.07142, 0.39954, -0.46131,
    -0.39568, -0.01351, -0.33392, 0.05597, -0.26443, 0.22197, -0.20653, 0.15248,
    0.04439, -0.46517, -0.16406, -0.04439, -0.34936, 0.37252, -0.01351, -0.30689,
    0.29917, 0.20653, -0.26829, 0.26443, 0.13318, -0.39954, 0.30303, -0.08686,
    0.44972, 0.20267, 0.04825, -0.21425, 0.24513, -0.07142, 0.39954, -0.46131,
    -0.39568, -0.01351, -0.33392, 0.05597, -0.26443, 0.22197, -0.20653, 0.15248,
    -0.42656, 0.12932, -0.14476, -0.46903, -0.00579, 0.34936, -0.18722, 0.28373,
    -0.23741, 0.22969, -0.16020, -0.38024, -0.08300, -0.48447, -0.02123, -0.14862,
    0.48061, -0.31847, 0.39568, -0.24899, 0.18722, -0.41884, 0.10230, -0.08300,
    -0.38796, 0.06369, -0.19881, -0.44972, 0.00579, -0.33392, 0.37252, -0.19108,
    -0.42656, 0.12932, -0.14476, -0.46903, -0.00579, 0.34936, -0.18722, 0.28373,
    -0.23741, 0.22969, -0.16020, -0.38024, -0.08300, -0.48447, -0.02123, -0.14862,
    -0.02509, -0.35708, 0.32619, 0.46517, 0.17178, -0.28373, 0.10616, 0.47675,
    -0.09458, 0.15248, 0.43428, 0.35322, 0.17564, 0.27215, 0.41112, -0.36480,
    0.24899, 0.11774, 0.01351, 0.33006, -0.11388, -0.18336, 0.41884, -0.23355,
    0.16406, 0.46131, 0.38410, -0.04825, -0.15634, 0.49219, 0.17564, 0.03667,
    -0.02509, -0.35708, 0.32619, 0.46517, 0.17178, -0.28373, 0.10616, 0.47675,
    -0.09458, 0.15248, 0.43428, 0.35322, 0.17564, 0.27215, 0.41112, -0.36480,
    0.40726, 0.23355, -0.25285, -0.08300, -0.41112, -0.12160, -0.35708, 0.05211,
    -0.41884, -0.29531, 0.02123, -0.21425, 0.09844, -0.30689, -0.11388, 0.34550,
    -0.26443, -0.07142, -0.39954, 0.44586, 0.05983, -0.48833, 0.24127, 0.34936,
    -0.44200, -0.12546, 0.12160, -0.30303, 0.27215, 0.07528, -0.48447, -0.29145,
    0.40726, 0.23355, -0.25285, -0.08300, -0.41112, -0.12160, -0.35708, 0.05211,
    -0.41884, -0.29531, 0.02123, -0.21425, 0.09844, -0.30689, -0.11388, 0.34550,
    0.28373, -0.17564, 0.09458, 0.02123, 0.30689, 0.41884, 0.20653, -0.03667,
    0.32233, 0.25671, -0.45744, -0.05597, 0.46517, -0.41498, 0.00965, 0.07142,
    -0.44586, 0.16406, -0.20653, 0.21811, -0.29917, 0.28759, -0.05597, 0.03281,
    -0.32619, -0.00965, 0.31847, -0.37252, 0.18722, -0.11002, -0.22969, -0.06369,
    0.28373, -0.17564, 0.09458, 0.02123, 0.30689, 0.41884, 0.20653, -0.03667,
    0.32233, 0.25671, -0.45744, -0.05597, 0.46517, -0.41498, 0.00965, 0.07142,
    -0.39568, 0.36866, -0.45744, -0.31847, 0.14476, -0.22583, -0.49219, 0.37638,
    -0.19494, -0.13318, 0.39182, -0.35322, 0.29531, -0.24127, 0.21039, -0.18722,
    0.45358, 0.31461, -0.13318, -0.01737, -0.36094, 0.12932, -0.25671, 0.43814,
    -0.16792, 0.23355, -0.22197, 0.44972, -0.42270, 0.33392, 0.42656, 0.11774,
    -0.39568, 0.36866, -0.45744, -0.31847, 0.14476, -0.22583, -0.49219, 0.37638,
    -0.19494, -0.13318, 0.39182, -0.35322, 0.29531, -0.24127, 0.21039, -0.18722,
    -0.13318, 0.19494, -0.03667, 0.44972, 0.24513, -0.15248, 0.08300, -0.33006,
    0.00579, 0.12546, 0.19494, 0.05983, -0.15634, 0.14476, 0.36480, -0.04053,
    -0.33006, 0.25671, -0.46903, 0.37252, 0.48833, -0.09458, -0.41112, 0.19108,
    0.08686, -0.46903, -0.07528, 0.04053, -0.26829, -0.02895, 0.22197, -0.34164,
    -0.13318, 0.19494, -0.03667, 0.44972, 0.24513, -0.15248, 0.08300, -0.33006,
    0.00579, 0.12546, 0.19494, 0.05983, -0.15634, 0.14476, 0.36480, -0.04053,
    0.47289, -0.21811, 0.06756, -0.38410, -0.27987, -0.06369, 0.27987, 0.43814,
    -0.25671, -0.39182, 0.49219, -0.27601, -0.07914, -0.48061, 0.42656, -0.38410,
    0.11002, 0.03667, -0.27215, 0.15634, 0.07528, -0.22197, 0.33006, 0.38410,
    -0.34936, 0.27987, 0.15248, 0.40340, 0.09844, -0.16406, -0.46131, 0.03281,
    0.47289, -0.21811, 0.06756, -0.38410, -0.27987, -0.06369, 0.27987, 0.43814,
    -0.25671, -0.39182, 0.49219, -0.27601, -0.07914, -0.48061, 0.42656, -0.38410,
    -0.29531, 0.31461, -0.10616, 0.39954, 0.01351, 0.33778, -0.43814, 0.17178,
    -0.08686, 0.23741, -0.44586, 0.33778, -0.00193, -0.31461, 0.23741, -0.12932,
    -0.22583, -0.06756, 0.40340, -0.16792, -0.43428, 0.01351, -0.14476, -0.04053,
    -0.29145, 0.46517, -0.13704, -0.39182, -0.32233, 0.29531, 0.38410, 0.16020,
    -0.29531, 0.31461, -0.10616, 0.39954, 0.01351, 0.33778, -0.43814, 0.17178,
    -0.08686, 0.23741, -0.44586, 0.33778, -0.00193, -0.31461, 0.23741, -0.12932,
    -0.44200, 0.26443, 0.12546, -0.42270, 0.21425, -0.19881, -0.35708, 0.04825,
    0.36480, -0.02895, -0.21425, 0.09072, 0.41498, 0.18336, 0.04439, 0.29917,
    0.47675, -0.40340, 0.27601, -0.31461, 0.31075, 0.17564, 0.24899, -0.45744,
    0.05597, -0.19494, 0.00193, 0.36094, 0.24127, -0.09844, -0.24513, -0.00965,
    -0.44200, 0.26443, 0.12546, -0.42270, 0.21425, -0.19881, -0.35708, 0.04825,
    0.36480, -0.02895, -0.21425, 0.09072, 0.41498, 0.18336, 0.04439, 0.29917,
    -0.17564, -0.05597, -0.34550, -0.24899, 0.48061, 0.15248, -0.11388, 0.45358,
    -0.16406, -0.32233, 0.31461, -0.11774, -0.36866, -0.18722, -0.25671, -0.44200,
    0.13318, -0.02123, 0.19881, -0.10616, 0.43042, -0.36866, -0.24899, 0.41112,
    0.11002, 0.21425, -0.25671, -0.47675, -0.04439, 0.13704, -0.37252, 0.43814,
    -0.17564, -0.05597, -0.34550, -0.24899, 0.48061, 0.15248, -0.11388, 0.45358,
    -0.16406, -0.32233, 0.31461, -0.11774, -0.36866, -0.18722, -0.25671, -0.44200,
    0.19108, 0.03667, 0.35708, -0.14090, 0.08300, -0.02123, -0.30303, -0.48061,
    0.11774, 0.20267, -0.43042, 0.25285, 0.14090, -0.04439, 0.38796, 0.34550,
    -0.34164, -0.19494, 0.05983, -0.48447, 0.09844, -0.00579, -0.07914, 0.33778,
    -0.41498, -0.10230, 0.30689, 0.17178, 0.48833, -0.20267, 0.07914, 0.33392,
    0.19108, 0.03667, 0.35708, -0.14090, 0.08300, -0.02123, -0.30303, -0.48061,
    0.11774, 0.20267, -0.43042, 0.25285, 0.14090, -0.04439, 0.38796, 0.34550,
    -0.48833, -0.30689, 0.41498, 0.22969, -0.44586, 0.32233, 0.25285, 0.39182,
    -0.23355, 0.01737, 0.42270, -0.27987, 0.46903, -0.47289, 0.02123, -0.09072,
    0.21811, 0.44586, -0.25285, 0.36480, -0.29145, 0.47289, -0.18722, 0.14476,
    -0.31461, 0.43814, -0.36094, 0.04439, -0.29917, -0.41884, 0.25285, -0.11774,
    -0.48833, -0.30689, 0.41498, 0.22969, -0.44586, 0.32233, 0.25285, 0.39182,
    -0.23355, 0.01737, 0.42270, -0.27987, 0.46903, -0.47289, 0.02123, -0.09072,
    0.46131, 0.11388, -0.21039, -0.07528, -0.38024, -0.26057, 0.06369, -0.05983,
    0.29145, -0.40340, -0.09072, 0.06756, -0.16020, 0.27601, -0.31075, 0.10616,
    -0.14090, -0.43042, 0.25671, -0.05211, -0.13318, 0.23355, -0.44972, 0.02895,
    0.26829, -0.02895, -0.17950, 0.37252, -0.13704, 0.40726, 0.01351, -0.26443,
    0.46131, 0.11388, -0.21039, -0.07528, -0.38024, -0.26057, 0.06369, -0.05983,
    0.29145, -0.40340, -0.09072, 0.06756, -0.16020, 0.27601, -0.31075, 0.10616,
    -0.03281, -0.40340, 0.27987, 0.17564, 0.02509, 0.44200, -0.15248, -0.34550,
    0.14862, -0.19881, -0.01351, 0.36866, -0.38796, 0.19494, -0.22197, 0.32619,
    -0.37638, 0.00193, 0.30689, 0.12160, -0.39182, 0.16792, -0.34550, 0.39954,
    -0.23355, 0.09072, -0.43428, 0.22969, -0.06369, 0.12546, -0.35322, 0.30689,
    -0.03281, -0.40340, 0.27987, 0.17564, 0.02509, 0.44200, -0.15248, -0.34550,
    0.14862, -0.19881, -0.01351, 0.36866, -0.38796, 0.19494, -0.22197, 0.32619,
    -0.09844, 0.06756, 0.38410, -0.33392, -0.18336, 0.35322, 0.21039, -0.42270,
    0.48833, 0.33006, 0.21811, -0.33392, 0.12932, -0.05211, 0.39568, 0.04825,
    0.48061, 0.17950, -0.31847, -0.21811, 0.38024, 0.05211, 0.32233, -0.06756,
    -0.12546, 0.46131, 0.16020, -0.25285, 0.29531, -0.44972, 0.17950, -0.16406,
    -0.09844, 0.06756, 0.38410, -0.33392, -0.18336, 0.35322, 0.21039, -0.42270,
    0.48833, 0.33006, 0.21811, -0.33392, 0.12932, -0.05211, 0.39568, 0.04825,
    0.22583, -0.46131, -0.27601, -0.00579, 0.12932, -0.47289, -0.09844, 0.10230,
    -0.28759, -0.12160, -0.49219, -0.24127, 0.44586, -0.11388, -0.45358, -0.27215,
    -0.17178, -0.07528, -0.47675, 0.43042, -0.02509, -0.27215, -0.19108, 0.19881,
    -0.49219, -0.37252, 0.33392, -0.00193, -0.33006, -0.20267, 0.48061, 0.34164,
    0.22583, -0.46131, -0.27601, -0.00579, 0.12932, -0.47289, -0.09844, 0.10230,
    -0.28759, -0.12160, -0.49219, -0.24127, 0.44586, -0.11388, -0.45358, -0.27215,
    -0.22969, 0.42270, -0.12160, 0.31075, 0.46903, -0.22583, 0.27215, -0.02509,
    0.03281, 0.40340, 0.25671, 0.08686, 0.00965, 0.29145, -0.41112, 0.14090,
    0.24513, 0.34164, 0.08686, -0.14862, 0.27601, -0.42656, 0.48447, 0.09844,
    0.26443, -0.27987, 0.05597, -0.10230, 0.43428, 0.08686, 0.02895, -0.38024,
    -0.22969, 0.42270, -0.12160, 0.31075, 0.46903, -0.22583, 0.27215, -0.02509,
    0.03281, 0.40340, 0.25671, 0.08686, 0.00965, 0.29145, -0.41112, 0.14090,
    0.15634, 0.09458, -0.36480, 0.18336, -0.05211, -0.40726, 0.36866, -0.33778,
    -0.19881, 0.16020, -0.37638, -0.16020, -0.29917, 0.20267, 0.41884, -0.01737,
    -0.34936, -0.24127, 0.02509, 0.20653, -0.36480, -0.08686, 0.01737, -0.33778,
    0.41498, -0.03667, 0.37638, -0.17178, -0.47289, 0.26829, -0.28759, -0.05597,
    0.15634, 0.09458, -0.36480, 0.18336, -0.05211, -0.40726, 0.36866, -0.33778,
    -0.19881, 0.16020, -0.37638, -0.16020, -0.29917, 0.20267, 0.41884, -0.01737,
    0.35708, 0.00193, 0.25285, -0.15634, -0.30303, 0.06369, 0.22197, 0.45358,
    -0.43814, 0.30303, -0.04053, 0.46517, 0.35322, -0.21039, 0.06756, -0.14090,
    0.37638, -0.43042, 0.45744, -0.29531, 0.39568, 0.14862, 0.23741, -0.13704,
    -0.21425, 0.16406, -0.40726, 0.22583, 0.13318, 0.38796, -0.12932, -0.43428,
    0.35708, 0.00193, 0.25285, -0.15634, -0.30303, 0.06369, 0.22197, 0.45358,
    -0.43814, 0.30303, -0.04053, 0.46517, 0.35322, -0.21039, 0.06756, -0.14090,
    -0.31461, -0.20653, 0.46131, -0.45358, 0.39568, -0.24513, -0.14090, 0.11002,
    -0.08300, -0.26829, 0.05211, -0.46517, -0.09844, -0.39568, -0.32619, -0.06369,
    0.16792, 0.28373, 0.11388, -0.04439, -0.18336, -0.44200, 0.35322, -0.26057,
    -0.46517, 0.31075, -0.07914, -0.34164, -0.24513, -0.02123, 0.19108, 0.44200,
    -0.31461, -0.20653, 0.46131, -0.45358, 0.39568, -0.24513, -0.14090, 0.11002,
    -0.08300, -0.26829, 0.05211, -0.46517, -0.09844, -0.39568, -0.32619, -0.06369,
    0.04825, -0.07914, -0.39954, 0.12160, 0.29145, 0.00965, -0.37638, 0.32233,
    0.20267, -0.17564, 0.39182, 0.12160, 0.18336, 0.32619, 0.26057, 0.49219,
    -0.48447, -0.20653, -0.10616, -0.38796, 0.31847, 0.07528, -0.01737, 0.44586,
    0.11774, 0.02509, 0.47289, 0.07142, 0.33392, -0.38410, -0.17950, 0.28373,
    0.04825, -0.07914, -0.39954, 0.12160, 0.29145, 0.00965, -0.37638, 0.32233,
    0.20267, -0.17564, 0.39182, 0.12160, 0.18336, 0.32619, 0.26057, 0.49219
};

namespace {
constexpr static size_t kChunkSize = 1024;
}  // namespace

using DF = HWY_FULL(float);

// Converts `v` to an appropriate value for the given unsigned type.
// If the unsigned type is an 8-bit type, performs ordered dithering.
template <typename T>
VFromD<Rebind<T, DF>> MakeUnsigned(VFromD<DF> v, size_t x0, size_t y0,
                                   VFromD<DF> mul, size_t c) {
  static_assert(std::is_unsigned<T>::value, "T must be an unsigned type");
  using DI32 = RebindToSigned<DF>;
  using DU32 = RebindToUnsigned<DF>;
  using DU = Rebind<T, DF>;
  v = Mul(v, mul);
  // TODO(veluca): if constexpr with C++17
  if (sizeof(T) == 1) {
    size_t x_off = (x0 + c * 23) % 32;
    size_t y_off = (y0 + c * 13) % 32;
    size_t pos = y_off * 48 + x_off;
    auto dither = LoadU(DF(), kDither + pos);
    v = Add(v, dither);
  }
  v = Clamp(Zero(DF()), v, mul);
  VFromD<DI32> ni = NearestInt(v);
  VFromD<DU32> nu = BitCast(DU32(), ni);
  return DemoteTo(DU(), nu);
}

class WriteToOutputStage : public RenderPipelineStage {
 public:
  WriteToOutputStage(const ImageOutput& main_output, size_t width,
                     size_t height, bool has_alpha, bool unpremul_alpha,
                     size_t alpha_c, Orientation undo_orientation,
                     const std::vector<ImageOutput>& extra_output,
                     JxlMemoryManager* memory_manager)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        width_(width),
        height_(height),
        main_(main_output),
        num_color_(main_.num_channels_ < 3 ? 1 : 3),
        want_alpha_(main_.num_channels_ == 2 || main_.num_channels_ == 4),
        has_alpha_(has_alpha),
        unpremul_alpha_(unpremul_alpha),
        alpha_c_(alpha_c),
        flip_x_(ShouldFlipX(undo_orientation)),
        flip_y_(ShouldFlipY(undo_orientation)),
        transpose_(ShouldTranspose(undo_orientation)),
        opaque_alpha_(kChunkSize, 1.0f),
        memory_manager_(memory_manager) {
    for (size_t ec = 0; ec < extra_output.size(); ++ec) {
      if (extra_output[ec].callback.IsPresent() || extra_output[ec].buffer) {
        Output extra(extra_output[ec]);
        extra.channel_index_ = 3 + ec;
        extra_channels_.push_back(extra);
      }
    }
  }

  WriteToOutputStage(const WriteToOutputStage&) = delete;
  WriteToOutputStage& operator=(const WriteToOutputStage&) = delete;
  WriteToOutputStage(WriteToOutputStage&&) = delete;
  WriteToOutputStage& operator=(WriteToOutputStage&&) = delete;

  ~WriteToOutputStage() override {
    if (main_.run_opaque_) {
      main_.pixel_callback_.destroy(main_.run_opaque_);
    }
    for (auto& extra : extra_channels_) {
      if (extra.run_opaque_) {
        extra.pixel_callback_.destroy(extra.run_opaque_);
      }
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    JXL_ENSURE(main_.run_opaque_ || main_.buffer_);
    if (ypos >= height_) return true;
    if (xpos >= width_) return true;
    if (flip_y_) {
      ypos = height_ - 1u - ypos;
    }
    size_t limit = std::min(xsize, width_ - xpos);
    for (size_t x0 = 0; x0 < limit; x0 += kChunkSize) {
      size_t xstart = xpos + x0;
      size_t len = std::min<size_t>(kChunkSize, limit - x0);

      const float* line_buffers[4];
      for (size_t c = 0; c < num_color_; c++) {
        line_buffers[c] = GetInputRow(input_rows, c, 0) + x0;
      }
      if (has_alpha_) {
        line_buffers[num_color_] = GetInputRow(input_rows, alpha_c_, 0) + x0;
      } else {
        // opaque_alpha_ is a way to set all values to 1.0f.
        line_buffers[num_color_] = opaque_alpha_.data();
      }
      if (has_alpha_ && want_alpha_ && unpremul_alpha_) {
        UnpremulAlpha(thread_id, len, line_buffers);
      }
      OutputBuffers(main_, thread_id, ypos, xstart, len, line_buffers);
      for (const auto& extra : extra_channels_) {
        line_buffers[0] = GetInputRow(input_rows, extra.channel_index_, 0) + x0;
        OutputBuffers(extra, thread_id, ypos, xstart, len, line_buffers);
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    if (c < num_color_ || (has_alpha_ && c == alpha_c_)) {
      return RenderPipelineChannelMode::kInput;
    }
    for (const auto& extra : extra_channels_) {
      if (c == extra.channel_index_) {
        return RenderPipelineChannelMode::kInput;
      }
    }
    return RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WritePixelCB"; }

 private:
  struct Output {
    explicit Output(const ImageOutput& image_out)
        : pixel_callback_(image_out.callback),
          buffer_(image_out.buffer),
          buffer_size_(image_out.buffer_size),
          stride_(image_out.stride),
          num_channels_(image_out.format.num_channels),
          swap_endianness_(SwapEndianness(image_out.format.endianness)),
          data_type_(image_out.format.data_type),
          bits_per_sample_(image_out.bits_per_sample) {}

    Status PrepareForThreads(size_t num_threads) {
      if (pixel_callback_.IsPresent()) {
        run_opaque_ =
            pixel_callback_.Init(num_threads, /*num_pixels=*/kChunkSize);
        JXL_RETURN_IF_ERROR(run_opaque_ != nullptr);
      } else {
        JXL_RETURN_IF_ERROR(buffer_ != nullptr);
      }
      return true;
    }

    PixelCallback pixel_callback_;
    void* run_opaque_ = nullptr;
    void* buffer_ = nullptr;
    size_t buffer_size_;
    size_t stride_;
    size_t num_channels_;
    bool swap_endianness_;
    JxlDataType data_type_;
    size_t bits_per_sample_;
    size_t channel_index_;  // used for extra_channels
  };

  Status PrepareForThreads(size_t num_threads) override {
    JXL_RETURN_IF_ERROR(main_.PrepareForThreads(num_threads));
    for (auto& extra : extra_channels_) {
      JXL_RETURN_IF_ERROR(extra.PrepareForThreads(num_threads));
    }
    temp_out_.resize(num_threads);
    size_t alloc_size = sizeof(float) * kChunkSize;
    for (AlignedMemory& temp : temp_out_) {
      JXL_ASSIGN_OR_RETURN(
          temp, AlignedMemory::Create(memory_manager_,
                                      alloc_size * main_.num_channels_));
    }
    if ((has_alpha_ && want_alpha_ && unpremul_alpha_) || flip_x_) {
      temp_in_.resize(num_threads * main_.num_channels_);
      for (AlignedMemory& temp : temp_in_) {
        JXL_ASSIGN_OR_RETURN(
            temp, AlignedMemory::Create(memory_manager_, alloc_size));
      }
    }
    return true;
  }
  static bool ShouldFlipX(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipHorizontal ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldFlipY(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipVertical ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldTranspose(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kTranspose ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }

  void UnpremulAlpha(size_t thread_id, size_t len,
                     const float** line_buffers) const {
    const HWY_FULL(float) d;
    auto one = Set(d, 1.0f);
    float* temp_in[4];
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
    }
    auto small_alpha = Set(d, kSmallAlpha);
    for (size_t ix = 0; ix < len; ix += Lanes(d)) {
      auto alpha = LoadU(d, temp_in[num_color_] + ix);
      auto mul = Div(one, Max(small_alpha, alpha));
      for (size_t c = 0; c < num_color_; ++c) {
        auto val = LoadU(d, temp_in[c] + ix);
        StoreU(Mul(val, mul), d, temp_in[c] + ix);
      }
    }
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
  }

  void OutputBuffers(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, const float* input[4]) const {
    if (flip_x_) {
      FlipX(out, thread_id, len, &xstart, input);
    }
    if (out.data_type_ == JXL_TYPE_UINT8) {
      uint8_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint8_t>();
      StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_UINT16 ||
               out.data_type_ == JXL_TYPE_FLOAT16) {
      uint16_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint16_t>();
      if (out.data_type_ == JXL_TYPE_UINT16) {
        StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      } else {
        StoreFloat16Row(out, input, len, temp);
      }
      if (out.swap_endianness_) {
        const HWY_FULL(uint16_t) du;
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; j += Lanes(du)) {
          auto v = LoadU(du, temp + j);
          auto vswap = Or(ShiftRightSame(v, 8), ShiftLeftSame(v, 8));
          StoreU(vswap, du, temp + j);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_FLOAT) {
      float* JXL_RESTRICT temp = temp_out_[thread_id].address<float>();
      StoreFloatRow(out, input, len, temp);
      if (out.swap_endianness_) {
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; ++j) {
          temp[j] = BSwapFloat(temp[j]);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    }
  }

  void FlipX(const Output& out, size_t thread_id, size_t len, size_t* xstart,
             const float** line_buffers) const {
    float* temp_in[4];
    for (size_t c = 0; c < out.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      if (temp_in[c] != line_buffers[c]) {
        memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
      }
    }
    size_t last = (len - 1u);
    size_t num = (len / 2);
    for (size_t i = 0; i < num; ++i) {
      for (size_t c = 0; c < out.num_channels_; ++c) {
        std::swap(temp_in[c][i], temp_in[c][last - i]);
      }
    }
    for (size_t c = 0; c < out.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
    *xstart = width_ - *xstart - len;
  }

  template <typename T>
  void StoreUnsignedRow(const Output& out, const float* input[4], size_t len,
                        T* output, size_t xstart, size_t ypos) const {
    const HWY_FULL(float) d;
    auto mul = Set(d, (1u << (out.bits_per_sample_)) - 1);
    const Rebind<T, decltype(d)> du;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreU(MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
               du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1), du,
            &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul, 2), du,
            &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul, 2),
            MakeUnsigned<T>(LoadU(d, &input[3][i]), xstart + i, ypos, mul, 3), du,
            &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloat16Row(const Output& out, const float* input[4],
                              size_t len, uint16_t* output) {
    const HWY_FULL(float) d;
    const Rebind<uint16_t, decltype(d)> du;
    const Rebind<hwy::float16_t, decltype(d)> df16;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        StoreU(BitCast(du, DemoteTo(df16, v0)), du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        StoreInterleaved2(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)), du, &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        StoreInterleaved3(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)), du, &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        auto v3 = LoadU(d, &input[3][i]);
        StoreInterleaved4(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)),
                          BitCast(du, DemoteTo(df16, v3)), du, &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloatRow(const Output& out, const float* input[4],
                            size_t len, float* output) {
    const HWY_FULL(float) d;
    if (out.num_channels_ == 1) {
      memcpy(output, input[0], len * sizeof(output[0]));
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]), d,
                          &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), d, &output[3 * i]);
      }
    } else {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), LoadU(d, &input[3][i]), d,
                          &output[4 * i]);
      }
    }
  }

  template <typename T>
  void WriteToOutput(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, T* output) const {
    if (transpose_) {
      // TODO(szabadka) Buffer 8x8 chunks and transpose with SIMD.
      if (out.run_opaque_) {
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          out.pixel_callback_.run(out.run_opaque_, thread_id, ypos, xstart + i,
                                  1, output + j);
        }
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = xstart * out.stride_ + ypos * pixel_stride;
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          const size_t ix = offset + i * out.stride_;
          JXL_DASSERT(ix + pixel_stride <= out.buffer_size_);
          memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + ix, output + j,
                 pixel_stride);
        }
      }
    } else {
      if (out.run_opaque_) {
        out.pixel_callback_.run(out.run_opaque_, thread_id, xstart, ypos, len,
                                output);
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = ypos * out.stride_ + xstart * pixel_stride;
        JXL_DASSERT(offset + len * pixel_stride <= out.buffer_size_);
        memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + offset, output,
               len * pixel_stride);
      }
    }
  }

  // Process row in chunks to keep per-thread buffers compact.
  size_t width_;
  size_t height_;
  Output main_;  // color + alpha
  size_t num_color_;
  bool want_alpha_;
  bool has_alpha_;
  bool unpremul_alpha_;
  size_t alpha_c_;
  bool flip_x_;
  bool flip_y_;
  bool transpose_;
  std::vector<Output> extra_channels_;
  std::vector<float> opaque_alpha_;
  JxlMemoryManager* memory_manager_;
  std::vector<AlignedMemory> temp_in_;
  std::vector<AlignedMemory> temp_out_;
};

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return jxl::make_unique<WriteToOutputStage>(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(GetWriteToOutputStage);

namespace {
class WriteToImageBundleStage : public RenderPipelineStage {
 public:
  explicit WriteToImageBundleStage(
      ImageBundle* image_bundle, const ColorEncoding* color_encoding)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        image_bundle_(image_bundle),
        color_encoding_(*color_encoding) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JxlMemoryManager* memory_manager = image_bundle_->memory_manager();
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < input_sizes.size(); c++) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    // TODO(eustas): what should we do in the case of "want only ECs"?
    JXL_ASSIGN_OR_RETURN(Image3F tmp,
                         Image3F::Create(memory_manager, input_sizes[0].first,
                                         input_sizes[0].second));
    JXL_RETURN_IF_ERROR(
        image_bundle_->SetFromImage(std::move(tmp), color_encoding_));
    // TODO(veluca): consider not reallocating ECs if not needed.
    image_bundle_->extra_channels().clear();
    for (size_t c = 3; c < input_sizes.size(); c++) {
      JXL_ASSIGN_OR_RETURN(ImageF ch,
                           ImageF::Create(memory_manager, input_sizes[c].first,
                                          input_sizes[c].second));
      image_bundle_->extra_channels().emplace_back(std::move(ch));
    }
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_bundle_->color()->PlaneRow(c, ypos) + xpos,
             GetInputRow(input_rows, c, 0), sizeof(float) * xsize);
    }
    for (size_t ec = 0; ec < image_bundle_->extra_channels().size(); ec++) {
      JXL_ENSURE(image_bundle_->extra_channels()[ec].xsize() >= xpos + xsize);
      memcpy(image_bundle_->extra_channels()[ec].Row(ypos) + xpos - xextra_left,
             GetInputRow(input_rows, 3 + ec, 0), sizeof(float) * xsize);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }

  const char* GetName() const override { return "WriteIB"; }

 private:
  ImageBundle* image_bundle_;
  ColorEncoding color_encoding_;
};

class WriteToImage3FStage : public RenderPipelineStage {
 public:
  WriteToImage3FStage(JxlMemoryManager* memory_manager, Image3F* image)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        memory_manager_(memory_manager),
        image_(image) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < 3; ++c) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    JXL_ASSIGN_OR_RETURN(*image_,
                         Image3F::Create(memory_manager_, input_sizes[0].first,
                                         input_sizes[0].second));
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_->PlaneRow(c, ypos) + xpos, GetInputRow(input_rows, c, 0),
             sizeof(float) * xsize);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInput
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WriteI3F"; }

 private:
  JxlMemoryManager* memory_manager_;
  Image3F* image_;
};

}  // namespace

std::unique_ptr<RenderPipelineStage> GetWriteToImageBundleStage(
    ImageBundle* image_bundle, const ColorEncoding* color_encoding) {
  return jxl::make_unique<WriteToImageBundleStage>(image_bundle,
                                                   color_encoding);
}

std::unique_ptr<RenderPipelineStage> GetWriteToImage3FStage(
    JxlMemoryManager* memory_manager, Image3F* image) {
  return jxl::make_unique<WriteToImage3FStage>(memory_manager, image);
}

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return HWY_DYNAMIC_DISPATCH(GetWriteToOutputStage)(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

}  // namespace jxl

#endif
