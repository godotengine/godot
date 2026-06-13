// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_H_
#define LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_H_

// Haar-like transform: halves the resolution in one direction
// A B   -> (A+B)>>1              in one channel (average)  -> same range as
// original channel
//          A-B - tendency        in a new channel ('residual' needed to make
//          the transform reversible)
//                                        -> theoretically range could be 2.5
//                                        times larger (2 times without the
//                                        'tendency'), but there should be lots
//                                        of zeroes
// Repeated application (alternating horizontal and vertical squeezes) results
// in downscaling
//
// The default coefficient ordering is low-frequency to high-frequency, as in
// M. Antonini, M. Barlaud, P. Mathieu and I. Daubechies, "Image coding using
// wavelet transform", IEEE Transactions on Image Processing, vol. 1, no. 2, pp.
// 205-220, April 1992, doi: 10.1109/83.136597.

#include <cstdlib>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

constexpr size_t kMaxFirstPreviewSize = 8;

/*
        int avg=(A+B)>>1;
        int diff=(A-B);
        int rA=(diff+(avg<<1)+(diff&1))>>1;
        int rB=rA-diff;

*/
//         |A B|C D|E F|
//           p   a   n             p=avg(A,B), a=avg(C,D), n=avg(E,F)
//
// Goal: estimate C-D (avoiding ringing artifacts)
// (ensuring that in smooth areas, a zero residual corresponds to a smooth
// gradient)

// best estimate for C: (B + 2*a)/3
// best estimate for D: (n + 3*a)/4
// best estimate for C-D:  4*B - 3*n - a /12

// avoid ringing by 1) only doing this if B <= a <= n  or  B >= a >= n
// (otherwise, this is not a smooth area and we cannot really estimate C-D)
//                  2) making sure that B <= C <= D <= n  or B >= C >= D >= n

inline pixel_type_w SmoothTendency(pixel_type_w B, pixel_type_w a,
                                   pixel_type_w n) {
  pixel_type_w diff = 0;
  if (B >= a && a >= n) {
    diff = (4 * B - 3 * n - a + 6) / 12;
    //      2C = a<<1 + diff - diff&1 <= 2B  so diff - diff&1 <= 2B - 2a
    //      2D = a<<1 - diff - diff&1 >= 2n  so diff + diff&1 <= 2a - 2n
    if (diff - (diff & 1) > 2 * (B - a)) diff = 2 * (B - a) + 1;
    if (diff + (diff & 1) > 2 * (a - n)) diff = 2 * (a - n);
  } else if (B <= a && a <= n) {
    diff = (4 * B - 3 * n - a - 6) / 12;
    //      2C = a<<1 + diff + diff&1 >= 2B  so diff + diff&1 >= 2B - 2a
    //      2D = a<<1 - diff + diff&1 <= 2n  so diff - diff&1 >= 2a - 2n
    if (diff + (diff & 1) < 2 * (B - a)) diff = 2 * (B - a) - 1;
    if (diff - (diff & 1) < 2 * (a - n)) diff = 2 * (a - n);
  }
  return diff;
}

void DefaultSqueezeParameters(std::vector<SqueezeParams> *parameters,
                              const Image &image);

Status CheckMetaSqueezeParams(const SqueezeParams &parameter, int num_channels);

Status MetaSqueeze(Image &image, std::vector<SqueezeParams> *parameters);

Status InvSqueeze(Image &input, const std::vector<SqueezeParams> &parameters,
                  ThreadPool *pool);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_H_
