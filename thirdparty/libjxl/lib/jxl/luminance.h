// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_LUMINANCE_H_
#define LIB_JXL_LUMINANCE_H_

namespace jxl {

// Chooses a default intensity target based on the transfer function of the
// image, if known. For SDR images or images not known to be HDR, returns
// kDefaultIntensityTarget, for images known to have PQ or HLG transfer function
// returns a higher value.

struct ImageMetadata;
// TODO(eustas): rename
void SetIntensityTarget(ImageMetadata* m);

}  // namespace jxl

#endif  // LIB_JXL_LUMINANCE_H_
