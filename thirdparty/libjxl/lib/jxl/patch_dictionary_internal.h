// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_PATCH_DICTIONARY_INTERNAL_H_
#define LIB_JXL_PATCH_DICTIONARY_INTERNAL_H_

namespace jxl {

// Context numbers as specified in Section C.4.5, Listing C.2:
enum Contexts {
  kNumRefPatchContext = 0,
  kReferenceFrameContext = 1,
  kPatchSizeContext = 2,
  kPatchReferencePositionContext = 3,
  kPatchPositionContext = 4,
  kPatchBlendModeContext = 5,
  kPatchOffsetContext = 6,
  kPatchCountContext = 7,
  kPatchAlphaChannelContext = 8,
  kPatchClampContext = 9,
  kNumPatchDictionaryContexts
};

}  // namespace jxl

#endif  // LIB_JXL_PATCH_DICTIONARY_INTERNAL_H_
