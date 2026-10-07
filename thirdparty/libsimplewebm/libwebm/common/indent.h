/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef LIBWEBM_COMMON_INDENT_H_
#define LIBWEBM_COMMON_INDENT_H_

#include <string>

#include "mkvmuxer/mkvmuxertypes.h"

namespace libwebm {

const int kIncreaseIndent = 2;
const int kDecreaseIndent = -2;

// Used for formatting output so objects only have to keep track of spacing
// within their scope.
class Indent {
 public:
  explicit Indent(int indent);

  // Changes the number of spaces output. The value adjusted is relative to
  // |indent_|.
  void Adjust(int indent);

  std::string indent_str() const { return indent_str_; }

 private:
  // Called after |indent_| is changed. This will set |indent_str_| to the
  // proper number of spaces.
  void Update();

  int indent_;
  std::string indent_str_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Indent);
};

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_INDENT_H_
