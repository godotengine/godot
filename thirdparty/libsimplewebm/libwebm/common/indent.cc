/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "common/indent.h"

#include <string>

namespace libwebm {

Indent::Indent(int indent) : indent_(indent), indent_str_() { Update(); }

void Indent::Adjust(int indent) {
  indent_ += indent;
  if (indent_ < 0)
    indent_ = 0;

  Update();
}

void Indent::Update() { indent_str_ = std::string(indent_, ' '); }

}  // namespace libwebm
