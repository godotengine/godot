// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_WEBMIDS_HPP_
#define LIBWEBM_WEBMIDS_HPP_

// This file is a wrapper for the file included immediately after this comment.
// New projects should not include this file: include the file included below.
#include "common/webmids.h"

namespace mkvmuxer {
// MkvId moved from the mkvmuxer namespace to the libwebm namespace. Pull all
// of libwebm into mkvmuxer to ease transition to the new namespace. New
// projects should use libwebm::MkvId and should not expect to find MkvId in
// mkvmuxer.
using namespace libwebm;
}  // namespace mkvmuxer

#endif  // LIBWEBM_WEBMIDS_HPP_
