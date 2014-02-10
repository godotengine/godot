// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Enhancement layer (for YUV444/422)
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>

#include "./vp8i.h"

//------------------------------------------------------------------------------

int VP8DecodeLayer(VP8Decoder* const dec) {
  assert(dec);
  assert(dec->layer_data_size_ > 0);
  (void)dec;

  // TODO: handle enhancement layer here.

  return 1;
}

