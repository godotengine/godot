// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Enhancement layer (for YUV444/422)
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>

#include "./vp8i.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------

int VP8DecodeLayer(VP8Decoder* const dec) {
  assert(dec);
  assert(dec->layer_data_size_ > 0);
  (void)dec;

  // TODO: handle enhancement layer here.

  return 1;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
