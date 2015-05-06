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

#include <stdlib.h>

#include "./vp8enci.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------

void VP8EncInitLayer(VP8Encoder* const enc) {
  enc->use_layer_ = (enc->pic_->u0 != NULL);
  enc->layer_data_size_ = 0;
  enc->layer_data_ = NULL;
  if (enc->use_layer_) {
    VP8BitWriterInit(&enc->layer_bw_, enc->mb_w_ * enc->mb_h_ * 3);
  }
}

void VP8EncCodeLayerBlock(VP8EncIterator* it) {
  (void)it;   // remove a warning
}

int VP8EncFinishLayer(VP8Encoder* const enc) {
  if (enc->use_layer_) {
    enc->layer_data_ = VP8BitWriterFinish(&enc->layer_bw_);
    enc->layer_data_size_ = VP8BitWriterSize(&enc->layer_bw_);
  }
  return 1;
}

void VP8EncDeleteLayer(VP8Encoder* enc) {
  free(enc->layer_data_);
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
