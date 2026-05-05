/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed  by a BSD-style license that can be
 *  found in the LICENSE file in the root of the source tree. An additional
 *  intellectual property  rights grant can  be found in the  file PATENTS.
 *  All contributing  project authors may be  found in the AUTHORS  file in
 *  the root of the source tree.
 */

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_alt_ref_aq.h"

struct ALT_REF_AQ {
  int dummy;
};

struct ALT_REF_AQ *vp9_alt_ref_aq_create(void) {
  return (struct ALT_REF_AQ *)vpx_malloc(sizeof(struct ALT_REF_AQ));
}

void vp9_alt_ref_aq_destroy(struct ALT_REF_AQ *const self) { vpx_free(self); }

void vp9_alt_ref_aq_upload_map(struct ALT_REF_AQ *const self,
                               const struct MATX_8U *segmentation_map) {
  (void)self;
  (void)segmentation_map;
}

void vp9_alt_ref_aq_set_nsegments(struct ALT_REF_AQ *const self,
                                  int nsegments) {
  (void)self;
  (void)nsegments;
}

void vp9_alt_ref_aq_setup_mode(struct ALT_REF_AQ *const self,
                               struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

// set basic segmentation to the altref's one
void vp9_alt_ref_aq_setup_map(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

// restore cpi->aq_mode
void vp9_alt_ref_aq_unset_all(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi) {
  (void)cpi;
  (void)self;
}

int vp9_alt_ref_aq_disable_if(const struct ALT_REF_AQ *self,
                              int segmentation_overhead, int bandwidth) {
  (void)bandwidth;
  (void)self;
  (void)segmentation_overhead;

  return 0;
}
