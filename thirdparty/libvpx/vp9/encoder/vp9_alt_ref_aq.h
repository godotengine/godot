/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed  by a BSD-style license that can be
 *  found in the LICENSE file in the root of the source tree. An additional
 *  intellectual property  rights grant can  be found in the  file PATENTS.
 *  All contributing  project authors may be  found in the AUTHORS  file in
 *  the root of the source tree.
 */

/*
 *  \file vp9_alt_ref_aq.h
 *
 *  This file  contains public interface  for setting up  adaptive segmentation
 *  for altref frames.  Go to alt_ref_aq_private.h for implmentation details.
 */

#ifndef VPX_VP9_ENCODER_VP9_ALT_REF_AQ_H_
#define VPX_VP9_ENCODER_VP9_ALT_REF_AQ_H_

#include "vpx/vpx_integer.h"

// Where to disable segmentation
#define ALT_REF_AQ_LOW_BITRATE_BOUNDARY 150

// Last frame always has overall quality = 0,
// so it is questionable if I can process it
#define ALT_REF_AQ_APPLY_TO_LAST_FRAME 1

// If I should try to compare gain
// against segmentation overhead
#define ALT_REF_AQ_PROTECT_GAIN 0

// Threshold to disable segmentation
#define ALT_REF_AQ_PROTECT_GAIN_THRESH 0.5

#ifdef __cplusplus
extern "C" {
#endif

// Simple structure for storing images
struct MATX_8U {
  int rows;
  int cols;
  int stride;

  uint8_t *data;
};

struct VP9_COMP;
struct ALT_REF_AQ;

/*!\brief Constructor
 *
 * \return Instance of the class
 */
struct ALT_REF_AQ *vp9_alt_ref_aq_create(void);

/*!\brief Upload segmentation_map to self object
 *
 * \param    self             Instance of the class
 * \param    segmentation_map Segmentation map to upload
 */
void vp9_alt_ref_aq_upload_map(struct ALT_REF_AQ *const self,
                               const struct MATX_8U *segmentation_map);

/*!\brief Return pointer to the altref segmentation map
 *
 * \param    self                    Instance of the class
 * \param    segmentation_overhead   Segmentation overhead in bytes
 * \param    bandwidth               Current frame bandwidth in bytes
 *
 * \return  Boolean value to disable segmentation
 */
int vp9_alt_ref_aq_disable_if(const struct ALT_REF_AQ *self,
                              int segmentation_overhead, int bandwidth);

/*!\brief Set number of segments
 *
 * It is used for delta quantizer computations
 * and thus it can be larger than
 * maximum value of the segmentation map
 *
 * \param    self        Instance of the class
 * \param    nsegments   Maximum number of segments
 */
void vp9_alt_ref_aq_set_nsegments(struct ALT_REF_AQ *const self, int nsegments);

/*!\brief Set up LOOKAHEAD_AQ segmentation mode
 *
 * Set up segmentation mode to LOOKAHEAD_AQ
 * (expected future frames prediction
 *  quality refering to the current frame).
 *
 * \param    self    Instance of the class
 * \param    cpi     Encoder context
 */
void vp9_alt_ref_aq_setup_mode(struct ALT_REF_AQ *const self,
                               struct VP9_COMP *const cpi);

/*!\brief Set up LOOKAHEAD_AQ segmentation map and delta quantizers
 *
 * \param    self    Instance of the class
 * \param    cpi     Encoder context
 */
void vp9_alt_ref_aq_setup_map(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi);

/*!\brief Restore main segmentation map mode and reset the class variables
 *
 * \param    self    Instance of the class
 * \param    cpi     Encoder context
 */
void vp9_alt_ref_aq_unset_all(struct ALT_REF_AQ *const self,
                              struct VP9_COMP *const cpi);

/*!\brief Destructor
 *
 * \param    self    Instance of the class
 */
void vp9_alt_ref_aq_destroy(struct ALT_REF_AQ *const self);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ALT_REF_AQ_H_
