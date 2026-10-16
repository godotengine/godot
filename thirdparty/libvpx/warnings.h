/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_WARNINGS_H_
#define VPX_WARNINGS_H_

#ifdef __cplusplus
extern "C" {
#endif

struct vpx_codec_enc_cfg;
struct VpxEncoderConfig;

/*
 * Checks config for improperly used settings. Warns user upon encountering
 * settings that will lead to poor output quality. Prompts user to continue
 * when warnings are issued.
 */
void check_encoder_config(int disable_prompt,
                          const struct VpxEncoderConfig *global_config,
                          const struct vpx_codec_enc_cfg *stream_config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_WARNINGS_H_
