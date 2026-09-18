// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_FRAME_H_
#define LIB_JXL_ENC_FRAME_H_

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

struct AuxOut;

// Information needed for encoding a frame that is not contained elsewhere and
// does not belong to `cparams`.
// TODO(lode): if possible, it might be better to replace FrameInfo and several
// fields from ImageBundle (such as frame name and duration) by direct usage of
// jxl::FrameHeader itself.
struct FrameInfo {
  // TODO(veluca): consider adding more parameters, such as custom patches.
  bool save_before_color_transform = false;
  // Whether or not the input image bundle is already in the codestream
  // colorspace (as deduced by cparams).
  // TODO(veluca): this is a hack - ImageBundle doesn't have a simple way to say
  // "this is already in XYB".
  bool ib_needs_color_transform = true;
  FrameType frame_type = FrameType::kRegularFrame;
  size_t dc_level = 0;
  // Only used for kRegularFrame.
  bool is_last = true;
  bool is_preview = false;
  // Information for storing this frame for future use (only for non-DC frames).
  size_t save_as_reference = 0;
  // The source frame for blending of a next frame, matching the
  // save_as_reference value of a previous frame. Animated frames can use
  // save_as_reference values 1, 2 and 3, while composite still frames can use
  // save_as_reference values 0, 1, 2 and 3. The current C++ encoder
  // implementation is assuming and using 1 for all frames of animations, so
  // using that as the default value here.
  // Corresponds to BlendingInfo::source from the FrameHeader.
  size_t source = 1;
  // Corresponds to BlendingInfo::clamp from the FrameHeader.
  bool clamp = true;
  // Corresponds to BlendingInfo::alpha_channel from the FrameHeader, or set to
  // -1 to automatically choose it as the index of the first extra channel of
  // type alpha.
  int alpha_channel = -1;

  FrameOrigin origin{0, 0};

  bool blend = false;
  BlendMode blendmode = BlendMode::kBlend;

  JxlBitDepth image_bit_depth = {};

  // Animation-related information, corresponding to the timecode and duration
  // fields of the jxl::AnimationFrame of the jxl::FrameHeader.
  uint32_t duration = 0;
  uint32_t timecode = 0;

  std::string name;

  // If non-empty, uses this blending info for the extra channels, otherwise
  // automatically chooses it. The encoder API will fill this vector with the
  // extra channel info and allows more options. The non-API cjxl leaves it
  // empty and relies on the default behavior.
  std::vector<BlendingInfo> extra_channel_blending_info;
};

// Checks and adjusts CompressParams when they are all initialized.
Status ParamsPostInit(CompressParams* p);

// Encodes a single frame (including its header) into a byte stream.  Groups may
// be processed in parallel by `pool`. metadata is the ImageMetadata encoded in
// the codestream, and must be used for the FrameHeaders, do not use
// ib.metadata.
Status EncodeFrame(JxlMemoryManager* memory_manager,
                   const CompressParams& cparams_orig,
                   const FrameInfo& frame_info, const CodecMetadata* metadata,
                   JxlEncoderChunkedFrameAdapter& frame_data,
                   const JxlCmsInterface& cms, ThreadPool* pool,
                   JxlEncoderOutputProcessorWrapper* output_processor,
                   AuxOut* aux_out);

Status EncodeFrame(JxlMemoryManager* memory_manager,
                   const CompressParams& cparams_orig,
                   const FrameInfo& frame_info, const CodecMetadata* metadata,
                   ImageBundle& ib, const JxlCmsInterface& cms,
                   ThreadPool* pool, BitWriter* writer, AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_FRAME_H_
