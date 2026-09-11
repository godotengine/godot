// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_MODULAR_H_
#define LIB_JXL_ENC_MODULAR_H_

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

class ModularFrameEncoder {
 public:
  static StatusOr<ModularFrameEncoder> Create(
      JxlMemoryManager* memory_manager, const FrameHeader& frame_header,
      const CompressParams& cparams_orig, bool streaming_mode);
  Status ComputeEncodingData(
      const FrameHeader& frame_header, const ImageMetadata& metadata,
      Image3F* JXL_RESTRICT color, const std::vector<ImageF>& extra_channels,
      const Rect& group_rect, const FrameDimensions& patch_dim,
      const Rect& frame_area_rect, PassesEncoderState* JXL_RESTRICT enc_state,
      const JxlCmsInterface& cms, ThreadPool* pool, AuxOut* aux_out,
      bool do_color);
  Status ComputeTree(ThreadPool* pool);
  Status ComputeTokens(ThreadPool* pool);
  // Encodes global info (tree + histograms) in the `writer`.
  Status EncodeGlobalInfo(bool streaming_mode, BitWriter* writer,
                          AuxOut* aux_out);
  // Encodes a specific modular image (identified by `stream`) in the `writer`,
  // assigning bits to the provided `layer`.
  Status EncodeStream(BitWriter* writer, AuxOut* aux_out, LayerType layer,
                      const ModularStreamId& stream);

  void ClearStreamData(const ModularStreamId& stream);
  void ClearModularStreamData();
  size_t ComputeStreamingAbsoluteAcGroupId(
      size_t dc_group_id, size_t ac_group_id,
      const FrameDimensions& patch_dim) const;

  // Creates a modular image for a given DC group of VarDCT mode. `dc` is the
  // input DC image, not quantized; the group is specified by `group_index`, and
  // `nl_dc` decides whether to apply a near-lossless processing to the DC or
  // not.
  Status AddVarDCTDC(const FrameHeader& frame_header, const Image3F& dc,
                     const Rect& r, size_t group_index, bool nl_dc,
                     PassesEncoderState* enc_state, bool jpeg_transcode);
  // Creates a modular image for the AC metadata of the given group
  // (`group_index`).
  Status AddACMetadata(const Rect& r, size_t group_index, bool jpeg_transcode,
                       PassesEncoderState* enc_state);
  // Encodes a RAW quantization table in `writer`. If `modular_frame_encoder` is
  // null, the quantization table in `encoding` is used, with dimensions `size_x
  // x size_y`. Otherwise, the table with ID `idx` is encoded from the given
  // `modular_frame_encoder`.
  static Status EncodeQuantTable(JxlMemoryManager* memory_manager,
                                 size_t size_x, size_t size_y,
                                 BitWriter* writer,
                                 const QuantEncoding& encoding, size_t idx,
                                 ModularFrameEncoder* modular_frame_encoder);
  // Stores a quantization table for future usage with `EncodeQuantTable`.
  Status AddQuantTable(size_t size_x, size_t size_y,
                       const QuantEncoding& encoding, size_t idx);

  std::vector<size_t> ac_metadata_size;
  std::vector<uint8_t> extra_dc_precision;

  JxlMemoryManager* memory_manager() const { return memory_manager_; }

 private:
  explicit ModularFrameEncoder(JxlMemoryManager* memory_manager);
  Status Init(const FrameHeader& frame_header,
              const CompressParams& cparams_orig, bool streaming_mode);

  Status PrepareStreamParams(const Rect& rect, const CompressParams& cparams,
                             int minShift, int maxShift,
                             const ModularStreamId& stream, bool do_color,
                             bool groupwise);
  JxlMemoryManager* memory_manager_;
  std::vector<Image> stream_images_;
  std::vector<ModularOptions> stream_options_;
  std::vector<uint32_t> quants_;

  Tree tree_;
  std::vector<std::vector<Token>> tree_tokens_;
  std::vector<GroupHeader> stream_headers_;
  std::vector<std::vector<Token>> tokens_;
  EntropyEncodingData code_;
  std::vector<uint8_t> context_map_;
  FrameDimensions frame_dim_;
  CompressParams cparams_;
  std::vector<size_t> tree_splits_;
  std::vector<std::vector<uint32_t>> gi_channel_;
  std::vector<size_t> image_widths_;

  struct GroupParams {
    Rect rect;
    int minShift;
    int maxShift;
    ModularStreamId id;
  };
  std::vector<GroupParams> stream_params_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_MODULAR_H_
