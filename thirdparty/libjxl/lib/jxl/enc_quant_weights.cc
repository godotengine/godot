// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_quant_weights.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstdlib>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

namespace {

Status EncodeDctParams(const DctQuantWeightParams& params, BitWriter* writer) {
  JXL_ENSURE(params.num_distance_bands >= 1);
  writer->Write(DctQuantWeightParams::kLog2MaxDistanceBands,
                params.num_distance_bands - 1);
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_distance_bands; i++) {
      JXL_RETURN_IF_ERROR(F16Coder::Write(
          params.distance_bands[c][i] * (i == 0 ? (1 / 64.0f) : 1.0f), writer));
    }
  }
  return true;
}

Status EncodeQuant(JxlMemoryManager* memory_manager,
                   const QuantEncoding& encoding, size_t idx, size_t size_x,
                   size_t size_y, BitWriter* writer,
                   ModularFrameEncoder* modular_frame_encoder) {
  writer->Write(kLog2NumQuantModes, encoding.mode);
  size_x *= kBlockDim;
  size_y *= kBlockDim;
  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary: {
      writer->Write(kCeilLog2NumPredefinedTables, encoding.predefined);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          JXL_RETURN_IF_ERROR(
              F16Coder::Write(encoding.idweights[c][i] * (1.0f / 64), writer));
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Write(
              encoding.dct2weights[c][i] * (1.0f / 64), writer));
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4X8: {
      for (size_t c = 0; c < 3; c++) {
        JXL_RETURN_IF_ERROR(
            F16Coder::Write(encoding.dct4x8multipliers[c], writer));
      }
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          JXL_RETURN_IF_ERROR(
              F16Coder::Write(encoding.dct4multipliers[c][i], writer));
        }
      }
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeRAW: {
      JXL_RETURN_IF_ERROR(ModularFrameEncoder::EncodeQuantTable(
          memory_manager, size_x, size_y, writer, encoding, idx,
          modular_frame_encoder));
      break;
    }
    case QuantEncoding::kQuantModeAFV: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 9; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Write(
              encoding.afv_weights[c][i] * (i < 6 ? 1.0f / 64 : 1.0f), writer));
        }
      }
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params_afv_4x4, writer));
      break;
    }
  }
  return true;
}

}  // namespace

Status DequantMatricesEncode(JxlMemoryManager* memory_manager,
                             const DequantMatrices& matrices, BitWriter* writer,
                             LayerType layer, AuxOut* aux_out,
                             ModularFrameEncoder* modular_frame_encoder) {
  bool all_default = true;
  const std::vector<QuantEncoding>& encodings = matrices.encodings();

  for (const auto& encoding : encodings) {
    if (encoding.mode != QuantEncoding::kQuantModeLibrary ||
        encoding.predefined != 0) {
      all_default = false;
    }
  }
  // TODO(janwas): better bound
  return writer->WithMaxBits(512 * 1024, layer, aux_out, [&]() -> Status {
    writer->Write(1, TO_JXL_BOOL(all_default));
    if (!all_default) {
      for (size_t i = 0; i < encodings.size(); i++) {
        JXL_RETURN_IF_ERROR(EncodeQuant(memory_manager, encodings[i], i,
                                        DequantMatrices::required_size_x[i],
                                        DequantMatrices::required_size_y[i],
                                        writer, modular_frame_encoder));
      }
    }
    return true;
  });
}

Status DequantMatricesEncodeDC(const DequantMatrices& matrices,
                               BitWriter* writer, LayerType layer,
                               AuxOut* aux_out) {
  bool all_default = true;
  const float* dc_quant = matrices.DCQuants();
  for (size_t c = 0; c < 3; c++) {
    if (dc_quant[c] != kDCQuant[c]) {
      all_default = false;
    }
  }
  return writer->WithMaxBits(
      1 + sizeof(float) * kBitsPerByte * 3, layer, aux_out, [&]() -> Status {
        writer->Write(1, TO_JXL_BOOL(all_default));
        if (!all_default) {
          for (size_t c = 0; c < 3; c++) {
            JXL_RETURN_IF_ERROR(F16Coder::Write(dc_quant[c] * 128.0f, writer));
          }
        }
        return true;
      });
}

Status DequantMatricesSetCustomDC(JxlMemoryManager* memory_manager,
                                  DequantMatrices* matrices, const float* dc) {
  matrices->SetDCQuant(dc);
  // Roundtrip encode/decode DC to ensure same values as decoder.
  BitWriter writer{memory_manager};
  // TODO(eustas): should it be LayerType::Quant?
  JXL_RETURN_IF_ERROR(
      DequantMatricesEncodeDC(*matrices, &writer, LayerType::Header, nullptr));
  writer.ZeroPadToByte();
  BitReader br(writer.GetSpan());
  // Called only in the encoder: should fail only for programmer errors.
  JXL_RETURN_IF_ERROR(matrices->DecodeDC(&br));
  JXL_RETURN_IF_ERROR(br.Close());
  return true;
}

Status DequantMatricesScaleDC(JxlMemoryManager* memory_manager,
                              DequantMatrices* matrices, const float scale) {
  float dc[3];
  for (size_t c = 0; c < 3; ++c) {
    dc[c] = matrices->InvDCQuant(c) * (1.0f / scale);
  }
  JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(memory_manager, matrices, dc));
  return true;
}

Status DequantMatricesRoundtrip(JxlMemoryManager* memory_manager,
                                DequantMatrices* matrices) {
  // Do not pass modular en/decoder, as they only change entropy and not
  // values.
  BitWriter writer{memory_manager};
  // TODO(eustas): should it be LayerType::Quant?
  JXL_RETURN_IF_ERROR(DequantMatricesEncode(memory_manager, *matrices, &writer,
                                            LayerType::Header, nullptr));
  writer.ZeroPadToByte();
  BitReader br(writer.GetSpan());
  // Called only in the encoder: should fail only for programmer errors.
  JXL_RETURN_IF_ERROR(matrices->Decode(memory_manager, &br));
  JXL_RETURN_IF_ERROR(br.Close());
  return true;
}

Status DequantMatricesSetCustom(DequantMatrices* matrices,
                                const std::vector<QuantEncoding>& encodings,
                                ModularFrameEncoder* encoder) {
  JXL_ENSURE(encoder != nullptr);
  JXL_ENSURE(encodings.size() == kNumQuantTables);
  JxlMemoryManager* memory_manager = encoder->memory_manager();
  matrices->SetEncodings(encodings);
  for (size_t i = 0; i < encodings.size(); i++) {
    if (encodings[i].mode == QuantEncodingInternal::kQuantModeRAW) {
      JXL_RETURN_IF_ERROR(encoder->AddQuantTable(
          DequantMatrices::required_size_x[i] * kBlockDim,
          DequantMatrices::required_size_y[i] * kBlockDim, encodings[i], i));
    }
  }
  JXL_RETURN_IF_ERROR(DequantMatricesRoundtrip(memory_manager, matrices));
  return true;
}

}  // namespace jxl
