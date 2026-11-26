// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_fields.h"

#include <cinttypes>  // PRIu64
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/quantizer.h"

namespace jxl {

namespace {
using ::jxl::fields_internal::VisitorBase;
class WriteVisitor : public VisitorBase {
 public:
  WriteVisitor(const size_t extension_bits, BitWriter* JXL_RESTRICT writer)
      : extension_bits_(extension_bits), writer_(writer) {}

  Status Bits(const size_t bits, const uint32_t /*default_value*/,
              uint32_t* JXL_RESTRICT value) override {
    ok_ &= BitsCoder::Write(bits, *value, writer_);
    return true;
  }
  Status U32(const U32Enc enc, const uint32_t /*default_value*/,
             uint32_t* JXL_RESTRICT value) override {
    ok_ &= U32Coder::Write(enc, *value, writer_);
    return true;
  }

  Status U64(const uint64_t /*default_value*/,
             uint64_t* JXL_RESTRICT value) override {
    ok_ &= U64Coder::Write(*value, writer_);
    return true;
  }

  Status F16(const float /*default_value*/,
             float* JXL_RESTRICT value) override {
    ok_ &= F16Coder::Write(*value, writer_);
    return true;
  }

  Status BeginExtensions(uint64_t* JXL_RESTRICT extensions) override {
    JXL_QUIET_RETURN_IF_ERROR(VisitorBase::BeginExtensions(extensions));
    if (*extensions == 0) {
      JXL_ENSURE(extension_bits_ == 0);
      return true;
    }
    // TODO(janwas): extend API to pass in array of extension_bits, one per
    // extension. We currently ascribe all bits to the first extension, but
    // this is only an encoder limitation. NOTE: extension_bits_ can be zero
    // if an extension does not require any additional fields.
    ok_ &= U64Coder::Write(extension_bits_, writer_);
    // For each nonzero bit except the lowest/first (already written):
    for (uint64_t remaining_extensions = *extensions & (*extensions - 1);
         remaining_extensions != 0;
         remaining_extensions &= remaining_extensions - 1) {
      ok_ &= U64Coder::Write(0, writer_);
    }
    return true;
  }
  // EndExtensions = default.

  Status OK() const { return ok_; }

 private:
  const size_t extension_bits_;
  BitWriter* JXL_RESTRICT writer_;
  bool ok_ = true;
};
}  // namespace

Status Bundle::Write(const Fields& fields, BitWriter* writer, LayerType layer,
                     AuxOut* aux_out) {
  size_t extension_bits;
  size_t total_bits;
  JXL_RETURN_IF_ERROR(Bundle::CanEncode(fields, &extension_bits, &total_bits));

  return writer->WithMaxBits(total_bits, layer, aux_out, [&] {
    WriteVisitor visitor(extension_bits, writer);
    JXL_RETURN_IF_ERROR(visitor.VisitConst(fields));
    return visitor.OK();
  });
}

// Returns false if the value is too large to encode.
Status BitsCoder::Write(const size_t bits, const uint32_t value,
                        BitWriter* JXL_RESTRICT writer) {
  if (value >= (1ULL << bits)) {
    return JXL_FAILURE("Value %d too large to encode in %" PRIu64 " bits",
                       value, static_cast<uint64_t>(bits));
  }
  writer->Write(bits, value);
  return true;
}

// Returns false if the value is too large to encode.
Status U32Coder::Write(const U32Enc enc, const uint32_t value,
                       BitWriter* JXL_RESTRICT writer) {
  uint32_t selector;
  size_t total_bits;
  JXL_RETURN_IF_ERROR(ChooseSelector(enc, value, &selector, &total_bits));

  writer->Write(2, selector);

  const U32Distr d = enc.GetDistr(selector);
  if (!d.IsDirect()) {  // Nothing more to write for direct encoding
    const uint32_t offset = d.Offset();
    JXL_ENSURE(value >= offset);
    writer->Write(total_bits - 2, value - offset);
  }

  return true;
}

// Returns false if the value is too large to encode.
Status U64Coder::Write(uint64_t value, BitWriter* JXL_RESTRICT writer) {
  if (value == 0) {
    // Selector: use 0 bits, value 0
    writer->Write(2, 0);
  } else if (value <= 16) {
    // Selector: use 4 bits, value 1..16
    writer->Write(2, 1);
    writer->Write(4, value - 1);
  } else if (value <= 272) {
    // Selector: use 8 bits, value 17..272
    writer->Write(2, 2);
    writer->Write(8, value - 17);
  } else {
    // Selector: varint, first a 12-bit group, after that per 8-bit group.
    writer->Write(2, 3);
    writer->Write(12, value & 4095);
    value >>= 12;
    int shift = 12;
    while (value > 0 && shift < 60) {
      // Indicate varint not done
      writer->Write(1, 1);
      writer->Write(8, value & 255);
      value >>= 8;
      shift += 8;
    }
    if (value > 0) {
      // This only could happen if shift == N - 4.
      writer->Write(1, 1);
      writer->Write(4, value & 15);
      // Implicitly closed sequence, no extra stop bit is required.
    } else {
      // Indicate end of varint
      writer->Write(1, 0);
    }
  }

  return true;
}

Status F16Coder::Write(float value, BitWriter* JXL_RESTRICT writer) {
  uint32_t bits32;
  memcpy(&bits32, &value, sizeof(bits32));
  const uint32_t sign = bits32 >> 31;
  const uint32_t biased_exp32 = (bits32 >> 23) & 0xFF;
  const uint32_t mantissa32 = bits32 & 0x7FFFFF;

  const int32_t exp = static_cast<int32_t>(biased_exp32) - 127;
  if (JXL_UNLIKELY(exp > 15)) {
    return JXL_FAILURE("Too big to encode, CanEncode should return false");
  }

  // Tiny or zero => zero.
  if (exp < -24) {
    writer->Write(16, 0);
    return true;
  }

  uint32_t biased_exp16;
  uint32_t mantissa16;

  // exp = [-24, -15] => subnormal
  if (JXL_UNLIKELY(exp < -14)) {
    biased_exp16 = 0;
    const uint32_t sub_exp = static_cast<uint32_t>(-14 - exp);
    JXL_ENSURE(1 <= sub_exp && sub_exp < 11);
    mantissa16 = (1 << (10 - sub_exp)) + (mantissa32 >> (13 + sub_exp));
  } else {
    // exp = [-14, 15]
    biased_exp16 = static_cast<uint32_t>(exp + 15);
    JXL_ENSURE(1 <= biased_exp16 && biased_exp16 < 31);
    mantissa16 = mantissa32 >> 13;
  }

  JXL_ENSURE(mantissa16 < 1024);
  const uint32_t bits16 = (sign << 15) | (biased_exp16 << 10) | mantissa16;
  JXL_ENSURE(bits16 < 0x10000);
  writer->Write(16, bits16);
  return true;
}

Status WriteCodestreamHeaders(CodecMetadata* metadata, BitWriter* writer,
                              AuxOut* aux_out) {
  // Marker/signature
  JXL_RETURN_IF_ERROR(writer->WithMaxBits(16, LayerType::Header, aux_out, [&] {
    writer->Write(8, 0xFF);
    writer->Write(8, kCodestreamMarker);
    return true;
  }));

  JXL_RETURN_IF_ERROR(
      WriteSizeHeader(metadata->size, writer, LayerType::Header, aux_out));

  JXL_RETURN_IF_ERROR(
      WriteImageMetadata(metadata->m, writer, LayerType::Header, aux_out));

  metadata->transform_data.nonserialized_xyb_encoded = metadata->m.xyb_encoded;
  JXL_RETURN_IF_ERROR(Bundle::Write(metadata->transform_data, writer,
                                    LayerType::Header, aux_out));

  return true;
}

Status WriteFrameHeader(const FrameHeader& frame,
                        BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return Bundle::Write(frame, writer, LayerType::Header, aux_out);
}

Status WriteImageMetadata(const ImageMetadata& metadata,
                          BitWriter* JXL_RESTRICT writer, LayerType layer,
                          AuxOut* aux_out) {
  return Bundle::Write(metadata, writer, layer, aux_out);
}

Status WriteQuantizerParams(const QuantizerParams& params,
                            BitWriter* JXL_RESTRICT writer, LayerType layer,
                            AuxOut* aux_out) {
  return Bundle::Write(params, writer, layer, aux_out);
}

Status WriteSizeHeader(const SizeHeader& size, BitWriter* JXL_RESTRICT writer,
                       LayerType layer, AuxOut* aux_out) {
  return Bundle::Write(size, writer, layer, aux_out);
}

}  // namespace jxl
