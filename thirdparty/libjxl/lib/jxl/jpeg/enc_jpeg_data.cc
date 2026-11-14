// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/enc_jpeg_data.h"

#if JPEGXL_ENABLE_TRANSCODE_JPEG
#include <brotli/encode.h>
#endif
#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/jpeg/enc_jpeg_data_reader.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {
namespace jpeg {

namespace {

// TODO(eustas): move to jpeg_data, to use from codec_jpg as well.
// See if there is a canonically chunked ICC profile and mark corresponding
// app-tags with AppMarkerType::kICC.
Status DetectIccProfile(JPEGData& jpeg_data) {
  JXL_ENSURE(jpeg_data.app_data.size() == jpeg_data.app_marker_type.size());
  size_t num_icc = 0;
  size_t num_icc_jpeg = 0;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    const auto& app = jpeg_data.app_data[i];
    // At least APPn + size; otherwise it should be intermarker-data.
    JXL_ENSURE(app.size() >= 3);
    size_t pos = 0;
    if (app[pos++] != 0xE2) continue;
    size_t tag_length = (app[pos] << 8) + app[pos + 1];
    pos += 2;
    JXL_ENSURE(app.size() == tag_length + 1);
    // Minimum is 2 bytes for tag length itself + signature + 2 bytes for
    // chunk_id and num_chunks (read below).
    if (tag_length < 2 + sizeof kIccProfileTag + 2) continue;

    if (memcmp(&app[pos], kIccProfileTag, sizeof kIccProfileTag) != 0) continue;
    pos += sizeof kIccProfileTag;
    uint8_t chunk_id = app[pos++];
    uint8_t num_chunks = app[pos++];
    if (chunk_id != num_icc + 1) continue;
    if (num_icc_jpeg == 0) num_icc_jpeg = num_chunks;
    if (num_icc_jpeg != num_chunks) continue;
    num_icc++;
    jpeg_data.app_marker_type[i] = AppMarkerType::kICC;
  }
  if (num_icc != num_icc_jpeg) {
    return JXL_FAILURE("Invalid ICC chunks");
  }
  return true;
}

bool GetMarkerPayload(const uint8_t* data, size_t size, Bytes* payload) {
  if (size < 3) {
    return false;
  }
  size_t hi = data[1];
  size_t lo = data[2];
  size_t internal_size = (hi << 8u) | lo;
  // Second byte of marker is not counted towards size.
  if (internal_size != size - 1) {
    return false;
  }
  // cut second marker byte and "length" from payload.
  *payload = Bytes(data, size);
  if (!payload->remove_prefix(3)) return false;
  return true;
}

Status DetectBlobs(jpeg::JPEGData& jpeg_data) {
  JXL_ENSURE(jpeg_data.app_data.size() == jpeg_data.app_marker_type.size());
  bool have_exif = false;
  bool have_xmp = false;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    auto& marker = jpeg_data.app_data[i];
    if (marker.empty() || marker[0] != kApp1) {
      continue;
    }
    Bytes payload;
    if (!GetMarkerPayload(marker.data(), marker.size(), &payload)) {
      // Something is wrong with this marker; does not care.
      continue;
    }
    if (!have_exif && payload.size() > sizeof kExifTag &&
        !memcmp(payload.data(), kExifTag, sizeof kExifTag)) {
      jpeg_data.app_marker_type[i] = AppMarkerType::kExif;
      have_exif = true;
    }
    if (!have_xmp && payload.size() >= sizeof kXMPTag &&
        !memcmp(payload.data(), kXMPTag, sizeof kXMPTag)) {
      jpeg_data.app_marker_type[i] = AppMarkerType::kXMP;
      have_xmp = true;
    }
  }
  return true;
}

Status ParseChunkedMarker(const jpeg::JPEGData& src, uint8_t marker_type,
                          const Bytes& tag, IccBytes* output,
                          bool allow_permutations = false) {
  output->clear();

  std::vector<Bytes> chunks;
  std::vector<bool> presence;
  size_t expected_number_of_parts = 0;
  bool is_first_chunk = true;
  size_t ordinal = 0;
  for (const auto& marker : src.app_data) {
    if (marker.empty() || marker[0] != marker_type) {
      continue;
    }
    Bytes payload;
    if (!GetMarkerPayload(marker.data(), marker.size(), &payload)) {
      // Something is wrong with this marker; does not care.
      continue;
    }
    if ((payload.size() < tag.size()) ||
        memcmp(payload.data(), tag.data(), tag.size()) != 0) {
      continue;
    }
    JXL_RETURN_IF_ERROR(payload.remove_prefix(tag.size()));
    if (payload.size() < 2) {
      return JXL_FAILURE("Chunk is too small.");
    }
    uint8_t index = payload[0];
    uint8_t total = payload[1];
    ordinal++;
    if (!allow_permutations) {
      if (index != ordinal) return JXL_FAILURE("Invalid chunk order.");
    }

    JXL_RETURN_IF_ERROR(payload.remove_prefix(2));

    JXL_RETURN_IF_ERROR(total != 0);
    if (is_first_chunk) {
      is_first_chunk = false;
      expected_number_of_parts = total;
      // 1-based indices; 0-th element is added for convenience.
      chunks.resize(total + 1);
      presence.resize(total + 1);
    } else {
      JXL_RETURN_IF_ERROR(expected_number_of_parts == total);
    }

    if (index == 0 || index > total) {
      return JXL_FAILURE("Invalid chunk index.");
    }

    if (presence[index]) {
      return JXL_FAILURE("Duplicate chunk.");
    }
    presence[index] = true;
    chunks[index] = payload;
  }

  for (size_t i = 0; i < expected_number_of_parts; ++i) {
    // 0-th element is not used.
    size_t index = i + 1;
    if (!presence[index]) {
      return JXL_FAILURE("Missing chunk.");
    }
    chunks[index].AppendTo(*output);
  }

  return true;
}

inline bool IsJPG(const Bytes bytes) {
  return bytes.size() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8;
}

}  // namespace

Status SetColorEncodingFromJpegData(const jpeg::JPEGData& jpg,
                                    ColorEncoding* color_encoding) {
  IccBytes icc_profile;
  if (!ParseChunkedMarker(jpg, kApp2, Bytes(kIccProfileTag), &icc_profile)) {
    JXL_WARNING("ReJPEG: corrupted ICC profile\n");
    icc_profile.clear();
  }

  if (icc_profile.empty()) {
    bool is_gray = (jpg.components.size() == 1);
    *color_encoding = ColorEncoding::SRGB(is_gray);
  } else {
    JXL_RETURN_IF_ERROR(
        color_encoding->SetICC(std::move(icc_profile), JxlGetDefaultCms()));
  }
  return true;
}

Status SetChromaSubsamplingFromJpegData(const JPEGData& jpg,
                                        YCbCrChromaSubsampling* cs) {
  size_t nbcomp = jpg.components.size();
  if (nbcomp != 1 && nbcomp != 3) {
    return JXL_FAILURE("Cannot recompress JPEGs with neither 1 nor 3 channels");
  }
  if (nbcomp == 3) {
    uint8_t hsample[3];
    uint8_t vsample[3];
    for (size_t i = 0; i < nbcomp; i++) {
      hsample[i] = jpg.components[i].h_samp_factor;
      vsample[i] = jpg.components[i].v_samp_factor;
    }
    JXL_RETURN_IF_ERROR(cs->Set(hsample, vsample));
  } else if (nbcomp == 1) {
    uint8_t hsample[3];
    uint8_t vsample[3];
    for (size_t i = 0; i < 3; i++) {
      hsample[i] = jpg.components[0].h_samp_factor;
      vsample[i] = jpg.components[0].v_samp_factor;
    }
    JXL_RETURN_IF_ERROR(cs->Set(hsample, vsample));
  }
  return true;
}

Status SetColorTransformFromJpegData(const JPEGData& jpg,
                                     ColorTransform* color_transform) {
  size_t nbcomp = jpg.components.size();
  if (nbcomp != 1 && nbcomp != 3) {
    return JXL_UNSUPPORTED(
        "Cannot recompress JPEGs with neither 1 nor 3 channels");
  }
  bool is_rgb = false;
  {
    const auto& markers = jpg.marker_order;
    // If there is a JFIF marker, this is YCbCr. Otherwise...
    if (std::find(markers.begin(), markers.end(), 0xE0) == markers.end()) {
      // Try to find an 'Adobe' marker.
      size_t app_markers = 0;
      size_t i = 0;
      for (; i < markers.size(); i++) {
        // This is an APP marker.
        if ((markers[i] & 0xF0) == 0xE0) {
          JXL_ENSURE(app_markers < jpg.app_data.size());
          // APP14 marker
          if (markers[i] == 0xEE) {
            const auto& data = jpg.app_data[app_markers];
            if (data.size() == 15 && data[3] == 'A' && data[4] == 'd' &&
                data[5] == 'o' && data[6] == 'b' && data[7] == 'e') {
              // 'Adobe' marker.
              is_rgb = data[14] == 0;
              break;
            }
          }
          app_markers++;
        }
      }

      if (i == markers.size()) {
        // No 'Adobe' marker, guess from component IDs.
        is_rgb = nbcomp == 3 && jpg.components[0].id == 'R' &&
                 jpg.components[1].id == 'G' && jpg.components[2].id == 'B';
      }
    }
  }
  *color_transform =
      (!is_rgb || nbcomp == 1) ? ColorTransform::kYCbCr : ColorTransform::kNone;
  return true;
}

Status EncodeJPEGData(JxlMemoryManager* memory_manager, JPEGData& jpeg_data,
                      std::vector<uint8_t>* bytes,
                      const CompressParams& cparams) {
#if !JPEGXL_ENABLE_TRANSCODE_JPEG
  return JXL_FAILURE("JPEG transcoding is not available");
#else
  bytes->clear();
  jpeg_data.app_marker_type.resize(jpeg_data.app_data.size(),
                                   AppMarkerType::kUnknown);
  JXL_RETURN_IF_ERROR(DetectIccProfile(jpeg_data));
  JXL_RETURN_IF_ERROR(DetectBlobs(jpeg_data));

  size_t total_data = 0;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    if (jpeg_data.app_marker_type[i] != AppMarkerType::kUnknown) {
      continue;
    }
    total_data += jpeg_data.app_data[i].size();
  }
  for (const auto& data : jpeg_data.com_data) {
    total_data += data.size();
  }
  for (const auto& data : jpeg_data.inter_marker_data) {
    total_data += data.size();
  }
  total_data += jpeg_data.tail_data.size();
  size_t brotli_capacity = BrotliEncoderMaxCompressedSize(total_data);

  BitWriter writer{memory_manager};
  JXL_RETURN_IF_ERROR(
      Bundle::Write(jpeg_data, &writer, LayerType::Header, nullptr));
  writer.ZeroPadToByte();
  {
    PaddedBytes serialized_jpeg_data = std::move(writer).TakeBytes();
    bytes->reserve(serialized_jpeg_data.size() + brotli_capacity);
    Bytes(serialized_jpeg_data).AppendTo(*bytes);
  }

  BrotliEncoderState* brotli_enc =
      BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
  int effort = cparams.brotli_effort;
  if (effort < 0) effort = 11 - static_cast<int>(cparams.speed_tier);
  BrotliEncoderSetParameter(brotli_enc, BROTLI_PARAM_QUALITY, effort);
  size_t initial_size = bytes->size();
  BrotliEncoderSetParameter(brotli_enc, BROTLI_PARAM_SIZE_HINT, total_data);
  bytes->resize(initial_size + brotli_capacity);
  size_t enc_size = 0;
  auto br_append = [&](const std::vector<uint8_t>& data, bool last) -> Status {
    size_t available_in = data.size();
    const uint8_t* in = data.data();
    uint8_t* out = &(*bytes)[initial_size + enc_size];
    do {
      uint8_t* out_before = out;
      msan::MemoryIsInitialized(in, available_in);
      JXL_ENSURE(BrotliEncoderCompressStream(
          brotli_enc, last ? BROTLI_OPERATION_FINISH : BROTLI_OPERATION_PROCESS,
          &available_in, &in, &brotli_capacity, &out, &enc_size));
      msan::UnpoisonMemory(out_before, out - out_before);
    } while (FROM_JXL_BOOL(BrotliEncoderHasMoreOutput(brotli_enc)) ||
             available_in > 0);
    return true;
  };

  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    if (jpeg_data.app_marker_type[i] != AppMarkerType::kUnknown) {
      continue;
    }
    JXL_RETURN_IF_ERROR(br_append(jpeg_data.app_data[i], /*last=*/false));
  }
  for (const auto& data : jpeg_data.com_data) {
    JXL_RETURN_IF_ERROR(br_append(data, /*last=*/false));
  }
  for (const auto& data : jpeg_data.inter_marker_data) {
    JXL_RETURN_IF_ERROR(br_append(data, /*last=*/false));
  }
  JXL_RETURN_IF_ERROR(br_append(jpeg_data.tail_data, /*last=*/true));
  BrotliEncoderDestroyInstance(brotli_enc);
  bytes->resize(initial_size + enc_size);
  return true;
#endif
}

StatusOr<std::unique_ptr<JPEGData>> ParseJPG(JxlMemoryManager* memory_manager,
                                             const Bytes bytes) {
  if (!IsJPG(bytes)) return JXL_FAILURE("Not JPEG");
  auto jpeg_data = jxl::make_unique<jxl::jpeg::JPEGData>();
  JXL_RETURN_IF_ERROR(jpeg::ReadJpeg(bytes.data(), bytes.size(),
                                     jpeg::JpegReadMode::kReadAll,
                                     jpeg_data.get()));
  return jpeg_data;
}

Status SetBlobsFromJpegData(const jpeg::JPEGData& jpeg_data, Blobs* blobs) {
  for (const auto& marker : jpeg_data.app_data) {
    if (marker.empty() || marker[0] != kApp1) {
      continue;
    }
    Bytes payload;
    if (!GetMarkerPayload(marker.data(), marker.size(), &payload)) {
      // Something is wrong with this marker; does not care.
      continue;
    }
    if (payload.size() >= sizeof kExifTag &&
        !memcmp(payload.data(), kExifTag, sizeof kExifTag)) {
      if (blobs->exif.empty()) {
        blobs->exif.resize(payload.size() - sizeof kExifTag);
        memcpy(blobs->exif.data(), payload.data() + sizeof kExifTag,
               payload.size() - sizeof kExifTag);
      } else {
        JXL_WARNING(
            "ReJPEG: multiple Exif blobs, storing only first one in the JPEG "
            "XL container\n");
      }
    }
    if (payload.size() >= sizeof kXMPTag &&
        !memcmp(payload.data(), kXMPTag, sizeof kXMPTag)) {
      if (blobs->xmp.empty()) {
        blobs->xmp.resize(payload.size() - sizeof kXMPTag);
        memcpy(blobs->xmp.data(), payload.data() + sizeof kXMPTag,
               payload.size() - sizeof kXMPTag);
      } else {
        JXL_WARNING(
            "ReJPEG: multiple XMP blobs, storing only first one in the JPEG "
            "XL container\n");
      }
    }
  }
  return true;
}

}  // namespace jpeg
}  // namespace jxl
