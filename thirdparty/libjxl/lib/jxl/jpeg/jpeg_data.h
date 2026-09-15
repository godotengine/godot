// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Data structures that represent the non-pixel contents of a jpeg file.

#ifndef LIB_JXL_JPEG_JPEG_DATA_H_
#define LIB_JXL_JPEG_JPEG_DATA_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"  // JPEGXL_ENABLE_TRANSCODE_JPEG
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {
namespace jpeg {

constexpr int kMaxComponents = 4;
constexpr int kMaxQuantTables = 4;
constexpr int kMaxHuffmanTables = 4;
constexpr size_t kJpegHuffmanMaxBitLength = 16;
constexpr int kJpegHuffmanAlphabetSize = 256;
constexpr int kJpegDCAlphabetSize = 12;
constexpr int kMaxDHTMarkers = 512;
constexpr int kMaxDimPixels = 65535;
constexpr uint8_t kApp1 = 0xE1;
constexpr uint8_t kApp2 = 0xE2;
const uint8_t kIccProfileTag[12] = "ICC_PROFILE";
const uint8_t kExifTag[6] = "Exif\0";
const uint8_t kXMPTag[29] = "http://ns.adobe.com/xap/1.0/";

/* clang-format off */
constexpr uint32_t kJPEGNaturalOrder[80] = {
  0,   1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
  // extra entries for safety in decoder
  63, 63, 63, 63, 63, 63, 63, 63,
  63, 63, 63, 63, 63, 63, 63, 63
};

constexpr uint32_t kJPEGZigZagOrder[64] = {
  0,   1,  5,  6, 14, 15, 27, 28,
  2,   4,  7, 13, 16, 26, 29, 42,
  3,   8, 12, 17, 25, 30, 41, 43,
  9,  11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63
};
/* clang-format on */

// Quantization values for an 8x8 pixel block.
struct JPEGQuantTable {
  std::array<int32_t, kDCTBlockSize> values;
  uint32_t precision = 0;
  // The index of this quantization table as it was parsed from the input JPEG.
  // Each DQT marker segment contains an 'index' field, and we save this index
  // here. Valid values are 0 to 3.
  uint32_t index = 0;
  // Set to true if this table is the last one within its marker segment.
  bool is_last = true;
};

// Huffman code and decoding lookup table used for DC and AC coefficients.
struct JPEGHuffmanCode {
  // Bit length histogram.
  std::array<uint32_t, kJpegHuffmanMaxBitLength + 1> counts = {};
  // Symbol values sorted by increasing bit lengths.
  std::array<uint32_t, kJpegHuffmanAlphabetSize + 1> values = {};
  // The index of the Huffman code in the current set of Huffman codes. For AC
  // component Huffman codes, 0x10 is added to the index.
  int slot_id = 0;
  // Set to true if this Huffman code is the last one within its marker segment.
  bool is_last = true;
};

// Huffman table indexes used for one component of one scan.
struct JPEGComponentScanInfo {
  uint32_t comp_idx;
  uint32_t dc_tbl_idx;
  uint32_t ac_tbl_idx;
};

// Contains information that is used in one scan.
struct JPEGScanInfo {
  // Parameters used for progressive scans (named the same way as in the spec):
  //   Ss : Start of spectral band in zig-zag sequence.
  //   Se : End of spectral band in zig-zag sequence.
  //   Ah : Successive approximation bit position, high.
  //   Al : Successive approximation bit position, low.
  uint32_t Ss;
  uint32_t Se;
  uint32_t Ah;
  uint32_t Al;
  uint32_t num_components = 0;
  std::array<JPEGComponentScanInfo, 4> components;
  // Last codestream pass that is needed to write this scan.
  uint32_t last_needed_pass = 0;

  // Extra information required for bit-precise JPEG file reconstruction.

  // Set of block indexes where the JPEG encoder has to flush the end-of-block
  // runs and refinement bits.
  std::vector<uint32_t> reset_points;
  // The number of extra zero runs (Huffman symbol 0xf0) before the end of
  // block (if nonzero), indexed by block index.
  // All of these symbols can be omitted without changing the pixel values, but
  // some jpeg encoders put these at the end of blocks.
  typedef struct {
    uint32_t block_idx;
    uint32_t num_extra_zero_runs;
  } ExtraZeroRunInfo;
  std::vector<ExtraZeroRunInfo> extra_zero_runs;
};

typedef int16_t coeff_t;

// Represents one component of a jpeg file.
struct JPEGComponent {
  JPEGComponent()
      : id(0),
        h_samp_factor(1),
        v_samp_factor(1),
        quant_idx(0),
        width_in_blocks(0),
        height_in_blocks(0) {}

  // One-byte id of the component.
  uint32_t id;
  // Horizontal and vertical sampling factors.
  // In interleaved mode, each minimal coded unit (MCU) has
  // h_samp_factor x v_samp_factor DCT blocks from this component.
  int h_samp_factor;
  int v_samp_factor;
  // The index of the quantization table used for this component.
  uint32_t quant_idx;
  // The dimensions of the component measured in 8x8 blocks.
  uint32_t width_in_blocks;
  uint32_t height_in_blocks;
  // The DCT coefficients of this component, laid out block-by-block, divided
  // through the quantization matrix values.
  std::vector<coeff_t> coeffs;
};

enum class AppMarkerType : uint32_t {
  kUnknown = 0,
  kICC = 1,
  kExif = 2,
  kXMP = 3,
};

// Represents a parsed jpeg file.
struct JPEGData : public Fields {
  JPEGData()
      : width(0), height(0), restart_interval(0), has_zero_padding_bit(false) {}

  JXL_FIELDS_NAME(JPEGData)
#if JPEGXL_ENABLE_TRANSCODE_JPEG
  // Doesn't serialize everything - skips brotli-encoded data and what is
  // already encoded in the codestream.
  Status VisitFields(Visitor* visitor) override;
#else
  Status VisitFields(Visitor* /* visitor */) override {
    return JXL_UNREACHABLE("JPEG transcoding support not enabled");
  }
#endif  // JPEGXL_ENABLE_TRANSCODE_JPEG

  void CalculateMcuSize(const JPEGScanInfo& scan, int* MCUs_per_row,
                        int* MCU_rows) const;

  int width;
  int height;
  uint32_t restart_interval;
  std::vector<std::vector<uint8_t>> app_data;
  std::vector<AppMarkerType> app_marker_type;
  std::vector<std::vector<uint8_t>> com_data;
  std::vector<JPEGQuantTable> quant;
  std::vector<JPEGHuffmanCode> huffman_code;
  std::vector<JPEGComponent> components;
  std::vector<JPEGScanInfo> scan_info;
  std::vector<uint8_t> marker_order;
  std::vector<std::vector<uint8_t>> inter_marker_data;
  std::vector<uint8_t> tail_data;

  // Extra information required for bit-precise JPEG file reconstruction.

  bool has_zero_padding_bit;
  std::vector<uint8_t> padding_bits;
};

#if JPEGXL_ENABLE_TRANSCODE_JPEG
// Set ICC profile in jpeg_data.
Status SetJPEGDataFromICC(const std::vector<uint8_t>& icc,
                          jpeg::JPEGData* jpeg_data);
#else
static JXL_INLINE Status SetJPEGDataFromICC(
    const std::vector<uint8_t>& /* icc */, jpeg::JPEGData* /* jpeg_data */) {
  return JXL_UNREACHABLE("JPEG transcoding support not enabled");
}
#endif  // JPEGXL_ENABLE_TRANSCODE_JPEG

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_JPEG_DATA_H_
