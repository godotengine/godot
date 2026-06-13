// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JPEG_DEC_JPEG_SERIALIZATION_STATE_H_
#define LIB_JXL_JPEG_DEC_JPEG_SERIALIZATION_STATE_H_

#include <algorithm>
#include <deque>
#include <vector>

#include "lib/jxl/jpeg/dec_jpeg_output_chunk.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

struct HuffmanCodeTable {
  int8_t depth[256];
  uint16_t code[256];
  bool initialized = false;
  void InitDepths(int8_t value = 0) {
    std::fill(std::begin(depth), std::end(depth), value);
  }
};

// Handles the packing of bits into output bytes.
struct JpegBitWriter {
  bool healthy;
  std::deque<OutputChunk>* output;
  OutputChunk chunk;
  uint8_t* data;
  size_t pos;
  uint64_t put_buffer;
  int put_bits;
};

// Holds data that is buffered between 8x8 blocks in progressive mode.
struct DCTCodingState {
  // The run length of end-of-band symbols in a progressive scan.
  int eob_run_;
  // The huffman table to be used when flushing the state.
  HuffmanCodeTable* cur_ac_huff_;
  // The sequence of currently buffered refinement bits for a successive
  // approximation scan (one where Ah > 0).
  std::vector<uint16_t> refinement_bits_;
  size_t refinement_bits_count_ = 0;
};

struct EncodeScanState {
  enum Stage { HEAD, BODY };

  Stage stage = HEAD;

  int mcu_y;
  JpegBitWriter bw;
  coeff_t last_dc_coeff[kMaxComponents] = {0};
  int restarts_to_go;
  int next_restart_marker;
  int block_scan_index;
  DCTCodingState coding_state;
  size_t extra_zero_runs_pos;
  int next_extra_zero_run_index;
  size_t next_reset_point_pos;
  int next_reset_point;
};

struct SerializationState {
  enum Stage {
    STAGE_INIT,
    STAGE_SERIALIZE_SECTION,
    STAGE_DONE,
    STAGE_ERROR,
  };

  Stage stage = STAGE_INIT;

  std::deque<OutputChunk> output_queue;

  size_t section_index = 0;
  int dht_index = 0;
  int dqt_index = 0;
  int app_index = 0;
  int com_index = 0;
  int data_index = 0;
  int scan_index = 0;
  std::vector<HuffmanCodeTable> dc_huff_table;
  std::vector<HuffmanCodeTable> ac_huff_table;
  const uint8_t* pad_bits = nullptr;
  const uint8_t* pad_bits_end = nullptr;
  bool seen_dri_marker = false;
  bool is_progressive = false;

  EncodeScanState scan_state;
};

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_SERIALIZATION_STATE_H_
