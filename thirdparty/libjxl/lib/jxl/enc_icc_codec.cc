// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_icc_codec.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/icc_codec_common.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {
namespace {

// Unshuffles or de-interleaves bytes, for example with width 2, turns
// "AaBbCcDc" into "ABCDabcd", this for example de-interleaves UTF-16 bytes into
// first all the high order bytes, then all the low order bytes.
// Transposes a matrix of width columns and ceil(size / width) rows. There are
// size elements, size may be < width * height, if so the
// last elements of the bottom row are missing, the missing spots are
// transposed along with the filled spots, and the result has the missing
// elements at the bottom of the rightmost column. The input is the input matrix
// in scanline order, the output is the result matrix in scanline order, with
// missing elements skipped over (this may occur at multiple positions).
Status Unshuffle(JxlMemoryManager* memory_manager, uint8_t* data, size_t size,
                 size_t width) {
  size_t height = (size + width - 1) / width;  // amount of rows of input
  PaddedBytes result(memory_manager);
  JXL_ASSIGN_OR_RETURN(result,
                       PaddedBytes::WithInitialSpace(memory_manager, size));

  // i = input index, j output index
  size_t s = 0;
  size_t j = 0;
  for (size_t i = 0; i < size; i++) {
    result[j] = data[i];
    j += height;
    if (j >= size) j = ++s;
  }

  for (size_t i = 0; i < size; i++) {
    data[i] = result[i];
  }
  return true;
}

// This is performed by the encoder, the encoder must be able to encode any
// random byte stream (not just byte streams that are a valid ICC profile), so
// an error returned by this function is an implementation error.
Status PredictAndShuffle(size_t stride, size_t width, int order, size_t num,
                         const uint8_t* data, size_t size, size_t* pos,
                         PaddedBytes* result) {
  JXL_RETURN_IF_ERROR(CheckOutOfBounds(*pos, num, size));
  JxlMemoryManager* memory_manager = result->memory_manager();
  // Required by the specification, see decoder. stride * 4 must be < *pos.
  if (!*pos || ((*pos - 1u) >> 2u) < stride) {
    return JXL_FAILURE("Invalid stride");
  }
  if (*pos < stride * 4) return JXL_FAILURE("Too large stride");
  size_t start = result->size();
  for (size_t i = 0; i < num; i++) {
    uint8_t predicted =
        LinearPredictICCValue(data, *pos, i, stride, width, order);
    JXL_RETURN_IF_ERROR(result->push_back(data[*pos + i] - predicted));
  }
  *pos += num;
  if (width > 1) {
    JXL_RETURN_IF_ERROR(
        Unshuffle(memory_manager, result->data() + start, num, width));
  }
  return true;
}

inline Status EncodeVarInt(uint64_t value, PaddedBytes* data) {
  size_t pos = data->size();
  JXL_RETURN_IF_ERROR(data->resize(data->size() + 9));
  size_t output_size = data->size();
  uint8_t* output = data->data();

  // While more than 7 bits of data are left,
  // store 7 bits and set the next byte flag
  while (value > 127) {
    // TODO(eustas): should it be `<` ?
    JXL_ENSURE(pos <= output_size);
    // |128: Set the next byte flag
    output[pos++] = (static_cast<uint8_t>(value & 127)) | 128;
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  // TODO(eustas): should it be `<` ?
  JXL_ENSURE(pos <= output_size);
  output[pos++] = static_cast<uint8_t>(value & 127);

  return data->resize(pos);
}

constexpr size_t kSizeLimit = std::numeric_limits<uint32_t>::max() >> 2;

}  // namespace

// Outputs a transformed form of the given icc profile. The result itself is
// not particularly smaller than the input data in bytes, but it will be in a
// form that is easier to compress (more zeroes, ...) and will compress better
// with brotli.
Status PredictICC(const uint8_t* icc, size_t size, PaddedBytes* result) {
  JxlMemoryManager* memory_manager = result->memory_manager();
  PaddedBytes commands{memory_manager};
  PaddedBytes data{memory_manager};

  static_assert(sizeof(size_t) >= 4, "size_t is too short");
  // Fuzzer expects that PredictICC can accept any input,
  // but 1GB should be enough for any purpose.
  if (size > kSizeLimit) {
    return JXL_FAILURE("ICC profile is too large");
  }

  JXL_RETURN_IF_ERROR(EncodeVarInt(size, result));

  // Header
  PaddedBytes header{memory_manager};
  JXL_RETURN_IF_ERROR(header.append(ICCInitialHeaderPrediction(size)));
  for (size_t i = 0; i < kICCHeaderSize && i < size; i++) {
    ICCPredictHeader(icc, size, header.data(), i);
    JXL_RETURN_IF_ERROR(data.push_back(icc[i] - header[i]));
  }
  if (size <= kICCHeaderSize) {
    JXL_RETURN_IF_ERROR(EncodeVarInt(0, result));  // 0 commands
    for (uint8_t b : data) {
      JXL_RETURN_IF_ERROR(result->push_back(b));
    }
    return true;
  }

  std::vector<Tag> tags;
  std::vector<size_t> tagstarts;
  std::vector<size_t> tagsizes;
  std::map<size_t, size_t> tagmap;

  // Tag list
  size_t pos = kICCHeaderSize;
  if (pos + 4 <= size) {
    uint64_t numtags = DecodeUint32(icc, size, pos);
    pos += 4;
    JXL_RETURN_IF_ERROR(EncodeVarInt(numtags + 1, &commands));
    uint64_t prevtagstart = kICCHeaderSize + numtags * 12;
    uint32_t prevtagsize = 0;
    for (size_t i = 0; i < numtags; i++) {
      if (pos + 12 > size) break;

      Tag tag = DecodeKeyword(icc, size, pos + 0);
      uint32_t tagstart = DecodeUint32(icc, size, pos + 4);
      uint32_t tagsize = DecodeUint32(icc, size, pos + 8);
      pos += 12;

      tags.push_back(tag);
      tagstarts.push_back(tagstart);
      tagsizes.push_back(tagsize);
      tagmap[tagstart] = tags.size() - 1;

      uint8_t tagcode = kCommandTagUnknown;
      for (size_t j = 0; j < kNumTagStrings; j++) {
        if (tag == *kTagStrings[j]) {
          tagcode = j + kCommandTagStringFirst;
          break;
        }
      }

      if (tag == kRtrcTag && pos + 24 < size) {
        bool ok = true;
        ok &= DecodeKeyword(icc, size, pos + 0) == kGtrcTag;
        ok &= DecodeKeyword(icc, size, pos + 12) == kBtrcTag;
        if (ok) {
          for (size_t kk = 0; kk < 8; kk++) {
            if (icc[pos - 8 + kk] != icc[pos + 4 + kk]) ok = false;
            if (icc[pos - 8 + kk] != icc[pos + 16 + kk]) ok = false;
          }
        }
        if (ok) {
          tagcode = kCommandTagTRC;
          pos += 24;
          i += 2;
        }
      }

      if (tag == kRxyzTag && pos + 24 < size) {
        bool ok = true;
        ok &= DecodeKeyword(icc, size, pos + 0) == kGxyzTag;
        ok &= DecodeKeyword(icc, size, pos + 12) == kBxyzTag;
        uint32_t offsetr = tagstart;
        uint32_t offsetg = DecodeUint32(icc, size, pos + 4);
        uint32_t offsetb = DecodeUint32(icc, size, pos + 16);
        uint32_t sizer = tagsize;
        uint32_t sizeg = DecodeUint32(icc, size, pos + 8);
        uint32_t sizeb = DecodeUint32(icc, size, pos + 20);
        ok &= sizer == 20;
        ok &= sizeg == 20;
        ok &= sizeb == 20;
        ok &= (offsetg == offsetr + 20);
        ok &= (offsetb == offsetr + 40);
        if (ok) {
          tagcode = kCommandTagXYZ;
          pos += 24;
          i += 2;
        }
      }

      uint8_t command = tagcode;
      uint64_t predicted_tagstart = prevtagstart + prevtagsize;
      if (predicted_tagstart != tagstart) command |= kFlagBitOffset;
      size_t predicted_tagsize = prevtagsize;
      if (tag == kRxyzTag || tag == kGxyzTag || tag == kBxyzTag ||
          tag == kKxyzTag || tag == kWtptTag || tag == kBkptTag ||
          tag == kLumiTag) {
        predicted_tagsize = 20;
      }
      if (predicted_tagsize != tagsize) command |= kFlagBitSize;
      JXL_RETURN_IF_ERROR(commands.push_back(command));
      if (tagcode == 1) {
        JXL_RETURN_IF_ERROR(AppendKeyword(tag, &data));
      }
      if (command & kFlagBitOffset)
        JXL_RETURN_IF_ERROR(EncodeVarInt(tagstart, &commands));
      if (command & kFlagBitSize)
        JXL_RETURN_IF_ERROR(EncodeVarInt(tagsize, &commands));

      prevtagstart = tagstart;
      prevtagsize = tagsize;
    }
  }
  // Indicate end of tag list or varint indicating there's none
  JXL_RETURN_IF_ERROR(commands.push_back(0));

  // Main content
  // The main content in a valid ICC profile contains tagged elements, with the
  // tag types (4 letter names) given by the tag list above, and the tag list
  // pointing to the start and indicating the size of each tagged element. It is
  // allowed for tagged elements to overlap, e.g. the curve for R, G and B could
  // all point to the same one.
  Tag tag;
  size_t tagstart = 0;
  size_t tagsize = 0;
  size_t clutstart = 0;

  // Should always check tag_sane before doing math with tagsize.
  const auto tag_sane = [&tagsize]() {
    return (tagsize > 8) && (tagsize < kSizeLimit);
  };

  size_t last0 = pos;
  // This loop appends commands to the output, processing some sub-section of a
  // current tagged element each time. We need to keep track of the tagtype of
  // the current element, and update it when we encounter the boundary of a
  // next one.
  // It is not required that the input data is a valid ICC profile, if the
  // encoder does not recognize the data it will still be able to output bytes
  // but will not predict as well.
  while (pos <= size) {
    size_t last1 = pos;
    PaddedBytes commands_add{memory_manager};
    PaddedBytes data_add{memory_manager};

    // This means the loop brought the position beyond the tag end.
    // If tagsize is nonsensical, any pos looks "ok-ish".
    if ((pos > tagstart + tagsize) && (tagsize < kSizeLimit)) {
      tag = {{0, 0, 0, 0}};  // nonsensical value
    }

    if (commands_add.empty() && data_add.empty() && tagmap.count(pos) &&
        pos + 4 <= size) {
      size_t index = tagmap[pos];
      tag = DecodeKeyword(icc, size, pos);
      tagstart = tagstarts[index];
      tagsize = tagsizes[index];

      if (tag == kMlucTag && tag_sane() && pos + tagsize <= size &&
          icc[pos + 4] == 0 && icc[pos + 5] == 0 && icc[pos + 6] == 0 &&
          icc[pos + 7] == 0) {
        size_t num = tagsize - 8;
        JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandTypeStartFirst + 3));
        pos += 8;
        JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandShuffle2));
        JXL_RETURN_IF_ERROR(EncodeVarInt(num, &commands_add));
        size_t start = data_add.size();
        for (size_t i = 0; i < num; i++) {
          JXL_RETURN_IF_ERROR(data_add.push_back(icc[pos]));
          pos++;
        }
        JXL_RETURN_IF_ERROR(
            Unshuffle(memory_manager, data_add.data() + start, num, 2));
      }

      if (tag == kCurvTag && tag_sane() && pos + tagsize <= size &&
          icc[pos + 4] == 0 && icc[pos + 5] == 0 && icc[pos + 6] == 0 &&
          icc[pos + 7] == 0) {
        size_t num = tagsize - 8;
        if (num > 16 && num < (1 << 28) && pos + num <= size && pos > 0) {
          JXL_RETURN_IF_ERROR(
              commands_add.push_back(kCommandTypeStartFirst + 5));
          pos += 8;
          JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandPredict));
          int order = 1;
          int width = 2;
          int stride = width;
          JXL_RETURN_IF_ERROR(
              commands_add.push_back((order << 2) | (width - 1)));
          JXL_RETURN_IF_ERROR(EncodeVarInt(num, &commands_add));
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }
    }

    if (tag == kMab_Tag || tag == kMba_Tag) {
      Tag subTag = DecodeKeyword(icc, size, pos);
      if (pos + 12 < size && (subTag == kCurvTag || subTag == kVcgtTag) &&
          DecodeUint32(icc, size, pos + 4) == 0) {
        uint32_t num = DecodeUint32(icc, size, pos + 8) * 2;
        if (num > 16 && num < (1 << 28) && pos + 12 + num <= size) {
          pos += 12;
          last1 = pos;
          JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandPredict));
          int order = 1;
          int width = 2;
          int stride = width;
          JXL_RETURN_IF_ERROR(
              commands_add.push_back((order << 2) | (width - 1)));
          JXL_RETURN_IF_ERROR(EncodeVarInt(num, &commands_add));
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }

      if (pos == tagstart + 24 && pos + 4 < size) {
        // Note that this value can be remembered for next iterations of the
        // loop, so the "pos == clutstart" if below can trigger during a later
        // iteration.
        clutstart = tagstart + DecodeUint32(icc, size, pos);
      }

      if (pos == clutstart && clutstart + 16 < size) {
        size_t numi = icc[tagstart + 8];
        size_t numo = icc[tagstart + 9];
        size_t width = icc[clutstart + 16];
        size_t stride = width * numo;
        size_t num = width * numo;
        for (size_t i = 0; i < numi && clutstart + i < size; i++) {
          num *= icc[clutstart + i];
        }
        if ((width == 1 || width == 2) && num > 64 && num < (1 << 28) &&
            pos + num <= size && pos > stride * 4) {
          JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandPredict));
          int order = 1;
          uint8_t flags =
              (order << 2) | (width - 1) | (stride == width ? 0 : 16);
          JXL_RETURN_IF_ERROR(commands_add.push_back(flags));
          if (flags & 16) {
            JXL_RETURN_IF_ERROR(EncodeVarInt(stride, &commands_add));
          }
          JXL_RETURN_IF_ERROR(EncodeVarInt(num, &commands_add));
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }
    }

    if (commands_add.empty() && data_add.empty() && tag == kGbd_Tag &&
        tag_sane() && pos == tagstart + 8 && pos + tagsize - 8 <= size &&
        pos > 16) {
      size_t width = 4;
      size_t order = 0;
      size_t stride = width;
      size_t num = tagsize - 8;
      uint8_t flags = (order << 2) | (width - 1) | (stride == width ? 0 : 16);
      JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandPredict));
      JXL_RETURN_IF_ERROR(commands_add.push_back(flags));
      if (flags & 16) {
        JXL_RETURN_IF_ERROR(EncodeVarInt(stride, &commands_add));
      }
      JXL_RETURN_IF_ERROR(EncodeVarInt(num, &commands_add));
      JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                            size, &pos, &data_add));
    }

    if (commands_add.empty() && data_add.empty() && pos + 20 <= size) {
      Tag subTag = DecodeKeyword(icc, size, pos);
      if (subTag == kXyz_Tag && DecodeUint32(icc, size, pos + 4) == 0) {
        JXL_RETURN_IF_ERROR(commands_add.push_back(kCommandXYZ));
        pos += 8;
        for (size_t j = 0; j < 12; j++) {
          JXL_RETURN_IF_ERROR(data_add.push_back(icc[pos++]));
        }
      }
    }

    if (commands_add.empty() && data_add.empty() && pos + 8 <= size) {
      if (DecodeUint32(icc, size, pos + 4) == 0) {
        Tag subTag = DecodeKeyword(icc, size, pos);
        for (size_t i = 0; i < kNumTypeStrings; i++) {
          if (subTag == *kTypeStrings[i]) {
            JXL_RETURN_IF_ERROR(
                commands_add.push_back(kCommandTypeStartFirst + i));
            pos += 8;
            break;
          }
        }
      }
    }

    if (!(commands_add.empty() && data_add.empty()) || pos == size) {
      if (last0 < last1) {
        JXL_RETURN_IF_ERROR(commands.push_back(kCommandInsert));
        JXL_RETURN_IF_ERROR(EncodeVarInt(last1 - last0, &commands));
        while (last0 < last1) {
          JXL_RETURN_IF_ERROR(data.push_back(icc[last0++]));
        }
      }
      for (uint8_t b : commands_add) {
        JXL_RETURN_IF_ERROR(commands.push_back(b));
      }
      for (uint8_t b : data_add) {
        JXL_RETURN_IF_ERROR(data.push_back(b));
      }
      last0 = pos;
    }
    if (commands_add.empty() && data_add.empty()) {
      pos++;
    }
  }

  JXL_RETURN_IF_ERROR(EncodeVarInt(commands.size(), result));
  for (uint8_t b : commands) {
    JXL_RETURN_IF_ERROR(result->push_back(b));
  }
  for (uint8_t b : data) {
    JXL_RETURN_IF_ERROR(result->push_back(b));
  }

  return true;
}

Status WriteICC(const Span<const uint8_t> icc, BitWriter* JXL_RESTRICT writer,
                LayerType layer, AuxOut* JXL_RESTRICT aux_out) {
  if (icc.empty()) return JXL_FAILURE("ICC must be non-empty");
  JxlMemoryManager* memory_manager = writer->memory_manager();
  PaddedBytes enc{memory_manager};
  JXL_RETURN_IF_ERROR(PredictICC(icc.data(), icc.size(), &enc));
  std::vector<std::vector<Token>> tokens(1);
  JXL_RETURN_IF_ERROR(writer->WithMaxBits(128, layer, aux_out, [&] {
    return U64Coder::Write(enc.size(), writer);
  }));

  for (size_t i = 0; i < enc.size(); i++) {
    uint32_t ctx = static_cast<uint32_t>(
        ICCANSContext(i, i > 0 ? enc[i - 1] : 0, i > 1 ? enc[i - 2] : 0));
    tokens[0].emplace_back(ctx, enc[i]);
  }
  HistogramParams params;
  params.lz77_method = enc.size() < 16384 ? HistogramParams::LZ77Method::kOptimal
                                         : HistogramParams::LZ77Method::kLZ77;
  EntropyEncodingData code;
  params.force_huffman = true;
  JXL_ASSIGN_OR_RETURN(size_t cost, BuildAndEncodeHistograms(
                                        memory_manager, params, kNumICCContexts,
                                        tokens, &code, writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], code, 0, writer, layer, aux_out));
  return true;
}

}  // namespace jxl
