// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/icc_codec.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/icc_codec_common.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {
namespace {

// Shuffles or interleaves bytes, for example with width 2, turns "ABCDabcd"
// into "AaBbCcDd". Transposes a matrix of ceil(size / width) columns and
// width rows. There are size elements, size may be < width * height, if so the
// last elements of the rightmost column are missing, the missing spots are
// transposed along with the filled spots, and the result has the missing
// elements at the end of the bottom row. The input is the input matrix in
// scanline order but with missing elements skipped (which may occur in multiple
// locations), the output is the result matrix in scanline order (with
// no need to skip missing elements as they are past the end of the data).
Status Shuffle(JxlMemoryManager* memory_manager, uint8_t* data, size_t size,
               size_t width) {
  size_t height = (size + width - 1) / width;  // amount of rows of output
  PaddedBytes result(memory_manager);
  JXL_ASSIGN_OR_RETURN(result,
                       PaddedBytes::WithInitialSpace(memory_manager, size));
  // i = output index, j input index
  size_t s = 0;
  size_t j = 0;
  for (size_t i = 0; i < size; i++) {
    result[i] = data[j];
    j += height;
    if (j >= size) j = ++s;
  }

  for (size_t i = 0; i < size; i++) {
    data[i] = result[i];
  }
  return true;
}

// Two base-128 varints at up to 10 bytes each.
constexpr size_t kPreambleSize = 20;

// Decodes a base-128 unsigned varint into *out, advancing *pos by the exact
// number of bytes consumed. Returns an error if the input is truncated, if
// the terminator (top-bit-clear byte) is not seen within 10 bytes, or if the
// 10th byte encodes a value that does not fit in a uint64_t.
Status DecodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos,
                    uint64_t* out) {
  uint64_t ret = 0;
  // 9 bytes cover bits 0..62; the 10th byte may only contribute bit 63.
  for (size_t i = 0; i < 9; ++i) {
    if (*pos >= inputSize) {
      return JXL_FAILURE("DecodeVarInt: truncated input");
    }
    const uint8_t byte = input[(*pos)++];
    ret |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);
    if ((byte & 0x80) == 0) {
      *out = ret;
      return true;
    }
  }
  if (*pos >= inputSize) {
    return JXL_FAILURE("DecodeVarInt: truncated input (10th byte)");
  }
  const uint8_t byte = input[(*pos)++];
  if ((byte & 0x80) != 0) {
    return JXL_FAILURE("DecodeVarInt: varint exceeds 10 bytes");
  }
  if ((byte & 0x7E) != 0) {
    return JXL_FAILURE("DecodeVarInt: value exceeds 2^64 - 1");
  }
  ret |= static_cast<uint64_t>(byte & 0x01) << 63;
  *out = ret;
  return true;
}

}  // namespace

// Mimics the beginning of UnpredictICC for quick validity check.
// At least kPreambleSize bytes of data should be valid at invocation time.
Status CheckPreamble(const PaddedBytes& data, size_t enc_size) {
  const uint8_t* enc = data.data();
  size_t size = data.size();
  size_t pos = 0;
  uint64_t osize;
  JXL_RETURN_IF_ERROR(DecodeVarInt(enc, size, &pos, &osize));
  JXL_RETURN_IF_ERROR(CheckIs32Bit(osize));
  uint64_t csize;
  JXL_RETURN_IF_ERROR(DecodeVarInt(enc, size, &pos, &csize));
  JXL_RETURN_IF_ERROR(CheckIs32Bit(csize));
  JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, csize, size));
  // We expect that UnpredictICC inflates input, not the other way round.
  if (osize + 65536 < enc_size) return JXL_FAILURE("Malformed ICC");

  // NB(eustas): 64 MiB ICC should be enough for everything!?
  const size_t output_limit = 1 << 28;
  if (output_limit && osize > output_limit) {
    return JXL_FAILURE("Decoded ICC is too large");
  }
  return true;
}

// Decodes the result of PredictICC back to a valid ICC profile.
Status UnpredictICC(const uint8_t* enc, size_t size, PaddedBytes* result) {
  if (!result->empty()) return JXL_FAILURE("result must be empty initially");
  JxlMemoryManager* memory_manager = result->memory_manager();
  size_t pos = 0;
  uint64_t osize;
  JXL_RETURN_IF_ERROR(DecodeVarInt(enc, size, &pos, &osize));  // Output size
  JXL_RETURN_IF_ERROR(CheckIs32Bit(osize));
  uint64_t csize;
  JXL_RETURN_IF_ERROR(DecodeVarInt(enc, size, &pos, &csize));  // Commands size
  // Every command is translated to at least one byte.
  JXL_RETURN_IF_ERROR(CheckIs32Bit(csize));
  size_t cpos = pos;  // pos in commands stream
  JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, csize, size));
  size_t commands_end = cpos + csize;
  pos = commands_end;  // pos in data stream

  // Header
  PaddedBytes header{memory_manager};
  JXL_RETURN_IF_ERROR(header.append(ICCInitialHeaderPrediction(osize)));
  for (size_t i = 0; i <= kICCHeaderSize; i++) {
    if (result->size() == osize) {
      if (cpos != commands_end) return JXL_FAILURE("Not all commands used");
      if (pos != size) return JXL_FAILURE("Not all data used");
      return true;  // Valid end
    }
    if (i == kICCHeaderSize) break;  // Done
    ICCPredictHeader(result->data(), result->size(), header.data(), i);
    if (pos >= size) return JXL_FAILURE("Out of bounds");
    JXL_RETURN_IF_ERROR(result->push_back(enc[pos++] + header[i]));
  }
  if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");

  // Tag list
  uint64_t numtags;
  JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &numtags));

  if (numtags != 0) {
    numtags--;
    JXL_RETURN_IF_ERROR(CheckIs32Bit(numtags));
    JXL_RETURN_IF_ERROR(AppendUint32(numtags, result));
    uint64_t prevtagstart = kICCHeaderSize + numtags * 12;
    uint64_t prevtagsize = 0;
    for (;;) {
      if (result->size() > osize) return JXL_FAILURE("Invalid result size");
      if (cpos > commands_end) return JXL_FAILURE("Out of bounds");
      if (cpos == commands_end) break;  // Valid end
      uint8_t command = enc[cpos++];
      uint8_t tagcode = command & 63;
      Tag tag;
      if (tagcode == 0) {
        break;
      } else if (tagcode == kCommandTagUnknown) {
        JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, 4, size));
        tag = DecodeKeyword(enc, size, pos);
        pos += 4;
      } else if (tagcode == kCommandTagTRC) {
        tag = kRtrcTag;
      } else if (tagcode == kCommandTagXYZ) {
        tag = kRxyzTag;
      } else {
        if (tagcode - kCommandTagStringFirst >= kNumTagStrings) {
          return JXL_FAILURE("Unknown tagcode");
        }
        tag = *kTagStrings[tagcode - kCommandTagStringFirst];
      }
      JXL_RETURN_IF_ERROR(AppendKeyword(tag, result));

      uint64_t tagstart;
      uint64_t tagsize = prevtagsize;
      if (tag == kRxyzTag || tag == kGxyzTag || tag == kBxyzTag ||
          tag == kKxyzTag || tag == kWtptTag || tag == kBkptTag ||
          tag == kLumiTag) {
        tagsize = 20;
      }

      if (command & kFlagBitOffset) {
        JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &tagstart));
      } else {
        JXL_RETURN_IF_ERROR(CheckIs32Bit(prevtagstart));
        tagstart = prevtagstart + prevtagsize;
      }
      JXL_RETURN_IF_ERROR(CheckIs32Bit(tagstart));
      JXL_RETURN_IF_ERROR(AppendUint32(tagstart, result));
      if (command & kFlagBitSize) {
        JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &tagsize));
      }
      JXL_RETURN_IF_ERROR(CheckIs32Bit(tagsize));
      JXL_RETURN_IF_ERROR(AppendUint32(tagsize, result));
      prevtagstart = tagstart;
      prevtagsize = tagsize;

      if (tagcode == kCommandTagTRC) {
        JXL_RETURN_IF_ERROR(AppendKeyword(kGtrcTag, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagstart, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagsize, result));
        JXL_RETURN_IF_ERROR(AppendKeyword(kBtrcTag, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagstart, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagsize, result));
      }

      if (tagcode == kCommandTagXYZ) {
        JXL_RETURN_IF_ERROR(CheckIs32Bit(tagstart + tagsize * 2));
        JXL_RETURN_IF_ERROR(AppendKeyword(kGxyzTag, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagstart + tagsize, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagsize, result));
        JXL_RETURN_IF_ERROR(AppendKeyword(kBxyzTag, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagstart + tagsize * 2, result));
        JXL_RETURN_IF_ERROR(AppendUint32(tagsize, result));
      }
    }
  }

  // Main Content
  for (;;) {
    if (result->size() > osize) return JXL_FAILURE("Invalid result size");
    if (cpos > commands_end) return JXL_FAILURE("Out of bounds");
    if (cpos == commands_end) break;  // Valid end
    uint8_t command = enc[cpos++];
    if (command == kCommandInsert) {
      uint64_t num;
      JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &num));
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));
      for (size_t i = 0; i < num; i++) {
        JXL_RETURN_IF_ERROR(result->push_back(enc[pos++]));
      }
    } else if (command == kCommandShuffle2 || command == kCommandShuffle4) {
      uint64_t num;
      JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &num));
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));
      PaddedBytes shuffled(memory_manager);
      JXL_ASSIGN_OR_RETURN(shuffled,
                           PaddedBytes::WithInitialSpace(memory_manager, num));
      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (command == kCommandShuffle2) {
        JXL_RETURN_IF_ERROR(Shuffle(memory_manager, shuffled.data(), num, 2));
      } else if (command == kCommandShuffle4) {
        JXL_RETURN_IF_ERROR(Shuffle(memory_manager, shuffled.data(), num, 4));
      }
      for (size_t i = 0; i < num; i++) {
        JXL_RETURN_IF_ERROR(result->push_back(shuffled[i]));
        pos++;
      }
    } else if (command == kCommandPredict) {
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(cpos, 2, commands_end));
      uint8_t flags = enc[cpos++];

      size_t width = (flags & 3) + 1;
      if (width == 3) return JXL_FAILURE("Invalid width");

      int order = (flags & 12) >> 2;
      if (order == 3) return JXL_FAILURE("Invalid order");

      uint64_t stride = width;
      if (flags & 16) {
        JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &stride));
        if (stride < width) {
          return JXL_FAILURE("Invalid stride");
        }
      }
      // If stride * 4 >= result->size(), return failure. The check
      // "size == 0 || ((size - 1) >> 2) < stride" corresponds to
      // "stride * 4 >= size", but does not suffer from integer overflow.
      // This check is more strict than necessary but follows the specification
      // and the encoder should ensure this is followed.
      if (result->empty() || ((result->size() - 1u) >> 2u) < stride) {
        return JXL_FAILURE("Invalid stride");
      }

      uint64_t num;
      JXL_RETURN_IF_ERROR(DecodeVarInt(enc, commands_end, &cpos, &num));  // in bytes
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));

      PaddedBytes shuffled(memory_manager);
      JXL_ASSIGN_OR_RETURN(shuffled,
                           PaddedBytes::WithInitialSpace(memory_manager, num));

      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (width > 1) {
        JXL_RETURN_IF_ERROR(
            Shuffle(memory_manager, shuffled.data(), num, width));
      }

      size_t start = result->size();
      for (size_t i = 0; i < num; i++) {
        uint8_t predicted = LinearPredictICCValue(result->data(), start, i,
                                                  stride, width, order);
        JXL_RETURN_IF_ERROR(result->push_back(predicted + shuffled[i]));
      }
      pos += num;
    } else if (command == kCommandXYZ) {
      JXL_RETURN_IF_ERROR(AppendKeyword(kXyz_Tag, result));
      for (int i = 0; i < 4; i++) {
        JXL_RETURN_IF_ERROR(result->push_back(0));
      }
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, 12, size));
      for (size_t i = 0; i < 12; i++) {
        JXL_RETURN_IF_ERROR(result->push_back(enc[pos++]));
      }
    } else if (command >= kCommandTypeStartFirst &&
               command < kCommandTypeStartFirst + kNumTypeStrings) {
      JXL_RETURN_IF_ERROR(AppendKeyword(
          *kTypeStrings[command - kCommandTypeStartFirst], result));
      for (size_t i = 0; i < 4; i++) {
        JXL_RETURN_IF_ERROR(result->push_back(0));
      }
    } else {
      return JXL_FAILURE("Unknown command");
    }
  }

  if (pos != size) return JXL_FAILURE("Not all data used");
  if (result->size() != osize) return JXL_FAILURE("Invalid result size");

  return true;
}

Status ICCReader::Init(BitReader* reader) {
  JXL_RETURN_IF_ERROR(CheckEOI(reader));
  JxlMemoryManager* memory_manager = decompressed_.memory_manager();
  used_bits_base_ = reader->TotalBitsConsumed();
  if (bits_to_skip_ == 0) {
    enc_size_ = U64Coder::Read(reader);
    if (enc_size_ > 268435456) {
      // Avoid too large memory allocation for invalid file.
      return JXL_FAILURE("Too large encoded profile");
    }
    JXL_RETURN_IF_ERROR(DecodeHistograms(
        memory_manager, reader, kNumICCContexts, &code_, &context_map_));
    JXL_ASSIGN_OR_RETURN(ans_reader_, ANSSymbolReader::Create(&code_, reader));
    i_ = 0;
    JXL_RETURN_IF_ERROR(
        decompressed_.resize(std::min<size_t>(i_ + 0x400, enc_size_)));
    for (; i_ < std::min<size_t>(2, enc_size_); i_++) {
      decompressed_[i_] = ans_reader_.ReadHybridUint(
          ICCANSContext(i_, i_ > 0 ? decompressed_[i_ - 1] : 0,
                        i_ > 1 ? decompressed_[i_ - 2] : 0),
          reader, context_map_);
    }
    if (enc_size_ > kPreambleSize) {
      for (; i_ < kPreambleSize; i_++) {
        decompressed_[i_] = ans_reader_.ReadHybridUint(
            ICCANSContext(i_, decompressed_[i_ - 1], decompressed_[i_ - 2]),
            reader, context_map_);
      }
      JXL_RETURN_IF_ERROR(CheckEOI(reader));
      JXL_RETURN_IF_ERROR(CheckPreamble(decompressed_, enc_size_));
    }
    bits_to_skip_ = reader->TotalBitsConsumed() - used_bits_base_;
  } else {
    reader->SkipBits(bits_to_skip_);
  }
  return true;
}

Status ICCReader::Process(BitReader* reader, PaddedBytes* icc) {
  auto checkpoint = jxl::make_unique<ANSSymbolReader::Checkpoint>();
  size_t saved_i = 0;
  auto save = [&]() {
    ans_reader_.Save(checkpoint.get());
    bits_to_skip_ = reader->TotalBitsConsumed() - used_bits_base_;
    saved_i = i_;
  };
  save();
  auto check_and_restore = [&]() -> Status {
    Status status = CheckEOI(reader);
    if (!status) {
      // not enough bytes.
      ans_reader_.Restore(*checkpoint);
      i_ = saved_i;
      return status;
    }
    return true;
  };
  for (; i_ < enc_size_; i_++) {
    if (i_ % ANSSymbolReader::kMaxCheckpointInterval == 0 && i_ > 0) {
      JXL_RETURN_IF_ERROR(check_and_restore());
      save();
      if ((i_ > 0) && (((i_ & 0xFFFF) == 0))) {
        float used_bytes =
            (reader->TotalBitsConsumed() - used_bits_base_) / 8.0f;
        if (i_ > used_bytes * 256) return JXL_FAILURE("Corrupted stream");
      }
      JXL_RETURN_IF_ERROR(
          decompressed_.resize(std::min<size_t>(i_ + 0x400, enc_size_)));
    }
    JXL_ENSURE(i_ >= 2);
    decompressed_[i_] = ans_reader_.ReadHybridUint(
        ICCANSContext(i_, decompressed_[i_ - 1], decompressed_[i_ - 2]), reader,
        context_map_);
  }
  JXL_RETURN_IF_ERROR(check_and_restore());
  bits_to_skip_ = reader->TotalBitsConsumed() - used_bits_base_;
  if (!ans_reader_.CheckANSFinalState()) {
    return JXL_FAILURE("Corrupted ICC profile");
  }

  icc->clear();
  return UnpredictICC(decompressed_.data(), decompressed_.size(), icc);
}

Status ICCReader::CheckEOI(BitReader* reader) {
  if (reader->AllReadsWithinBounds()) return true;
  return JXL_NOT_ENOUGH_BYTES("Not enough bytes for reading ICC profile");
}

}  // namespace jxl
