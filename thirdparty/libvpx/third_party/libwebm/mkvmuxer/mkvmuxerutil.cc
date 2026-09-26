// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "mkvmuxer/mkvmuxerutil.h"

#ifdef __ANDROID__
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>

#include "common/webmids.h"
#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvwriter.h"

namespace mkvmuxer {

namespace {

// Date elements are always 8 octets in size.
const int kDateElementSize = 8;

uint64 WriteBlock(IMkvWriter* writer, const Frame* const frame, int64 timecode,
                  uint64 timecode_scale) {
  uint64 block_additional_elem_size = 0;
  uint64 block_addid_elem_size = 0;
  uint64 block_more_payload_size = 0;
  uint64 block_more_elem_size = 0;
  uint64 block_additions_payload_size = 0;
  uint64 block_additions_elem_size = 0;
  if (frame->additional()) {
    block_additional_elem_size =
        EbmlElementSize(libwebm::kMkvBlockAdditional, frame->additional(),
                        frame->additional_length());
    block_addid_elem_size = EbmlElementSize(
        libwebm::kMkvBlockAddID, static_cast<uint64>(frame->add_id()));

    block_more_payload_size =
        block_addid_elem_size + block_additional_elem_size;
    block_more_elem_size =
        EbmlMasterElementSize(libwebm::kMkvBlockMore, block_more_payload_size) +
        block_more_payload_size;
    block_additions_payload_size = block_more_elem_size;
    block_additions_elem_size =
        EbmlMasterElementSize(libwebm::kMkvBlockAdditions,
                              block_additions_payload_size) +
        block_additions_payload_size;
  }

  uint64 discard_padding_elem_size = 0;
  if (frame->discard_padding() != 0) {
    discard_padding_elem_size =
        EbmlElementSize(libwebm::kMkvDiscardPadding,
                        static_cast<int64>(frame->discard_padding()));
  }

  const uint64 reference_block_timestamp =
      frame->reference_block_timestamp() / timecode_scale;
  uint64 reference_block_elem_size = 0;
  if (!frame->is_key()) {
    reference_block_elem_size =
        EbmlElementSize(libwebm::kMkvReferenceBlock, reference_block_timestamp);
  }

  const uint64 duration = frame->duration() / timecode_scale;
  uint64 block_duration_elem_size = 0;
  if (duration > 0)
    block_duration_elem_size =
        EbmlElementSize(libwebm::kMkvBlockDuration, duration);

  const uint64 block_payload_size = 4 + frame->length();
  const uint64 block_elem_size =
      EbmlMasterElementSize(libwebm::kMkvBlock, block_payload_size) +
      block_payload_size;

  const uint64 block_group_payload_size =
      block_elem_size + block_additions_elem_size + block_duration_elem_size +
      discard_padding_elem_size + reference_block_elem_size;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvBlockGroup,
                              block_group_payload_size)) {
    return 0;
  }

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvBlock, block_payload_size))
    return 0;

  if (WriteUInt(writer, frame->track_number()))
    return 0;

  if (SerializeInt(writer, timecode, 2))
    return 0;

  // For a Block, flags is always 0.
  if (SerializeInt(writer, 0, 1))
    return 0;

  if (writer->Write(frame->frame(), static_cast<uint32>(frame->length())))
    return 0;

  if (frame->additional()) {
    if (!WriteEbmlMasterElement(writer, libwebm::kMkvBlockAdditions,
                                block_additions_payload_size)) {
      return 0;
    }

    if (!WriteEbmlMasterElement(writer, libwebm::kMkvBlockMore,
                                block_more_payload_size))
      return 0;

    if (!WriteEbmlElement(writer, libwebm::kMkvBlockAddID,
                          static_cast<uint64>(frame->add_id())))
      return 0;

    if (!WriteEbmlElement(writer, libwebm::kMkvBlockAdditional,
                          frame->additional(), frame->additional_length())) {
      return 0;
    }
  }

  if (frame->discard_padding() != 0 &&
      !WriteEbmlElement(writer, libwebm::kMkvDiscardPadding,
                        static_cast<int64>(frame->discard_padding()))) {
    return false;
  }

  if (!frame->is_key() && !WriteEbmlElement(writer, libwebm::kMkvReferenceBlock,
                                            reference_block_timestamp)) {
    return false;
  }

  if (duration > 0 &&
      !WriteEbmlElement(writer, libwebm::kMkvBlockDuration, duration)) {
    return false;
  }
  return EbmlMasterElementSize(libwebm::kMkvBlockGroup,
                               block_group_payload_size) +
         block_group_payload_size;
}

uint64 WriteSimpleBlock(IMkvWriter* writer, const Frame* const frame,
                        int64 timecode) {
  if (WriteID(writer, libwebm::kMkvSimpleBlock))
    return 0;

  const int32 size = static_cast<int32>(frame->length()) + 4;
  if (WriteUInt(writer, size))
    return 0;

  if (WriteUInt(writer, static_cast<uint64>(frame->track_number())))
    return 0;

  if (SerializeInt(writer, timecode, 2))
    return 0;

  uint64 flags = 0;
  if (frame->is_key())
    flags |= 0x80;

  if (SerializeInt(writer, flags, 1))
    return 0;

  if (writer->Write(frame->frame(), static_cast<uint32>(frame->length())))
    return 0;

  return GetUIntSize(libwebm::kMkvSimpleBlock) + GetCodedUIntSize(size) + 4 +
         frame->length();
}

}  // namespace

int32 GetCodedUIntSize(uint64 value) {
  if (value < 0x000000000000007FULL)
    return 1;
  else if (value < 0x0000000000003FFFULL)
    return 2;
  else if (value < 0x00000000001FFFFFULL)
    return 3;
  else if (value < 0x000000000FFFFFFFULL)
    return 4;
  else if (value < 0x00000007FFFFFFFFULL)
    return 5;
  else if (value < 0x000003FFFFFFFFFFULL)
    return 6;
  else if (value < 0x0001FFFFFFFFFFFFULL)
    return 7;
  return 8;
}

int32 GetUIntSize(uint64 value) {
  if (value < 0x0000000000000100ULL)
    return 1;
  else if (value < 0x0000000000010000ULL)
    return 2;
  else if (value < 0x0000000001000000ULL)
    return 3;
  else if (value < 0x0000000100000000ULL)
    return 4;
  else if (value < 0x0000010000000000ULL)
    return 5;
  else if (value < 0x0001000000000000ULL)
    return 6;
  else if (value < 0x0100000000000000ULL)
    return 7;
  return 8;
}

int32 GetIntSize(int64 value) {
  // Doubling the requested value ensures positive values with their high bit
  // set are written with 0-padding to avoid flipping the signedness.
  const uint64 v = (value < 0) ? value ^ -1LL : value;
  return GetUIntSize(2 * v);
}

uint64 EbmlMasterElementSize(uint64 type, uint64 value) {
  // Size of EBML ID
  int32 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += GetCodedUIntSize(value);

  return ebml_size;
}

uint64 EbmlElementSize(uint64 type, int64 value) {
  // Size of EBML ID
  int32 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += GetIntSize(value);

  // Size of Datasize
  ebml_size++;

  return ebml_size;
}

uint64 EbmlElementSize(uint64 type, uint64 value) {
  return EbmlElementSize(type, value, 0);
}

uint64 EbmlElementSize(uint64 type, uint64 value, uint64 fixed_size) {
  // Size of EBML ID
  uint64 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += (fixed_size > 0) ? fixed_size : GetUIntSize(value);

  // Size of Datasize
  ebml_size++;

  return ebml_size;
}

uint64 EbmlElementSize(uint64 type, float /* value */) {
  // Size of EBML ID
  uint64 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += sizeof(float);

  // Size of Datasize
  ebml_size++;

  return ebml_size;
}

uint64 EbmlElementSize(uint64 type, const char* value) {
  if (!value)
    return 0;

  // Size of EBML ID
  uint64 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += strlen(value);

  // Size of Datasize
  ebml_size += GetCodedUIntSize(strlen(value));

  return ebml_size;
}

uint64 EbmlElementSize(uint64 type, const uint8* value, uint64 size) {
  if (!value)
    return 0;

  // Size of EBML ID
  uint64 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += size;

  // Size of Datasize
  ebml_size += GetCodedUIntSize(size);

  return ebml_size;
}

uint64 EbmlDateElementSize(uint64 type) {
  // Size of EBML ID
  uint64 ebml_size = GetUIntSize(type);

  // Datasize
  ebml_size += kDateElementSize;

  // Size of Datasize
  ebml_size++;

  return ebml_size;
}

int32 SerializeInt(IMkvWriter* writer, int64 value, int32 size) {
  if (!writer || size < 1 || size > 8)
    return -1;

  for (int32 i = 1; i <= size; ++i) {
    const int32 byte_count = size - i;
    const int32 bit_count = byte_count * 8;

    const int64 bb = value >> bit_count;
    const uint8 b = static_cast<uint8>(bb);

    const int32 status = writer->Write(&b, 1);

    if (status < 0)
      return status;
  }

  return 0;
}

int32 SerializeFloat(IMkvWriter* writer, float f) {
  if (!writer)
    return -1;

  assert(sizeof(uint32) == sizeof(float));
  // This union is merely used to avoid a reinterpret_cast from float& to
  // uint32& which will result in violation of strict aliasing.
  union U32 {
    uint32 u32;
    float f;
  } value;
  value.f = f;

  for (int32 i = 1; i <= 4; ++i) {
    const int32 byte_count = 4 - i;
    const int32 bit_count = byte_count * 8;

    const uint8 byte = static_cast<uint8>(value.u32 >> bit_count);

    const int32 status = writer->Write(&byte, 1);

    if (status < 0)
      return status;
  }

  return 0;
}

int32 WriteUInt(IMkvWriter* writer, uint64 value) {
  if (!writer)
    return -1;

  int32 size = GetCodedUIntSize(value);

  return WriteUIntSize(writer, value, size);
}

int32 WriteUIntSize(IMkvWriter* writer, uint64 value, int32 size) {
  if (!writer || size < 0 || size > 8)
    return -1;

  if (size > 0) {
    const uint64 bit = 1LL << (size * 7);

    if (value > (bit - 2))
      return -1;

    value |= bit;
  } else {
    size = 1;
    int64 bit;

    for (;;) {
      bit = 1LL << (size * 7);
      const uint64 max = bit - 2;

      if (value <= max)
        break;

      ++size;
    }

    if (size > 8)
      return false;

    value |= bit;
  }

  return SerializeInt(writer, value, size);
}

int32 WriteID(IMkvWriter* writer, uint64 type) {
  if (!writer)
    return -1;

  writer->ElementStartNotify(type, writer->Position());

  const int32 size = GetUIntSize(type);

  return SerializeInt(writer, type, size);
}

bool WriteEbmlMasterElement(IMkvWriter* writer, uint64 type, uint64 size) {
  if (!writer)
    return false;

  if (WriteID(writer, type))
    return false;

  if (WriteUInt(writer, size))
    return false;

  return true;
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, uint64 value) {
  return WriteEbmlElement(writer, type, value, 0);
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, uint64 value,
                      uint64 fixed_size) {
  if (!writer)
    return false;

  if (WriteID(writer, type))
    return false;

  uint64 size = GetUIntSize(value);
  if (fixed_size > 0) {
    if (size > fixed_size)
      return false;
    size = fixed_size;
  }
  if (WriteUInt(writer, size))
    return false;

  if (SerializeInt(writer, value, static_cast<int32>(size)))
    return false;

  return true;
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, int64 value) {
  if (!writer)
    return false;

  if (WriteID(writer, type))
    return 0;

  const uint64 size = GetIntSize(value);
  if (WriteUInt(writer, size))
    return false;

  if (SerializeInt(writer, value, static_cast<int32>(size)))
    return false;

  return true;
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, float value) {
  if (!writer)
    return false;

  if (WriteID(writer, type))
    return false;

  if (WriteUInt(writer, 4))
    return false;

  if (SerializeFloat(writer, value))
    return false;

  return true;
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, const char* value) {
  if (!writer || !value)
    return false;

  if (WriteID(writer, type))
    return false;

  const uint64 length = strlen(value);
  if (WriteUInt(writer, length))
    return false;

  if (writer->Write(value, static_cast<uint32>(length)))
    return false;

  return true;
}

bool WriteEbmlElement(IMkvWriter* writer, uint64 type, const uint8* value,
                      uint64 size) {
  if (!writer || !value || size < 1)
    return false;

  if (WriteID(writer, type))
    return false;

  if (WriteUInt(writer, size))
    return false;

  if (writer->Write(value, static_cast<uint32>(size)))
    return false;

  return true;
}

bool WriteEbmlDateElement(IMkvWriter* writer, uint64 type, int64 value) {
  if (!writer)
    return false;

  if (WriteID(writer, type))
    return false;

  if (WriteUInt(writer, kDateElementSize))
    return false;

  if (SerializeInt(writer, value, kDateElementSize))
    return false;

  return true;
}

uint64 WriteFrame(IMkvWriter* writer, const Frame* const frame,
                  Cluster* cluster) {
  if (!writer || !frame || !frame->IsValid() || !cluster ||
      !cluster->timecode_scale())
    return 0;

  //  Technically the timecode for a block can be less than the
  //  timecode for the cluster itself (remember that block timecode
  //  is a signed, 16-bit integer).  However, as a simplification we
  //  only permit non-negative cluster-relative timecodes for blocks.
  const int64 relative_timecode = cluster->GetRelativeTimecode(
      frame->timestamp() / cluster->timecode_scale());
  if (relative_timecode < 0 || relative_timecode > kMaxBlockTimecode)
    return 0;

  return frame->CanBeSimpleBlock()
             ? WriteSimpleBlock(writer, frame, relative_timecode)
             : WriteBlock(writer, frame, relative_timecode,
                          cluster->timecode_scale());
}

uint64 WriteVoidElement(IMkvWriter* writer, uint64 size) {
  if (!writer)
    return false;

  // Subtract one for the void ID and the coded size.
  uint64 void_entry_size = size - 1 - GetCodedUIntSize(size - 1);
  uint64 void_size = EbmlMasterElementSize(libwebm::kMkvVoid, void_entry_size) +
                     void_entry_size;

  if (void_size != size)
    return 0;

  const int64 payload_position = writer->Position();
  if (payload_position < 0)
    return 0;

  if (WriteID(writer, libwebm::kMkvVoid))
    return 0;

  if (WriteUInt(writer, void_entry_size))
    return 0;

  const uint8 value = 0;
  for (int32 i = 0; i < static_cast<int32>(void_entry_size); ++i) {
    if (writer->Write(&value, 1))
      return 0;
  }

  const int64 stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64>(void_size))
    return 0;

  return void_size;
}

void GetVersion(int32* major, int32* minor, int32* build, int32* revision) {
  *major = 0;
  *minor = 3;
  *build = 3;
  *revision = 0;
}

uint64 MakeUID(unsigned int* seed) {
  uint64 uid = 0;

  for (int i = 0; i < 7; ++i) {  // avoid problems with 8-byte values
    uid <<= 8;

// TODO(fgalligan): Move random number generation to platform specific code.
#ifdef _WIN32
    (void)seed;
    const int32 nn = rand();
#elif defined(__ANDROID__)
    (void)seed;
    int32 temp_num = 1;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd != -1) {
      read(fd, &temp_num, sizeof(temp_num));
      close(fd);
    }
    const int32 nn = temp_num;
#else
    const int32 nn = rand_r(seed);
#endif
    const int32 n = 0xFF & (nn >> 4);  // throw away low-order bits

    uid |= n;
  }

  return uid;
}

bool IsMatrixCoefficientsValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kGbr:
    case mkvmuxer::Colour::kBt709:
    case mkvmuxer::Colour::kUnspecifiedMc:
    case mkvmuxer::Colour::kReserved:
    case mkvmuxer::Colour::kFcc:
    case mkvmuxer::Colour::kBt470bg:
    case mkvmuxer::Colour::kSmpte170MMc:
    case mkvmuxer::Colour::kSmpte240MMc:
    case mkvmuxer::Colour::kYcocg:
    case mkvmuxer::Colour::kBt2020NonConstantLuminance:
    case mkvmuxer::Colour::kBt2020ConstantLuminance:
      return true;
  }
  return false;
}

bool IsChromaSitingHorzValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kUnspecifiedCsh:
    case mkvmuxer::Colour::kLeftCollocated:
    case mkvmuxer::Colour::kHalfCsh:
      return true;
  }
  return false;
}

bool IsChromaSitingVertValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kUnspecifiedCsv:
    case mkvmuxer::Colour::kTopCollocated:
    case mkvmuxer::Colour::kHalfCsv:
      return true;
  }
  return false;
}

bool IsColourRangeValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kUnspecifiedCr:
    case mkvmuxer::Colour::kBroadcastRange:
    case mkvmuxer::Colour::kFullRange:
    case mkvmuxer::Colour::kMcTcDefined:
      return true;
  }
  return false;
}

bool IsTransferCharacteristicsValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kIturBt709Tc:
    case mkvmuxer::Colour::kUnspecifiedTc:
    case mkvmuxer::Colour::kReservedTc:
    case mkvmuxer::Colour::kGamma22Curve:
    case mkvmuxer::Colour::kGamma28Curve:
    case mkvmuxer::Colour::kSmpte170MTc:
    case mkvmuxer::Colour::kSmpte240MTc:
    case mkvmuxer::Colour::kLinear:
    case mkvmuxer::Colour::kLog:
    case mkvmuxer::Colour::kLogSqrt:
    case mkvmuxer::Colour::kIec6196624:
    case mkvmuxer::Colour::kIturBt1361ExtendedColourGamut:
    case mkvmuxer::Colour::kIec6196621:
    case mkvmuxer::Colour::kIturBt202010bit:
    case mkvmuxer::Colour::kIturBt202012bit:
    case mkvmuxer::Colour::kSmpteSt2084:
    case mkvmuxer::Colour::kSmpteSt4281Tc:
    case mkvmuxer::Colour::kAribStdB67Hlg:
      return true;
  }
  return false;
}

bool IsPrimariesValueValid(uint64_t value) {
  switch (value) {
    case mkvmuxer::Colour::kReservedP0:
    case mkvmuxer::Colour::kIturBt709P:
    case mkvmuxer::Colour::kUnspecifiedP:
    case mkvmuxer::Colour::kReservedP3:
    case mkvmuxer::Colour::kIturBt470M:
    case mkvmuxer::Colour::kIturBt470Bg:
    case mkvmuxer::Colour::kSmpte170MP:
    case mkvmuxer::Colour::kSmpte240MP:
    case mkvmuxer::Colour::kFilm:
    case mkvmuxer::Colour::kIturBt2020:
    case mkvmuxer::Colour::kSmpteSt4281P:
    case mkvmuxer::Colour::kJedecP22Phosphors:
      return true;
  }
  return false;
}

}  // namespace mkvmuxer
