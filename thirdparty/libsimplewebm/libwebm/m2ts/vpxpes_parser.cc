// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "vpxpes_parser.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "common/file_util.h"

namespace libwebm {

VpxPesParser::BcmvHeader::BcmvHeader(std::uint32_t len) : length(len) {
  id[0] = 'B';
  id[1] = 'C';
  id[2] = 'M';
  id[3] = 'V';
}

bool VpxPesParser::BcmvHeader::operator==(const BcmvHeader& other) const {
  return (other.length == length && other.id[0] == id[0] &&
          other.id[1] == id[1] && other.id[2] == id[2] && other.id[3] == id[3]);
}

bool VpxPesParser::BcmvHeader::Valid() const {
  return (length > 0 && id[0] == 'B' && id[1] == 'C' && id[2] == 'M' &&
          id[3] == 'V');
}

// TODO(tomfinegan): Break Open() into separate functions. One that opens the
// file, and one that reads one packet at a time. As things are files larger
// than the maximum availble memory for the current process cannot be loaded.
bool VpxPesParser::Open(const std::string& pes_file) {
  pes_file_size_ = static_cast<size_t>(libwebm::GetFileSize(pes_file));
  if (pes_file_size_ <= 0)
    return false;
  pes_file_data_.reserve(static_cast<size_t>(pes_file_size_));
  libwebm::FilePtr file = libwebm::FilePtr(std::fopen(pes_file.c_str(), "rb"),
                                           libwebm::FILEDeleter());
  int byte;
  while ((byte = fgetc(file.get())) != EOF) {
    pes_file_data_.push_back(static_cast<std::uint8_t>(byte));
  }

  if (!feof(file.get()) || ferror(file.get()) ||
      pes_file_size_ != pes_file_data_.size()) {
    return false;
  }

  read_pos_ = 0;
  parse_state_ = kFindStartCode;
  return true;
}

bool VpxPesParser::VerifyPacketStartCode() const {
  if (read_pos_ + 2 > pes_file_data_.size())
    return false;

  // PES packets all start with the byte sequence 0x0 0x0 0x1.
  if (pes_file_data_[read_pos_] != 0 || pes_file_data_[read_pos_ + 1] != 0 ||
      pes_file_data_[read_pos_ + 2] != 1) {
    return false;
  }

  return true;
}

bool VpxPesParser::ReadStreamId(std::uint8_t* stream_id) const {
  if (!stream_id || BytesAvailable() < 4)
    return false;

  *stream_id = pes_file_data_[read_pos_ + 3];
  return true;
}

bool VpxPesParser::ReadPacketLength(std::uint16_t* packet_length) const {
  if (!packet_length || BytesAvailable() < 6)
    return false;

  // Read and byte swap 16 bit big endian length.
  *packet_length =
      (pes_file_data_[read_pos_ + 4] << 8) | pes_file_data_[read_pos_ + 5];

  return true;
}

bool VpxPesParser::ParsePesHeader(PesHeader* header) {
  if (!header || parse_state_ != kParsePesHeader)
    return false;

  if (!VerifyPacketStartCode())
    return false;

  std::size_t pos = read_pos_;
  for (auto& a : header->start_code) {
    a = pes_file_data_[pos++];
  }

  // PES Video stream IDs start at E0.
  if (!ReadStreamId(&header->stream_id))
    return false;

  if (header->stream_id < kMinVideoStreamId ||
      header->stream_id > kMaxVideoStreamId)
    return false;

  if (!ReadPacketLength(&header->packet_length))
    return false;

  read_pos_ += kPesHeaderSize;
  parse_state_ = kParsePesOptionalHeader;
  return true;
}

// TODO(tomfinegan): Make these masks constants.
bool VpxPesParser::ParsePesOptionalHeader(PesOptionalHeader* header) {
  if (!header || parse_state_ != kParsePesOptionalHeader ||
      read_pos_ >= pes_file_size_) {
    return false;
  }

  std::size_t consumed = 0;
  PacketData poh_buffer;
  if (!RemoveStartCodeEmulationPreventionBytes(&pes_file_data_[read_pos_],
                                               kPesOptionalHeaderSize,
                                               &poh_buffer, &consumed)) {
    return false;
  }

  std::size_t offset = 0;
  header->marker = (poh_buffer[offset] & 0x80) >> 6;
  header->scrambling = (poh_buffer[offset] & 0x30) >> 4;
  header->priority = (poh_buffer[offset] & 0x8) >> 3;
  header->data_alignment = (poh_buffer[offset] & 0xc) >> 2;
  header->copyright = (poh_buffer[offset] & 0x2) >> 1;
  header->original = poh_buffer[offset] & 0x1;
  offset++;

  header->has_pts = (poh_buffer[offset] & 0x80) >> 7;
  header->has_dts = (poh_buffer[offset] & 0x40) >> 6;
  header->unused_fields = poh_buffer[offset] & 0x3f;
  offset++;

  header->remaining_size = poh_buffer[offset];
  if (header->remaining_size !=
      static_cast<int>(kWebm2PesOptHeaderRemainingSize))
    return false;

  size_t bytes_left = header->remaining_size;
  offset++;

  if (header->has_pts) {
    // Read PTS markers. Format:
    // PTS: 5 bytes
    //   4 bits (flag: PTS present, but no DTS): 0x2 ('0010')
    //   36 bits (90khz PTS):
    //     top 3 bits
    //     marker ('1')
    //     middle 15 bits
    //     marker ('1')
    //     bottom 15 bits
    //     marker ('1')
    // TODO(tomfinegan): read/store the timestamp.
    header->pts_dts_flag = (poh_buffer[offset] & 0x20) >> 4;
    // Check the marker bits.
    if ((poh_buffer[offset + 0] & 1) != 1 ||
        (poh_buffer[offset + 2] & 1) != 1 ||
        (poh_buffer[offset + 4] & 1) != 1) {
      return false;
    }

    header->pts = (poh_buffer[offset] & 0xe) << 29 |
                  ((ReadUint16(&poh_buffer[offset + 1]) & ~1) << 14) |
                  (ReadUint16(&poh_buffer[offset + 3]) >> 1);
    offset += 5;
    bytes_left -= 5;
  }

  // Validate stuffing byte(s).
  for (size_t i = 0; i < bytes_left; ++i) {
    if (poh_buffer[offset + i] != 0xff)
      return false;
  }

  read_pos_ += consumed;
  parse_state_ = kParseBcmvHeader;

  return true;
}

// Parses and validates a BCMV header.
bool VpxPesParser::ParseBcmvHeader(BcmvHeader* header) {
  if (!header || parse_state_ != kParseBcmvHeader)
    return false;

  PacketData bcmv_buffer;
  std::size_t consumed = 0;
  if (!RemoveStartCodeEmulationPreventionBytes(&pes_file_data_[read_pos_],
                                               kBcmvHeaderSize, &bcmv_buffer,
                                               &consumed)) {
    return false;
  }

  std::size_t offset = 0;
  header->id[0] = bcmv_buffer[offset++];
  header->id[1] = bcmv_buffer[offset++];
  header->id[2] = bcmv_buffer[offset++];
  header->id[3] = bcmv_buffer[offset++];

  header->length = 0;
  header->length |= bcmv_buffer[offset++] << 24;
  header->length |= bcmv_buffer[offset++] << 16;
  header->length |= bcmv_buffer[offset++] << 8;
  header->length |= bcmv_buffer[offset++];

  // Length stored in the BCMV header is followed by 2 bytes of 0 padding.
  if (bcmv_buffer[offset++] != 0 || bcmv_buffer[offset++] != 0)
    return false;

  if (!header->Valid())
    return false;

  parse_state_ = kFindStartCode;
  read_pos_ += consumed;

  return true;
}

bool VpxPesParser::FindStartCode(std::size_t origin,
                                 std::size_t* offset) const {
  if (read_pos_ + 2 >= pes_file_size_)
    return false;

  const std::size_t length = pes_file_size_ - origin;
  if (length < 3)
    return false;

  const uint8_t* const data = &pes_file_data_[origin];
  for (std::size_t i = 0; i < length - 3; ++i) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
      *offset = origin + i;
      return true;
    }
  }

  return false;
}

bool VpxPesParser::IsPayloadFragmented(const PesHeader& header) const {
  return (header.packet_length != 0 &&
          (header.packet_length - kPesOptionalHeaderSize) !=
              header.bcmv_header.length);
}

bool VpxPesParser::AccumulateFragmentedPayload(std::size_t pes_packet_length,
                                               std::size_t payload_length) {
  const std::size_t first_fragment_length =
      pes_packet_length - kPesOptionalHeaderSize - kBcmvHeaderSize;
  for (std::size_t i = 0; i < first_fragment_length; ++i) {
    payload_.push_back(pes_file_data_[read_pos_ + i]);
  }
  read_pos_ += first_fragment_length;
  parse_state_ = kFindStartCode;

  while (payload_.size() < payload_length) {
    PesHeader header;
    std::size_t packet_start_pos = read_pos_;
    if (!FindStartCode(read_pos_, &packet_start_pos)) {
      return false;
    }
    parse_state_ = kParsePesHeader;
    read_pos_ = packet_start_pos;

    if (!ParsePesHeader(&header)) {
      return false;
    }
    if (!ParsePesOptionalHeader(&header.opt_header)) {
      return false;
    }

    const std::size_t fragment_length =
        header.packet_length - kPesOptionalHeaderSize;
    std::size_t consumed = 0;
    if (!RemoveStartCodeEmulationPreventionBytes(&pes_file_data_[read_pos_],
                                                 fragment_length, &payload_,
                                                 &consumed)) {
      return false;
    }
    read_pos_ += consumed;
  }
  return true;
}

bool VpxPesParser::RemoveStartCodeEmulationPreventionBytes(
    const std::uint8_t* raw_data, std::size_t bytes_required,
    PacketData* processed_data, std::size_t* bytes_consumed) const {
  if (bytes_required == 0 || !processed_data)
    return false;

  std::size_t num_zeros = 0;
  std::size_t bytes_copied = 0;
  const std::uint8_t* const end_of_input =
      &pes_file_data_[0] + pes_file_data_.size();
  std::size_t i;
  for (i = 0; bytes_copied < bytes_required; ++i) {
    if (raw_data + i > end_of_input)
      return false;

    bool skip = false;

    const std::uint8_t byte = raw_data[i];
    if (byte == 0) {
      ++num_zeros;
    } else if (byte == 0x3 && num_zeros == 2) {
      skip = true;
      num_zeros = 0;
    } else {
      num_zeros = 0;
    }

    if (skip == false) {
      processed_data->push_back(byte);
      ++bytes_copied;
    }
  }
  *bytes_consumed = i;
  return true;
}

int VpxPesParser::BytesAvailable() const {
  return static_cast<int>(pes_file_data_.size() - read_pos_);
}

bool VpxPesParser::ParseNextPacket(PesHeader* header, VideoFrame* frame) {
  if (!header || !frame || parse_state_ != kFindStartCode ||
      BytesAvailable() == 0) {
    return false;
  }

  std::size_t packet_start_pos = read_pos_;
  if (!FindStartCode(read_pos_, &packet_start_pos)) {
    return false;
  }
  parse_state_ = kParsePesHeader;
  read_pos_ = packet_start_pos;

  if (!ParsePesHeader(header)) {
    return false;
  }
  if (!ParsePesOptionalHeader(&header->opt_header)) {
    return false;
  }
  if (!ParseBcmvHeader(&header->bcmv_header)) {
    return false;
  }

  // BCMV header length includes the length of the BCMVHeader itself. Adjust:
  const std::size_t payload_length =
      header->bcmv_header.length - BcmvHeader::size();

  // Make sure there's enough input data to read the entire frame.
  if (read_pos_ + payload_length > pes_file_data_.size()) {
    // Need more data.
    printf("VpxPesParser: Not enough data. Required: %u Available: %u\n",
           static_cast<unsigned int>(payload_length),
           static_cast<unsigned int>(pes_file_data_.size() - read_pos_));
    parse_state_ = kFindStartCode;
    read_pos_ = packet_start_pos;
    return false;
  }

  if (IsPayloadFragmented(*header)) {
    if (!AccumulateFragmentedPayload(header->packet_length, payload_length)) {
      fprintf(stderr, "VpxPesParser: Failed parsing fragmented payload!\n");
      return false;
    }
  } else {
    std::size_t consumed = 0;
    if (!RemoveStartCodeEmulationPreventionBytes(
            &pes_file_data_[read_pos_], payload_length, &payload_, &consumed)) {
      return false;
    }
    read_pos_ += consumed;
  }

  if (frame->buffer().capacity < payload_.size()) {
    if (frame->Init(payload_.size()) == false) {
      fprintf(stderr, "VpxPesParser: Out of memory.\n");
      return false;
    }
  }
  frame->set_nanosecond_pts(Khz90TicksToNanoseconds(header->opt_header.pts));
  std::memcpy(frame->buffer().data.get(), &payload_[0], payload_.size());
  frame->SetBufferLength(payload_.size());

  payload_.clear();
  parse_state_ = kFindStartCode;

  return true;
}

}  // namespace libwebm
