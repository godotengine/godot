// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "m2ts/vpxpes2ts.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

#include "m2ts/webm2pes.h"

namespace libwebm {
// TODO(tomfinegan): Dedupe this and PesHeaderField.
// Stores a value and its size in bits for writing into a MPEG2 TS Header.
// Maximum size is 64 bits. Users may call the Check() method to perform minimal
// validation (size > 0 and <= 64).
struct TsHeaderField {
  TsHeaderField(std::uint64_t value, std::uint32_t size_in_bits,
                std::uint8_t byte_index, std::uint8_t bits_to_shift)
      : bits(value),
        num_bits(size_in_bits),
        index(byte_index),
        shift(bits_to_shift) {}
  TsHeaderField() = delete;
  TsHeaderField(const TsHeaderField&) = default;
  TsHeaderField(TsHeaderField&&) = default;
  ~TsHeaderField() = default;
  bool Check() const {
    return num_bits > 0 && num_bits <= 64 && shift >= 0 && shift < 64;
  }

  // Value to be stored in the field.
  std::uint64_t bits;

  // Number of bits in the value.
  const int num_bits;

  // Index into the header for the byte in which |bits| will be written.
  const std::uint8_t index;

  // Number of bits to left shift value before or'ing. Ignored for whole bytes.
  const int shift;
};

// Data storage for MPEG2 Transport Stream headers.
// https://en.wikipedia.org/wiki/MPEG_transport_stream#Packet
struct TsHeader {
  TsHeader(bool payload_start, bool adaptation_flag, std::uint8_t counter)
      : is_payload_start(payload_start),
        has_adaptation(adaptation_flag),
        counter_value(counter) {}
  TsHeader() = delete;
  TsHeader(const TsHeader&) = default;
  TsHeader(TsHeader&&) = default;
  ~TsHeader() = default;

  void Write(PacketDataBuffer* buffer) const;

  // Indicates the packet is the beginning of a new fragmented payload.
  const bool is_payload_start;

  // Indicates the packet contains an adaptation field.
  const bool has_adaptation;

  // The sync byte is the bit pattern of 0x47 (ASCII char 'G').
  const std::uint8_t kTsHeaderSyncByte = 0x47;
  const std::uint8_t sync_byte = kTsHeaderSyncByte;

  // Value for |continuity_counter|. Used to detect gaps when demuxing.
  const std::uint8_t counter_value;

  // Set when FEC is impossible. Always 0.
  const TsHeaderField transport_error_indicator = TsHeaderField(0, 1, 1, 7);

  // This MPEG2 TS header is the start of a new payload (aka PES packet).
  const TsHeaderField payload_unit_start_indicator =
      TsHeaderField(is_payload_start ? 1 : 0, 1, 1, 6);

  // Set when the current packet has a higher priority than other packets with
  // the same PID. Always 0 for VPX.
  const TsHeaderField transport_priority = TsHeaderField(0, 1, 1, 5);

  // https://en.wikipedia.org/wiki/MPEG_transport_stream#Packet_Identifier_.28PID.29
  // 0x0020-0x1FFA May be assigned as needed to Program Map Tables, elementary
  // streams and other data tables.
  // Note: Though we hard code to 0x20, this value is actually 13 bits-- the
  // buffer for the header is always set to 0, so it doesn't matter in practice.
  const TsHeaderField pid = TsHeaderField(0x20, 8, 2, 0);

  // Indicates scrambling key. Unused; always 0.
  const TsHeaderField scrambling_control = TsHeaderField(0, 2, 3, 6);

  // Adaptation field flag. Unused; always 0.
  // TODO(tomfinegan): Not sure this is OK. Might need to add support for
  // writing the Adaptation Field.
  const TsHeaderField adaptation_field_flag =
      TsHeaderField(has_adaptation ? 1 : 0, 1, 3, 5);

  // Payload flag. All output packets created here have payloads. Always 1.
  const TsHeaderField payload_flag = TsHeaderField(1, 1, 3, 4);

  // Continuity counter. Two bit field that is incremented for every packet.
  const TsHeaderField continuity_counter =
      TsHeaderField(counter_value, 4, 3, 3);
};

void TsHeader::Write(PacketDataBuffer* buffer) const {
  std::uint8_t* byte = &(*buffer)[0];
  *byte = sync_byte;

  *++byte = 0;
  *byte |= transport_error_indicator.bits << transport_error_indicator.shift;
  *byte |= payload_unit_start_indicator.bits
           << payload_unit_start_indicator.shift;
  *byte |= transport_priority.bits << transport_priority.shift;

  *++byte = pid.bits & 0xff;

  *++byte = 0;
  *byte |= scrambling_control.bits << scrambling_control.shift;
  *byte |= adaptation_field_flag.bits << adaptation_field_flag.shift;
  *byte |= payload_flag.bits << payload_flag.shift;
  *byte |= continuity_counter.bits;  // last 4 bits.
}

bool VpxPes2Ts::ConvertToFile() {
  output_file_ = FilePtr(fopen(output_file_name_.c_str(), "wb"), FILEDeleter());
  if (output_file_ == nullptr) {
    std::fprintf(stderr, "VpxPes2Ts: Cannot open %s for output.\n",
                 output_file_name_.c_str());
    return false;
  }
  pes_converter_.reset(new (std::nothrow) Webm2Pes(input_file_name_, this));
  if (pes_converter_ == nullptr) {
    std::fprintf(stderr, "VpxPes2Ts: Out of memory.\n");
    return false;
  }
  return pes_converter_->ConvertToPacketReceiver();
}

bool VpxPes2Ts::ReceivePacket(const PacketDataBuffer& packet_data) {
  const int kTsHeaderSize = 4;
  const int kTsPayloadSize = 184;
  const int kTsPacketSize = kTsHeaderSize + kTsPayloadSize;
  int bytes_to_packetize = static_cast<int>(packet_data.size());
  std::uint8_t continuity_counter = 0;
  std::size_t read_pos = 0;

  ts_buffer_.reserve(kTsPacketSize);

  while (bytes_to_packetize > 0) {
    if (continuity_counter > 0xf)
      continuity_counter = 0;

    // Calculate payload size (need to know if we'll have to pad with an empty
    // adaptation field).
    int payload_size = std::min(bytes_to_packetize, kTsPayloadSize);

    // Write the TS header.
    const TsHeader header(
        bytes_to_packetize == static_cast<int>(packet_data.size()),
        payload_size != kTsPayloadSize, continuity_counter);
    header.Write(&ts_buffer_);
    int write_pos = kTsHeaderSize;

    // (pre)Pad payload with an empty adaptation field. All packets must be
    // |kTsPacketSize| (188).
    if (payload_size < kTsPayloadSize) {
      // We need at least 2 bytes to write an empty adaptation field.
      if (payload_size == (kTsPayloadSize - 1)) {
        payload_size--;
      }

      // Padding adaptation field:
      //   8 bits: number of adaptation field bytes following this byte.
      //   8 bits: unused (in this program) flags.
      // This is followed by a run of 0xff to reach |kTsPayloadSize| (184)
      // bytes.
      const int pad_size = kTsPayloadSize - payload_size - 1 - 1;
      ts_buffer_[write_pos++] = pad_size + 1;
      ts_buffer_[write_pos++] = 0;

      const std::uint8_t kStuffingByte = 0xff;
      for (int i = 0; i < pad_size; ++i) {
        ts_buffer_[write_pos++] = kStuffingByte;
      }
    }

    for (int i = 0; i < payload_size; ++i) {
      ts_buffer_[write_pos++] = packet_data[read_pos++];
    }

    bytes_to_packetize -= payload_size;
    continuity_counter++;

    if (write_pos != kTsPacketSize) {
      fprintf(stderr, "VpxPes2Ts: Invalid packet length.\n");
      return false;
    }

    // Write contents of |ts_buffer_| to |output_file_|.
    // TODO(tomfinegan): Writing 188 bytes at a time isn't exactly efficient...
    // Fix me.
    if (static_cast<int>(std::fwrite(&ts_buffer_[0], 1, kTsPacketSize,
                                     output_file_.get())) != kTsPacketSize) {
      std::fprintf(stderr, "VpxPes2Ts: TS packet write failed.\n");
      return false;
    }
  }

  return true;
}

}  // namespace libwebm
