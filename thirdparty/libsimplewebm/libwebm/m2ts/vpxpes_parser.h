// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_M2TS_VPXPES_PARSER_H_
#define LIBWEBM_M2TS_VPXPES_PARSER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common/libwebm_util.h"
#include "common/video_frame.h"

namespace libwebm {

// Parser for VPx PES. Requires that the _entire_ PES stream can be stored in
// a std::vector<std::uint8_t> and read into memory when Open() is called.
// TODO(tomfinegan): Support incremental parse.
class VpxPesParser {
 public:
  typedef std::vector<std::uint8_t> PesFileData;
  typedef std::vector<std::uint8_t> PacketData;

  enum ParseState {
    kFindStartCode,
    kParsePesHeader,
    kParsePesOptionalHeader,
    kParseBcmvHeader,
  };

  struct PesOptionalHeader {
    int marker = 0;
    int scrambling = 0;
    int priority = 0;
    int data_alignment = 0;
    int copyright = 0;
    int original = 0;
    int has_pts = 0;
    int has_dts = 0;
    int unused_fields = 0;
    int remaining_size = 0;
    int pts_dts_flag = 0;
    std::uint64_t pts = 0;
    int stuffing_byte = 0;
  };

  struct BcmvHeader {
    BcmvHeader() = default;
    ~BcmvHeader() = default;
    BcmvHeader(const BcmvHeader&) = delete;
    BcmvHeader(BcmvHeader&&) = delete;

    // Convenience ctor for quick validation of expected values via operator==
    // after parsing input.
    explicit BcmvHeader(std::uint32_t len);

    bool operator==(const BcmvHeader& other) const;

    void Reset();
    bool Valid() const;
    static std::size_t size() { return 10; }

    char id[4] = {0};
    std::uint32_t length = 0;
  };

  struct PesHeader {
    std::uint8_t start_code[4] = {0};
    std::uint16_t packet_length = 0;
    std::uint8_t stream_id = 0;
    PesOptionalHeader opt_header;
    BcmvHeader bcmv_header;
  };

  // Constants for validating known values from input data.
  const std::uint8_t kMinVideoStreamId = 0xE0;
  const std::uint8_t kMaxVideoStreamId = 0xEF;
  const std::size_t kPesHeaderSize = 6;
  const std::size_t kPesOptionalHeaderStartOffset = kPesHeaderSize;
  const std::size_t kPesOptionalHeaderSize = 9;
  const std::size_t kPesOptionalHeaderMarkerValue = 0x2;
  const std::size_t kWebm2PesOptHeaderRemainingSize = 6;
  const std::size_t kBcmvHeaderSize = 10;

  VpxPesParser() = default;
  ~VpxPesParser() = default;

  // Opens file specified by |pes_file_path| and reads its contents. Returns
  // true after successful read of input file.
  bool Open(const std::string& pes_file_path);

  // Parses the next packet in the PES. PES header information is stored in
  // |header|, and the frame payload is stored in |frame|. Returns true when
  // a full frame has been consumed from the PES.
  bool ParseNextPacket(PesHeader* header, VideoFrame* frame);

  // PES Header parsing utility functions.
  // PES Header structure:
  // Start code         Stream ID   Packet length (16 bits)
  // /                  /      ____/
  // |                  |     /
  // Byte0 Byte1  Byte2 Byte3 Byte4 Byte5
  //     0     0      1     X           Y
  bool VerifyPacketStartCode() const;
  bool ReadStreamId(std::uint8_t* stream_id) const;
  bool ReadPacketLength(std::uint16_t* packet_length) const;

  std::uint64_t pes_file_size() const { return pes_file_size_; }
  const PesFileData& pes_file_data() const { return pes_file_data_; }

  // Returns number of unparsed bytes remaining.
  int BytesAvailable() const;

 private:
  // Parses and verifies the static 6 byte portion that begins every PES packet.
  bool ParsePesHeader(PesHeader* header);

  // Parses a PES optional header, the optional header following the static
  // header that begins the VPX PES packet.
  // https://en.wikipedia.org/wiki/Packetized_elementary_stream
  bool ParsePesOptionalHeader(PesOptionalHeader* header);

  // Parses and validates the BCMV header. This immediately follows the optional
  // header.
  bool ParseBcmvHeader(BcmvHeader* header);

  // Returns true when a start code is found and sets |offset| to the position
  // of the start code relative to |pes_file_data_[read_pos_]|.
  // Does not set |offset| value if the end of |pes_file_data_| is reached
  // without locating a start code.
  // Note: A start code is the byte sequence 0x00 0x00 0x01.
  bool FindStartCode(std::size_t origin, std::size_t* offset) const;

  // Returns true when a PES packet containing a BCMV header contains only a
  // portion of the frame payload length reported by the BCMV header.
  bool IsPayloadFragmented(const PesHeader& header) const;

  // Parses PES and PES Optional header while accumulating payload data in
  // |payload_|.
  // Returns true once all payload fragments have been stored in |payload_|.
  // Returns false if unable to accumulate full payload.
  bool AccumulateFragmentedPayload(std::size_t pes_packet_length,
                                   std::size_t payload_length);

  // The byte sequence 0x0 0x0 0x1 is a start code in PES. When PES muxers
  // encounter 0x0 0x0 0x1 or 0x0 0x0 0x3, an additional 0x3 is inserted into
  // the PES. The following change occurs:
  //    0x0 0x0 0x1  =>  0x0 0x0 0x3 0x1
  //    0x0 0x0 0x3  =>  0x0 0x0 0x3 0x3
  // PES demuxers must reverse the change:
  //    0x0 0x0 0x3 0x1  =>  0x0 0x0 0x1
  //    0x0 0x0 0x3 0x3  =>  0x0 0x0 0x3
  // PES optional header, BCMV header, and payload data must be preprocessed to
  // avoid potentially invalid data due to the presence of inserted bytes.
  //
  // Removes start code emulation prevention bytes while copying data from
  // |raw_data| to |processed_data|. Returns true when |bytes_required| bytes
  // have been written to |processed_data|. Reports bytes consumed during the
  // operation via |bytes_consumed|.
  bool RemoveStartCodeEmulationPreventionBytes(
      const std::uint8_t* raw_data, std::size_t bytes_required,
      PacketData* processed_data, std::size_t* bytes_consumed) const;

  std::size_t pes_file_size_ = 0;
  PacketData payload_;
  PesFileData pes_file_data_;
  std::size_t read_pos_ = 0;
  ParseState parse_state_ = kFindStartCode;
};

}  // namespace libwebm

#endif  // LIBWEBM_M2TS_VPXPES_PARSER_H_
