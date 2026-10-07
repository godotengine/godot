// Copyright (c) 2015 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "m2ts/webm2pes.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "common/libwebm_util.h"

namespace libwebm {

const std::size_t Webm2Pes::kMaxPayloadSize = 32768;

namespace {

std::string ToString(const char* str) {
  return std::string((str == nullptr) ? "" : str);
}

}  // namespace

//
// PesOptionalHeader methods.
//

void PesOptionalHeader::SetPtsBits(std::int64_t pts_90khz) {
  // PTS is broken up and stored in 40 bits as shown:
  //
  //  PES PTS Only flag
  // /                  Marker              Marker              Marker
  // |                 /                   /                   /
  // |                 |                   |                   |
  // 7654  321         0  765432107654321  0  765432107654321  0
  // 0010  PTS 32-30   1  PTS 29-15        1  PTS 14-0         1
  const std::uint32_t pts1 = (pts_90khz >> 30) & 0x7;
  const std::uint32_t pts2 = (pts_90khz >> 15) & 0x7FFF;
  const std::uint32_t pts3 = pts_90khz & 0x7FFF;

  pts.bits = 0;

  // bottom 7 bits of second PTS chunk.
  pts.bits |= (pts3 << 1) & 0xff;
  // Marker.
  pts.bits |= 1;

  // Last 15 bits of pts and 1 bit marker.
  // Top 8 bits of second PTS chunk.
  pts.bits <<= 8;
  pts.bits |= (pts3 >> 7) & 0xff;

  // bottom 7 bits of second PTS chunk.
  pts.bits <<= 8;
  pts.bits |= (pts2 << 1);
  // Marker.
  pts.bits |= 1;

  // Next 15 bits of pts and 1 bit marker.
  // Top 8 bits of second PTS chunk.
  pts.bits <<= 8;
  pts.bits |= (pts2 >> 7) & 0xff;

  // PTS only flag.
  pts.bits <<= 8;
  pts.bits |= 1 << 5;
  // Top 3 bits of PTS and 1 bit marker.
  pts.bits |= pts1 << 1;
  // Marker.
  pts.bits |= 1;
}

// Writes fields to |buffer| and returns true. Returns false when write or
// field value validation fails.
bool PesOptionalHeader::Write(bool write_pts, PacketDataBuffer* buffer) const {
  if (buffer == nullptr) {
    std::fprintf(stderr, "Webm2Pes: nullptr in opt header writer.\n");
    return false;
  }

  const int kHeaderSize = 9;
  std::uint8_t header[kHeaderSize] = {0};
  std::uint8_t* byte = header;

  if (marker.Check() != true || scrambling.Check() != true ||
      priority.Check() != true || data_alignment.Check() != true ||
      copyright.Check() != true || original.Check() != true ||
      has_pts.Check() != true || has_dts.Check() != true ||
      pts.Check() != true || stuffing_byte.Check() != true) {
    std::fprintf(stderr, "Webm2Pes: Invalid PES Optional Header field.\n");
    return false;
  }

  // TODO(tomfinegan): As noted in above, the PesHeaderFields should be an
  // array (or some data structure) that can be iterated over.

  // First byte of header, fields: marker, scrambling, priority, alignment,
  // copyright, original.
  *byte = 0;
  *byte |= marker.bits << marker.shift;
  *byte |= scrambling.bits << scrambling.shift;
  *byte |= priority.bits << priority.shift;
  *byte |= data_alignment.bits << data_alignment.shift;
  *byte |= copyright.bits << copyright.shift;
  *byte |= original.bits << original.shift;

  // Second byte of header, fields: has_pts, has_dts, unused fields.
  *++byte = 0;
  if (write_pts == true)
    *byte |= has_pts.bits << has_pts.shift;

  *byte |= has_dts.bits << has_dts.shift;

  // Third byte of header, fields: remaining size of header.
  *++byte = remaining_size.bits & 0xff;  // Field is 8 bits wide.

  int num_stuffing_bytes =
      (pts.num_bits + 7) / 8 + 1 /* always 1 stuffing byte */;
  if (write_pts == true) {
    // Write the PTS value as big endian and adjust stuffing byte count
    // accordingly.
    *++byte = pts.bits & 0xff;
    *++byte = (pts.bits >> 8) & 0xff;
    *++byte = (pts.bits >> 16) & 0xff;
    *++byte = (pts.bits >> 24) & 0xff;
    *++byte = (pts.bits >> 32) & 0xff;
    num_stuffing_bytes = 1;
  }

  // Add the stuffing byte(s).
  for (int i = 0; i < num_stuffing_bytes; ++i)
    *++byte = stuffing_byte.bits & 0xff;

  return CopyAndEscapeStartCodes(&header[0], kHeaderSize, buffer);
}

//
// BCMVHeader methods.
//

bool BCMVHeader::Write(PacketDataBuffer* buffer) const {
  if (buffer == nullptr) {
    std::fprintf(stderr, "Webm2Pes: nullptr for buffer in BCMV Write.\n");
    return false;
  }
  const int kBcmvSize = 4;
  for (int i = 0; i < kBcmvSize; ++i)
    buffer->push_back(bcmv[i]);

  // Note: The 4 byte length field must include the size of the BCMV header.
  const int kRemainingBytes = 6;
  const uint32_t bcmv_total_length = length + static_cast<uint32_t>(size());
  const uint8_t bcmv_buffer[kRemainingBytes] = {
      static_cast<std::uint8_t>((bcmv_total_length >> 24) & 0xff),
      static_cast<std::uint8_t>((bcmv_total_length >> 16) & 0xff),
      static_cast<std::uint8_t>((bcmv_total_length >> 8) & 0xff),
      static_cast<std::uint8_t>(bcmv_total_length & 0xff),
      0,
      0 /* 2 bytes 0 padding */};

  return CopyAndEscapeStartCodes(bcmv_buffer, kRemainingBytes, buffer);
}

//
// PesHeader methods.
//

// Writes out the header to |buffer|. Calls PesOptionalHeader::Write() to write
// |optional_header| contents. Returns true when successful, false otherwise.
bool PesHeader::Write(bool write_pts, PacketDataBuffer* buffer) const {
  if (buffer == nullptr) {
    std::fprintf(stderr, "Webm2Pes: nullptr in header writer.\n");
    return false;
  }

  // Write |start_code|.
  const int kStartCodeLength = 4;
  for (int i = 0; i < kStartCodeLength; ++i)
    buffer->push_back(start_code[i]);

  // The length field here reports number of bytes following the field. The
  // length of the optional header must be added to the payload length set by
  // the user.
  const std::size_t header_length =
      packet_length + optional_header.size_in_bytes();
  if (header_length > UINT16_MAX)
    return false;

  // Write |header_length| as big endian.
  std::uint8_t byte = (header_length >> 8) & 0xff;
  buffer->push_back(byte);
  byte = header_length & 0xff;
  buffer->push_back(byte);

  // Write the (not really) optional header.
  if (optional_header.Write(write_pts, buffer) != true) {
    std::fprintf(stderr, "Webm2Pes: PES optional header write failed.");
    return false;
  }
  return true;
}

//
// Webm2Pes methods.
//

bool Webm2Pes::ConvertToFile() {
  if (input_file_name_.empty() || output_file_name_.empty()) {
    std::fprintf(stderr, "Webm2Pes: input and/or output file name(s) empty.\n");
    return false;
  }

  output_file_ = FilePtr(fopen(output_file_name_.c_str(), "wb"), FILEDeleter());
  if (output_file_ == nullptr) {
    std::fprintf(stderr, "Webm2Pes: Cannot open %s for output.\n",
                 output_file_name_.c_str());
    return false;
  }

  if (InitWebmParser() != true) {
    std::fprintf(stderr, "Webm2Pes: Cannot initialize WebM parser.\n");
    return false;
  }

  // Walk clusters in segment.
  const mkvparser::Cluster* cluster = webm_parser_->GetFirst();
  while (cluster != nullptr && cluster->EOS() == false) {
    const mkvparser::BlockEntry* block_entry = nullptr;
    std::int64_t block_status = cluster->GetFirst(block_entry);
    if (block_status < 0) {
      std::fprintf(stderr, "Webm2Pes: Cannot parse first block in %s.\n",
                   input_file_name_.c_str());
      return false;
    }

    // Walk blocks in cluster.
    while (block_entry != nullptr && block_entry->EOS() == false) {
      const mkvparser::Block* block = block_entry->GetBlock();
      if (block->GetTrackNumber() == video_track_num_) {
        const int frame_count = block->GetFrameCount();

        // Walk frames in block.
        for (int frame_num = 0; frame_num < frame_count; ++frame_num) {
          const mkvparser::Block::Frame& mkvparser_frame =
              block->GetFrame(frame_num);

          // Read the frame.
          VideoFrame vpx_frame(block->GetTime(cluster), codec_);
          if (ReadVideoFrame(mkvparser_frame, &vpx_frame) == false) {
            fprintf(stderr, "Webm2Pes: frame read failed.\n");
            return false;
          }

          // Write frame out as PES packet(s).
          if (WritePesPacket(vpx_frame, &packet_data_) == false) {
            std::fprintf(stderr, "Webm2Pes: WritePesPacket failed.\n");
            return false;
          }

          // Write contents of |packet_data_| to |output_file_|.
          if (std::fwrite(&packet_data_[0], 1, packet_data_.size(),
                          output_file_.get()) != packet_data_.size()) {
            std::fprintf(stderr, "Webm2Pes: packet payload write failed.\n");
            return false;
          }
          bytes_written_ += packet_data_.size();
        }
      }
      block_status = cluster->GetNext(block_entry, block_entry);
      if (block_status < 0) {
        std::fprintf(stderr, "Webm2Pes: Cannot parse block in %s.\n",
                     input_file_name_.c_str());
        return false;
      }
    }

    cluster = webm_parser_->GetNext(cluster);
  }

  std::fflush(output_file_.get());
  return true;
}

bool Webm2Pes::ConvertToPacketReceiver() {
  if (input_file_name_.empty() || packet_sink_ == nullptr) {
    std::fprintf(stderr, "Webm2Pes: input file name empty or null sink.\n");
    return false;
  }

  if (InitWebmParser() != true) {
    std::fprintf(stderr, "Webm2Pes: Cannot initialize WebM parser.\n");
    return false;
  }

  // Walk clusters in segment.
  const mkvparser::Cluster* cluster = webm_parser_->GetFirst();
  while (cluster != nullptr && cluster->EOS() == false) {
    const mkvparser::BlockEntry* block_entry = nullptr;
    std::int64_t block_status = cluster->GetFirst(block_entry);
    if (block_status < 0) {
      std::fprintf(stderr, "Webm2Pes: Cannot parse first block in %s.\n",
                   input_file_name_.c_str());
      return false;
    }

    // Walk blocks in cluster.
    while (block_entry != nullptr && block_entry->EOS() == false) {
      const mkvparser::Block* block = block_entry->GetBlock();
      if (block->GetTrackNumber() == video_track_num_) {
        const int frame_count = block->GetFrameCount();

        // Walk frames in block.
        for (int frame_num = 0; frame_num < frame_count; ++frame_num) {
          const mkvparser::Block::Frame& mkvparser_frame =
              block->GetFrame(frame_num);

          // Read the frame.
          VideoFrame frame(block->GetTime(cluster), codec_);
          if (ReadVideoFrame(mkvparser_frame, &frame) == false) {
            fprintf(stderr, "Webm2Pes: frame read failed.\n");
            return false;
          }

          // Write frame out as PES packet(s).
          if (WritePesPacket(frame, &packet_data_) == false) {
            std::fprintf(stderr, "Webm2Pes: WritePesPacket failed.\n");
            return false;
          }
          if (packet_sink_->ReceivePacket(packet_data_) != true) {
            std::fprintf(stderr, "Webm2Pes: ReceivePacket failed.\n");
            return false;
          }
          bytes_written_ += packet_data_.size();
        }
      }
      block_status = cluster->GetNext(block_entry, block_entry);
      if (block_status < 0) {
        std::fprintf(stderr, "Webm2Pes: Cannot parse block in %s.\n",
                     input_file_name_.c_str());
        return false;
      }
    }

    cluster = webm_parser_->GetNext(cluster);
  }

  return true;
}

bool Webm2Pes::InitWebmParser() {
  if (webm_reader_.Open(input_file_name_.c_str()) != 0) {
    std::fprintf(stderr, "Webm2Pes: Cannot open %s as input.\n",
                 input_file_name_.c_str());
    return false;
  }

  using mkvparser::Segment;
  Segment* webm_parser = nullptr;
  if (Segment::CreateInstance(&webm_reader_, 0 /* pos */,
                              webm_parser /* Segment*& */) != 0) {
    std::fprintf(stderr, "Webm2Pes: Cannot create WebM parser.\n");
    return false;
  }
  webm_parser_.reset(webm_parser);

  if (webm_parser_->Load() != 0) {
    std::fprintf(stderr, "Webm2Pes: Cannot parse %s.\n",
                 input_file_name_.c_str());
    return false;
  }

  // Make sure there's a video track.
  const mkvparser::Tracks* tracks = webm_parser_->GetTracks();
  if (tracks == nullptr) {
    std::fprintf(stderr, "Webm2Pes: %s has no tracks.\n",
                 input_file_name_.c_str());
    return false;
  }

  timecode_scale_ = webm_parser_->GetInfo()->GetTimeCodeScale();

  for (int track_index = 0;
       track_index < static_cast<int>(tracks->GetTracksCount());
       ++track_index) {
    const mkvparser::Track* track = tracks->GetTrackByIndex(track_index);
    if (track && track->GetType() == mkvparser::Track::kVideo) {
      const std::string codec_id = ToString(track->GetCodecId());
      if (codec_id == std::string("V_VP8")) {
        codec_ = VideoFrame::kVP8;
      } else if (codec_id == std::string("V_VP9")) {
        codec_ = VideoFrame::kVP9;
      } else {
        fprintf(stderr, "Webm2Pes: Codec must be VP8 or VP9.\n");
        return false;
      }
      video_track_num_ = track_index + 1;
      break;
    }
  }
  if (video_track_num_ < 1) {
    std::fprintf(stderr, "Webm2Pes: No video track found in %s.\n",
                 input_file_name_.c_str());
    return false;
  }
  return true;
}

bool Webm2Pes::ReadVideoFrame(const mkvparser::Block::Frame& mkvparser_frame,
                              VideoFrame* frame) {
  if (mkvparser_frame.len < 1 || frame == nullptr)
    return false;

  const std::size_t mkv_len = static_cast<std::size_t>(mkvparser_frame.len);
  if (mkv_len > frame->buffer().capacity) {
    const std::size_t new_size = 2 * mkv_len;
    if (frame->Init(new_size) == false) {
      std::fprintf(stderr, "Webm2Pes: Out of memory.\n");
      return false;
    }
  }
  if (mkvparser_frame.Read(&webm_reader_, frame->buffer().data.get()) != 0) {
    std::fprintf(stderr, "Webm2Pes: Error reading VPx frame!\n");
    return false;
  }
  return frame->SetBufferLength(mkv_len);
}

bool Webm2Pes::WritePesPacket(const VideoFrame& frame,
                              PacketDataBuffer* packet_data) {
  if (frame.buffer().data.get() == nullptr || frame.buffer().length < 1)
    return false;

  Ranges frame_ranges;
  if (frame.codec() == VideoFrame::kVP9) {
    bool error = false;
    const bool has_superframe_index =
        ParseVP9SuperFrameIndex(frame.buffer().data.get(),
                                frame.buffer().length, &frame_ranges, &error);
    if (error) {
      std::fprintf(stderr, "Webm2Pes: Superframe index parse failed.\n");
      return false;
    }
    if (has_superframe_index == false) {
      frame_ranges.push_back(Range(0, frame.buffer().length));
    }
  } else {
    frame_ranges.push_back(Range(0, frame.buffer().length));
  }

  const std::int64_t khz90_pts =
      NanosecondsTo90KhzTicks(frame.nanosecond_pts());
  PesHeader header;
  header.optional_header.SetPtsBits(khz90_pts);

  packet_data->clear();

  for (const Range& packet_payload_range : frame_ranges) {
    std::size_t extra_bytes = 0;
    if (packet_payload_range.length > kMaxPayloadSize) {
      extra_bytes = packet_payload_range.length - kMaxPayloadSize;
    }
    if (packet_payload_range.length + packet_payload_range.offset >
        frame.buffer().length) {
      std::fprintf(stderr, "Webm2Pes: Invalid frame length.\n");
      return false;
    }

    // First packet of new frame. Always include PTS and BCMV header.
    header.packet_length =
        packet_payload_range.length - extra_bytes + BCMVHeader::size();
    if (header.Write(true, packet_data) != true) {
      std::fprintf(stderr, "Webm2Pes: packet header write failed.\n");
      return false;
    }

    BCMVHeader bcmv_header(static_cast<uint32_t>(packet_payload_range.length));
    if (bcmv_header.Write(packet_data) != true) {
      std::fprintf(stderr, "Webm2Pes: BCMV write failed.\n");
      return false;
    }

    // Insert the payload at the end of |packet_data|.
    const std::uint8_t* const payload_start =
        frame.buffer().data.get() + packet_payload_range.offset;

    const std::size_t bytes_to_copy = packet_payload_range.length - extra_bytes;
    if (CopyAndEscapeStartCodes(payload_start, bytes_to_copy, packet_data) ==
        false) {
      fprintf(stderr, "Webm2Pes: Payload write failed.\n");
      return false;
    }

    std::size_t bytes_copied = bytes_to_copy;
    while (extra_bytes) {
      // Write PES packets for the remaining data, but omit the PTS and BCMV
      // header.
      const std::size_t extra_bytes_to_copy =
          std::min(kMaxPayloadSize, extra_bytes);
      extra_bytes -= extra_bytes_to_copy;
      header.packet_length = extra_bytes_to_copy;
      if (header.Write(false, packet_data) != true) {
        fprintf(stderr, "Webm2pes: fragment write failed.\n");
        return false;
      }

      const std::uint8_t* fragment_start = payload_start + bytes_copied;
      if (CopyAndEscapeStartCodes(fragment_start, extra_bytes_to_copy,
                                  packet_data) == false) {
        fprintf(stderr, "Webm2Pes: Payload write failed.\n");
        return false;
      }

      bytes_copied += extra_bytes_to_copy;
    }
  }

  return true;
}

bool CopyAndEscapeStartCodes(const std::uint8_t* raw_input,
                             std::size_t raw_input_length,
                             PacketDataBuffer* packet_buffer) {
  if (raw_input == nullptr || raw_input_length < 1 || packet_buffer == nullptr)
    return false;

  int num_zeros = 0;
  for (std::size_t i = 0; i < raw_input_length; ++i) {
    const uint8_t byte = raw_input[i];

    if (byte == 0) {
      ++num_zeros;
    } else if (num_zeros >= 2 && (byte == 0x1 || byte == 0x3)) {
      packet_buffer->push_back(0x3);
      num_zeros = 0;
    } else {
      num_zeros = 0;
    }

    packet_buffer->push_back(byte);
  }

  return true;
}

}  // namespace libwebm
