// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Syoyo Fujita and many contributors.
// All rights reserved.
//
// StreamReader: Simple header-only stream reader with endian support
//
// Part of TinyEXR V2 API (EXPERIMENTAL)

#ifndef TINYEXR_STREAMREADER_HH_
#define TINYEXR_STREAMREADER_HH_

#include <cstdint>
#include <cstring>

namespace tinyexr {

enum class Endian {
  Little,
  Big,
  Native
};

class StreamReader {
public:
  // Constructor: takes memory address, length, and endianness
  StreamReader(const uint8_t* data, size_t length, Endian endian = Endian::Little)
      : data_(data), length_(length), pos_(0), endian_(endian) {
    // Detect native endian
    uint16_t test = 0x0001;
    native_is_little_ = (*reinterpret_cast<uint8_t*>(&test) == 0x01);

    // Determine if we need to swap bytes
    if (endian_ == Endian::Native) {
      needs_swap_ = false;
    } else {
      bool data_is_little = (endian_ == Endian::Little);
      needs_swap_ = (data_is_little != native_is_little_);
    }
  }

  // Read n bytes into destination buffer
  // Returns false on out-of-bounds or error
  bool read(size_t n, uint8_t* dst) {
    if (!dst || n == 0) {
      return false;
    }
    if (pos_ + n > length_) {
      return false;  // Out of bounds
    }
    std::memcpy(dst, data_ + pos_, n);
    pos_ += n;
    return true;
  }

  // Read 1 byte (uint8_t)
  bool read1(uint8_t* dst) {
    return read(1, dst);
  }

  // Read 2 bytes (uint16_t) with endian swap if needed
  bool read2(uint16_t* dst) {
    if (!dst) {
      return false;
    }
    uint8_t buf[2];
    if (!read(2, buf)) {
      return false;
    }

    if (needs_swap_) {
      *dst = static_cast<uint16_t>(buf[1]) << 8 |
             static_cast<uint16_t>(buf[0]);
    } else {
      std::memcpy(dst, buf, 2);
    }
    return true;
  }

  // Read 4 bytes (uint32_t) with endian swap if needed
  bool read4(uint32_t* dst) {
    if (!dst) {
      return false;
    }
    uint8_t buf[4];
    if (!read(4, buf)) {
      return false;
    }

    if (needs_swap_) {
      *dst = static_cast<uint32_t>(buf[3]) << 24 |
             static_cast<uint32_t>(buf[2]) << 16 |
             static_cast<uint32_t>(buf[1]) << 8  |
             static_cast<uint32_t>(buf[0]);
    } else {
      std::memcpy(dst, buf, 4);
    }
    return true;
  }

  // Read 8 bytes (uint64_t) with endian swap if needed
  bool read8(uint64_t* dst) {
    if (!dst) {
      return false;
    }
    uint8_t buf[8];
    if (!read(8, buf)) {
      return false;
    }

    if (needs_swap_) {
      *dst = static_cast<uint64_t>(buf[7]) << 56 |
             static_cast<uint64_t>(buf[6]) << 48 |
             static_cast<uint64_t>(buf[5]) << 40 |
             static_cast<uint64_t>(buf[4]) << 32 |
             static_cast<uint64_t>(buf[3]) << 24 |
             static_cast<uint64_t>(buf[2]) << 16 |
             static_cast<uint64_t>(buf[1]) << 8  |
             static_cast<uint64_t>(buf[0]);
    } else {
      std::memcpy(dst, buf, 8);
    }
    return true;
  }

  // Seek to absolute position
  // Returns false if position is out of bounds
  bool seek(size_t pos) {
    if (pos > length_) {
      return false;
    }
    pos_ = pos;
    return true;
  }

  // Rewind to beginning
  void rewind() {
    pos_ = 0;
  }

  // Get current position
  size_t tell() const {
    return pos_;
  }

  // Get remaining bytes
  size_t remaining() const {
    return length_ - pos_;
  }

  // Check if at end
  bool eof() const {
    return pos_ >= length_;
  }

  // Get total length
  size_t length() const {
    return length_;
  }

private:
  const uint8_t* data_;
  size_t length_;
  size_t pos_;
  Endian endian_;
  bool native_is_little_;
  bool needs_swap_;
};

}  // namespace tinyexr

#endif  // TINYEXR_STREAMREADER_HH_
