// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Syoyo Fujita and many contributors.
// All rights reserved.
//
// EXR Reader: Reader class with error stack for safe memory reading

#ifndef TINYEXR_EXR_READER_HH_
#define TINYEXR_EXR_READER_HH_

#include <vector>
#include <string>
#include "streamreader.hh"

namespace tinyexr {

// Reader class that wraps StreamReader and accumulates errors
class Reader {
public:
  Reader(const uint8_t* data, size_t length, Endian endian = Endian::Little)
      : stream_(data, length, endian), has_error_(false) {}

  // Check if any errors have occurred
  bool has_error() const { return has_error_; }

  // Get all accumulated errors
  const std::vector<std::string>& errors() const { return errors_; }

  // Get the most recent error
  std::string last_error() const {
    return errors_.empty() ? "" : errors_.back();
  }

  // Get all errors as a single string
  std::string all_errors() const {
    std::string result;
    for (size_t i = 0; i < errors_.size(); i++) {
      if (i > 0) result += "\n";
      result += errors_[i];
    }
    return result;
  }

  // Clear error stack
  void clear_errors() {
    errors_.clear();
    has_error_ = false;
  }

  // Read n bytes into destination buffer
  bool read(size_t n, uint8_t* dst) {
    if (!stream_.read(n, dst)) {
      add_error("Failed to read " + std::to_string(n) + " bytes at position " +
                std::to_string(stream_.tell()));
      return false;
    }
    return true;
  }

  // Read 1 byte
  bool read1(uint8_t* dst) {
    if (!stream_.read1(dst)) {
      add_error("Failed to read 1 byte at position " +
                std::to_string(stream_.tell()));
      return false;
    }
    return true;
  }

  // Read 2 bytes with endian swap
  bool read2(uint16_t* dst) {
    if (!stream_.read2(dst)) {
      add_error("Failed to read 2 bytes at position " +
                std::to_string(stream_.tell()));
      return false;
    }
    return true;
  }

  // Read 4 bytes with endian swap
  bool read4(uint32_t* dst) {
    if (!stream_.read4(dst)) {
      add_error("Failed to read 4 bytes at position " +
                std::to_string(stream_.tell()));
      return false;
    }
    return true;
  }

  // Read 8 bytes with endian swap
  bool read8(uint64_t* dst) {
    if (!stream_.read8(dst)) {
      add_error("Failed to read 8 bytes at position " +
                std::to_string(stream_.tell()));
      return false;
    }
    return true;
  }

  // Read a null-terminated string up to max_len bytes
  // Returns false if no null terminator found within max_len
  bool read_string(std::string* str, size_t max_len = 256) {
    if (!str) {
      add_error("Null pointer passed to read_string");
      return false;
    }

    str->clear();
    size_t start_pos = stream_.tell();

    for (size_t i = 0; i < max_len; i++) {
      uint8_t c;
      if (!stream_.read1(&c)) {
        add_error("Failed to read string at position " + std::to_string(start_pos));
        return false;
      }
      if (c == '\0') {
        return true;
      }
      str->push_back(static_cast<char>(c));
    }

    add_error("String not null-terminated within " + std::to_string(max_len) +
              " bytes at position " + std::to_string(start_pos));
    return false;
  }

  // Seek to absolute position
  bool seek(size_t pos) {
    if (!stream_.seek(pos)) {
      add_error("Failed to seek to position " + std::to_string(pos));
      return false;
    }
    return true;
  }

  // Seek relative to current position
  bool seek_relative(int64_t offset) {
    size_t current = stream_.tell();
    int64_t new_pos = static_cast<int64_t>(current) + offset;

    if (new_pos < 0) {
      add_error("Seek would move before start of stream");
      return false;
    }

    return seek(static_cast<size_t>(new_pos));
  }

  // Rewind to beginning
  void rewind() {
    stream_.rewind();
  }

  // Get current position
  size_t tell() const {
    return stream_.tell();
  }

  // Get remaining bytes
  size_t remaining() const {
    return stream_.remaining();
  }

  // Check if at end
  bool eof() const {
    return stream_.eof();
  }

  // Get total length
  size_t length() const {
    return stream_.length();
  }

  // Add a custom error message
  void add_error(const std::string& msg) {
    errors_.push_back(msg);
    has_error_ = true;
  }

  // Get direct access to underlying StreamReader (use with caution)
  const StreamReader& stream() const { return stream_; }

private:
  StreamReader stream_;
  std::vector<std::string> errors_;
  bool has_error_;
};

}  // namespace tinyexr

#endif  // TINYEXR_EXR_READER_HH_
