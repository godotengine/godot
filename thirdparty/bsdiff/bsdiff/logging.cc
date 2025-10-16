// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "bsdiff/logging.h"

#include <ctime>

LogMessage::LogMessage(const char* file,
                       unsigned int line,
                       const char* severity)
    : LogMessage(file, line, severity, -1) {}


LogMessage::LogMessage(const char* file,
                       unsigned int line,
                       const char* severity,
                       int error)
    : error_(error) {
  std::time_t t = std::time(nullptr);
  char timestamp[32];
  strftime(timestamp, sizeof(timestamp), "%m-%d %H:%M:%S", std::localtime(&t));

  stream_ << severity << " " << timestamp << " " << file << ":" << line << ": ";
}

LogMessage::~LogMessage() {
  if (error_ != -1) {
    stream_ << ": " << strerror(error_);
  }
  std::cerr << stream_.str() << std::endl;
}
