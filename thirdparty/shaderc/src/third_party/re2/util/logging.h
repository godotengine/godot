// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_LOGGING_H_
#define UTIL_LOGGING_H_

// Simplified version of Google's logging.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ostream>
#include <sstream>

#include "util/util.h"

// Debug-only checking.
#define DCHECK(condition) assert(condition)
#define DCHECK_EQ(val1, val2) assert((val1) == (val2))
#define DCHECK_NE(val1, val2) assert((val1) != (val2))
#define DCHECK_LE(val1, val2) assert((val1) <= (val2))
#define DCHECK_LT(val1, val2) assert((val1) < (val2))
#define DCHECK_GE(val1, val2) assert((val1) >= (val2))
#define DCHECK_GT(val1, val2) assert((val1) > (val2))

// Always-on checking
#define CHECK(x)	if(x){}else LogMessageFatal(__FILE__, __LINE__).stream() << "Check failed: " #x
#define CHECK_LT(x, y)	CHECK((x) < (y))
#define CHECK_GT(x, y)	CHECK((x) > (y))
#define CHECK_LE(x, y)	CHECK((x) <= (y))
#define CHECK_GE(x, y)	CHECK((x) >= (y))
#define CHECK_EQ(x, y)	CHECK((x) == (y))
#define CHECK_NE(x, y)	CHECK((x) != (y))

#define LOG_INFO LogMessage(__FILE__, __LINE__)
#define LOG_WARNING LogMessage(__FILE__, __LINE__)
#define LOG_ERROR LogMessage(__FILE__, __LINE__)
#define LOG_FATAL LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

// It seems that one of the Windows header files defines ERROR as 0.
#ifdef _WIN32
#define LOG_0 LOG_INFO
#endif

#ifdef NDEBUG
#define LOG_DFATAL LOG_ERROR
#else
#define LOG_DFATAL LOG_FATAL
#endif

#define LOG(severity) LOG_ ## severity.stream()

#define VLOG(x) if((x)>0){}else LOG_INFO.stream()

class LogMessage {
 public:
  LogMessage(const char* file, int line)
      : flushed_(false) {
    stream() << file << ":" << line << ": ";
  }
  void Flush() {
    stream() << "\n";
    std::string s = str_.str();
    size_t n = s.size();
    if (fwrite(s.data(), 1, n, stderr) < n) {}  // shut up gcc
    flushed_ = true;
  }
  ~LogMessage() {
    if (!flushed_) {
      Flush();
    }
  }
  std::ostream& stream() { return str_; }

 private:
  bool flushed_;
  std::ostringstream str_;

  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;
};

// Silence "destructor never returns" warning for ~LogMessageFatal().
// Since this is a header file, push and then pop to limit the scope.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4722)
#endif

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line)
      : LogMessage(file, line) {}
  ATTRIBUTE_NORETURN ~LogMessageFatal() {
    Flush();
    abort();
  }
 private:
  LogMessageFatal(const LogMessageFatal&) = delete;
  LogMessageFatal& operator=(const LogMessageFatal&) = delete;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif  // UTIL_LOGGING_H_
