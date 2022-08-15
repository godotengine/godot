//===- Unix/TimeValue.cpp - Unix TimeValue Implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the TimeValue class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"

namespace llvm {
  using namespace sys;

std::string TimeValue::str() const {
  time_t OurTime = time_t(this->toEpochTime());
  struct tm Storage;
  struct tm *LT = ::localtime_r(&OurTime, &Storage);
  assert(LT);
  char Buffer1[sizeof("YYYY-MM-DD HH:MM:SS")];
  strftime(Buffer1, sizeof(Buffer1), "%Y-%m-%d %H:%M:%S", LT);
  char Buffer2[sizeof("YYYY-MM-DD HH:MM:SS.MMMUUUNNN")];
  snprintf(Buffer2, sizeof(Buffer2), "%s.%.9u", Buffer1, this->nanoseconds());
  return std::string(Buffer2);
}

TimeValue TimeValue::now() {
  struct timeval the_time;
  timerclear(&the_time);
  if (0 != ::gettimeofday(&the_time,nullptr)) {
    // This is *really* unlikely to occur because the only gettimeofday
    // errors concern the timezone parameter which we're passing in as 0.
    // In the unlikely case it does happen, just return MinTime, no error
    // message needed.
    return MinTime();
  }

  return TimeValue(
    static_cast<TimeValue::SecondsType>( the_time.tv_sec +
      PosixZeroTimeSeconds ),
    static_cast<TimeValue::NanoSecondsType>( the_time.tv_usec *
      NANOSECONDS_PER_MICROSECOND ) );
}

}
