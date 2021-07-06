//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef SAMPLE_UTIL_TIMER_H
#define SAMPLE_UTIL_TIMER_H

#include "util/util_export.h"

class ANGLE_UTIL_EXPORT Timer final
{
  public:
    Timer();
    ~Timer() {}

    // Use start() and stop() to record the duration and use getElapsedTime() to query that
    // duration.  If getElapsedTime() is called in between, it will report the elapsed time since
    // start().
    void start();
    void stop();
    double getElapsedTime() const;

  private:
    bool mRunning;
    double mStartTime;
    double mStopTime;
};

#endif  // SAMPLE_UTIL_TIMER_H
