//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Timer.cpp: Implementation of a high precision timer class.
//

#include "util/Timer.h"

#include "common/system_utils.h"

Timer::Timer() : mRunning(false), mStartTime(0), mStopTime(0) {}

void Timer::start()
{
    mStartTime = angle::GetCurrentTime();
    mRunning   = true;
}

void Timer::stop()
{
    mStopTime = angle::GetCurrentTime();
    mRunning  = false;
}

double Timer::getElapsedTime() const
{
    double endTime;
    if (mRunning)
    {
        endTime = angle::GetCurrentTime();
    }
    else
    {
        endTime = mStopTime;
    }

    return endTime - mStartTime;
}
