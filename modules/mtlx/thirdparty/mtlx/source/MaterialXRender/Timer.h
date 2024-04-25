//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_TIMER_H
#define MATERIALX_TIMER_H

/// @file
/// Support for event timing

#include <MaterialXRender/Export.h>

#include <chrono>

MATERIALX_NAMESPACE_BEGIN

/// @class ScopedTimer
/// A class for scoped event timing
class MX_RENDER_API ScopedTimer
{
  public:
    using clock = std::chrono::high_resolution_clock;

    ScopedTimer(double* externalCounter = nullptr);
    ~ScopedTimer();

    /// Return the elapsed time in seconds since our start time.
    double elapsedTime();

    /// Activate the timer, and set our start time to the current moment.
    void startTimer();

    /// Deactivate the timer, and add the elapsed time to our external counter.
    void endTimer();

  protected:
    bool _active;
    double* _externalCounter;
    std::chrono::time_point<clock> _startTime;
};

MATERIALX_NAMESPACE_END

#endif
