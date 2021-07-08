// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "common/event_tracer.h"

#include "common/debug.h"

namespace angle
{

const unsigned char *GetTraceCategoryEnabledFlag(PlatformMethods *platform, const char *name)
{
    ASSERT(platform);

    const unsigned char *categoryEnabledFlag =
        platform->getTraceCategoryEnabledFlag(platform, name);
    if (categoryEnabledFlag != nullptr)
    {
        return categoryEnabledFlag;
    }

    static unsigned char disabled = 0;
    return &disabled;
}

angle::TraceEventHandle AddTraceEvent(PlatformMethods *platform,
                                      char phase,
                                      const unsigned char *categoryGroupEnabled,
                                      const char *name,
                                      unsigned long long id,
                                      int numArgs,
                                      const char **argNames,
                                      const unsigned char *argTypes,
                                      const unsigned long long *argValues,
                                      unsigned char flags)
{
    ASSERT(platform);

    double timestamp = platform->monotonicallyIncreasingTime(platform);

    if (timestamp != 0)
    {
        angle::TraceEventHandle handle =
            platform->addTraceEvent(platform, phase, categoryGroupEnabled, name, id, timestamp,
                                    numArgs, argNames, argTypes, argValues, flags);
        return handle;
    }

    return static_cast<angle::TraceEventHandle>(0);
}

}  // namespace angle
