//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// trace.h: Wrappers for ANGLE trace event functions.
//

#ifndef LIBANGLE_TRACE_H_
#define LIBANGLE_TRACE_H_

#include <platform/Platform.h>
#include "third_party/trace_event/trace_event.h"

// TODO: Pass platform directly to these methods. http://anglebug.com/1892
#define ANGLE_TRACE_EVENT_BEGIN0(CATEGORY, EVENT) \
    TRACE_EVENT_BEGIN0(ANGLEPlatformCurrent(), CATEGORY, EVENT)
#define ANGLE_TRACE_EVENT_END0(CATEGORY, EVENT) \
    TRACE_EVENT_END0(ANGLEPlatformCurrent(), CATEGORY, EVENT)
#define ANGLE_TRACE_EVENT_INSTANT0(CATEGORY, EVENT) \
    TRACE_EVENT_INSTANT0(ANGLEPlatformCurrent(), CATEGORY, EVENT)
#define ANGLE_TRACE_EVENT0(CATEGORY, EVENT) TRACE_EVENT0(ANGLEPlatformCurrent(), CATEGORY, EVENT)

#endif  // LIBANGLE_TRACE_H_
