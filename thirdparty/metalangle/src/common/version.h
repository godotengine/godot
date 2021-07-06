//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMMON_VERSION_H_
#define COMMON_VERSION_H_

#include "id/commit.h"

#define ANGLE_MAJOR_VERSION 2
#define ANGLE_MINOR_VERSION 1

#ifndef ANGLE_REVISION
#    define ANGLE_REVISION 0
#endif

#define ANGLE_STRINGIFY(x) #x
#define ANGLE_MACRO_STRINGIFY(x) ANGLE_STRINGIFY(x)

#define ANGLE_VERSION_STRING                                                  \
    ANGLE_MACRO_STRINGIFY(ANGLE_MAJOR_VERSION)                                \
    "." ANGLE_MACRO_STRINGIFY(ANGLE_MINOR_VERSION) "." ANGLE_MACRO_STRINGIFY( \
        ANGLE_REVISION) "." ANGLE_COMMIT_HASH

#endif  // COMMON_VERSION_H_
