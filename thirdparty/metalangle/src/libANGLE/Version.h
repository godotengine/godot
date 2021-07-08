//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Version.h: Encapsulation of a GL version.

#ifndef LIBANGLE_VERSION_H_
#define LIBANGLE_VERSION_H_

namespace gl
{

struct Version
{
    constexpr Version();
    constexpr Version(unsigned int major, unsigned int minor);

    unsigned int major;
    unsigned int minor;
};

bool operator==(const Version &a, const Version &b);
bool operator!=(const Version &a, const Version &b);
bool operator>=(const Version &a, const Version &b);
bool operator<=(const Version &a, const Version &b);
bool operator<(const Version &a, const Version &b);
bool operator>(const Version &a, const Version &b);
}  // namespace gl

#include "Version.inc"

#endif  // LIBANGLE_VERSION_H_
