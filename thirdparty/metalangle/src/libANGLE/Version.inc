//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Version.inc: Encapsulation of a GL version.

#include <tuple>

namespace gl
{

constexpr Version::Version()
    : Version(0, 0)
{
}

// Avoid conflicts with linux system defines
#undef major
#undef minor

constexpr Version::Version(unsigned int major_, unsigned int minor_)
    : major(major_),
      minor(minor_)
{
}

inline bool operator==(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) == std::tie(b.major, b.minor);
}

inline bool operator!=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) != std::tie(b.major, b.minor);
}

inline bool operator>=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) >= std::tie(b.major, b.minor);
}

inline bool operator<=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) <= std::tie(b.major, b.minor);
}

inline bool operator<(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) < std::tie(b.major, b.minor);
}

inline bool operator>(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) > std::tie(b.major, b.minor);
}

}  // namespace gl
