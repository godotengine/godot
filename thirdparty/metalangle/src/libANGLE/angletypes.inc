//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// angletypes.inc : Inline definitions of some functions from angletypes.h

namespace gl
{

inline bool operator==(const Rectangle &a, const Rectangle &b)
{
    return a.x == b.x &&
           a.y == b.y &&
           a.width == b.width &&
           a.height == b.height;
}

inline bool operator!=(const Rectangle &a, const Rectangle &b)
{
    return !(a == b);
}

inline bool operator==(const SamplerState &a, const SamplerState &b)
{
    return memcmp(&a, &b, sizeof(SamplerState)) == 0;
}

inline bool operator!=(const SamplerState &a, const SamplerState &b)
{
    return !(a == b);
}

}  // namespace gl
