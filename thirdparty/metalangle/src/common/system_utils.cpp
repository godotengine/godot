//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils.cpp: Implementation of common functions

#include "system_utils.h"

namespace angle
{
bool PrependPathToEnvironmentVar(const char *variableName, const char *path)
{
    std::string oldValue = GetEnvironmentVar(variableName);
    const char *newValue = nullptr;
    std::string buf;
    if (oldValue.empty())
    {
        newValue = path;
    }
    else
    {
        buf = path;
        buf += GetPathSeparator();
        buf += oldValue;
        newValue = buf.c_str();
    }
    return SetEnvironmentVar(variableName, newValue);
}
}  // namespace angle
