//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_SOURCELOCATION_H_
#define COMPILER_PREPROCESSOR_SOURCELOCATION_H_

namespace angle
{

namespace pp
{

struct SourceLocation
{
    SourceLocation() : file(0), line(0) {}
    SourceLocation(int f, int l) : file(f), line(l) {}

    bool equals(const SourceLocation &other) const
    {
        return (file == other.file) && (line == other.line);
    }

    int file;
    int line;
};

inline bool operator==(const SourceLocation &lhs, const SourceLocation &rhs)
{
    return lhs.equals(rhs);
}

inline bool operator!=(const SourceLocation &lhs, const SourceLocation &rhs)
{
    return !lhs.equals(rhs);
}

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_SOURCELOCATION_H_
