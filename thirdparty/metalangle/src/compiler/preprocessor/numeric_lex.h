//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// numeric_lex.h: Functions to extract numeric values from string.

#ifndef COMPILER_PREPROCESSOR_NUMERICLEX_H_
#define COMPILER_PREPROCESSOR_NUMERICLEX_H_

#include <sstream>

namespace angle
{

namespace pp
{

inline std::ios::fmtflags numeric_base_int(const std::string &str)
{
    if ((str.size() >= 2) && (str[0] == '0') && (str[1] == 'x' || str[1] == 'X'))
    {
        return std::ios::hex;
    }
    if ((str.size() >= 1) && (str[0] == '0'))
    {
        return std::ios::oct;
    }
    return std::ios::dec;
}

// The following functions parse the given string to extract a numerical
// value of the given type. These functions assume that the string is
// of the correct form. They can only fail if the parsed value is too big,
// in which case false is returned.

template <typename IntType>
bool numeric_lex_int(const std::string &str, IntType *value)
{
    std::istringstream stream(str);
    // This should not be necessary, but MSVS has a buggy implementation.
    // It returns incorrect results if the base is not specified.
    stream.setf(numeric_base_int(str), std::ios::basefield);

    stream >> (*value);
    return !stream.fail();
}

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_NUMERICLEX_H_
