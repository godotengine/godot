//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_INFOSINK_H_
#define COMPILER_TRANSLATOR_INFOSINK_H_

#include <math.h>
#include <stdlib.h>
#include "compiler/translator/Common.h"
#include "compiler/translator/Severity.h"

namespace sh
{

class ImmutableString;
class TType;

// Returns the fractional part of the given floating-point number.
inline float fractionalPart(float f)
{
    float intPart = 0.0f;
    return modff(f, &intPart);
}

class ImmutableString;

//
// Encapsulate info logs for all objects that have them.
//
// The methods are a general set of tools for getting a variety of
// messages and types inserted into the log.
//
class TInfoSinkBase
{
  public:
    TInfoSinkBase() {}

    template <typename T>
    TInfoSinkBase &operator<<(const T &t)
    {
        TPersistStringStream stream = sh::InitializeStream<TPersistStringStream>();
        stream << t;
        sink.append(stream.str());
        return *this;
    }
    // Override << operator for specific types. It is faster to append strings
    // and characters directly to the sink.
    TInfoSinkBase &operator<<(char c)
    {
        sink.append(1, c);
        return *this;
    }
    TInfoSinkBase &operator<<(const char *str)
    {
        sink.append(str);
        return *this;
    }
    TInfoSinkBase &operator<<(const TPersistString &str)
    {
        sink.append(str);
        return *this;
    }
    TInfoSinkBase &operator<<(const TString &str)
    {
        sink.append(str.c_str());
        return *this;
    }
    TInfoSinkBase &operator<<(const ImmutableString &str);

    TInfoSinkBase &operator<<(const TType &type);

    // Make sure floats are written with correct precision.
    TInfoSinkBase &operator<<(float f)
    {
        // Make sure that at least one decimal point is written. If a number
        // does not have a fractional part, the default precision format does
        // not write the decimal portion which gets interpreted as integer by
        // the compiler.
        TPersistStringStream stream = sh::InitializeStream<TPersistStringStream>();
        if (fractionalPart(f) == 0.0f)
        {
            stream.precision(1);
            stream << std::showpoint << std::fixed << f;
        }
        else
        {
            stream.unsetf(std::ios::fixed);
            stream.unsetf(std::ios::scientific);
            stream.precision(8);
            stream << f;
        }
        sink.append(stream.str());
        return *this;
    }
    // Write boolean values as their names instead of integral value.
    TInfoSinkBase &operator<<(bool b)
    {
        const char *str = b ? "true" : "false";
        sink.append(str);
        return *this;
    }

    void erase() { sink.clear(); }
    int size() { return static_cast<int>(sink.size()); }

    const TPersistString &str() const { return sink; }
    const char *c_str() const { return sink.c_str(); }

    void prefix(Severity severity);
    void location(int file, int line);

  private:
    TPersistString sink;
};

class TInfoSink
{
  public:
    TInfoSinkBase info;
    TInfoSinkBase debug;
    TInfoSinkBase obj;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_INFOSINK_H_
