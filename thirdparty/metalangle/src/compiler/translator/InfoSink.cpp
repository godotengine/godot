//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/InfoSink.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Types.h"

namespace sh
{

void TInfoSinkBase::prefix(Severity severity)
{
    switch (severity)
    {
        case SH_WARNING:
            sink.append("WARNING: ");
            break;
        case SH_ERROR:
            sink.append("ERROR: ");
            break;
        default:
            sink.append("UNKOWN ERROR: ");
            break;
    }
}

TInfoSinkBase &TInfoSinkBase::operator<<(const ImmutableString &str)
{
    sink.append(str.data());
    return *this;
}

TInfoSinkBase &TInfoSinkBase::operator<<(const TType &type)
{
    if (type.isInvariant())
        sink.append("invariant ");
    if (type.getQualifier() != EvqTemporary && type.getQualifier() != EvqGlobal)
    {
        sink.append(type.getQualifierString());
        sink.append(" ");
    }
    if (type.getPrecision() != EbpUndefined)
    {
        sink.append(type.getPrecisionString());
        sink.append(" ");
    }
    if (type.isArray())
    {
        for (auto arraySizeIter = type.getArraySizes()->rbegin();
             arraySizeIter != type.getArraySizes()->rend(); ++arraySizeIter)
        {
            *this << "array[" << (*arraySizeIter) << "] of ";
        }
    }
    if (type.isMatrix())
    {
        *this << type.getCols() << "X" << type.getRows() << " matrix of ";
    }
    else if (type.isVector())
        *this << type.getNominalSize() << "-component vector of ";

    sink.append(type.getBasicString());
    return *this;
}

void TInfoSinkBase::location(int file, int line)
{
    TPersistStringStream stream = sh::InitializeStream<TPersistStringStream>();
    if (line)
        stream << file << ":" << line;
    else
        stream << file << ":? ";
    stream << ": ";

    sink.append(stream.str());
}

}  // namespace sh
