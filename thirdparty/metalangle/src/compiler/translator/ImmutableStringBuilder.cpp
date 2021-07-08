//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImmutableStringBuilder.cpp: Stringstream-like utility for building pool allocated strings where
// the maximum length is known in advance.
//

#include "compiler/translator/ImmutableStringBuilder.h"

#include <stdio.h>

namespace sh
{

ImmutableStringBuilder &ImmutableStringBuilder::operator<<(const ImmutableString &str)
{
    ASSERT(mData != nullptr);
    ASSERT(mPos + str.length() <= mMaxLength);
    memcpy(mData + mPos, str.data(), str.length());
    mPos += str.length();
    return *this;
}

ImmutableStringBuilder &ImmutableStringBuilder::operator<<(const char *str)
{
    ASSERT(mData != nullptr);
    size_t len = strlen(str);
    ASSERT(mPos + len <= mMaxLength);
    memcpy(mData + mPos, str, len);
    mPos += len;
    return *this;
}

ImmutableStringBuilder &ImmutableStringBuilder::operator<<(const char &c)
{
    ASSERT(mData != nullptr);
    ASSERT(mPos + 1 <= mMaxLength);
    mData[mPos++] = c;
    return *this;
}

void ImmutableStringBuilder::appendDecimal(const uint32_t &u)
{
    int numChars = snprintf(mData + mPos, mMaxLength - mPos, "%d", u);
    ASSERT(numChars >= 0);
    ASSERT(mPos + numChars <= mMaxLength);
    mPos += numChars;
}

ImmutableStringBuilder::operator ImmutableString()
{
    mData[mPos] = '\0';
    ImmutableString str(static_cast<const char *>(mData), mPos);
#if defined(ANGLE_ENABLE_ASSERTS)
    // Make sure that nothing is added to the string after it is finalized.
    mData = nullptr;
#endif
    return str;
}

}  // namespace sh
