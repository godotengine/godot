//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImmutableStringBuilder.h: Stringstream-like utility for building pool allocated strings where the
// maximum length is known in advance.
//

#ifndef COMPILER_TRANSLATOR_IMMUTABLESTRINGBUILDER_H_
#define COMPILER_TRANSLATOR_IMMUTABLESTRINGBUILDER_H_

#include "compiler/translator/ImmutableString.h"

namespace sh
{

class ImmutableStringBuilder
{
  public:
    ImmutableStringBuilder(size_t maxLength)
        : mPos(0u), mMaxLength(maxLength), mData(AllocateEmptyPoolCharArray(maxLength))
    {}

    ImmutableStringBuilder &operator<<(const ImmutableString &str);

    ImmutableStringBuilder &operator<<(const char *str);

    ImmutableStringBuilder &operator<<(const char &c);

    // This invalidates the ImmutableStringBuilder, so it should only be called once.
    operator ImmutableString();

    void appendDecimal(const uint32_t &i);

    template <typename T>
    void appendHex(T number)
    {
        ASSERT(mData != nullptr);
        ASSERT(mPos + sizeof(T) * 2u <= mMaxLength);
        int index = static_cast<int>(sizeof(T)) * 2 - 1;
        // Loop through leading zeroes.
        while (((number >> (index * 4)) & 0xfu) == 0 && index > 0)
        {
            --index;
        }
        // Write the rest of the hex digits.
        while (index >= 0)
        {
            char digit     = static_cast<char>((number >> (index * 4)) & 0xfu);
            char digitChar = (digit < 10) ? (digit + '0') : (digit + ('a' - 10));
            mData[mPos++]  = digitChar;
            --index;
        }
    }

    template <typename T>
    static constexpr size_t GetHexCharCount()
    {
        return sizeof(T) * 2;
    }

  private:
    inline static char *AllocateEmptyPoolCharArray(size_t strLength)
    {
        size_t requiredSize = strLength + 1u;
        return static_cast<char *>(GetGlobalPoolAllocator()->allocate(requiredSize));
    }

    size_t mPos;
    size_t mMaxLength;
    char *mData;
};

// GLSL ES 3.00.6 section 3.9: the maximum length of an identifier is 1024 characters.
constexpr unsigned int kESSLMaxIdentifierLength = 1024u;

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_IMMUTABLESTRINGBUILDER_H_
