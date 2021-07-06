//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImmutableString.h: Wrapper for static or pool allocated char arrays, that are guaranteed to be
// valid and unchanged for the duration of the compilation.
//

#ifndef COMPILER_TRANSLATOR_IMMUTABLESTRING_H_
#define COMPILER_TRANSLATOR_IMMUTABLESTRING_H_

#include <string>

#include "common/string_utils.h"
#include "compiler/translator/Common.h"

namespace sh
{

namespace
{
constexpr size_t constStrlen(const char *str)
{
    if (str == nullptr)
    {
        return 0u;
    }
    size_t len = 0u;
    while (*(str + len) != '\0')
    {
        ++len;
    }
    return len;
}
}  // namespace

class ImmutableString
{
  public:
    // The data pointer passed in must be one of:
    //  1. nullptr (only valid with length 0).
    //  2. a null-terminated static char array like a string literal.
    //  3. a null-terminated pool allocated char array. This can't be c_str() of a local TString,
    //     since when a TString goes out of scope it clears its first character.
    explicit constexpr ImmutableString(const char *data) : mData(data), mLength(constStrlen(data))
    {}

    constexpr ImmutableString(const char *data, size_t length) : mData(data), mLength(length) {}

    ImmutableString(const std::string &str)
        : mData(AllocatePoolCharArray(str.c_str(), str.size())), mLength(str.size())
    {}

    constexpr ImmutableString(const ImmutableString &) = default;

    ImmutableString &operator=(const ImmutableString &) = default;

    constexpr const char *data() const { return mData ? mData : ""; }
    constexpr size_t length() const { return mLength; }

    char operator[](size_t index) const { return data()[index]; }

    constexpr bool empty() const { return mLength == 0; }
    bool beginsWith(const char *prefix) const { return angle::BeginsWith(data(), prefix); }
    constexpr bool beginsWith(const ImmutableString &prefix) const
    {
        return mLength >= prefix.length() && memcmp(data(), prefix.data(), prefix.length()) == 0;
    }
    bool contains(const char *substr) const { return strstr(data(), substr) != nullptr; }

    constexpr bool operator==(const ImmutableString &b) const
    {
        if (mLength != b.mLength)
        {
            return false;
        }
        return memcmp(data(), b.data(), mLength) == 0;
    }
    constexpr bool operator!=(const ImmutableString &b) const { return !(*this == b); }
    constexpr bool operator==(const char *b) const
    {
        if (b == nullptr)
        {
            return empty();
        }
        return strcmp(data(), b) == 0;
    }
    constexpr bool operator!=(const char *b) const { return !(*this == b); }
    bool operator==(const std::string &b) const
    {
        return mLength == b.length() && memcmp(data(), b.c_str(), mLength) == 0;
    }
    bool operator!=(const std::string &b) const { return !(*this == b); }

    constexpr bool operator<(const ImmutableString &b) const
    {
        if (mLength < b.mLength)
        {
            return true;
        }
        if (mLength > b.mLength)
        {
            return false;
        }
        return (memcmp(data(), b.data(), mLength) < 0);
    }

    template <size_t hashBytes>
    struct FowlerNollVoHash
    {
        static const size_t kFnvOffsetBasis;
        static const size_t kFnvPrime;

        constexpr size_t operator()(const ImmutableString &a) const
        {
            const char *data = a.data();
            size_t hash      = kFnvOffsetBasis;
            while ((*data) != '\0')
            {
                hash = hash ^ (*data);
                hash = hash * kFnvPrime;
                ++data;
            }
            return hash;
        }
    };

    // Perfect hash functions
    uint32_t mangledNameHash() const;
    uint32_t unmangledNameHash() const;

  private:
    const char *mData;
    size_t mLength;
};

constexpr ImmutableString kEmptyImmutableString("");
}  // namespace sh

std::ostream &operator<<(std::ostream &os, const sh::ImmutableString &str);

#endif  // COMPILER_TRANSLATOR_IMMUTABLESTRING_H_
