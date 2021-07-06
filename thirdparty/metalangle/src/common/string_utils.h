//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// string_utils:
//   String helper functions.
//

#ifndef LIBANGLE_STRING_UTILS_H_
#define LIBANGLE_STRING_UTILS_H_

#include <string>
#include <vector>

#include "common/Optional.h"

namespace angle
{

extern const char kWhitespaceASCII[];

enum WhitespaceHandling
{
    KEEP_WHITESPACE,
    TRIM_WHITESPACE,
};

enum SplitResult
{
    SPLIT_WANT_ALL,
    SPLIT_WANT_NONEMPTY,
};

std::vector<std::string> SplitString(const std::string &input,
                                     const std::string &delimiters,
                                     WhitespaceHandling whitespace,
                                     SplitResult resultType);

void SplitStringAlongWhitespace(const std::string &input, std::vector<std::string> *tokensOut);

std::string TrimString(const std::string &input, const std::string &trimChars);

// Return the substring starting at offset and up to the first occurance of the |delimeter|.
std::string GetPrefix(const std::string &input, size_t offset, const char *delimiter);
std::string GetPrefix(const std::string &input, size_t offset, char delimiter);

bool HexStringToUInt(const std::string &input, unsigned int *uintOut);

bool ReadFileToString(const std::string &path, std::string *stringOut);

// Check if the string str begins with the given prefix.
// The comparison is case sensitive.
bool BeginsWith(const std::string &str, const std::string &prefix);

// Check if the string str begins with the given prefix.
// Prefix may not be NULL and needs to be NULL terminated.
// The comparison is case sensitive.
bool BeginsWith(const std::string &str, const char *prefix);

// Check if the string str begins with the given prefix.
// str and prefix may not be NULL and need to be NULL terminated.
// The comparison is case sensitive.
bool BeginsWith(const char *str, const char *prefix);

// Check if the string str begins with the first prefixLength characters of the given prefix.
// The length of the prefix string should be greater than or equal to prefixLength.
// The comparison is case sensitive.
bool BeginsWith(const std::string &str, const std::string &prefix, const size_t prefixLength);

// Check if the string str ends with the given suffix.
// Suffix may not be NUL and needs to be NULL terminated.
// The comparison is case sensitive.
bool EndsWith(const std::string &str, const char *suffix);

// Convert to lower-case.
void ToLower(std::string *str);

// Replaces the substring 'substring' in 'str' with 'replacement'. Returns true if successful.
bool ReplaceSubstring(std::string *str,
                      const std::string &substring,
                      const std::string &replacement);

}  // namespace angle

#endif  // LIBANGLE_STRING_UTILS_H_
