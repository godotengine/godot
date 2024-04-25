//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UTIL
#define MATERIALX_UTIL

/// @file
/// Utility methods

#include <MaterialXCore/Export.h>

MATERIALX_NAMESPACE_BEGIN

extern MX_CORE_API const string EMPTY_STRING;

/// Return the version of the MaterialX library as a string.
MX_CORE_API string getVersionString();

/// Return the major, minor, and build versions of the MaterialX library
/// as an integer tuple.
MX_CORE_API std::tuple<int, int, int> getVersionIntegers();

/// Create a valid MaterialX name from the given string.
MX_CORE_API string createValidName(string name, char replaceChar = '_');

/// Return true if the given string is a valid MaterialX name.
MX_CORE_API bool isValidName(const string& name);

/// Increment the numeric suffix of a name
MX_CORE_API string incrementName(const string& name);

/// Split a string into a vector of substrings using the given set of
/// separator characters.
MX_CORE_API StringVec splitString(const string& str, const string& sep);

/// Join a vector of substrings into a single string, placing the given
/// separator between each substring.
MX_CORE_API string joinStrings(const StringVec& strVec, const string& sep);

/// Apply the given substring substitutions to the input string.
MX_CORE_API string replaceSubstrings(string str, const StringMap& stringMap);

/// Return a copy of the given string with letters converted to lower case.
MX_CORE_API string stringToLower(string str);

/// Return true if the given string starts with the given prefix.
MX_CORE_API bool stringStartsWith(const string& str, const string& prefix);

/// Return true if the given string ends with the given suffix.
MX_CORE_API bool stringEndsWith(const string& str, const string& suffix);

/// Trim leading and trailing spaces from a string.
MX_CORE_API string trimSpaces(const string& str);

/// Combine the hash of a value with an existing seed.
template <typename T> void hashCombine(size_t& seed, const T& value)
{
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Split a name path into string vector
MX_CORE_API StringVec splitNamePath(const string& namePath);

/// Create a name path from a string vector
MX_CORE_API string createNamePath(const StringVec& nameVec);

/// Given a name path, return the parent name path
MX_CORE_API string parentNamePath(const string& namePath);

MATERIALX_NAMESPACE_END

#endif
