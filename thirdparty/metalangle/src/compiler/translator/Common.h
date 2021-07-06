//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_COMMON_H_
#define COMPILER_TRANSLATOR_COMMON_H_

#include <stdio.h>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/angleutils.h"
#include "common/debug.h"
#include "common/third_party/smhasher/src/PMurHash.h"
#include "compiler/translator/PoolAlloc.h"

namespace sh
{

struct TSourceLoc
{
    int first_file;
    int first_line;
    int last_file;
    int last_line;
};

//
// Put POOL_ALLOCATOR_NEW_DELETE in base classes to make them use this scheme.
//
#define POOL_ALLOCATOR_NEW_DELETE                                                    \
    void *operator new(size_t s) { return GetGlobalPoolAllocator()->allocate(s); }   \
    void *operator new(size_t, void *_Where) { return (_Where); }                    \
    void operator delete(void *) {}                                                  \
    void operator delete(void *, void *) {}                                          \
    void *operator new[](size_t s) { return GetGlobalPoolAllocator()->allocate(s); } \
    void *operator new[](size_t, void *_Where) { return (_Where); }                  \
    void operator delete[](void *) {}                                                \
    void operator delete[](void *, void *) {}

//
// Pool version of string.
//
typedef pool_allocator<char> TStringAllocator;
typedef std::basic_string<char, std::char_traits<char>, TStringAllocator> TString;
typedef std::basic_ostringstream<char, std::char_traits<char>, TStringAllocator> TStringStream;

//
// Persistent string memory.  Should only be used for strings that survive
// across compiles.
//
#define TPersistString std::string
#define TPersistStringStream std::ostringstream

//
// Pool allocator versions of vectors, lists, and maps
//
template <class T>
class TVector : public std::vector<T, pool_allocator<T>>
{
  public:
    POOL_ALLOCATOR_NEW_DELETE

    typedef typename std::vector<T, pool_allocator<T>>::size_type size_type;
    TVector() : std::vector<T, pool_allocator<T>>() {}
    TVector(const pool_allocator<T> &a) : std::vector<T, pool_allocator<T>>(a) {}
    TVector(size_type i) : std::vector<T, pool_allocator<T>>(i) {}
    TVector(size_type i, const T &value) : std::vector<T, pool_allocator<T>>(i, value) {}
    template <typename InputIt>
    TVector(InputIt first, InputIt last) : std::vector<T, pool_allocator<T>>(first, last)
    {}
    TVector(std::initializer_list<T> init) : std::vector<T, pool_allocator<T>>(init) {}
};

template <class K, class D, class H = std::hash<K>, class CMP = std::equal_to<K>>
class TUnorderedMap : public std::unordered_map<K, D, H, CMP, pool_allocator<std::pair<const K, D>>>
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    typedef pool_allocator<std::pair<const K, D>> tAllocator;

    TUnorderedMap() : std::unordered_map<K, D, H, CMP, tAllocator>() {}
    // use correct two-stage name lookup supported in gcc 3.4 and above
    TUnorderedMap(const tAllocator &a)
        : std::unordered_map<K, D, H, CMP, tAllocator>(
              std::unordered_map<K, D, H, CMP, tAllocator>::key_compare(),
              a)
    {}
};

template <class K, class D, class CMP = std::less<K>>
class TMap : public std::map<K, D, CMP, pool_allocator<std::pair<const K, D>>>
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    typedef pool_allocator<std::pair<const K, D>> tAllocator;

    TMap() : std::map<K, D, CMP, tAllocator>() {}
    // use correct two-stage name lookup supported in gcc 3.4 and above
    TMap(const tAllocator &a)
        : std::map<K, D, CMP, tAllocator>(std::map<K, D, CMP, tAllocator>::key_compare(), a)
    {}
};

// Integer to TString conversion
template <typename T>
inline TString str(T i)
{
    ASSERT(std::numeric_limits<T>::is_integer);
    char buffer[((8 * sizeof(T)) / 3) + 3];
    const char *formatStr = std::numeric_limits<T>::is_signed ? "%d" : "%u";
    snprintf(buffer, sizeof(buffer), formatStr, i);
    return buffer;
}

// Allocate a char array in the global memory pool. str must be a null terminated string. strLength
// is the length without the null terminator.
inline const char *AllocatePoolCharArray(const char *str, size_t strLength)
{
    size_t requiredSize = strLength + 1;
    char *buffer        = static_cast<char *>(GetGlobalPoolAllocator()->allocate(requiredSize));
    memcpy(buffer, str, requiredSize);
    ASSERT(buffer[strLength] == '\0');
    return buffer;
}

// Initialize a new stream which must be imbued with the classic locale
template <typename T>
T InitializeStream()
{
    T stream;
    stream.imbue(std::locale::classic());
    return stream;
}

}  // namespace sh

namespace std
{
template <>
struct hash<sh::TString>
{
    size_t operator()(const sh::TString &s) const
    {
        return angle::PMurHash32(0, s.data(), static_cast<int>(s.length()));
    }
};
}  // namespace std

#endif  // COMPILER_TRANSLATOR_COMMON_H_
