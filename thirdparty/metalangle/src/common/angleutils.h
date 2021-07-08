//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// angleutils.h: Common ANGLE utilities.

#ifndef COMMON_ANGLEUTILS_H_
#define COMMON_ANGLEUTILS_H_

#include "common/platform.h"

#include <climits>
#include <cstdarg>
#include <cstddef>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// A helper class to disallow copy and assignment operators
namespace angle
{

#if defined(ANGLE_ENABLE_D3D9) || defined(ANGLE_ENABLE_D3D11)
using Microsoft::WRL::ComPtr;
#endif  // defined(ANGLE_ENABLE_D3D9) || defined(ANGLE_ENABLE_D3D11)

class NonCopyable
{
  protected:
    constexpr NonCopyable() = default;
    ~NonCopyable()          = default;

  private:
    NonCopyable(const NonCopyable &) = delete;
    void operator=(const NonCopyable &) = delete;
};

extern const uintptr_t DirtyPointer;

}  // namespace angle

template <typename T, size_t N>
constexpr inline size_t ArraySize(T (&)[N])
{
    return N;
}

template <typename T>
class WrappedArray final : angle::NonCopyable
{
  public:
    template <size_t N>
    constexpr WrappedArray(const T (&data)[N]) : mArray(&data[0]), mSize(N)
    {}

    constexpr WrappedArray() : mArray(nullptr), mSize(0) {}
    constexpr WrappedArray(const T *data, size_t size) : mArray(data), mSize(size) {}

    WrappedArray(WrappedArray &&other) : WrappedArray()
    {
        std::swap(mArray, other.mArray);
        std::swap(mSize, other.mSize);
    }

    ~WrappedArray() {}

    constexpr const T *get() const { return mArray; }
    constexpr size_t size() const { return mSize; }

  private:
    const T *mArray;
    size_t mSize;
};

template <typename T, unsigned int N>
void SafeRelease(T (&resourceBlock)[N])
{
    for (unsigned int i = 0; i < N; i++)
    {
        SafeRelease(resourceBlock[i]);
    }
}

template <typename T>
void SafeRelease(T &resource)
{
    if (resource)
    {
        resource->Release();
        resource = nullptr;
    }
}

template <typename T>
void SafeDelete(T *&resource)
{
    delete resource;
    resource = nullptr;
}

template <typename T>
void SafeDeleteContainer(T &resource)
{
    for (auto &element : resource)
    {
        SafeDelete(element);
    }
    resource.clear();
}

template <typename T>
void SafeDeleteArray(T *&resource)
{
    delete[] resource;
    resource = nullptr;
}

// Provide a less-than function for comparing structs
// Note: struct memory must be initialized to zero, because of packing gaps
template <typename T>
inline bool StructLessThan(const T &a, const T &b)
{
    return (memcmp(&a, &b, sizeof(T)) < 0);
}

// Provide a less-than function for comparing structs
// Note: struct memory must be initialized to zero, because of packing gaps
template <typename T>
inline bool StructEquals(const T &a, const T &b)
{
    return (memcmp(&a, &b, sizeof(T)) == 0);
}

template <typename T>
inline void StructZero(T *obj)
{
    memset(obj, 0, sizeof(T));
}

template <typename T>
inline bool IsMaskFlagSet(T mask, T flag)
{
    // Handles multibit flags as well
    return (mask & flag) == flag;
}

inline const char *MakeStaticString(const std::string &str)
{
    // On the heap so that no destructor runs on application exit.
    static std::set<std::string> *strings = new std::set<std::string>;
    std::set<std::string>::iterator it    = strings->find(str);
    if (it != strings->end())
    {
        return it->c_str();
    }

    return strings->insert(str).first->c_str();
}

std::string ArrayString(unsigned int i);

// Indices are stored in vectors with the outermost index in the back. In the output of the function
// the indices are reversed.
std::string ArrayIndexString(const std::vector<unsigned int> &indices);

inline std::string Str(int i)
{
    std::stringstream strstr;
    strstr << i;
    return strstr.str();
}

size_t FormatStringIntoVector(const char *fmt, va_list vararg, std::vector<char> &buffer);

template <typename T>
std::string ToString(const T &value)
{
    std::ostringstream o;
    o << value;
    return o.str();
}

// snprintf is not defined with MSVC prior to to msvc14
#if defined(_MSC_VER) && _MSC_VER < 1900
#    define snprintf _snprintf
#endif

#define GL_A1RGB5_ANGLEX 0x6AC5
#define GL_BGRX8_ANGLEX 0x6ABA
#define GL_BGR565_ANGLEX 0x6ABB
#define GL_BGRA4_ANGLEX 0x6ABC
#define GL_BGR5_A1_ANGLEX 0x6ABD
#define GL_INT_64_ANGLEX 0x6ABE
#define GL_UINT_64_ANGLEX 0x6ABF
#define GL_BGRA8_SRGB_ANGLEX 0x6AC0

// These are dummy formats used to fit typeless D3D textures that can be bound to EGL pbuffers into
// the format system (for extension EGL_ANGLE_d3d_texture_client_buffer):
#define GL_RGBA8_TYPELESS_ANGLEX 0x6AC1
#define GL_RGBA8_TYPELESS_SRGB_ANGLEX 0x6AC2
#define GL_BGRA8_TYPELESS_ANGLEX 0x6AC3
#define GL_BGRA8_TYPELESS_SRGB_ANGLEX 0x6AC4

#define GL_R8_SSCALED_ANGLEX 0x6AC6
#define GL_RG8_SSCALED_ANGLEX 0x6AC7
#define GL_RGB8_SSCALED_ANGLEX 0x6AC8
#define GL_RGBA8_SSCALED_ANGLEX 0x6AC9
#define GL_R8_USCALED_ANGLEX 0x6ACA
#define GL_RG8_USCALED_ANGLEX 0x6ACB
#define GL_RGB8_USCALED_ANGLEX 0x6ACC
#define GL_RGBA8_USCALED_ANGLEX 0x6ACD

#define GL_R16_SSCALED_ANGLEX 0x6ACE
#define GL_RG16_SSCALED_ANGLEX 0x6ACF
#define GL_RGB16_SSCALED_ANGLEX 0x6AD0
#define GL_RGBA16_SSCALED_ANGLEX 0x6AD1
#define GL_R16_USCALED_ANGLEX 0x6AD2
#define GL_RG16_USCALED_ANGLEX 0x6AD3
#define GL_RGB16_USCALED_ANGLEX 0x6AD4
#define GL_RGBA16_USCALED_ANGLEX 0x6AD5

#define GL_R32_SSCALED_ANGLEX 0x6AD6
#define GL_RG32_SSCALED_ANGLEX 0x6AD7
#define GL_RGB32_SSCALED_ANGLEX 0x6AD8
#define GL_RGBA32_SSCALED_ANGLEX 0x6AD9
#define GL_R32_USCALED_ANGLEX 0x6ADA
#define GL_RG32_USCALED_ANGLEX 0x6ADB
#define GL_RGB32_USCALED_ANGLEX 0x6ADC
#define GL_RGBA32_USCALED_ANGLEX 0x6ADD

#define GL_R32_SNORM_ANGLEX 0x6ADE
#define GL_RG32_SNORM_ANGLEX 0x6ADF
#define GL_RGB32_SNORM_ANGLEX 0x6AE0
#define GL_RGBA32_SNORM_ANGLEX 0x6AE1
#define GL_R32_UNORM_ANGLEX 0x6AE2
#define GL_RG32_UNORM_ANGLEX 0x6AE3
#define GL_RGB32_UNORM_ANGLEX 0x6AE4
#define GL_RGBA32_UNORM_ANGLEX 0x6AE5

#define GL_R32_FIXED_ANGLEX 0x6AE6
#define GL_RG32_FIXED_ANGLEX 0x6AE7
#define GL_RGB32_FIXED_ANGLEX 0x6AE8
#define GL_RGBA32_FIXED_ANGLEX 0x6AE9

#define GL_RGB10_A2_SINT_ANGLEX 0x6AEA
#define GL_RGB10_A2_SNORM_ANGLEX 0x6AEB
#define GL_RGB10_A2_SSCALED_ANGLEX 0x6AEC
#define GL_RGB10_A2_USCALED_ANGLEX 0x6AED

// EXT_texture_type_2_10_10_10_REV
#define GL_RGB10_UNORM_ANGLEX 0x6AEE

// These are dummy formats for OES_vertex_type_10_10_10_2
#define GL_A2_RGB10_UNORM_ANGLEX 0x6AEF
#define GL_A2_RGB10_SNORM_ANGLEX 0x6AF0
#define GL_A2_RGB10_USCALED_ANGLEX 0x6AF1
#define GL_A2_RGB10_SSCALED_ANGLEX 0x6AF2
#define GL_X2_RGB10_UINT_ANGLEX 0x6AF3
#define GL_X2_RGB10_SINT_ANGLEX 0x6AF4
#define GL_X2_RGB10_USCALED_ANGLEX 0x6AF5
#define GL_X2_RGB10_SSCALED_ANGLEX 0x6AF6
#define GL_X2_RGB10_UNORM_ANGLEX 0x6AF7
#define GL_X2_RGB10_SNORM_ANGLEX 0x6AF8

#define ANGLE_CHECK_GL_ALLOC(context, result) \
    ANGLE_CHECK(context, result, "Failed to allocate host memory", GL_OUT_OF_MEMORY)

#define ANGLE_CHECK_GL_MATH(context, result) \
    ANGLE_CHECK(context, result, "Integer overflow.", GL_INVALID_OPERATION)

#define ANGLE_GL_UNREACHABLE(context) \
    UNREACHABLE();                    \
    ANGLE_CHECK(context, false, "Unreachable Code.", GL_INVALID_OPERATION)

// The below inlining code lifted from V8.
#if defined(__clang__) || (defined(__GNUC__) && defined(__has_attribute))
#    define ANGLE_HAS_ATTRIBUTE_ALWAYS_INLINE (__has_attribute(always_inline))
#    define ANGLE_HAS___FORCEINLINE 0
#elif defined(_MSC_VER)
#    define ANGLE_HAS_ATTRIBUTE_ALWAYS_INLINE 0
#    define ANGLE_HAS___FORCEINLINE 1
#else
#    define ANGLE_HAS_ATTRIBUTE_ALWAYS_INLINE 0
#    define ANGLE_HAS___FORCEINLINE 0
#endif

#if defined(NDEBUG) && ANGLE_HAS_ATTRIBUTE_ALWAYS_INLINE
#    define ANGLE_INLINE inline __attribute__((always_inline))
#elif defined(NDEBUG) && ANGLE_HAS___FORCEINLINE
#    define ANGLE_INLINE __forceinline
#else
#    define ANGLE_INLINE inline
#endif

#if defined(__clang__) || (defined(__GNUC__) && defined(__has_attribute))
#    if __has_attribute(noinline)
#        define ANGLE_NOINLINE __attribute__((noinline))
#    else
#        define ANGLE_NOINLINE
#    endif
#elif defined(_MSC_VER)
#    define ANGLE_NOINLINE __declspec(noinline)
#else
#    define ANGLE_NOINLINE
#endif

#if defined(__clang__) || (defined(__GNUC__) && defined(__has_attribute))
#    if __has_attribute(format)
#        define ANGLE_FORMAT_PRINTF(fmt, args) __attribute__((format(__printf__, fmt, args)))
#    else
#        define ANGLE_FORMAT_PRINTF(fmt, args)
#    endif
#else
#    define ANGLE_FORMAT_PRINTF(fmt, args)
#endif

// Format messes up the # inside the macro.
// clang-format off
#ifndef ANGLE_STRINGIFY
#    define ANGLE_STRINGIFY(x) #x
#endif
// clang-format on

#ifndef ANGLE_MACRO_STRINGIFY
#    define ANGLE_MACRO_STRINGIFY(x) ANGLE_STRINGIFY(x)
#endif

// Detect support for C++17 [[nodiscard]]
#if !defined(__has_cpp_attribute)
#    define __has_cpp_attribute(name) 0
#endif  // !defined(__has_cpp_attribute)

#if __has_cpp_attribute(nodiscard)
#    define ANGLE_NO_DISCARD [[nodiscard]]
#else
#    define ANGLE_NO_DISCARD
#endif  // __has_cpp_attribute(nodiscard)

#if __has_cpp_attribute(maybe_unused)
#    define ANGLE_MAYBE_UNUSED [[maybe_unused]]
#else
#    define ANGLE_MAYBE_UNUSED
#endif  // __has_cpp_attribute(maybe_unused)

#if __has_cpp_attribute(require_constant_initialization)
#    define ANGLE_REQUIRE_CONSTANT_INIT [[require_constant_initialization]]
#else
#    define ANGLE_REQUIRE_CONSTANT_INIT
#endif  // __has_cpp_attribute(require_constant_initialization)

#endif  // COMMON_ANGLEUTILS_H_
