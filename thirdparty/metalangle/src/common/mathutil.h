//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// mathutil.h: Math and bit manipulation functions.

#ifndef COMMON_MATHUTIL_H_
#define COMMON_MATHUTIL_H_

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <limits>

#include <anglebase/numerics/safe_math.h>

#include "common/debug.h"
#include "common/platform.h"

namespace angle
{
using base::CheckedNumeric;
using base::IsValueInRangeForNumericType;
}  // namespace angle

namespace gl
{

const unsigned int Float32One   = 0x3F800000;
const unsigned short Float16One = 0x3C00;

template <typename T>
inline constexpr bool isPow2(T x)
{
    static_assert(std::is_integral<T>::value, "isPow2 must be called on an integer type.");
    return (x & (x - 1)) == 0 && (x != 0);
}

inline int log2(int x)
{
    int r = 0;
    while ((x >> r) > 1)
        r++;
    return r;
}

inline unsigned int ceilPow2(unsigned int x)
{
    if (x != 0)
        x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

template <typename DestT, typename SrcT>
inline DestT clampCast(SrcT value)
{
    // For floating-point types with denormalization, min returns the minimum positive normalized
    // value. To find the value that has no values less than it, use numeric_limits::lowest.
    constexpr const long double destLo =
        static_cast<long double>(std::numeric_limits<DestT>::lowest());
    constexpr const long double destHi =
        static_cast<long double>(std::numeric_limits<DestT>::max());
    constexpr const long double srcLo =
        static_cast<long double>(std::numeric_limits<SrcT>::lowest());
    constexpr long double srcHi = static_cast<long double>(std::numeric_limits<SrcT>::max());

    if (destHi < srcHi)
    {
        DestT destMax = std::numeric_limits<DestT>::max();
        if (value >= static_cast<SrcT>(destMax))
        {
            return destMax;
        }
    }

    if (destLo > srcLo)
    {
        DestT destLow = std::numeric_limits<DestT>::lowest();
        if (value <= static_cast<SrcT>(destLow))
        {
            return destLow;
        }
    }

    return static_cast<DestT>(value);
}

// Specialize clampCast for bool->int conversion to avoid MSVS 2015 performance warning when the max
// value is casted to the source type.
template <>
inline unsigned int clampCast(bool value)
{
    return static_cast<unsigned int>(value);
}

template <>
inline int clampCast(bool value)
{
    return static_cast<int>(value);
}

template <typename T, typename MIN, typename MAX>
inline T clamp(T x, MIN min, MAX max)
{
    // Since NaNs fail all comparison tests, a NaN value will default to min
    return x > min ? (x > max ? max : x) : min;
}

inline float clamp01(float x)
{
    return clamp(x, 0.0f, 1.0f);
}

template <const int n>
inline unsigned int unorm(float x)
{
    const unsigned int max = 0xFFFFFFFF >> (32 - n);

    if (x > 1)
    {
        return max;
    }
    else if (x < 0)
    {
        return 0;
    }
    else
    {
        return (unsigned int)(max * x + 0.5f);
    }
}

inline bool supportsSSE2()
{
#if defined(ANGLE_USE_SSE)
    static bool checked  = false;
    static bool supports = false;

    if (checked)
    {
        return supports;
    }

#    if defined(ANGLE_PLATFORM_WINDOWS) && !defined(_M_ARM) && !defined(_M_ARM64)
    {
        int info[4];
        __cpuid(info, 0);

        if (info[0] >= 1)
        {
            __cpuid(info, 1);

            supports = (info[3] >> 26) & 1;
        }
    }
#    endif  // defined(ANGLE_PLATFORM_WINDOWS) && !defined(_M_ARM) && !defined(_M_ARM64)
    checked = true;
    return supports;
#else  // defined(ANGLE_USE_SSE)
    return false;
#endif
}

template <typename destType, typename sourceType>
destType bitCast(const sourceType &source)
{
    size_t copySize = std::min(sizeof(destType), sizeof(sourceType));
    destType output;
    memcpy(&output, &source, copySize);
    return output;
}

// https://stackoverflow.com/a/37581284
template <typename T>
static constexpr double normalize(T value)
{
    return value < 0 ? -static_cast<double>(value) / std::numeric_limits<T>::min()
                     : static_cast<double>(value) / std::numeric_limits<T>::max();
}

inline unsigned short float32ToFloat16(float fp32)
{
    unsigned int fp32i = bitCast<unsigned int>(fp32);
    unsigned int sign  = (fp32i & 0x80000000) >> 16;
    unsigned int abs   = fp32i & 0x7FFFFFFF;

    if (abs > 0x7F800000)
    {  // NaN
        return 0x7FFF;
    }
    else if (abs > 0x47FFEFFF)
    {  // Infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    else if (abs < 0x38800000)  // Denormal
    {
        unsigned int mantissa = (abs & 0x007FFFFF) | 0x00800000;
        int e                 = 113 - (abs >> 23);

        if (e < 24)
        {
            abs = mantissa >> e;
        }
        else
        {
            abs = 0;
        }

        return static_cast<unsigned short>(sign | (abs + 0x00000FFF + ((abs >> 13) & 1)) >> 13);
    }
    else
    {
        return static_cast<unsigned short>(
            sign | (abs + 0xC8000000 + 0x00000FFF + ((abs >> 13) & 1)) >> 13);
    }
}

float float16ToFloat32(unsigned short h);

unsigned int convertRGBFloatsTo999E5(float red, float green, float blue);
void convert999E5toRGBFloats(unsigned int input, float *red, float *green, float *blue);

inline unsigned short float32ToFloat11(float fp32)
{
    const unsigned int float32MantissaMask     = 0x7FFFFF;
    const unsigned int float32ExponentMask     = 0x7F800000;
    const unsigned int float32SignMask         = 0x80000000;
    const unsigned int float32ValueMask        = ~float32SignMask;
    const unsigned int float32ExponentFirstBit = 23;
    const unsigned int float32ExponentBias     = 127;

    const unsigned short float11Max          = 0x7BF;
    const unsigned short float11MantissaMask = 0x3F;
    const unsigned short float11ExponentMask = 0x7C0;
    const unsigned short float11BitMask      = 0x7FF;
    const unsigned int float11ExponentBias   = 14;

    const unsigned int float32Maxfloat11 = 0x477E0000;
    const unsigned int float32Minfloat11 = 0x38800000;

    const unsigned int float32Bits = bitCast<unsigned int>(fp32);
    const bool float32Sign         = (float32Bits & float32SignMask) == float32SignMask;

    unsigned int float32Val = float32Bits & float32ValueMask;

    if ((float32Val & float32ExponentMask) == float32ExponentMask)
    {
        // INF or NAN
        if ((float32Val & float32MantissaMask) != 0)
        {
            return float11ExponentMask |
                   (((float32Val >> 17) | (float32Val >> 11) | (float32Val >> 6) | (float32Val)) &
                    float11MantissaMask);
        }
        else if (float32Sign)
        {
            // -INF is clamped to 0 since float11 is positive only
            return 0;
        }
        else
        {
            return float11ExponentMask;
        }
    }
    else if (float32Sign)
    {
        // float11 is positive only, so clamp to zero
        return 0;
    }
    else if (float32Val > float32Maxfloat11)
    {
        // The number is too large to be represented as a float11, set to max
        return float11Max;
    }
    else
    {
        if (float32Val < float32Minfloat11)
        {
            // The number is too small to be represented as a normalized float11
            // Convert it to a denormalized value.
            const unsigned int shift = (float32ExponentBias - float11ExponentBias) -
                                       (float32Val >> float32ExponentFirstBit);
            float32Val =
                ((1 << float32ExponentFirstBit) | (float32Val & float32MantissaMask)) >> shift;
        }
        else
        {
            // Rebias the exponent to represent the value as a normalized float11
            float32Val += 0xC8000000;
        }

        return ((float32Val + 0xFFFF + ((float32Val >> 17) & 1)) >> 17) & float11BitMask;
    }
}

inline unsigned short float32ToFloat10(float fp32)
{
    const unsigned int float32MantissaMask     = 0x7FFFFF;
    const unsigned int float32ExponentMask     = 0x7F800000;
    const unsigned int float32SignMask         = 0x80000000;
    const unsigned int float32ValueMask        = ~float32SignMask;
    const unsigned int float32ExponentFirstBit = 23;
    const unsigned int float32ExponentBias     = 127;

    const unsigned short float10Max          = 0x3DF;
    const unsigned short float10MantissaMask = 0x1F;
    const unsigned short float10ExponentMask = 0x3E0;
    const unsigned short float10BitMask      = 0x3FF;
    const unsigned int float10ExponentBias   = 14;

    const unsigned int float32Maxfloat10 = 0x477C0000;
    const unsigned int float32Minfloat10 = 0x38800000;

    const unsigned int float32Bits = bitCast<unsigned int>(fp32);
    const bool float32Sign         = (float32Bits & float32SignMask) == float32SignMask;

    unsigned int float32Val = float32Bits & float32ValueMask;

    if ((float32Val & float32ExponentMask) == float32ExponentMask)
    {
        // INF or NAN
        if ((float32Val & float32MantissaMask) != 0)
        {
            return float10ExponentMask |
                   (((float32Val >> 18) | (float32Val >> 13) | (float32Val >> 3) | (float32Val)) &
                    float10MantissaMask);
        }
        else if (float32Sign)
        {
            // -INF is clamped to 0 since float11 is positive only
            return 0;
        }
        else
        {
            return float10ExponentMask;
        }
    }
    else if (float32Sign)
    {
        // float10 is positive only, so clamp to zero
        return 0;
    }
    else if (float32Val > float32Maxfloat10)
    {
        // The number is too large to be represented as a float11, set to max
        return float10Max;
    }
    else
    {
        if (float32Val < float32Minfloat10)
        {
            // The number is too small to be represented as a normalized float11
            // Convert it to a denormalized value.
            const unsigned int shift = (float32ExponentBias - float10ExponentBias) -
                                       (float32Val >> float32ExponentFirstBit);
            float32Val =
                ((1 << float32ExponentFirstBit) | (float32Val & float32MantissaMask)) >> shift;
        }
        else
        {
            // Rebias the exponent to represent the value as a normalized float11
            float32Val += 0xC8000000;
        }

        return ((float32Val + 0x1FFFF + ((float32Val >> 18) & 1)) >> 18) & float10BitMask;
    }
}

inline float float11ToFloat32(unsigned short fp11)
{
    unsigned short exponent = (fp11 >> 6) & 0x1F;
    unsigned short mantissa = fp11 & 0x3F;

    if (exponent == 0x1F)
    {
        // INF or NAN
        return bitCast<float>(0x7f800000 | (mantissa << 17));
    }
    else
    {
        if (exponent != 0)
        {
            // normalized
        }
        else if (mantissa != 0)
        {
            // The value is denormalized
            exponent = 1;

            do
            {
                exponent--;
                mantissa <<= 1;
            } while ((mantissa & 0x40) == 0);

            mantissa = mantissa & 0x3F;
        }
        else  // The value is zero
        {
            exponent = static_cast<unsigned short>(-112);
        }

        return bitCast<float>(((exponent + 112) << 23) | (mantissa << 17));
    }
}

inline float float10ToFloat32(unsigned short fp11)
{
    unsigned short exponent = (fp11 >> 5) & 0x1F;
    unsigned short mantissa = fp11 & 0x1F;

    if (exponent == 0x1F)
    {
        // INF or NAN
        return bitCast<float>(0x7f800000 | (mantissa << 17));
    }
    else
    {
        if (exponent != 0)
        {
            // normalized
        }
        else if (mantissa != 0)
        {
            // The value is denormalized
            exponent = 1;

            do
            {
                exponent--;
                mantissa <<= 1;
            } while ((mantissa & 0x20) == 0);

            mantissa = mantissa & 0x1F;
        }
        else  // The value is zero
        {
            exponent = static_cast<unsigned short>(-112);
        }

        return bitCast<float>(((exponent + 112) << 23) | (mantissa << 18));
    }
}

// Convers to and from float and 16.16 fixed point format.

inline float ConvertFixedToFloat(uint32_t fixedInput)
{
    return static_cast<float>(fixedInput) / 65536.0f;
}

inline uint32_t ConvertFloatToFixed(float floatInput)
{
    static constexpr uint32_t kHighest = 32767 * 65536 + 65535;
    static constexpr uint32_t kLowest  = static_cast<uint32_t>(-32768 * 65536 + 65535);

    if (floatInput > 32767.65535)
    {
        return kHighest;
    }
    else if (floatInput < -32768.65535)
    {
        return kLowest;
    }
    else
    {
        return static_cast<uint32_t>(floatInput * 65536);
    }
}

template <typename T>
inline float normalizedToFloat(T input)
{
    static_assert(std::numeric_limits<T>::is_integer, "T must be an integer.");

    if (sizeof(T) > 2)
    {
        // float has only a 23 bit mantissa, so we need to do the calculation in double precision
        constexpr double inverseMax = 1.0 / std::numeric_limits<T>::max();
        return static_cast<float>(input * inverseMax);
    }
    else
    {
        constexpr float inverseMax = 1.0f / std::numeric_limits<T>::max();
        return input * inverseMax;
    }
}

template <unsigned int inputBitCount, typename T>
inline float normalizedToFloat(T input)
{
    static_assert(std::numeric_limits<T>::is_integer, "T must be an integer.");
    static_assert(inputBitCount < (sizeof(T) * 8), "T must have more bits than inputBitCount.");

    if (inputBitCount > 23)
    {
        // float has only a 23 bit mantissa, so we need to do the calculation in double precision
        constexpr double inverseMax = 1.0 / ((1 << inputBitCount) - 1);
        return static_cast<float>(input * inverseMax);
    }
    else
    {
        constexpr float inverseMax = 1.0f / ((1 << inputBitCount) - 1);
        return input * inverseMax;
    }
}

template <typename T>
inline T floatToNormalized(float input)
{
    if (sizeof(T) > 2)
    {
        // float has only a 23 bit mantissa, so we need to do the calculation in double precision
        return static_cast<T>(std::numeric_limits<T>::max() * static_cast<double>(input) + 0.5);
    }
    else
    {
        return static_cast<T>(std::numeric_limits<T>::max() * input + 0.5f);
    }
}

template <unsigned int outputBitCount, typename T>
inline T floatToNormalized(float input)
{
    static_assert(outputBitCount < (sizeof(T) * 8), "T must have more bits than outputBitCount.");

    if (outputBitCount > 23)
    {
        // float has only a 23 bit mantissa, so we need to do the calculation in double precision
        return static_cast<T>(((1 << outputBitCount) - 1) * static_cast<double>(input) + 0.5);
    }
    else
    {
        return static_cast<T>(((1 << outputBitCount) - 1) * input + 0.5f);
    }
}

template <unsigned int inputBitCount, unsigned int inputBitStart, typename T>
inline T getShiftedData(T input)
{
    static_assert(inputBitCount + inputBitStart <= (sizeof(T) * 8),
                  "T must have at least as many bits as inputBitCount + inputBitStart.");
    const T mask = (1 << inputBitCount) - 1;
    return (input >> inputBitStart) & mask;
}

template <unsigned int inputBitCount, unsigned int inputBitStart, typename T>
inline T shiftData(T input)
{
    static_assert(inputBitCount + inputBitStart <= (sizeof(T) * 8),
                  "T must have at least as many bits as inputBitCount + inputBitStart.");
    const T mask = (1 << inputBitCount) - 1;
    return (input & mask) << inputBitStart;
}

inline unsigned int CountLeadingZeros(uint32_t x)
{
    // Use binary search to find the amount of leading zeros.
    unsigned int zeros = 32u;
    uint32_t y;

    y = x >> 16u;
    if (y != 0)
    {
        zeros = zeros - 16u;
        x     = y;
    }
    y = x >> 8u;
    if (y != 0)
    {
        zeros = zeros - 8u;
        x     = y;
    }
    y = x >> 4u;
    if (y != 0)
    {
        zeros = zeros - 4u;
        x     = y;
    }
    y = x >> 2u;
    if (y != 0)
    {
        zeros = zeros - 2u;
        x     = y;
    }
    y = x >> 1u;
    if (y != 0)
    {
        return zeros - 2u;
    }
    return zeros - x;
}

inline unsigned char average(unsigned char a, unsigned char b)
{
    return ((a ^ b) >> 1) + (a & b);
}

inline signed char average(signed char a, signed char b)
{
    return ((short)a + (short)b) / 2;
}

inline unsigned short average(unsigned short a, unsigned short b)
{
    return ((a ^ b) >> 1) + (a & b);
}

inline signed short average(signed short a, signed short b)
{
    return ((int)a + (int)b) / 2;
}

inline unsigned int average(unsigned int a, unsigned int b)
{
    return ((a ^ b) >> 1) + (a & b);
}

inline int average(int a, int b)
{
    long long average = (static_cast<long long>(a) + static_cast<long long>(b)) / 2ll;
    return static_cast<int>(average);
}

inline float average(float a, float b)
{
    return (a + b) * 0.5f;
}

inline unsigned short averageHalfFloat(unsigned short a, unsigned short b)
{
    return float32ToFloat16((float16ToFloat32(a) + float16ToFloat32(b)) * 0.5f);
}

inline unsigned int averageFloat11(unsigned int a, unsigned int b)
{
    return float32ToFloat11((float11ToFloat32(static_cast<unsigned short>(a)) +
                             float11ToFloat32(static_cast<unsigned short>(b))) *
                            0.5f);
}

inline unsigned int averageFloat10(unsigned int a, unsigned int b)
{
    return float32ToFloat10((float10ToFloat32(static_cast<unsigned short>(a)) +
                             float10ToFloat32(static_cast<unsigned short>(b))) *
                            0.5f);
}

template <typename T>
class Range
{
  public:
    Range() {}
    Range(T lo, T hi) : mLow(lo), mHigh(hi) {}

    T length() const { return (empty() ? 0 : (mHigh - mLow)); }

    bool intersects(Range<T> other)
    {
        if (mLow <= other.mLow)
        {
            return other.mLow < mHigh;
        }
        else
        {
            return mLow < other.mHigh;
        }
    }

    // Assumes that end is non-inclusive.. for example, extending to 5 will make "end" 6.
    void extend(T value)
    {
        mLow  = value < mLow ? value : mLow;
        mHigh = value >= mHigh ? (value + 1) : mHigh;
    }

    bool empty() const { return mHigh <= mLow; }

    bool contains(T value) const { return value >= mLow && value < mHigh; }

    class Iterator final
    {
      public:
        Iterator(T value) : mCurrent(value) {}

        Iterator &operator++()
        {
            mCurrent++;
            return *this;
        }
        bool operator==(const Iterator &other) const { return mCurrent == other.mCurrent; }
        bool operator!=(const Iterator &other) const { return mCurrent != other.mCurrent; }
        T operator*() const { return mCurrent; }

      private:
        T mCurrent;
    };

    Iterator begin() const { return Iterator(mLow); }

    Iterator end() const { return Iterator(mHigh); }

    T low() const { return mLow; }
    T high() const { return mHigh; }

    void invalidate()
    {
        mLow  = std::numeric_limits<T>::max();
        mHigh = std::numeric_limits<T>::min();
    }

  private:
    T mLow;
    T mHigh;
};

typedef Range<int> RangeI;
typedef Range<unsigned int> RangeUI;

struct IndexRange
{
    struct Undefined
    {};
    IndexRange(Undefined) {}
    IndexRange() : IndexRange(0, 0, 0) {}
    IndexRange(size_t start_, size_t end_, size_t vertexIndexCount_)
        : start(start_), end(end_), vertexIndexCount(vertexIndexCount_)
    {
        ASSERT(start <= end);
    }

    // Number of vertices in the range.
    size_t vertexCount() const { return (end - start) + 1; }

    // Inclusive range of indices that are not primitive restart
    size_t start;
    size_t end;

    // Number of non-primitive restart indices
    size_t vertexIndexCount;
};

// Combine a floating-point value representing a mantissa (x) and an integer exponent (exp) into a
// floating-point value. As in GLSL ldexp() built-in.
inline float Ldexp(float x, int exp)
{
    if (exp > 128)
    {
        return std::numeric_limits<float>::infinity();
    }
    if (exp < -126)
    {
        return 0.0f;
    }
    double result = static_cast<double>(x) * std::pow(2.0, static_cast<double>(exp));
    return static_cast<float>(result);
}

// First, both normalized floating-point values are converted into 16-bit integer values.
// Then, the results are packed into the returned 32-bit unsigned integer.
// The first float value will be written to the least significant bits of the output;
// the last float value will be written to the most significant bits.
// The conversion of each value to fixed point is done as follows :
// packSnorm2x16 : round(clamp(c, -1, +1) * 32767.0)
inline uint32_t packSnorm2x16(float f1, float f2)
{
    int16_t leastSignificantBits = static_cast<int16_t>(roundf(clamp(f1, -1.0f, 1.0f) * 32767.0f));
    int16_t mostSignificantBits  = static_cast<int16_t>(roundf(clamp(f2, -1.0f, 1.0f) * 32767.0f));
    return static_cast<uint32_t>(mostSignificantBits) << 16 |
           (static_cast<uint32_t>(leastSignificantBits) & 0xFFFF);
}

// First, unpacks a single 32-bit unsigned integer u into a pair of 16-bit unsigned integers. Then,
// each component is converted to a normalized floating-point value to generate the returned two
// float values. The first float value will be extracted from the least significant bits of the
// input; the last float value will be extracted from the most-significant bits. The conversion for
// unpacked fixed-point value to floating point is done as follows: unpackSnorm2x16 : clamp(f /
// 32767.0, -1, +1)
inline void unpackSnorm2x16(uint32_t u, float *f1, float *f2)
{
    int16_t leastSignificantBits = static_cast<int16_t>(u & 0xFFFF);
    int16_t mostSignificantBits  = static_cast<int16_t>(u >> 16);
    *f1 = clamp(static_cast<float>(leastSignificantBits) / 32767.0f, -1.0f, 1.0f);
    *f2 = clamp(static_cast<float>(mostSignificantBits) / 32767.0f, -1.0f, 1.0f);
}

// First, both normalized floating-point values are converted into 16-bit integer values.
// Then, the results are packed into the returned 32-bit unsigned integer.
// The first float value will be written to the least significant bits of the output;
// the last float value will be written to the most significant bits.
// The conversion of each value to fixed point is done as follows:
// packUnorm2x16 : round(clamp(c, 0, +1) * 65535.0)
inline uint32_t packUnorm2x16(float f1, float f2)
{
    uint16_t leastSignificantBits = static_cast<uint16_t>(roundf(clamp(f1, 0.0f, 1.0f) * 65535.0f));
    uint16_t mostSignificantBits  = static_cast<uint16_t>(roundf(clamp(f2, 0.0f, 1.0f) * 65535.0f));
    return static_cast<uint32_t>(mostSignificantBits) << 16 |
           static_cast<uint32_t>(leastSignificantBits);
}

// First, unpacks a single 32-bit unsigned integer u into a pair of 16-bit unsigned integers. Then,
// each component is converted to a normalized floating-point value to generate the returned two
// float values. The first float value will be extracted from the least significant bits of the
// input; the last float value will be extracted from the most-significant bits. The conversion for
// unpacked fixed-point value to floating point is done as follows: unpackUnorm2x16 : f / 65535.0
inline void unpackUnorm2x16(uint32_t u, float *f1, float *f2)
{
    uint16_t leastSignificantBits = static_cast<uint16_t>(u & 0xFFFF);
    uint16_t mostSignificantBits  = static_cast<uint16_t>(u >> 16);
    *f1                           = static_cast<float>(leastSignificantBits) / 65535.0f;
    *f2                           = static_cast<float>(mostSignificantBits) / 65535.0f;
}

// Helper functions intended to be used only here.
namespace priv
{

inline uint8_t ToPackedUnorm8(float f)
{
    return static_cast<uint8_t>(roundf(clamp(f, 0.0f, 1.0f) * 255.0f));
}

inline int8_t ToPackedSnorm8(float f)
{
    return static_cast<int8_t>(roundf(clamp(f, -1.0f, 1.0f) * 127.0f));
}

}  // namespace priv

// Packs 4 normalized unsigned floating-point values to a single 32-bit unsigned integer. Works
// similarly to packUnorm2x16. The floats are clamped to the range 0.0 to 1.0, and written to the
// unsigned integer starting from the least significant bits.
inline uint32_t PackUnorm4x8(float f1, float f2, float f3, float f4)
{
    uint8_t bits[4];
    bits[0]         = priv::ToPackedUnorm8(f1);
    bits[1]         = priv::ToPackedUnorm8(f2);
    bits[2]         = priv::ToPackedUnorm8(f3);
    bits[3]         = priv::ToPackedUnorm8(f4);
    uint32_t result = 0u;
    for (int i = 0; i < 4; ++i)
    {
        int shift = i * 8;
        result |= (static_cast<uint32_t>(bits[i]) << shift);
    }
    return result;
}

// Unpacks 4 normalized unsigned floating-point values from a single 32-bit unsigned integer into f.
// Works similarly to unpackUnorm2x16. The floats are unpacked starting from the least significant
// bits.
inline void UnpackUnorm4x8(uint32_t u, float *f)
{
    for (int i = 0; i < 4; ++i)
    {
        int shift    = i * 8;
        uint8_t bits = static_cast<uint8_t>((u >> shift) & 0xFF);
        f[i]         = static_cast<float>(bits) / 255.0f;
    }
}

// Packs 4 normalized signed floating-point values to a single 32-bit unsigned integer. The floats
// are clamped to the range -1.0 to 1.0, and written to the unsigned integer starting from the least
// significant bits.
inline uint32_t PackSnorm4x8(float f1, float f2, float f3, float f4)
{
    int8_t bits[4];
    bits[0]         = priv::ToPackedSnorm8(f1);
    bits[1]         = priv::ToPackedSnorm8(f2);
    bits[2]         = priv::ToPackedSnorm8(f3);
    bits[3]         = priv::ToPackedSnorm8(f4);
    uint32_t result = 0u;
    for (int i = 0; i < 4; ++i)
    {
        int shift = i * 8;
        result |= ((static_cast<uint32_t>(bits[i]) & 0xFF) << shift);
    }
    return result;
}

// Unpacks 4 normalized signed floating-point values from a single 32-bit unsigned integer into f.
// Works similarly to unpackSnorm2x16. The floats are unpacked starting from the least significant
// bits, and clamped to the range -1.0 to 1.0.
inline void UnpackSnorm4x8(uint32_t u, float *f)
{
    for (int i = 0; i < 4; ++i)
    {
        int shift   = i * 8;
        int8_t bits = static_cast<int8_t>((u >> shift) & 0xFF);
        f[i]        = clamp(static_cast<float>(bits) / 127.0f, -1.0f, 1.0f);
    }
}

// Returns an unsigned integer obtained by converting the two floating-point values to the 16-bit
// floating-point representation found in the OpenGL ES Specification, and then packing these
// two 16-bit integers into a 32-bit unsigned integer.
// f1: The 16 least-significant bits of the result;
// f2: The 16 most-significant bits.
inline uint32_t packHalf2x16(float f1, float f2)
{
    uint16_t leastSignificantBits = static_cast<uint16_t>(float32ToFloat16(f1));
    uint16_t mostSignificantBits  = static_cast<uint16_t>(float32ToFloat16(f2));
    return static_cast<uint32_t>(mostSignificantBits) << 16 |
           static_cast<uint32_t>(leastSignificantBits);
}

// Returns two floating-point values obtained by unpacking a 32-bit unsigned integer into a pair of
// 16-bit values, interpreting those values as 16-bit floating-point numbers according to the OpenGL
// ES Specification, and converting them to 32-bit floating-point values. The first float value is
// obtained from the 16 least-significant bits of u; the second component is obtained from the 16
// most-significant bits of u.
inline void unpackHalf2x16(uint32_t u, float *f1, float *f2)
{
    uint16_t leastSignificantBits = static_cast<uint16_t>(u & 0xFFFF);
    uint16_t mostSignificantBits  = static_cast<uint16_t>(u >> 16);

    *f1 = float16ToFloat32(leastSignificantBits);
    *f2 = float16ToFloat32(mostSignificantBits);
}

inline uint8_t sRGBToLinear(uint8_t srgbValue)
{
    float value = srgbValue / 255.0f;
    if (value <= 0.04045f)
    {
        value = value / 12.92f;
    }
    else
    {
        value = std::pow((value + 0.055f) / 1.055f, 2.4f);
    }
    return static_cast<uint8_t>(clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
}

inline uint8_t linearToSRGB(uint8_t linearValue)
{
    float value = linearValue / 255.0f;
    if (value <= 0.0f)
    {
        value = 0.0f;
    }
    else if (value < 0.0031308f)
    {
        value = value * 12.92f;
    }
    else if (value < 1.0f)
    {
        value = std::pow(value, 0.41666f) * 1.055f - 0.055f;
    }
    else
    {
        value = 1.0f;
    }
    return static_cast<uint8_t>(clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
}

// Reverse the order of the bits.
inline uint32_t BitfieldReverse(uint32_t value)
{
    // TODO(oetuaho@nvidia.com): Optimize this if needed. There don't seem to be compiler intrinsics
    // for this, and right now it's not used in performance-critical paths.
    uint32_t result = 0u;
    for (size_t j = 0u; j < 32u; ++j)
    {
        result |= (((value >> j) & 1u) << (31u - j));
    }
    return result;
}

// Count the 1 bits.
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#    define ANGLE_HAS_BITCOUNT_32
inline int BitCount(uint32_t bits)
{
    return static_cast<int>(__popcnt(bits));
}
#    if defined(_M_X64)
#        define ANGLE_HAS_BITCOUNT_64
inline int BitCount(uint64_t bits)
{
    return static_cast<int>(__popcnt64(bits));
}
#    endif  // defined(_M_X64)
#endif      // defined(_M_IX86) || defined(_M_X64)

#if defined(ANGLE_PLATFORM_POSIX)
#    define ANGLE_HAS_BITCOUNT_32
inline int BitCount(uint32_t bits)
{
    return __builtin_popcount(bits);
}

#    if defined(ANGLE_IS_64_BIT_CPU)
#        define ANGLE_HAS_BITCOUNT_64
inline int BitCount(uint64_t bits)
{
    return __builtin_popcountll(bits);
}
#    endif  // defined(ANGLE_IS_64_BIT_CPU)
#endif      // defined(ANGLE_PLATFORM_POSIX)

int BitCountPolyfill(uint32_t bits);

#if !defined(ANGLE_HAS_BITCOUNT_32)
inline int BitCount(const uint32_t bits)
{
    return BitCountPolyfill(bits);
}
#endif  // !defined(ANGLE_HAS_BITCOUNT_32)

#if !defined(ANGLE_HAS_BITCOUNT_64)
inline int BitCount(const uint64_t bits)
{
    return BitCount(static_cast<uint32_t>(bits >> 32)) + BitCount(static_cast<uint32_t>(bits));
}
#endif  // !defined(ANGLE_HAS_BITCOUNT_64)
#undef ANGLE_HAS_BITCOUNT_32
#undef ANGLE_HAS_BITCOUNT_64

inline int BitCount(uint8_t bits)
{
    return BitCount(static_cast<uint32_t>(bits));
}

inline int BitCount(uint16_t bits)
{
    return BitCount(static_cast<uint32_t>(bits));
}

#if defined(ANGLE_PLATFORM_WINDOWS)
// Return the index of the least significant bit set. Indexing is such that bit 0 is the least
// significant bit. Implemented for different bit widths on different platforms.
inline unsigned long ScanForward(uint32_t bits)
{
    ASSERT(bits != 0u);
    unsigned long firstBitIndex = 0ul;
    unsigned char ret           = _BitScanForward(&firstBitIndex, bits);
    ASSERT(ret != 0u);
    return firstBitIndex;
}

#    if defined(ANGLE_IS_64_BIT_CPU)
inline unsigned long ScanForward(uint64_t bits)
{
    ASSERT(bits != 0u);
    unsigned long firstBitIndex = 0ul;
    unsigned char ret           = _BitScanForward64(&firstBitIndex, bits);
    ASSERT(ret != 0u);
    return firstBitIndex;
}
#    endif  // defined(ANGLE_IS_64_BIT_CPU)
#endif      // defined(ANGLE_PLATFORM_WINDOWS)

#if defined(ANGLE_PLATFORM_POSIX)
inline unsigned long ScanForward(uint32_t bits)
{
    ASSERT(bits != 0u);
    return static_cast<unsigned long>(__builtin_ctz(bits));
}

#    if defined(ANGLE_IS_64_BIT_CPU)
inline unsigned long ScanForward(uint64_t bits)
{
    ASSERT(bits != 0u);
    return static_cast<unsigned long>(__builtin_ctzll(bits));
}
#    endif  // defined(ANGLE_IS_64_BIT_CPU)
#endif      // defined(ANGLE_PLATFORM_POSIX)

inline unsigned long ScanForward(uint8_t bits)
{
    return ScanForward(static_cast<uint32_t>(bits));
}

inline unsigned long ScanForward(uint16_t bits)
{
    return ScanForward(static_cast<uint32_t>(bits));
}

// Return the index of the most significant bit set. Indexing is such that bit 0 is the least
// significant bit.
inline unsigned long ScanReverse(unsigned long bits)
{
    ASSERT(bits != 0u);
#if defined(ANGLE_PLATFORM_WINDOWS)
    unsigned long lastBitIndex = 0ul;
    unsigned char ret          = _BitScanReverse(&lastBitIndex, bits);
    ASSERT(ret != 0u);
    return lastBitIndex;
#elif defined(ANGLE_PLATFORM_POSIX)
    return static_cast<unsigned long>(sizeof(unsigned long) * CHAR_BIT - 1 - __builtin_clzl(bits));
#else
#    error Please implement bit-scan-reverse for your platform!
#endif
}

// Returns -1 on 0, otherwise the index of the least significant 1 bit as in GLSL.
template <typename T>
int FindLSB(T bits)
{
    static_assert(std::is_integral<T>::value, "must be integral type.");
    if (bits == 0u)
    {
        return -1;
    }
    else
    {
        return static_cast<int>(ScanForward(bits));
    }
}

// Returns -1 on 0, otherwise the index of the most significant 1 bit as in GLSL.
template <typename T>
int FindMSB(T bits)
{
    static_assert(std::is_integral<T>::value, "must be integral type.");
    if (bits == 0u)
    {
        return -1;
    }
    else
    {
        return static_cast<int>(ScanReverse(bits));
    }
}

// Returns whether the argument is Not a Number.
// IEEE 754 single precision NaN representation: Exponent(8 bits) - 255, Mantissa(23 bits) -
// non-zero.
inline bool isNaN(float f)
{
    // Exponent mask: ((1u << 8) - 1u) << 23 = 0x7f800000u
    // Mantissa mask: ((1u << 23) - 1u) = 0x7fffffu
    return ((bitCast<uint32_t>(f) & 0x7f800000u) == 0x7f800000u) &&
           (bitCast<uint32_t>(f) & 0x7fffffu);
}

// Returns whether the argument is infinity.
// IEEE 754 single precision infinity representation: Exponent(8 bits) - 255, Mantissa(23 bits) -
// zero.
inline bool isInf(float f)
{
    // Exponent mask: ((1u << 8) - 1u) << 23 = 0x7f800000u
    // Mantissa mask: ((1u << 23) - 1u) = 0x7fffffu
    return ((bitCast<uint32_t>(f) & 0x7f800000u) == 0x7f800000u) &&
           !(bitCast<uint32_t>(f) & 0x7fffffu);
}

namespace priv
{
template <unsigned int N, unsigned int R>
struct iSquareRoot
{
    static constexpr unsigned int solve()
    {
        return (R * R > N)
                   ? 0
                   : ((R * R == N) ? R : static_cast<unsigned int>(iSquareRoot<N, R + 1>::value));
    }
    enum Result
    {
        value = iSquareRoot::solve()
    };
};

template <unsigned int N>
struct iSquareRoot<N, N>
{
    enum result
    {
        value = N
    };
};

}  // namespace priv

template <unsigned int N>
constexpr unsigned int iSquareRoot()
{
    return priv::iSquareRoot<N, 1>::value;
}

// Sum, difference and multiplication operations for signed ints that wrap on 32-bit overflow.
//
// Unsigned types are defined to do arithmetic modulo 2^n in C++. For signed types, overflow
// behavior is undefined.

template <typename T>
inline T WrappingSum(T lhs, T rhs)
{
    uint32_t lhsUnsigned = static_cast<uint32_t>(lhs);
    uint32_t rhsUnsigned = static_cast<uint32_t>(rhs);
    return static_cast<T>(lhsUnsigned + rhsUnsigned);
}

template <typename T>
inline T WrappingDiff(T lhs, T rhs)
{
    uint32_t lhsUnsigned = static_cast<uint32_t>(lhs);
    uint32_t rhsUnsigned = static_cast<uint32_t>(rhs);
    return static_cast<T>(lhsUnsigned - rhsUnsigned);
}

inline int32_t WrappingMul(int32_t lhs, int32_t rhs)
{
    int64_t lhsWide = static_cast<int64_t>(lhs);
    int64_t rhsWide = static_cast<int64_t>(rhs);
    // The multiplication is guaranteed not to overflow.
    int64_t resultWide = lhsWide * rhsWide;
    // Implement the desired wrapping behavior by masking out the high-order 32 bits.
    resultWide = resultWide & 0xffffffffll;
    // Casting to a narrower signed type is fine since the casted value is representable in the
    // narrower type.
    return static_cast<int32_t>(resultWide);
}

inline float scaleScreenDimensionToNdc(float dimensionScreen, float viewportDimension)
{
    return 2.0f * dimensionScreen / viewportDimension;
}

inline float scaleScreenCoordinateToNdc(float coordinateScreen, float viewportDimension)
{
    float halfShifted = coordinateScreen / viewportDimension;
    return 2.0f * (halfShifted - 0.5f);
}

}  // namespace gl

namespace rx
{

template <typename T>
T roundUp(const T value, const T alignment)
{
    auto temp = value + alignment - static_cast<T>(1);
    return temp - temp % alignment;
}

template <typename T>
constexpr T roundUpPow2(const T value, const T alignment)
{
    ASSERT(gl::isPow2(alignment));
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
angle::CheckedNumeric<T> CheckedRoundUp(const T value, const T alignment)
{
    angle::CheckedNumeric<T> checkedValue(value);
    angle::CheckedNumeric<T> checkedAlignment(alignment);
    return roundUp(checkedValue, checkedAlignment);
}

inline constexpr unsigned int UnsignedCeilDivide(unsigned int value, unsigned int divisor)
{
    unsigned int divided = value / divisor;
    return (divided + ((value % divisor == 0) ? 0 : 1));
}

#if defined(__has_builtin)
#    define ANGLE_HAS_BUILTIN(x) __has_builtin(x)
#else
#    define ANGLE_HAS_BUILTIN(x) 0
#endif

#if defined(_MSC_VER)

#    define ANGLE_ROTL(x, y) _rotl(x, y)
#    define ANGLE_ROTL64(x, y) _rotl64(x, y)
#    define ANGLE_ROTR16(x, y) _rotr16(x, y)

#elif defined(__clang__) && ANGLE_HAS_BUILTIN(__builtin_rotateleft32) && \
    ANGLE_HAS_BUILTIN(__builtin_rotateleft64) && ANGLE_HAS_BUILTIN(__builtin_rotateright16)

#    define ANGLE_ROTL(x, y) __builtin_rotateleft32(x, y)
#    define ANGLE_ROTL64(x, y) __builtin_rotateleft64(x, y)
#    define ANGLE_ROTR16(x, y) __builtin_rotateright16(x, y)

#else

inline uint32_t RotL(uint32_t x, int8_t r)
{
    return (x << r) | (x >> (32 - r));
}

inline uint64_t RotL64(uint64_t x, int8_t r)
{
    return (x << r) | (x >> (64 - r));
}

inline uint16_t RotR16(uint16_t x, int8_t r)
{
    return (x >> r) | (x << (16 - r));
}

#    define ANGLE_ROTL(x, y) ::rx::RotL(x, y)
#    define ANGLE_ROTL64(x, y) ::rx::RotL64(x, y)
#    define ANGLE_ROTR16(x, y) ::rx::RotR16(x, y)

#endif  // namespace rx

constexpr unsigned int Log2(unsigned int bytes)
{
    return bytes == 1 ? 0 : (1 + Log2(bytes / 2));
}
}  // namespace rx

#endif  // COMMON_MATHUTIL_H_
