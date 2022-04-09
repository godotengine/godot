/*
Convection Texture Tools
Copyright (c) 2018-2019 Eric Lasota

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
#pragma once
#ifndef __CVTT_PARALLELMATH_H__
#define __CVTT_PARALLELMATH_H__

#include "ConvectionKernels.h"
#include "ConvectionKernels_Config.h"

#ifdef CVTT_USE_SSE2
#include <emmintrin.h>
#endif

#include <float.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <math.h>

#define UNREFERENCED_PARAMETER(n) ((void)n)

// Parallel math implementation
//
// After preprocessor defs are handled, what this should do is expose the following types:
// SInt16 - Signed 16-bit integer
// UInt16 - Signed 16-bit integer
// UInt15 - Unsigned 15-bit integer
// SInt32 - Signed 32-bit integer
// UInt31 - Unsigned 31-bit integer
// AInt16 - 16-bit integer of unknown signedness (only used for storage)
// Int16CompFlag - Comparison flags from comparing 16-bit integers
// Int32CompFlag - Comparison flags from comparing 32-bit integers
// FloatCompFlag - Comparison flags from comparing 32-bit floats
//
// The reason for these distinctions are that depending on the instruction set, signed or unsigned versions of certain ops
// (particularly max, min, compares, and right shift) may not be available.  In cases where ops are not available, it's
// necessary to do high bit manipulations to accomplish the operation with 16-bit numbers.  The 15-bit and 31-bit uint types
// can elide the bit flips if unsigned versions are not available.

namespace cvtt
{
#ifdef CVTT_USE_SSE2
    // SSE2 version
    struct ParallelMath
    {
        typedef uint16_t ScalarUInt16;
        typedef int16_t ScalarSInt16;

        template<unsigned int TRoundingMode>
        struct RoundForScope
        {
            unsigned int m_oldCSR;

            RoundForScope()
            {
                m_oldCSR = _mm_getcsr();
                _mm_setcsr((m_oldCSR & ~_MM_ROUND_MASK) | (TRoundingMode));
            }

            ~RoundForScope()
            {
                _mm_setcsr(m_oldCSR);
            }
        };

        struct RoundTowardZeroForScope : RoundForScope<_MM_ROUND_TOWARD_ZERO>
        {
        };

        struct RoundTowardNearestForScope : RoundForScope<_MM_ROUND_NEAREST>
        {
        };

        struct RoundUpForScope : RoundForScope<_MM_ROUND_UP>
        {
        };

        struct RoundDownForScope : RoundForScope<_MM_ROUND_DOWN>
        {
        };

        static const int ParallelSize = 8;

        enum Int16Subtype
        {
            IntSubtype_Signed,
            IntSubtype_UnsignedFull,
            IntSubtype_UnsignedTruncated,
            IntSubtype_Abstract,
        };

        template<int TSubtype>
        struct VInt16
        {
            __m128i m_value;

            inline VInt16 operator+(int16_t other) const
            {
                VInt16 result;
                result.m_value = _mm_add_epi16(m_value, _mm_set1_epi16(static_cast<int16_t>(other)));
                return result;
            }

            inline VInt16 operator+(const VInt16 &other) const
            {
                VInt16 result;
                result.m_value = _mm_add_epi16(m_value, other.m_value);
                return result;
            }

            inline VInt16 operator|(const VInt16 &other) const
            {
                VInt16 result;
                result.m_value = _mm_or_si128(m_value, other.m_value);
                return result;
            }

            inline VInt16 operator&(const VInt16 &other) const
            {
                VInt16 result;
                result.m_value = _mm_and_si128(m_value, other.m_value);
                return result;
            }

            inline VInt16 operator-(const VInt16 &other) const
            {
                VInt16 result;
                result.m_value = _mm_sub_epi16(m_value, other.m_value);
                return result;
            }

            inline VInt16 operator<<(int bits) const
            {
                VInt16 result;
                result.m_value = _mm_slli_epi16(m_value, bits);
                return result;
            }

            inline VInt16 operator^(const VInt16 &other) const
            {
                VInt16 result;
                result.m_value = _mm_xor_si128(m_value, other.m_value);
                return result;
            }
        };

        typedef VInt16<IntSubtype_Signed> SInt16;
        typedef VInt16<IntSubtype_UnsignedFull> UInt16;
        typedef VInt16<IntSubtype_UnsignedTruncated> UInt15;
        typedef VInt16<IntSubtype_Abstract> AInt16;

        template<int TSubtype>
        struct VInt32
        {
            __m128i m_values[2];

            inline VInt32 operator+(const VInt32& other) const
            {
                VInt32 result;
                result.m_values[0] = _mm_add_epi32(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_add_epi32(m_values[1], other.m_values[1]);
                return result;
            }

            inline VInt32 operator-(const VInt32& other) const
            {
                VInt32 result;
                result.m_values[0] = _mm_sub_epi32(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_sub_epi32(m_values[1], other.m_values[1]);
                return result;
            }

            inline VInt32 operator<<(const int other) const
            {
                VInt32 result;
                result.m_values[0] = _mm_slli_epi32(m_values[0], other);
                result.m_values[1] = _mm_slli_epi32(m_values[1], other);
                return result;
            }

            inline VInt32 operator|(const VInt32& other) const
            {
                VInt32 result;
                result.m_values[0] = _mm_or_si128(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_or_si128(m_values[1], other.m_values[1]);
                return result;
            }
        };

        typedef VInt32<IntSubtype_Signed> SInt32;
        typedef VInt32<IntSubtype_UnsignedTruncated> UInt31;
        typedef VInt32<IntSubtype_UnsignedFull> UInt32;
        typedef VInt32<IntSubtype_Abstract> AInt32;

        template<class TTargetType>
        struct LosslessCast
        {
#ifdef CVTT_PERMIT_ALIASING
            template<int TSrcSubtype>
            static const TTargetType& Cast(const VInt32<TSrcSubtype> &src)
            {
                return reinterpret_cast<VInt32<TSubtype>&>(src);
            }

            template<int TSrcSubtype>
            static const TTargetType& Cast(const VInt16<TSrcSubtype> &src)
            {
                return reinterpret_cast<VInt16<TSubtype>&>(src);
            }
#else
            template<int TSrcSubtype>
            static TTargetType Cast(const VInt32<TSrcSubtype> &src)
            {
                TTargetType result;
                result.m_values[0] = src.m_values[0];
                result.m_values[1] = src.m_values[1];
                return result;
            }

            template<int TSrcSubtype>
            static TTargetType Cast(const VInt16<TSrcSubtype> &src)
            {
                TTargetType result;
                result.m_value = src.m_value;
                return result;
            }
#endif
        };

        struct Int64
        {
            __m128i m_values[4];
        };

        struct Float
        {
            __m128 m_values[2];

            inline Float operator+(const Float &other) const
            {
                Float result;
                result.m_values[0] = _mm_add_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_add_ps(m_values[1], other.m_values[1]);
                return result;
            }

            inline Float operator+(float other) const
            {
                Float result;
                result.m_values[0] = _mm_add_ps(m_values[0], _mm_set1_ps(other));
                result.m_values[1] = _mm_add_ps(m_values[1], _mm_set1_ps(other));
                return result;
            }

            inline Float operator-(const Float& other) const
            {
                Float result;
                result.m_values[0] = _mm_sub_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_sub_ps(m_values[1], other.m_values[1]);
                return result;
            }

            inline Float operator-() const
            {
                Float result;
                result.m_values[0] = _mm_sub_ps(_mm_setzero_ps(), m_values[0]);
                result.m_values[1] = _mm_sub_ps(_mm_setzero_ps(), m_values[1]);
                return result;
            }

            inline Float operator*(const Float& other) const
            {
                Float result;
                result.m_values[0] = _mm_mul_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_mul_ps(m_values[1], other.m_values[1]);
                return result;
            }

            inline Float operator*(float other) const
            {
                Float result;
                result.m_values[0] = _mm_mul_ps(m_values[0], _mm_set1_ps(other));
                result.m_values[1] = _mm_mul_ps(m_values[1], _mm_set1_ps(other));
                return result;
            }

            inline Float operator/(const Float &other) const
            {
                Float result;
                result.m_values[0] = _mm_div_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_div_ps(m_values[1], other.m_values[1]);
                return result;
            }

            inline Float operator/(float other) const
            {
                Float result;
                result.m_values[0] = _mm_div_ps(m_values[0], _mm_set1_ps(other));
                result.m_values[1] = _mm_div_ps(m_values[1], _mm_set1_ps(other));
                return result;
            }
        };

        struct Int16CompFlag
        {
            __m128i m_value;

            inline Int16CompFlag operator&(const Int16CompFlag &other) const
            {
                Int16CompFlag result;
                result.m_value = _mm_and_si128(m_value, other.m_value);
                return result;
            }

            inline Int16CompFlag operator|(const Int16CompFlag &other) const
            {
                Int16CompFlag result;
                result.m_value = _mm_or_si128(m_value, other.m_value);
                return result;
            }
        };

        struct Int32CompFlag
        {
            __m128i m_values[2];

            inline Int32CompFlag operator&(const Int32CompFlag &other) const
            {
                Int32CompFlag result;
                result.m_values[0] = _mm_and_si128(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_and_si128(m_values[1], other.m_values[1]);
                return result;
            }

            inline Int32CompFlag operator|(const Int32CompFlag &other) const
            {
                Int32CompFlag result;
                result.m_values[0] = _mm_or_si128(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_or_si128(m_values[1], other.m_values[1]);
                return result;
            }
        };

        struct FloatCompFlag
        {
            __m128 m_values[2];

            inline FloatCompFlag operator&(const FloatCompFlag &other) const
            {
                FloatCompFlag result;
                result.m_values[0] = _mm_and_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_and_ps(m_values[1], other.m_values[1]);
                return result;
            }

            inline FloatCompFlag operator|(const FloatCompFlag &other) const
            {
                FloatCompFlag result;
                result.m_values[0] = _mm_or_ps(m_values[0], other.m_values[0]);
                result.m_values[1] = _mm_or_ps(m_values[1], other.m_values[1]);
                return result;
            }
        };

        template<int TSubtype>
        static VInt16<TSubtype> AbstractAdd(const VInt16<TSubtype> &a, const VInt16<TSubtype> &b)
        {
            VInt16<TSubtype> result;
            result.m_value = _mm_add_epi16(a.m_value, b.m_value);
            return result;
        }

        template<int TSubtype>
        static VInt16<TSubtype> AbstractSubtract(const VInt16<TSubtype> &a, const VInt16<TSubtype> &b)
        {
            VInt16<TSubtype> result;
            result.m_value = _mm_sub_epi16(a.m_value, b.m_value);
            return result;
        }

        static Float Select(const FloatCompFlag &flag, const Float &a, const Float &b)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_or_ps(_mm_and_ps(flag.m_values[i], a.m_values[i]), _mm_andnot_ps(flag.m_values[i], b.m_values[i]));
            return result;
        }

        template<int TSubtype>
        static VInt16<TSubtype> Select(const Int16CompFlag &flag, const VInt16<TSubtype> &a, const VInt16<TSubtype> &b)
        {
            VInt16<TSubtype> result;
            result.m_value = _mm_or_si128(_mm_and_si128(flag.m_value, a.m_value), _mm_andnot_si128(flag.m_value, b.m_value));
            return result;
        }

        template<int TSubtype>
        static VInt16<TSubtype> SelectOrZero(const Int16CompFlag &flag, const VInt16<TSubtype> &a)
        {
            VInt16<TSubtype> result;
            result.m_value = _mm_and_si128(flag.m_value, a.m_value);
            return result;
        }

        template<int TSubtype>
        static void ConditionalSet(VInt16<TSubtype> &dest, const Int16CompFlag &flag, const VInt16<TSubtype> &src)
        {
            dest.m_value = _mm_or_si128(_mm_andnot_si128(flag.m_value, dest.m_value), _mm_and_si128(flag.m_value, src.m_value));
        }

        template<int TSubtype>
        static void ConditionalSet(VInt32<TSubtype> &dest, const Int16CompFlag &flag, const VInt32<TSubtype> &src)
        {
            __m128i lowFlags = _mm_unpacklo_epi16(flag.m_value, flag.m_value);
            __m128i highFlags = _mm_unpackhi_epi16(flag.m_value, flag.m_value);
            dest.m_values[0] = _mm_or_si128(_mm_andnot_si128(lowFlags, dest.m_values[0]), _mm_and_si128(lowFlags, src.m_values[0]));
            dest.m_values[1] = _mm_or_si128(_mm_andnot_si128(highFlags, dest.m_values[1]), _mm_and_si128(highFlags, src.m_values[1]));
        }

        static void ConditionalSet(ParallelMath::Int16CompFlag &dest, const Int16CompFlag &flag, const ParallelMath::Int16CompFlag &src)
        {
            dest.m_value = _mm_or_si128(_mm_andnot_si128(flag.m_value, dest.m_value), _mm_and_si128(flag.m_value, src.m_value));
        }

        static SInt16 ConditionalNegate(const Int16CompFlag &flag, const SInt16 &v)
        {
            SInt16 result;
            result.m_value = _mm_add_epi16(_mm_xor_si128(flag.m_value, v.m_value), _mm_srli_epi16(flag.m_value, 15));
            return result;
        }

        template<int TSubtype>
        static void NotConditionalSet(VInt16<TSubtype> &dest, const Int16CompFlag &flag, const VInt16<TSubtype> &src)
        {
            dest.m_value = _mm_or_si128(_mm_and_si128(flag.m_value, dest.m_value), _mm_andnot_si128(flag.m_value, src.m_value));
        }

        static void ConditionalSet(Float &dest, const FloatCompFlag &flag, const Float &src)
        {
            for (int i = 0; i < 2; i++)
                dest.m_values[i] = _mm_or_ps(_mm_andnot_ps(flag.m_values[i], dest.m_values[i]), _mm_and_ps(flag.m_values[i], src.m_values[i]));
        }

        static void NotConditionalSet(Float &dest, const FloatCompFlag &flag, const Float &src)
        {
            for (int i = 0; i < 2; i++)
                dest.m_values[i] = _mm_or_ps(_mm_and_ps(flag.m_values[i], dest.m_values[i]), _mm_andnot_ps(flag.m_values[i], src.m_values[i]));
        }

        static void MakeSafeDenominator(Float& v)
        {
            ConditionalSet(v, Equal(v, MakeFloatZero()), MakeFloat(1.0f));
        }

        static SInt16 TruncateToPrecisionSigned(const SInt16 &v, int precision)
        {
            int lostBits = 16 - precision;
            if (lostBits == 0)
                return v;

            SInt16 result;
            result.m_value = _mm_srai_epi16(_mm_slli_epi16(v.m_value, lostBits), lostBits);
            return result;
        }

        static UInt16 TruncateToPrecisionUnsigned(const UInt16 &v, int precision)
        {
            int lostBits = 16 - precision;
            if (lostBits == 0)
                return v;

            UInt16 result;
            result.m_value = _mm_srli_epi16(_mm_slli_epi16(v.m_value, lostBits), lostBits);
            return result;
        }

        static UInt16 Min(const UInt16 &a, const UInt16 &b)
        {
            __m128i bitFlip = _mm_set1_epi16(-32768);

            UInt16 result;
            result.m_value = _mm_xor_si128(_mm_min_epi16(_mm_xor_si128(a.m_value, bitFlip), _mm_xor_si128(b.m_value, bitFlip)), bitFlip);
            return result;
        }

        static SInt16 Min(const SInt16 &a, const SInt16 &b)
        {
            SInt16 result;
            result.m_value = _mm_min_epi16(a.m_value, b.m_value);
            return result;
        }

        static UInt15 Min(const UInt15 &a, const UInt15 &b)
        {
            UInt15 result;
            result.m_value = _mm_min_epi16(a.m_value, b.m_value);
            return result;
        }

        static Float Min(const Float &a, const Float &b)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_min_ps(a.m_values[i], b.m_values[i]);
            return result;
        }

        static UInt16 Max(const UInt16 &a, const UInt16 &b)
        {
            __m128i bitFlip = _mm_set1_epi16(-32768);

            UInt16 result;
            result.m_value = _mm_xor_si128(_mm_max_epi16(_mm_xor_si128(a.m_value, bitFlip), _mm_xor_si128(b.m_value, bitFlip)), bitFlip);
            return result;
        }

        static SInt16 Max(const SInt16 &a, const SInt16 &b)
        {
            SInt16 result;
            result.m_value = _mm_max_epi16(a.m_value, b.m_value);
            return result;
        }

        static UInt15 Max(const UInt15 &a, const UInt15 &b)
        {
            UInt15 result;
            result.m_value = _mm_max_epi16(a.m_value, b.m_value);
            return result;
        }

        static Float Max(const Float &a, const Float &b)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_max_ps(a.m_values[i], b.m_values[i]);
            return result;
        }

        static Float Clamp(const Float &v, float min, float max)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_max_ps(_mm_min_ps(v.m_values[i], _mm_set1_ps(max)), _mm_set1_ps(min));
            return result;
        }

        static Float Reciprocal(const Float &v)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_rcp_ps(v.m_values[i]);
            return result;
        }

        static void ConvertLDRInputs(const PixelBlockU8* inputBlocks, int pxOffset, int channel, UInt15 &chOut)
        {
            int16_t values[8];
            for (int i = 0; i < 8; i++)
                values[i] = inputBlocks[i].m_pixels[pxOffset][channel];

            chOut.m_value = _mm_set_epi16(values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0]);
        }

        static void ConvertHDRInputs(const PixelBlockF16* inputBlocks, int pxOffset, int channel, SInt16 &chOut)
        {
            int16_t values[8];
            for (int i = 0; i < 8; i++)
                values[i] = inputBlocks[i].m_pixels[pxOffset][channel];

            chOut.m_value = _mm_set_epi16(values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0]);
        }

        static Float MakeFloat(float v)
        {
            Float f;
            f.m_values[0] = f.m_values[1] = _mm_set1_ps(v);
            return f;
        }

        static Float MakeFloatZero()
        {
            Float f;
            f.m_values[0] = f.m_values[1] = _mm_setzero_ps();
            return f;
        }

        static UInt16 MakeUInt16(uint16_t v)
        {
            UInt16 result;
            result.m_value = _mm_set1_epi16(static_cast<short>(v));
            return result;
        }

        static SInt16 MakeSInt16(int16_t v)
        {
            SInt16 result;
            result.m_value = _mm_set1_epi16(static_cast<short>(v));
            return result;
        }

        static AInt16 MakeAInt16(int16_t v)
        {
            AInt16 result;
            result.m_value = _mm_set1_epi16(static_cast<short>(v));
            return result;
        }

        static UInt15 MakeUInt15(uint16_t v)
        {
            UInt15 result;
            result.m_value = _mm_set1_epi16(static_cast<short>(v));
            return result;
        }

        static SInt32 MakeSInt32(int32_t v)
        {
            SInt32 result;
            result.m_values[0] = _mm_set1_epi32(v);
            result.m_values[1] = _mm_set1_epi32(v);
            return result;
        }

        static UInt31 MakeUInt31(uint32_t v)
        {
            UInt31 result;
            result.m_values[0] = _mm_set1_epi32(v);
            result.m_values[1] = _mm_set1_epi32(v);
            return result;
        }

        static uint16_t Extract(const UInt16 &v, int offset)
        {
            return reinterpret_cast<const uint16_t*>(&v.m_value)[offset];
        }

        static int16_t Extract(const SInt16 &v, int offset)
        {
            return reinterpret_cast<const int16_t*>(&v.m_value)[offset];
        }

        static uint16_t Extract(const UInt15 &v, int offset)
        {
            return reinterpret_cast<const uint16_t*>(&v.m_value)[offset];
        }

        static int16_t Extract(const AInt16 &v, int offset)
        {
            return reinterpret_cast<const int16_t*>(&v.m_value)[offset];
        }

        static int32_t Extract(const SInt32 &v, int offset)
        {
            return reinterpret_cast<const int32_t*>(&v.m_values[offset >> 2])[offset & 3];
        }

        static float Extract(const Float &v, int offset)
        {
            return reinterpret_cast<const float*>(&v.m_values[offset >> 2])[offset & 3];
        }

        static bool Extract(const ParallelMath::Int16CompFlag &v, int offset)
        {
            return reinterpret_cast<const int16_t*>(&v.m_value)[offset] != 0;
        }

        static void PutUInt16(UInt16 &dest, int offset, uint16_t v)
        {
            reinterpret_cast<uint16_t*>(&dest)[offset] = v;
        }

        static void PutUInt15(UInt15 &dest, int offset, uint16_t v)
        {
            reinterpret_cast<uint16_t*>(&dest)[offset] = v;
        }

        static void PutSInt16(SInt16 &dest, int offset, int16_t v)
        {
            reinterpret_cast<int16_t*>(&dest)[offset] = v;
        }

        static float ExtractFloat(const Float& v, int offset)
        {
            return reinterpret_cast<const float*>(&v)[offset];
        }

        static void PutFloat(Float &dest, int offset, float v)
        {
            reinterpret_cast<float*>(&dest)[offset] = v;
        }

        static void PutBoolInt16(Int16CompFlag &dest, int offset, bool v)
        {
            reinterpret_cast<int16_t*>(&dest)[offset] = v ? -1 : 0;
        }

        static Int32CompFlag Less(const UInt31 &a, const UInt31 &b)
        {
            Int32CompFlag result;
            result.m_values[0] = _mm_cmplt_epi32(a.m_values[0], b.m_values[0]);
            result.m_values[1] = _mm_cmplt_epi32(a.m_values[1], b.m_values[1]);
            return result;
        }

        static Int16CompFlag Less(const SInt16 &a, const SInt16 &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_cmplt_epi16(a.m_value, b.m_value);
            return result;
        }

        static Int16CompFlag Less(const UInt15 &a, const UInt15 &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_cmplt_epi16(a.m_value, b.m_value);
            return result;
        }

        static Int16CompFlag LessOrEqual(const UInt15 &a, const UInt15 &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_cmplt_epi16(a.m_value, b.m_value);
            return result;
        }

        static FloatCompFlag Less(const Float &a, const Float &b)
        {
            FloatCompFlag result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_cmplt_ps(a.m_values[i], b.m_values[i]);
            return result;
        }

        static FloatCompFlag LessOrEqual(const Float &a, const Float &b)
        {
            FloatCompFlag result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_cmple_ps(a.m_values[i], b.m_values[i]);
            return result;
        }

        template<int TSubtype>
        static Int16CompFlag Equal(const VInt16<TSubtype> &a, const VInt16<TSubtype> &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_cmpeq_epi16(a.m_value, b.m_value);
            return result;
        }

        static FloatCompFlag Equal(const Float &a, const Float &b)
        {
            FloatCompFlag result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_cmpeq_ps(a.m_values[i], b.m_values[i]);
            return result;
        }

        static Int16CompFlag Equal(const Int16CompFlag &a, const Int16CompFlag &b)
        {
            Int16CompFlag notResult;
            notResult.m_value = _mm_xor_si128(a.m_value, b.m_value);
            return Not(notResult);
        }

        static Float ToFloat(const UInt16 &v)
        {
            Float result;
            result.m_values[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v.m_value, _mm_setzero_si128()));
            result.m_values[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v.m_value, _mm_setzero_si128()));
            return result;
        }

        static UInt31 ToUInt31(const UInt16 &v)
        {
            UInt31 result;
            result.m_values[0] = _mm_unpacklo_epi16(v.m_value, _mm_setzero_si128());
            result.m_values[1] = _mm_unpackhi_epi16(v.m_value, _mm_setzero_si128());
            return result;
        }

        static SInt32 ToInt32(const UInt16 &v)
        {
            SInt32 result;
            result.m_values[0] = _mm_unpacklo_epi16(v.m_value, _mm_setzero_si128());
            result.m_values[1] = _mm_unpackhi_epi16(v.m_value, _mm_setzero_si128());
            return result;
        }

        static SInt32 ToInt32(const UInt15 &v)
        {
            SInt32 result;
            result.m_values[0] = _mm_unpacklo_epi16(v.m_value, _mm_setzero_si128());
            result.m_values[1] = _mm_unpackhi_epi16(v.m_value, _mm_setzero_si128());
            return result;
        }

        static SInt32 ToInt32(const SInt16 &v)
        {
            SInt32 result;
            result.m_values[0] = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), v.m_value), 16);
            result.m_values[1] = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), v.m_value), 16);
            return result;
        }

        static Float ToFloat(const SInt16 &v)
        {
            Float result;
            result.m_values[0] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), v.m_value), 16));
            result.m_values[1] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), v.m_value), 16));
            return result;
        }

        static Float ToFloat(const UInt15 &v)
        {
            Float result;
            result.m_values[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v.m_value, _mm_setzero_si128()));
            result.m_values[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v.m_value, _mm_setzero_si128()));
            return result;
        }

        static Float ToFloat(const UInt31 &v)
        {
            Float result;
            result.m_values[0] = _mm_cvtepi32_ps(v.m_values[0]);
            result.m_values[1] = _mm_cvtepi32_ps(v.m_values[1]);
            return result;
        }

        static Int16CompFlag FloatFlagToInt16(const FloatCompFlag &v)
        {
            __m128i lo = _mm_castps_si128(v.m_values[0]);
            __m128i hi = _mm_castps_si128(v.m_values[1]);

            Int16CompFlag result;
            result.m_value = _mm_packs_epi32(lo, hi);
            return result;
        }

        static FloatCompFlag Int16FlagToFloat(const Int16CompFlag &v)
        {
            __m128i lo = _mm_unpacklo_epi16(v.m_value, v.m_value);
            __m128i hi = _mm_unpackhi_epi16(v.m_value, v.m_value);

            FloatCompFlag result;
            result.m_values[0] = _mm_castsi128_ps(lo);
            result.m_values[1] = _mm_castsi128_ps(hi);
            return result;
        }

        static Int16CompFlag Int32FlagToInt16(const Int32CompFlag &v)
        {
            __m128i lo = v.m_values[0];
            __m128i hi = v.m_values[1];

            Int16CompFlag result;
            result.m_value = _mm_packs_epi32(lo, hi);
            return result;
        }

        static Int16CompFlag MakeBoolInt16(bool b)
        {
            Int16CompFlag result;
            if (b)
                result.m_value = _mm_set1_epi16(-1);
            else
                result.m_value = _mm_setzero_si128();
            return result;
        }

        static FloatCompFlag MakeBoolFloat(bool b)
        {
            FloatCompFlag result;
            if (b)
                result.m_values[0] = result.m_values[1] = _mm_castsi128_ps(_mm_set1_epi32(-1));
            else
                result.m_values[0] = result.m_values[1] = _mm_setzero_ps();
            return result;
        }

        static Int16CompFlag AndNot(const Int16CompFlag &a, const Int16CompFlag &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_andnot_si128(b.m_value, a.m_value);
            return result;
        }

        static Int16CompFlag Not(const Int16CompFlag &b)
        {
            Int16CompFlag result;
            result.m_value = _mm_xor_si128(b.m_value, _mm_set1_epi32(-1));
            return result;
        }

        static Int32CompFlag Not(const Int32CompFlag &b)
        {
            Int32CompFlag result;
            result.m_values[0] = _mm_xor_si128(b.m_values[0], _mm_set1_epi32(-1));
            result.m_values[1] = _mm_xor_si128(b.m_values[1], _mm_set1_epi32(-1));
            return result;
        }

        static UInt16 RoundAndConvertToU16(const Float &v, const void* /*roundingMode*/)
        {
            __m128i lo = _mm_cvtps_epi32(_mm_add_ps(v.m_values[0], _mm_set1_ps(-32768)));
            __m128i hi = _mm_cvtps_epi32(_mm_add_ps(v.m_values[1], _mm_set1_ps(-32768)));

            __m128i packed = _mm_packs_epi32(lo, hi);

            UInt16 result;
            result.m_value = _mm_xor_si128(packed, _mm_set1_epi16(-32768));
            return result;
        }

        static UInt15 RoundAndConvertToU15(const Float &v, const void* /*roundingMode*/)
        {
            __m128i lo = _mm_cvtps_epi32(v.m_values[0]);
            __m128i hi = _mm_cvtps_epi32(v.m_values[1]);

            __m128i packed = _mm_packs_epi32(lo, hi);

            UInt15 result;
            result.m_value = _mm_packs_epi32(lo, hi);
            return result;
        }

        static SInt16 RoundAndConvertToS16(const Float &v, const void* /*roundingMode*/)
        {
            __m128i lo = _mm_cvtps_epi32(v.m_values[0]);
            __m128i hi = _mm_cvtps_epi32(v.m_values[1]);

            __m128i packed = _mm_packs_epi32(lo, hi);

            SInt16 result;
            result.m_value = _mm_packs_epi32(lo, hi);
            return result;
        }

        static Float Sqrt(const Float &f)
        {
            Float result;
            for (int i = 0; i < 2; i++)
                result.m_values[i] = _mm_sqrt_ps(f.m_values[i]);
            return result;
        }

        static UInt16 Abs(const SInt16 &a)
        {
            __m128i signBitsXor = _mm_srai_epi16(a.m_value, 15);
            __m128i signBitsAdd = _mm_srli_epi16(a.m_value, 15);

            UInt16 result;
            result.m_value = _mm_add_epi16(_mm_xor_si128(a.m_value, signBitsXor), signBitsAdd);
            return result;
        }

        static Float Abs(const Float& a)
        {
            __m128 invMask = _mm_set1_ps(-0.0f);

            Float result;
            result.m_values[0] = _mm_andnot_ps(invMask, a.m_values[0]);
            result.m_values[1] = _mm_andnot_ps(invMask, a.m_values[1]);
            return result;
        }

        static UInt16 SqDiffUInt8(const UInt15 &a, const UInt15 &b)
        {
            __m128i diff = _mm_sub_epi16(a.m_value, b.m_value);

            UInt16 result;
            result.m_value = _mm_mullo_epi16(diff, diff);
            return result;
        }

        static Float SqDiffSInt16(const SInt16 &a, const SInt16 &b)
        {
            __m128i diffU = _mm_sub_epi16(_mm_max_epi16(a.m_value, b.m_value), _mm_min_epi16(a.m_value, b.m_value));

            __m128i mulHi = _mm_mulhi_epu16(diffU, diffU);
            __m128i mulLo = _mm_mullo_epi16(diffU, diffU);
            __m128i sqDiffHi = _mm_unpackhi_epi16(mulLo, mulHi);
            __m128i sqDiffLo = _mm_unpacklo_epi16(mulLo, mulHi);

            Float result;
            result.m_values[0] = _mm_cvtepi32_ps(sqDiffLo);
            result.m_values[1] = _mm_cvtepi32_ps(sqDiffHi);

            return result;
        }

        static Float TwosCLHalfToFloat(const SInt16 &v)
        {
            __m128i absV = _mm_add_epi16(_mm_xor_si128(v.m_value, _mm_srai_epi16(v.m_value, 15)), _mm_srli_epi16(v.m_value, 15));

            __m128i signBits = _mm_and_si128(v.m_value, _mm_set1_epi16(-32768));
            __m128i mantissa = _mm_and_si128(v.m_value, _mm_set1_epi16(0x03ff));
            __m128i exponent = _mm_and_si128(v.m_value, _mm_set1_epi16(0x7c00));

            __m128i isDenormal = _mm_cmpeq_epi16(exponent, _mm_setzero_si128());

            // Convert exponent to high-bits 
            exponent = _mm_add_epi16(_mm_srli_epi16(exponent, 3), _mm_set1_epi16(14336));

            __m128i denormalCorrectionHigh = _mm_and_si128(isDenormal, _mm_or_si128(signBits, _mm_set1_epi16(14336)));

            __m128i highBits = _mm_or_si128(signBits, _mm_or_si128(exponent, _mm_srli_epi16(mantissa, 3)));
            __m128i lowBits = _mm_slli_epi16(mantissa, 13);

            __m128i flow = _mm_unpacklo_epi16(lowBits, highBits);
            __m128i fhigh = _mm_unpackhi_epi16(lowBits, highBits);

            __m128i correctionLow = _mm_unpacklo_epi16(_mm_setzero_si128(), denormalCorrectionHigh);
            __m128i correctionHigh = _mm_unpackhi_epi16(_mm_setzero_si128(), denormalCorrectionHigh);

            Float result;
            result.m_values[0] = _mm_sub_ps(_mm_castsi128_ps(flow), _mm_castsi128_ps(correctionLow));
            result.m_values[1] = _mm_sub_ps(_mm_castsi128_ps(fhigh), _mm_castsi128_ps(correctionHigh));

            return result;
        }

        static Float SqDiff2CLFloat(const SInt16 &a, const Float &b)
        {
            Float fa = TwosCLHalfToFloat(a);

            Float diff = fa - b;
            return diff * diff;
        }

        static Float SqDiff2CL(const SInt16 &a, const SInt16 &b)
        {
            Float fa = TwosCLHalfToFloat(a);
            Float fb = TwosCLHalfToFloat(b);

            Float diff = fa - fb;
            return diff * diff;
        }

        static Float SqDiff2CLFloat(const SInt16 &a, float aWeight, const Float &b)
        {
            Float fa = TwosCLHalfToFloat(a) * aWeight;

            Float diff = fa - b;
            return diff * diff;
        }

        static UInt16 RightShift(const UInt16 &v, int bits)
        {
            UInt16 result;
            result.m_value = _mm_srli_epi16(v.m_value, bits);
            return result;
        }

        static UInt31 RightShift(const UInt31 &v, int bits)
        {
            UInt31 result;
            result.m_values[0] = _mm_srli_epi32(v.m_values[0], bits);
            result.m_values[1] = _mm_srli_epi32(v.m_values[1], bits);
            return result;
        }

        static SInt16 RightShift(const SInt16 &v, int bits)
        {
            SInt16 result;
            result.m_value = _mm_srai_epi16(v.m_value, bits);
            return result;
        }

        static UInt15 RightShift(const UInt15 &v, int bits)
        {
            UInt15 result;
            result.m_value = _mm_srli_epi16(v.m_value, bits);
            return result;
        }

        static SInt32 RightShift(const SInt32 &v, int bits)
        {
            SInt32 result;
            result.m_values[0] = _mm_srai_epi32(v.m_values[0], bits);
            result.m_values[1] = _mm_srai_epi32(v.m_values[1], bits);
            return result;
        }

        static SInt16 ToSInt16(const SInt32 &v)
        {
            SInt16 result;
            result.m_value = _mm_packs_epi32(v.m_values[0], v.m_values[1]);
            return result;
        }

        static SInt16 ToSInt16(const UInt16 &v)
        {
            SInt16 result;
            result.m_value = v.m_value;
            return result;
        }

        static SInt16 ToSInt16(const UInt15 &v)
        {
            SInt16 result;
            result.m_value = v.m_value;
            return result;
        }

        static UInt16 ToUInt16(const UInt32 &v)
        {
            __m128i low = _mm_srai_epi32(_mm_slli_epi32(v.m_values[0], 16), 16);
            __m128i high = _mm_srai_epi32(_mm_slli_epi32(v.m_values[1], 16), 16);

            UInt16 result;
            result.m_value = _mm_packs_epi32(low, high);
            return result;
        }

        static UInt16 ToUInt16(const UInt31 &v)
        {
            __m128i low = _mm_srai_epi32(_mm_slli_epi32(v.m_values[0], 16), 16);
            __m128i high = _mm_srai_epi32(_mm_slli_epi32(v.m_values[1], 16), 16);

            UInt16 result;
            result.m_value = _mm_packs_epi32(low, high);
            return result;
        }

        static UInt15 ToUInt15(const UInt31 &v)
        {
            UInt15 result;
            result.m_value = _mm_packs_epi32(v.m_values[0], v.m_values[1]);
            return result;
        }

        static UInt15 ToUInt15(const SInt16 &v)
        {
            UInt15 result;
            result.m_value = v.m_value;
            return result;
        }

        static UInt15 ToUInt15(const UInt16 &v)
        {
            UInt15 result;
            result.m_value = v.m_value;
            return result;
        }

        static SInt32 XMultiply(const SInt16 &a, const SInt16 &b)
        {
            __m128i high = _mm_mulhi_epi16(a.m_value, b.m_value);
            __m128i low = _mm_mullo_epi16(a.m_value, b.m_value);

            SInt32 result;
            result.m_values[0] = _mm_unpacklo_epi16(low, high);
            result.m_values[1] = _mm_unpackhi_epi16(low, high);
            return result;
        }

        static SInt32 XMultiply(const SInt16 &a, const UInt15 &b)
        {
            __m128i high = _mm_mulhi_epi16(a.m_value, b.m_value);
            __m128i low = _mm_mullo_epi16(a.m_value, b.m_value);

            SInt32 result;
            result.m_values[0] = _mm_unpacklo_epi16(low, high);
            result.m_values[1] = _mm_unpackhi_epi16(low, high);
            return result;
        }

        static SInt32 XMultiply(const UInt15 &a, const SInt16 &b)
        {
            return XMultiply(b, a);
        }

        static UInt32 XMultiply(const UInt16 &a, const UInt16 &b)
        {
            __m128i high = _mm_mulhi_epu16(a.m_value, b.m_value);
            __m128i low = _mm_mullo_epi16(a.m_value, b.m_value);

            UInt32 result;
            result.m_values[0] = _mm_unpacklo_epi16(low, high);
            result.m_values[1] = _mm_unpackhi_epi16(low, high);
            return result;
        }

        static UInt16 CompactMultiply(const UInt16 &a, const UInt15 &b)
        {
            UInt16 result;
            result.m_value = _mm_mullo_epi16(a.m_value, b.m_value);
            return result;
        }

        static UInt16 CompactMultiply(const UInt15 &a, const UInt15 &b)
        {
            UInt16 result;
            result.m_value = _mm_mullo_epi16(a.m_value, b.m_value);
            return result;
        }

        static SInt16 CompactMultiply(const SInt16 &a, const UInt15 &b)
        {
            SInt16 result;
            result.m_value = _mm_mullo_epi16(a.m_value, b.m_value);
            return result;
        }

        static SInt16 CompactMultiply(const SInt16 &a, const SInt16 &b)
        {
            SInt16 result;
            result.m_value = _mm_mullo_epi16(a.m_value, b.m_value);
            return result;
        }

        static UInt31 XMultiply(const UInt15 &a, const UInt15 &b)
        {
            __m128i high = _mm_mulhi_epu16(a.m_value, b.m_value);
            __m128i low = _mm_mullo_epi16(a.m_value, b.m_value);

            UInt31 result;
            result.m_values[0] = _mm_unpacklo_epi16(low, high);
            result.m_values[1] = _mm_unpackhi_epi16(low, high);
            return result;
        }

        static UInt31 XMultiply(const UInt16 &a, const UInt15 &b)
        {
            __m128i high = _mm_mulhi_epu16(a.m_value, b.m_value);
            __m128i low = _mm_mullo_epi16(a.m_value, b.m_value);

            UInt31 result;
            result.m_values[0] = _mm_unpacklo_epi16(low, high);
            result.m_values[1] = _mm_unpackhi_epi16(low, high);
            return result;
        }

        static UInt31 XMultiply(const UInt15 &a, const UInt16 &b)
        {
            return XMultiply(b, a);
        }

        static bool AnySet(const Int16CompFlag &v)
        {
            return _mm_movemask_epi8(v.m_value) != 0;
        }

        static bool AllSet(const Int16CompFlag &v)
        {
            return _mm_movemask_epi8(v.m_value) == 0xffff;
        }

        static bool AnySet(const FloatCompFlag &v)
        {
            return _mm_movemask_ps(v.m_values[0]) != 0 || _mm_movemask_ps(v.m_values[1]) != 0;
        }

        static bool AllSet(const FloatCompFlag &v)
        {
            return _mm_movemask_ps(v.m_values[0]) == 0xf && _mm_movemask_ps(v.m_values[1]) == 0xf;
        }
    };

#else
    // Scalar version
    struct ParallelMath
    {
        struct RoundTowardZeroForScope
        {
        };

        struct RoundTowardNearestForScope
        {
        };

        struct RoundUpForScope
        {
        };

        struct RoundDownForScope
        {
        };

        static const int ParallelSize = 1;

        enum Int16Subtype
        {
            IntSubtype_Signed,
            IntSubtype_UnsignedFull,
            IntSubtype_UnsignedTruncated,
            IntSubtype_Abstract,
        };

        typedef int32_t SInt16;
        typedef int32_t UInt15;
        typedef int32_t UInt16;
        typedef int32_t AInt16;

        typedef int32_t SInt32;
        typedef int32_t UInt31;
        typedef int32_t UInt32;
        typedef int32_t AInt32;

        typedef int32_t ScalarUInt16;
        typedef int32_t ScalarSInt16;

        typedef float Float;

        template<class TTargetType>
        struct LosslessCast
        {
            static const int32_t& Cast(const int32_t &src)
            {
                return src;
            }
        };

        typedef bool Int16CompFlag;
        typedef bool FloatCompFlag;

        static int32_t AbstractAdd(const int32_t &a, const int32_t &b)
        {
            return a + b;
        }

        static int32_t AbstractSubtract(const int32_t &a, const int32_t &b)
        {
            return a - b;
        }

        static float Select(bool flag, float a, float b)
        {
            return flag ? a : b;
        }

        static int32_t Select(bool flag, int32_t a, int32_t b)
        {
            return flag ? a : b;
        }

        static int32_t SelectOrZero(bool flag, int32_t a)
        {
            return flag ? a : 0;
        }

        static void ConditionalSet(int32_t& dest, bool flag, int32_t src)
        {
            if (flag)
                dest = src;
        }

        static void ConditionalSet(bool& dest, bool flag, bool src)
        {
            if (flag)
                dest = src;
        }

        static int32_t ConditionalNegate(bool flag, int32_t v)
        {
            return (flag) ? -v : v;
        }

        static void NotConditionalSet(int32_t& dest, bool flag, int32_t src)
        {
            if (!flag)
                dest = src;
        }

        static void ConditionalSet(float& dest, bool flag, float src)
        {
            if (flag)
                dest = src;
        }

        static void NotConditionalSet(float& dest, bool flag, float src)
        {
            if (!flag)
                dest = src;
        }

        static void MakeSafeDenominator(float& v)
        {
            if (v == 0.0f)
                v = 1.0f;
        }

        static int32_t SignedRightShift(int32_t v, int bits)
        {
            return v >> bits;
        }

        static int32_t TruncateToPrecisionSigned(int32_t v, int precision)
        {
            v = (v << (32 - precision)) & 0xffffffff;
            return SignedRightShift(v, 32 - precision);
        }

        static int32_t TruncateToPrecisionUnsigned(int32_t v, int precision)
        {
            return v & ((1 << precision) - 1);
        }

        static int32_t Min(int32_t a, int32_t b)
        {
            if (a < b)
                return a;
            return b;
        }

        static float Min(float a, float b)
        {
            if (a < b)
                return a;
            return b;
        }

        static int32_t Max(int32_t a, int32_t b)
        {
            if (a > b)
                return a;
            return b;
        }

        static float Max(float a, float b)
        {
            if (a > b)
                return a;
            return b;
        }

        static float Abs(float a)
        {
            return fabsf(a);
        }

        static int32_t Abs(int32_t a)
        {
            if (a < 0)
                return -a;
            return a;
        }

        static float Clamp(float v, float min, float max)
        {
            if (v < min)
                return min;
            if (v > max)
                return max;
            return v;
        }

        static float Reciprocal(float v)
        {
            return 1.0f / v;
        }

        static void ConvertLDRInputs(const PixelBlockU8* inputBlocks, int pxOffset, int channel, int32_t& chOut)
        {
            chOut = inputBlocks[0].m_pixels[pxOffset][channel];
        }

        static void ConvertHDRInputs(const PixelBlockF16* inputBlocks, int pxOffset, int channel, int32_t& chOut)
        {
            chOut = inputBlocks[0].m_pixels[pxOffset][channel];
        }

        static float MakeFloat(float v)
        {
            return v;
        }

        static float MakeFloatZero()
        {
            return 0.0f;
        }

        static int32_t MakeUInt16(uint16_t v)
        {
            return v;
        }

        static int32_t MakeSInt16(int16_t v)
        {
            return v;
        }

        static int32_t MakeAInt16(int16_t v)
        {
            return v;
        }

        static int32_t MakeUInt15(uint16_t v)
        {
            return v;
        }

        static int32_t MakeSInt32(int32_t v)
        {
            return v;
        }

        static int32_t MakeUInt31(int32_t v)
        {
            return v;
        }

        static int32_t Extract(int32_t v, int offset)
        {
            UNREFERENCED_PARAMETER(offset);
            return v;
        }

        static bool Extract(bool v, int offset)
        {
            UNREFERENCED_PARAMETER(offset);
            return v;
        }

        static float Extract(float v, int offset)
        {
            UNREFERENCED_PARAMETER(offset);
            return v;
        }

        static void PutUInt16(int32_t &dest, int offset, ParallelMath::ScalarUInt16 v)
        {
            UNREFERENCED_PARAMETER(offset);
            dest = v;
        }

        static void PutUInt15(int32_t &dest, int offset, ParallelMath::ScalarUInt16 v)
        {
            UNREFERENCED_PARAMETER(offset);
            dest = v;
        }

        static void PutSInt16(int32_t &dest, int offset, ParallelMath::ScalarSInt16 v)
        {
            UNREFERENCED_PARAMETER(offset);
            dest = v;
        }

        static float ExtractFloat(float v, int offset)
        {
            UNREFERENCED_PARAMETER(offset);
            return v;
        }

        static void PutFloat(float &dest, int offset, float v)
        {
            UNREFERENCED_PARAMETER(offset);
            dest = v;
        }

        static void PutBoolInt16(bool &dest, int offset, bool v)
        {
            UNREFERENCED_PARAMETER(offset);
            dest = v;
        }

        static bool Less(int32_t a, int32_t b)
        {
            return a < b;
        }

        static bool Less(float a, float b)
        {
            return a < b;
        }

        static bool LessOrEqual(int32_t a, int32_t b)
        {
            return a < b;
        }

        static bool LessOrEqual(float a, float b)
        {
            return a < b;
        }

        static bool Equal(int32_t a, int32_t b)
        {
            return a == b;
        }

        static bool Equal(float a, float b)
        {
            return a == b;
        }

        static float ToFloat(int32_t v)
        {
            return static_cast<float>(v);
        }

        static int32_t ToUInt31(int32_t v)
        {
            return v;
        }

        static int32_t ToInt32(int32_t v)
        {
            return v;
        }

        static bool FloatFlagToInt16(bool v)
        {
            return v;
        }

        static bool Int32FlagToInt16(bool v)
        {
            return v;
        }

        static bool Int16FlagToFloat(bool v)
        {
            return v;
        }

        static bool MakeBoolInt16(bool b)
        {
            return b;
        }

        static bool MakeBoolFloat(bool b)
        {
            return b;
        }

        static bool AndNot(bool a, bool b)
        {
            return a && !b;
        }

        static bool Not(bool b)
        {
            return !b;
        }

        static int32_t RoundAndConvertToInt(float v, const ParallelMath::RoundTowardZeroForScope *rtz)
        {
            UNREFERENCED_PARAMETER(rtz);
            return static_cast<int>(v);
        }

        static int32_t RoundAndConvertToInt(float v, const ParallelMath::RoundUpForScope *ru)
        {
            UNREFERENCED_PARAMETER(ru);
            return static_cast<int>(ceilf(v));
        }

        static int32_t RoundAndConvertToInt(float v, const ParallelMath::RoundDownForScope *rd)
        {
            UNREFERENCED_PARAMETER(rd);
            return static_cast<int>(floorf(v));
        }

        static int32_t RoundAndConvertToInt(float v, const ParallelMath::RoundTowardNearestForScope *rtn)
        {
            UNREFERENCED_PARAMETER(rtn);
            return static_cast<int>(floorf(v + 0.5f));
        }

        template<class TRoundMode>
        static int32_t RoundAndConvertToU16(float v, const TRoundMode *roundingMode)
        {
            return RoundAndConvertToInt(v, roundingMode);
        }

        template<class TRoundMode>
        static int32_t RoundAndConvertToU15(float v, const TRoundMode *roundingMode)
        {
            return RoundAndConvertToInt(v, roundingMode);
        }

        template<class TRoundMode>
        static int32_t RoundAndConvertToS16(float v, const TRoundMode *roundingMode)
        {
            return RoundAndConvertToInt(v, roundingMode);
        }

        static float Sqrt(float f)
        {
            return sqrtf(f);
        }

        static int32_t SqDiffUInt8(int32_t a, int32_t b)
        {
            int32_t delta = a - b;
            return delta * delta;
        }

        static int32_t SqDiffInt16(int32_t a, int32_t b)
        {
            int32_t delta = a - b;
            return delta * delta;
        }

        static int32_t SqDiffSInt16(int32_t a, int32_t b)
        {
            int32_t delta = a - b;
            return delta * delta;
        }

        static float TwosCLHalfToFloat(int32_t v)
        {
            int32_t absV = (v < 0) ? -v : v;

            int32_t signBits = (absV & -32768);
            int32_t mantissa = (absV & 0x03ff);
            int32_t exponent = (absV & 0x7c00);

            bool isDenormal = (exponent == 0);

            // Convert exponent to high-bits
            exponent = (exponent >> 3) + 14336;

            int32_t denormalCorrection = (isDenormal ? (signBits | 14336) : 0) << 16;

            int32_t fBits = ((exponent | signBits) << 16) | (mantissa << 13);

            float f, correction;
            memcpy(&f, &fBits, 4);
            memcpy(&correction, &denormalCorrection, 4);

            return f - correction;
        }

        static Float SqDiff2CLFloat(const SInt16 &a, const Float &b)
        {
            Float fa = TwosCLHalfToFloat(a);

            Float diff = fa - b;
            return diff * diff;
        }

        static Float SqDiff2CL(const SInt16 &a, const SInt16 &b)
        {
            Float fa = TwosCLHalfToFloat(a);
            Float fb = TwosCLHalfToFloat(b);

            Float diff = fa - fb;
            return diff * diff;
        }

        static Float SqDiff2CLFloat(const SInt16 &a, float aWeight, const Float &b)
        {
            Float fa = TwosCLHalfToFloat(a) * aWeight;

            Float diff = fa - b;
            return diff * diff;
        }

        static int32_t RightShift(int32_t v, int bits)
        {
            return SignedRightShift(v, bits);
        }

        static int32_t ToSInt16(int32_t v)
        {
            return v;
        }

        static int32_t ToUInt16(int32_t v)
        {
            return v;
        }

        static int32_t ToUInt15(int32_t v)
        {
            return v;
        }

        static int32_t XMultiply(int32_t a, int32_t b)
        {
            return a * b;
        }

        static int32_t CompactMultiply(int32_t a, int32_t b)
        {
            return a * b;
        }

        static bool AnySet(bool v)
        {
            return v;
        }

        static bool AllSet(bool v)
        {
            return v;
        }
    };

#endif
}

#endif
