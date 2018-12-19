/*
Convection Texture Tools
Copyright (c) 2018 Eric Lasota

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

-------------------------------------------------------------------------------------

Portions based on DirectX Texture Library (DirectXTex)

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

http://go.microsoft.com/fwlink/?LinkId=248926
*/
#include "ConvectionKernels.h"
#include "ConvectionKernels_BC7_SingleColor.h"

#if (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64) || defined(__SSE2__)
#define CVTT_USE_SSE2
#endif

#ifdef CVTT_USE_SSE2
#include <emmintrin.h>
#endif

#include <float.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <math.h>

#define UNREFERENCED_PARAMETER(n) ((void)n)

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

        struct FloatCompFlag
        {
            __m128 m_values[2];
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

    namespace Internal
    {
        namespace BC7Data
        {
            enum AlphaMode
            {
                AlphaMode_Combined,
                AlphaMode_Separate,
                AlphaMode_None,
            };

            enum PBitMode
            {
                PBitMode_PerEndpoint,
                PBitMode_PerSubset,
                PBitMode_None
            };

            struct BC7ModeInfo
            {
                PBitMode m_pBitMode;
                AlphaMode m_alphaMode;
                int m_rgbBits;
                int m_alphaBits;
                int m_partitionBits;
                int m_numSubsets;
                int m_indexBits;
                int m_alphaIndexBits;
                bool m_hasIndexSelector;
            };

            BC7ModeInfo g_modes[] =
            {
                { PBitMode_PerEndpoint, AlphaMode_None, 4, 0, 4, 3, 3, 0, false },     // 0
                { PBitMode_PerSubset, AlphaMode_None, 6, 0, 6, 2, 3, 0, false },       // 1
                { PBitMode_None, AlphaMode_None, 5, 0, 6, 3, 2, 0, false },            // 2
                { PBitMode_PerEndpoint, AlphaMode_None, 7, 0, 6, 2, 2, 0, false },     // 3 (Mode reference has an error, P-bit is really per-endpoint)

                { PBitMode_None, AlphaMode_Separate, 5, 6, 0, 1, 2, 3, true },         // 4
                { PBitMode_None, AlphaMode_Separate, 7, 8, 0, 1, 2, 2, false },        // 5
                { PBitMode_PerEndpoint, AlphaMode_Combined, 7, 7, 0, 1, 4, 0, false }, // 6
                { PBitMode_PerEndpoint, AlphaMode_Combined, 5, 5, 6, 2, 2, 0, false }  // 7
            };

			const int g_weight2[] = { 0, 21, 43, 64 };
			const int g_weight3[] = { 0, 9, 18, 27, 37, 46, 55, 64 };
			const int g_weight4[] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

			const int *g_weightTables[] =
			{
				NULL,
				NULL,
				g_weight2,
				g_weight3,
				g_weight4
			};

            struct BC6HModeInfo
            {
                uint16_t m_modeID;
                bool m_partitioned;
                bool m_transformed;
                int m_aPrec;
                int m_bPrec[3];
            };

            // [partitioned][precision]
            bool g_hdrModesExistForPrecision[2][17] =
            {
                //0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16
                { false, false, false, false, false, false, false, false, false, false, true,  true,  true,  false, false, false, true },
                { false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  false, false, false, false, false },
            };

            BC6HModeInfo g_hdrModes[] =
            {
                { 0x00, true,  true,  10,{ 5, 5, 5 } },
                { 0x01, true,  true,  7,{ 6, 6, 6 } },
                { 0x02, true,  true,  11,{ 5, 4, 4 } },
                { 0x06, true,  true,  11,{ 4, 5, 4 } },
                { 0x0a, true,  true,  11,{ 4, 4, 5 } },
                { 0x0e, true,  true,  9,{ 5, 5, 5 } },
                { 0x12, true,  true,  8,{ 6, 5, 5 } },
                { 0x16, true,  true,  8,{ 5, 6, 5 } },
                { 0x1a, true,  true,  8,{ 5, 5, 6 } },
                { 0x1e, true,  false, 6,{ 6, 6, 6 } },
                { 0x03, false, false, 10,{ 10, 10, 10 } },
                { 0x07, false, true,  11,{ 9, 9, 9 } },
                { 0x0b, false, true,  12,{ 8, 8, 8 } },
                { 0x0f, false, true,  16,{ 4, 4, 4 } },
            };

            const int g_maxHDRPrecision = 16;

            static const size_t g_numHDRModes = sizeof(g_hdrModes) / sizeof(g_hdrModes[0]);

            static uint16_t g_partitionMap[64] =
            {
                0xCCCC, 0x8888, 0xEEEE, 0xECC8,
                0xC880, 0xFEEC, 0xFEC8, 0xEC80,
                0xC800, 0xFFEC, 0xFE80, 0xE800,
                0xFFE8, 0xFF00, 0xFFF0, 0xF000,
                0xF710, 0x008E, 0x7100, 0x08CE,
                0x008C, 0x7310, 0x3100, 0x8CCE,
                0x088C, 0x3110, 0x6666, 0x366C,
                0x17E8, 0x0FF0, 0x718E, 0x399C,
                0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
                0x3c3c, 0x55aa, 0x9696, 0xa55a,
                0x73ce, 0x13c8, 0x324c, 0x3bdc,
                0x6996, 0xc33c, 0x9966, 0x660,
                0x272, 0x4e4, 0x4e40, 0x2720,
                0xc936, 0x936c, 0x39c6, 0x639c,
                0x9336, 0x9cc6, 0x817e, 0xe718,
                0xccf0, 0xfcc, 0x7744, 0xee22,
            };

            static uint32_t g_partitionMap2[64] =
            {
                0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8,
                0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
                0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
                0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
                0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0,
                0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
                0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400,
                0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
                0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
                0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
                0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0,
                0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
                0xaa444444, 0x54a854a8, 0x95809580, 0x96969600,
                0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
                0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
                0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
            };

            static int g_fixupIndexes2[64] =
            {
                15,15,15,15,
                15,15,15,15,
                15,15,15,15,
                15,15,15,15,
                15, 2, 8, 2,
                2, 8, 8,15,
                2, 8, 2, 2,
                8, 8, 2, 2,

                15,15, 6, 8,
                2, 8,15,15,
                2, 8, 2, 2,
                2,15,15, 6,
                6, 2, 6, 8,
                15,15, 2, 2,
                15,15,15,15,
                15, 2, 2,15,
            };

            static int g_fixupIndexes3[64][2] =
            {
                { 3,15 },{ 3, 8 },{ 15, 8 },{ 15, 3 },
                { 8,15 },{ 3,15 },{ 15, 3 },{ 15, 8 },
                { 8,15 },{ 8,15 },{ 6,15 },{ 6,15 },
                { 6,15 },{ 5,15 },{ 3,15 },{ 3, 8 },
                { 3,15 },{ 3, 8 },{ 8,15 },{ 15, 3 },
                { 3,15 },{ 3, 8 },{ 6,15 },{ 10, 8 },
                { 5, 3 },{ 8,15 },{ 8, 6 },{ 6,10 },
                { 8,15 },{ 5,15 },{ 15,10 },{ 15, 8 },

                { 8,15 },{ 15, 3 },{ 3,15 },{ 5,10 },
                { 6,10 },{ 10, 8 },{ 8, 9 },{ 15,10 },
                { 15, 6 },{ 3,15 },{ 15, 8 },{ 5,15 },
                { 15, 3 },{ 15, 6 },{ 15, 6 },{ 15, 8 },
                { 3,15 },{ 15, 3 },{ 5,15 },{ 5,15 },
                { 5,15 },{ 8,15 },{ 5,15 },{ 10,15 },
                { 5,15 },{ 10,15 },{ 8,15 },{ 13,15 },
                { 15, 3 },{ 12,15 },{ 3,15 },{ 3, 8 },
            };

            static const unsigned char g_fragments[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 0, 16
                0, 1, 2, 3,  // 16, 4
                0, 1, 4,  // 20, 3
                0, 1, 2, 4,  // 23, 4
                2, 3, 7,  // 27, 3
                1, 2, 3, 7,  // 30, 4
                0, 1, 2, 3, 4, 5, 6, 7,  // 34, 8
                0, 1, 4, 8,  // 42, 4
                0, 1, 2, 4, 5, 8,  // 46, 6
                0, 1, 2, 3, 4, 5, 6, 8,  // 52, 8
                1, 4, 5, 6, 9,  // 60, 5
                2, 5, 6, 7, 10,  // 65, 5
                5, 6, 9, 10,  // 70, 4
                2, 3, 7, 11,  // 74, 4
                1, 2, 3, 6, 7, 11,  // 78, 6
                0, 1, 2, 3, 5, 6, 7, 11,  // 84, 8
                0, 1, 2, 3, 8, 9, 10, 11,  // 92, 8
                2, 3, 6, 7, 8, 9, 10, 11,  // 100, 8
                4, 5, 6, 7, 8, 9, 10, 11,  // 108, 8
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  // 116, 12
                0, 4, 8, 12,  // 128, 4
                0, 2, 3, 4, 6, 7, 8, 12,  // 132, 8
                0, 1, 2, 4, 5, 8, 9, 12,  // 140, 8
                0, 1, 2, 3, 4, 5, 6, 8, 9, 12,  // 148, 10
                3, 6, 7, 8, 9, 12,  // 158, 6
                3, 5, 6, 7, 8, 9, 10, 12,  // 164, 8
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,  // 172, 12
                0, 1, 2, 5, 6, 7, 11, 12,  // 184, 8
                5, 8, 9, 10, 13,  // 192, 5
                8, 12, 13,  // 197, 3
                4, 8, 12, 13,  // 200, 4
                2, 3, 6, 9, 12, 13,  // 204, 6
                0, 1, 2, 3, 8, 9, 12, 13,  // 210, 8
                0, 1, 4, 5, 8, 9, 12, 13,  // 218, 8
                2, 3, 6, 7, 8, 9, 12, 13,  // 226, 8
                2, 3, 5, 6, 9, 10, 12, 13,  // 234, 8
                0, 3, 6, 7, 9, 10, 12, 13,  // 242, 8
                0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13,  // 250, 12
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,  // 262, 13
                2, 3, 4, 7, 8, 11, 12, 13,  // 275, 8
                1, 2, 6, 7, 8, 11, 12, 13,  // 283, 8
                2, 3, 4, 6, 7, 8, 9, 11, 12, 13,  // 291, 10
                2, 3, 4, 5, 10, 11, 12, 13,  // 301, 8
                0, 1, 6, 7, 10, 11, 12, 13,  // 309, 8
                6, 9, 10, 11, 14,  // 317, 5
                0, 2, 4, 6, 8, 10, 12, 14,  // 322, 8
                1, 3, 5, 7, 8, 10, 12, 14,  // 330, 8
                1, 3, 4, 6, 9, 11, 12, 14,  // 338, 8
                0, 2, 5, 7, 9, 11, 12, 14,  // 346, 8
                0, 3, 4, 5, 8, 9, 13, 14,  // 354, 8
                2, 3, 4, 7, 8, 9, 13, 14,  // 362, 8
                1, 2, 5, 6, 9, 10, 13, 14,  // 370, 8
                0, 3, 4, 7, 9, 10, 13, 14,  // 378, 8
                0, 3, 5, 6, 8, 11, 13, 14,  // 386, 8
                1, 2, 4, 7, 8, 11, 13, 14,  // 394, 8
                0, 1, 4, 7, 10, 11, 13, 14,  // 402, 8
                0, 3, 6, 7, 10, 11, 13, 14,  // 410, 8
                8, 12, 13, 14,  // 418, 4
                1, 2, 3, 7, 8, 12, 13, 14,  // 422, 8
                4, 8, 9, 12, 13, 14,  // 430, 6
                0, 4, 5, 8, 9, 12, 13, 14,  // 436, 8
                1, 2, 3, 6, 7, 8, 9, 12, 13, 14,  // 444, 10
                2, 6, 8, 9, 10, 12, 13, 14,  // 454, 8
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,  // 462, 12
                0, 7, 9, 10, 11, 12, 13, 14,  // 474, 8
                1, 2, 3, 4, 5, 6, 8, 15,  // 482, 8
                3, 7, 11, 15,  // 490, 4
                0, 1, 3, 4, 5, 7, 11, 15,  // 494, 8
                0, 4, 5, 10, 11, 15,  // 502, 6
                1, 2, 3, 6, 7, 10, 11, 15,  // 508, 8
                0, 1, 2, 3, 5, 6, 7, 10, 11, 15,  // 516, 10
                0, 4, 5, 6, 9, 10, 11, 15,  // 526, 8
                0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15,  // 534, 12
                1, 2, 4, 5, 8, 9, 12, 15,  // 546, 8
                2, 3, 5, 6, 8, 9, 12, 15,  // 554, 8
                0, 3, 5, 6, 9, 10, 12, 15,  // 562, 8
                1, 2, 4, 7, 9, 10, 12, 15,  // 570, 8
                1, 2, 5, 6, 8, 11, 12, 15,  // 578, 8
                0, 3, 4, 7, 8, 11, 12, 15,  // 586, 8
                0, 1, 5, 6, 10, 11, 12, 15,  // 594, 8
                1, 2, 6, 7, 10, 11, 12, 15,  // 602, 8
                1, 3, 4, 6, 8, 10, 13, 15,  // 610, 8
                0, 2, 5, 7, 8, 10, 13, 15,  // 618, 8
                0, 2, 4, 6, 9, 11, 13, 15,  // 626, 8
                1, 3, 5, 7, 9, 11, 13, 15,  // 634, 8
                0, 1, 2, 3, 4, 5, 7, 8, 12, 13, 15,  // 642, 11
                2, 3, 4, 5, 8, 9, 14, 15,  // 653, 8
                0, 1, 6, 7, 8, 9, 14, 15,  // 661, 8
                0, 1, 5, 10, 14, 15,  // 669, 6
                0, 3, 4, 5, 9, 10, 14, 15,  // 675, 8
                0, 1, 5, 6, 9, 10, 14, 15,  // 683, 8
                11, 14, 15,  // 691, 3
                7, 11, 14, 15,  // 694, 4
                1, 2, 4, 5, 8, 11, 14, 15,  // 698, 8
                0, 1, 4, 7, 8, 11, 14, 15,  // 706, 8
                0, 1, 4, 5, 10, 11, 14, 15,  // 714, 8
                2, 3, 6, 7, 10, 11, 14, 15,  // 722, 8
                4, 5, 6, 7, 10, 11, 14, 15,  // 730, 8
                0, 1, 4, 5, 7, 8, 10, 11, 14, 15,  // 738, 10
                0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15,  // 748, 12
                0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15,  // 760, 13
                0, 1, 2, 3, 4, 6, 7, 11, 12, 14, 15,  // 773, 11
                3, 4, 8, 9, 10, 13, 14, 15,  // 784, 8
                11, 13, 14, 15,  // 792, 4
                0, 1, 2, 4, 11, 13, 14, 15,  // 796, 8
                0, 1, 2, 4, 5, 10, 11, 13, 14, 15,  // 804, 10
                7, 10, 11, 13, 14, 15,  // 814, 6
                3, 6, 7, 10, 11, 13, 14, 15,  // 820, 8
                1, 5, 9, 10, 11, 13, 14, 15,  // 828, 8
                1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,  // 836, 12
                12, 13, 14, 15,  // 848, 4
                0, 1, 2, 3, 12, 13, 14, 15,  // 852, 8
                0, 1, 4, 5, 12, 13, 14, 15,  // 860, 8
                4, 5, 6, 7, 12, 13, 14, 15,  // 868, 8
                4, 8, 9, 10, 12, 13, 14, 15,  // 876, 8
                0, 4, 5, 8, 9, 10, 12, 13, 14, 15,  // 884, 10
                0, 1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15,  // 894, 12
                0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15,  // 906, 12
                0, 1, 3, 4, 8, 9, 11, 12, 13, 14, 15,  // 918, 11
                0, 2, 3, 7, 8, 10, 11, 12, 13, 14, 15,  // 929, 11
                7, 9, 10, 11, 12, 13, 14, 15,  // 940, 8
                3, 6, 7, 9, 10, 11, 12, 13, 14, 15,  // 948, 10
                2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,  // 958, 12
                8, 9, 10, 11, 12, 13, 14, 15,  // 970, 8
                0, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,  // 978, 12
                0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,  // 990, 13
                3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1003, 12
                2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1015, 13
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1028, 12
                0, 2,  // 1040, 2
                1, 3,  // 1042, 2
                0, 1, 4, 5,  // 1044, 4
                0, 1, 2, 4, 5,  // 1048, 5
                2, 3, 6,  // 1053, 3
                0, 2, 4, 6,  // 1056, 4
                1, 2, 5, 6,  // 1060, 4
                0, 1, 2, 3, 5, 6,  // 1064, 6
                0, 1, 2, 4, 5, 6,  // 1070, 6
                0, 1, 2, 3, 4, 5, 6,  // 1076, 7
                0, 3, 4, 7,  // 1083, 4
                0, 1, 2, 3, 4, 7,  // 1087, 6
                1, 3, 5, 7,  // 1093, 4
                2, 3, 6, 7,  // 1097, 4
                1, 2, 3, 6, 7,  // 1101, 5
                1, 2, 3, 5, 6, 7,  // 1106, 6
                0, 1, 2, 3, 5, 6, 7,  // 1112, 7
                4, 5, 6, 7,  // 1119, 4
                0, 8,  // 1123, 2
                0, 1, 4, 5, 8,  // 1125, 5
                0, 1, 8, 9,  // 1130, 4
                4, 5, 8, 9,  // 1134, 4
                0, 1, 4, 5, 8, 9,  // 1138, 6
                2, 6, 8, 9,  // 1144, 4
                6, 7, 8, 9,  // 1148, 4
                0, 2, 4, 6, 8, 10,  // 1152, 6
                1, 2, 5, 6, 9, 10,  // 1158, 6
                0, 3, 4, 7, 9, 10,  // 1164, 6
                0, 1, 2, 8, 9, 10,  // 1170, 6
                4, 5, 6, 8, 9, 10,  // 1176, 6
                3, 11,  // 1182, 2
                2, 3, 6, 7, 11,  // 1184, 5
                0, 3, 8, 11,  // 1189, 4
                0, 3, 4, 7, 8, 11,  // 1193, 6
                1, 3, 5, 7, 9, 11,  // 1199, 6
                2, 3, 10, 11,  // 1205, 4
                1, 5, 10, 11,  // 1209, 4
                4, 5, 10, 11,  // 1213, 4
                6, 7, 10, 11,  // 1217, 4
                2, 3, 6, 7, 10, 11,  // 1221, 6
                1, 2, 3, 9, 10, 11,  // 1227, 6
                5, 6, 7, 9, 10, 11,  // 1233, 6
                8, 9, 10, 11,  // 1239, 4
                4, 12,  // 1243, 2
                0, 1, 2, 3, 4, 5, 8, 12,  // 1245, 8
                8, 9, 12,  // 1253, 3
                0, 4, 5, 8, 9, 12,  // 1256, 6
                0, 1, 4, 5, 8, 9, 12,  // 1262, 7
                2, 3, 5, 6, 8, 9, 12,  // 1269, 7
                1, 5, 9, 13,  // 1276, 4
                6, 7, 9, 13,  // 1280, 4
                1, 4, 7, 10, 13,  // 1284, 5
                1, 6, 8, 11, 13,  // 1289, 5
                0, 1, 12, 13,  // 1294, 4
                4, 5, 12, 13,  // 1298, 4
                0, 1, 6, 7, 12, 13,  // 1302, 6
                0, 1, 4, 8, 12, 13,  // 1308, 6
                8, 9, 12, 13,  // 1314, 4
                4, 8, 9, 12, 13,  // 1318, 5
                4, 5, 8, 9, 12, 13,  // 1323, 6
                0, 4, 5, 8, 9, 12, 13,  // 1329, 7
                0, 1, 6, 10, 12, 13,  // 1336, 6
                3, 6, 7, 9, 10, 12, 13,  // 1342, 7
                0, 1, 10, 11, 12, 13,  // 1349, 6
                2, 4, 7, 9, 14,  // 1355, 5
                4, 5, 10, 14,  // 1360, 4
                2, 6, 10, 14,  // 1364, 4
                2, 5, 8, 11, 14,  // 1368, 5
                0, 2, 12, 14,  // 1373, 4
                8, 10, 12, 14,  // 1377, 4
                4, 6, 8, 10, 12, 14,  // 1381, 6
                13, 14,  // 1387, 2
                9, 10, 13, 14,  // 1389, 4
                5, 6, 9, 10, 13, 14,  // 1393, 6
                0, 1, 2, 12, 13, 14,  // 1399, 6
                4, 5, 6, 12, 13, 14,  // 1405, 6
                8, 9, 12, 13, 14,  // 1411, 5
                8, 9, 10, 12, 13, 14,  // 1416, 6
                7, 15,  // 1422, 2
                0, 5, 10, 15,  // 1424, 4
                0, 1, 2, 3, 6, 7, 11, 15,  // 1428, 8
                10, 11, 15,  // 1436, 3
                0, 1, 5, 6, 10, 11, 15,  // 1439, 7
                3, 6, 7, 10, 11, 15,  // 1446, 6
                12, 15,  // 1452, 2
                0, 3, 12, 15,  // 1454, 4
                4, 7, 12, 15,  // 1458, 4
                0, 3, 6, 9, 12, 15,  // 1462, 6
                0, 3, 5, 10, 12, 15,  // 1468, 6
                8, 11, 12, 15,  // 1474, 4
                5, 6, 8, 11, 12, 15,  // 1478, 6
                4, 7, 8, 11, 12, 15,  // 1484, 6
                1, 3, 13, 15,  // 1490, 4
                9, 11, 13, 15,  // 1494, 4
                5, 7, 9, 11, 13, 15,  // 1498, 6
                2, 3, 14, 15,  // 1504, 4
                2, 3, 4, 5, 14, 15,  // 1508, 6
                6, 7, 14, 15,  // 1514, 4
                2, 3, 5, 9, 14, 15,  // 1518, 6
                2, 3, 8, 9, 14, 15,  // 1524, 6
                10, 14, 15,  // 1530, 3
                0, 4, 5, 9, 10, 14, 15,  // 1533, 7
                2, 3, 7, 11, 14, 15,  // 1540, 6
                10, 11, 14, 15,  // 1546, 4
                7, 10, 11, 14, 15,  // 1550, 5
                6, 7, 10, 11, 14, 15,  // 1555, 6
                1, 2, 3, 13, 14, 15,  // 1561, 6
                5, 6, 7, 13, 14, 15,  // 1567, 6
                10, 11, 13, 14, 15,  // 1573, 5
                9, 10, 11, 13, 14, 15,  // 1578, 6
                0, 4, 8, 9, 12, 13, 14, 15,  // 1584, 8
                9, 10, 12, 13, 14, 15,  // 1592, 6
                8, 11, 12, 13, 14, 15,  // 1598, 6
                3, 7, 10, 11, 12, 13, 14, 15,  // 1604, 8
            };
            static const int g_shapeRanges[][2] =
            {
                { 0, 16 },{ 16, 4 },{ 20, 3 },{ 23, 4 },{ 27, 3 },{ 30, 4 },{ 34, 8 },{ 42, 4 },{ 46, 6 },{ 52, 8 },{ 60, 5 },
                { 65, 5 },{ 70, 4 },{ 74, 4 },{ 78, 6 },{ 84, 8 },{ 92, 8 },{ 100, 8 },{ 108, 8 },{ 116, 12 },{ 128, 4 },{ 132, 8 },
                { 140, 8 },{ 148, 10 },{ 158, 6 },{ 164, 8 },{ 172, 12 },{ 184, 8 },{ 192, 5 },{ 197, 3 },{ 200, 4 },{ 204, 6 },{ 210, 8 },
                { 218, 8 },{ 226, 8 },{ 234, 8 },{ 242, 8 },{ 250, 12 },{ 262, 13 },{ 275, 8 },{ 283, 8 },{ 291, 10 },{ 301, 8 },{ 309, 8 },
                { 317, 5 },{ 322, 8 },{ 330, 8 },{ 338, 8 },{ 346, 8 },{ 354, 8 },{ 362, 8 },{ 370, 8 },{ 378, 8 },{ 386, 8 },{ 394, 8 },
                { 402, 8 },{ 410, 8 },{ 418, 4 },{ 422, 8 },{ 430, 6 },{ 436, 8 },{ 444, 10 },{ 454, 8 },{ 462, 12 },{ 474, 8 },{ 482, 8 },
                { 490, 4 },{ 494, 8 },{ 502, 6 },{ 508, 8 },{ 516, 10 },{ 526, 8 },{ 534, 12 },{ 546, 8 },{ 554, 8 },{ 562, 8 },{ 570, 8 },
                { 578, 8 },{ 586, 8 },{ 594, 8 },{ 602, 8 },{ 610, 8 },{ 618, 8 },{ 626, 8 },{ 634, 8 },{ 642, 11 },{ 653, 8 },{ 661, 8 },
                { 669, 6 },{ 675, 8 },{ 683, 8 },{ 691, 3 },{ 694, 4 },{ 698, 8 },{ 706, 8 },{ 714, 8 },{ 722, 8 },{ 730, 8 },{ 738, 10 },
                { 748, 12 },{ 760, 13 },{ 773, 11 },{ 784, 8 },{ 792, 4 },{ 796, 8 },{ 804, 10 },{ 814, 6 },{ 820, 8 },{ 828, 8 },{ 836, 12 },
                { 848, 4 },{ 852, 8 },{ 860, 8 },{ 868, 8 },{ 876, 8 },{ 884, 10 },{ 894, 12 },{ 906, 12 },{ 918, 11 },{ 929, 11 },{ 940, 8 },
                { 948, 10 },{ 958, 12 },{ 970, 8 },{ 978, 12 },{ 990, 13 },{ 1003, 12 },{ 1015, 13 },{ 1028, 12 },{ 1040, 2 },{ 1042, 2 },{ 1044, 4 },
                { 1048, 5 },{ 1053, 3 },{ 1056, 4 },{ 1060, 4 },{ 1064, 6 },{ 1070, 6 },{ 1076, 7 },{ 1083, 4 },{ 1087, 6 },{ 1093, 4 },{ 1097, 4 },
                { 1101, 5 },{ 1106, 6 },{ 1112, 7 },{ 1119, 4 },{ 1123, 2 },{ 1125, 5 },{ 1130, 4 },{ 1134, 4 },{ 1138, 6 },{ 1144, 4 },{ 1148, 4 },
                { 1152, 6 },{ 1158, 6 },{ 1164, 6 },{ 1170, 6 },{ 1176, 6 },{ 1182, 2 },{ 1184, 5 },{ 1189, 4 },{ 1193, 6 },{ 1199, 6 },{ 1205, 4 },
                { 1209, 4 },{ 1213, 4 },{ 1217, 4 },{ 1221, 6 },{ 1227, 6 },{ 1233, 6 },{ 1239, 4 },{ 1243, 2 },{ 1245, 8 },{ 1253, 3 },{ 1256, 6 },
                { 1262, 7 },{ 1269, 7 },{ 1276, 4 },{ 1280, 4 },{ 1284, 5 },{ 1289, 5 },{ 1294, 4 },{ 1298, 4 },{ 1302, 6 },{ 1308, 6 },{ 1314, 4 },
                { 1318, 5 },{ 1323, 6 },{ 1329, 7 },{ 1336, 6 },{ 1342, 7 },{ 1349, 6 },{ 1355, 5 },{ 1360, 4 },{ 1364, 4 },{ 1368, 5 },{ 1373, 4 },
                { 1377, 4 },{ 1381, 6 },{ 1387, 2 },{ 1389, 4 },{ 1393, 6 },{ 1399, 6 },{ 1405, 6 },{ 1411, 5 },{ 1416, 6 },{ 1422, 2 },{ 1424, 4 },
                { 1428, 8 },{ 1436, 3 },{ 1439, 7 },{ 1446, 6 },{ 1452, 2 },{ 1454, 4 },{ 1458, 4 },{ 1462, 6 },{ 1468, 6 },{ 1474, 4 },{ 1478, 6 },
                { 1484, 6 },{ 1490, 4 },{ 1494, 4 },{ 1498, 6 },{ 1504, 4 },{ 1508, 6 },{ 1514, 4 },{ 1518, 6 },{ 1524, 6 },{ 1530, 3 },{ 1533, 7 },
                { 1540, 6 },{ 1546, 4 },{ 1550, 5 },{ 1555, 6 },{ 1561, 6 },{ 1567, 6 },{ 1573, 5 },{ 1578, 6 },{ 1584, 8 },{ 1592, 6 },{ 1598, 6 },
                { 1604, 8 },
            };
            static const int g_shapes1[][2] =
            {
                { 0, 16 }
            };
            static const int g_shapes2[64][2] =
            {
                { 33, 96 },{ 63, 66 },{ 20, 109 },{ 22, 107 },{ 37, 92 },{ 7, 122 },{ 8, 121 },{ 23, 106 },
                { 38, 91 },{ 2, 127 },{ 9, 120 },{ 26, 103 },{ 3, 126 },{ 6, 123 },{ 1, 128 },{ 19, 110 },
                { 15, 114 },{ 124, 5 },{ 72, 57 },{ 115, 14 },{ 125, 4 },{ 70, 59 },{ 100, 29 },{ 60, 69 },
                { 116, 13 },{ 99, 30 },{ 78, 51 },{ 94, 35 },{ 104, 25 },{ 111, 18 },{ 71, 58 },{ 90, 39 },
                { 45, 84 },{ 16, 113 },{ 82, 47 },{ 95, 34 },{ 87, 42 },{ 83, 46 },{ 53, 76 },{ 48, 81 },
                { 68, 61 },{ 105, 24 },{ 98, 31 },{ 88, 41 },{ 75, 54 },{ 43, 86 },{ 52, 77 },{ 117, 12 },
                { 119, 10 },{ 118, 11 },{ 85, 44 },{ 101, 28 },{ 36, 93 },{ 55, 74 },{ 89, 40 },{ 79, 50 },
                { 56, 73 },{ 49, 80 },{ 64, 65 },{ 27, 102 },{ 32, 97 },{ 112, 17 },{ 67, 62 },{ 21, 108 },
            };
            static const int g_shapes3[64][3] =
            {
                { 148, 160, 240 },{ 132, 212, 205 },{ 136, 233, 187 },{ 175, 237, 143 },{ 6, 186, 232 },{ 33, 142, 232 },{ 131, 123, 142 },{ 131, 96, 186 },
                { 6, 171, 110 },{ 1, 18, 110 },{ 1, 146, 123 },{ 33, 195, 66 },{ 20, 51, 66 },{ 20, 178, 96 },{ 2, 177, 106 },{ 211, 4, 59 },
                { 8, 191, 91 },{ 230, 14, 29 },{ 1, 188, 234 },{ 151, 110, 168 },{ 20, 144, 238 },{ 137, 66, 206 },{ 173, 179, 232 },{ 209, 194, 186 },
                { 239, 165, 142 },{ 131, 152, 242 },{ 214, 54, 12 },{ 140, 219, 201 },{ 190, 150, 231 },{ 156, 135, 241 },{ 185, 227, 167 },{ 145, 210, 59 },
                { 138, 174, 106 },{ 189, 229, 14 },{ 176, 133, 106 },{ 78, 178, 195 },{ 111, 146, 171 },{ 216, 180, 196 },{ 217, 181, 193 },{ 184, 228, 166 },
                { 192, 225, 153 },{ 134, 141, 123 },{ 6, 222, 198 },{ 149, 183, 96 },{ 33, 226, 164 },{ 161, 215, 51 },{ 197, 221, 18 },{ 1, 223, 199 },
                { 154, 163, 110 },{ 20, 236, 169 },{ 157, 204, 66 },{ 1, 202, 220 },{ 20, 170, 235 },{ 203, 158, 66 },{ 162, 155, 110 },{ 6, 201, 218 },
                { 139, 135, 123 },{ 33, 167, 224 },{ 182, 150, 96 },{ 19, 200, 213 },{ 63, 207, 159 },{ 147, 172, 109 },{ 129, 130, 128 },{ 208, 14, 59 },
            };

            static const int g_shapeList1[] =
            {
                0,
            };

            static const int g_shapeList1Collapse[] =
            {
                0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1,
            };
            static const int g_shapeList2[] =
            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                122, 123, 124, 125, 126, 127, 128,
            };
            static const int g_shapeList2Collapse[] =
            {
                -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                120, 121, 122, 123, 124, 125, 126, 127, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1,
            };

            static const int g_shapeList12[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128,
            };

            static const int g_shapeList12Collapse[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1,
            };

            static const int g_shapeList3[] =
            {
                1, 2, 4, 6, 8, 12, 14, 18, 19, 20, 29,
                33, 51, 54, 59, 63, 66, 78, 91, 96, 106, 109,
                110, 111, 123, 128, 129, 130, 131, 132, 133, 134, 135,
                136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
                147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
                158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
                191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
                202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
                213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
                235, 236, 237, 238, 239, 240, 241, 242,
            };

            static const int g_shapeList3Collapse[] =
            {
                -1, 0, 1, -1, 2, -1, 3, -1, 4, -1, -1,
                -1, 5, -1, 6, -1, -1, -1, 7, 8, 9, -1,
                -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1,
                11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 12, -1, -1, 13,
                -1, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1,
                16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, 18, -1, -1, -1, -1, 19, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, 21,
                22, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, 24, -1, -1, -1, -1, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
                95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                139,
            };

            static const int g_shapeList3Short[] =
            {
                1, 2, 4, 6, 18, 20, 33, 51, 59, 66, 96,
                106, 110, 123, 131, 132, 136, 142, 143, 146, 148, 160,
                171, 175, 177, 178, 186, 187, 195, 205, 211, 212, 232,
                233, 237, 240,
            };

            static const int g_shapeList3ShortCollapse[] =
            {
                -1, 0, 1, -1, 2, -1, 3, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 4, -1, 5, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1,
                -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1,
                9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 11, -1, -1, -1,
                12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, 14,
                15, -1, -1, -1, 16, -1, -1, -1, -1, -1, 17,
                18, -1, -1, 19, -1, 20, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 21, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 22, -1, -1, -1, 23,
                -1, 24, 25, -1, -1, -1, -1, -1, -1, -1, 26,
                27, -1, -1, -1, -1, -1, -1, -1, 28, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 29, -1, -1, -1,
                -1, -1, 30, 31, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, 32, 33, -1, -1, -1, 34, -1, -1, 35, -1,
                -1,
            };

            static const int g_shapeListAll[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
                198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
                209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
                231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
                242,
            };

            static const int g_numShapes1 = sizeof(g_shapeList1) / sizeof(g_shapeList1[0]);
            static const int g_numShapes2 = sizeof(g_shapeList2) / sizeof(g_shapeList2[0]);
            static const int g_numShapes12 = sizeof(g_shapeList12) / sizeof(g_shapeList12[0]);
            static const int g_numShapes3 = sizeof(g_shapeList3) / sizeof(g_shapeList3[0]);
            static const int g_numShapes3Short = sizeof(g_shapeList3Short) / sizeof(g_shapeList3Short[0]);
            static const int g_numShapesAll = sizeof(g_shapeListAll) / sizeof(g_shapeListAll[0]);
            static const int g_numFragments = sizeof(g_fragments) / sizeof(g_fragments[0]);

            static const int g_maxFragmentsPerMode = (g_numShapes2 > g_numShapes3) ? g_numShapes2 : g_numShapes3;
        }

        namespace BC6HData
        {
            enum EField
            {
                NA, // N/A
                M,  // Mode
                D,  // Shape
                RW,
                RX,
                RY,
                RZ,
                GW,
                GX,
                GY,
                GZ,
                BW,
                BX,
                BY,
                BZ,
            };

            struct ModeDescriptor
            {
                EField m_eField;
                uint8_t   m_uBit;
            };

            const ModeDescriptor g_modeDescriptors[14][82] =
            {
                {   // Mode 1 (0x00) - 10 5 5 5
                    { M, 0 },{ M, 1 },{ GY, 4 },{ BY, 4 },{ BZ, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { GZ, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 2 (0x01) - 7 6 6 6
                    { M, 0 },{ M, 1 },{ GY, 5 },{ GZ, 4 },{ GZ, 5 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ BZ, 0 },{ BZ, 1 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ BY, 5 },{ BZ, 2 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BZ, 3 },{ BZ, 5 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { RY, 5 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ RZ, 5 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 3 (0x02) - 11 5 4 4
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RW,10 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GW,10 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BW,10 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 4 (0x06) - 11 4 5 4
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RW,10 },
                    { GZ, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GW,10 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BW,10 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ BZ, 0 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ GY, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 5 (0x0a) - 11 4 4 5
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RW,10 },
                    { BY, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GW,10 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BW,10 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ BZ, 1 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ BZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 6 (0x0e) - 9 5 5 5
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { GZ, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 7 (0x12) - 8 6 5 5
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ GZ, 4 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ BZ, 2 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BZ, 3 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { RY, 5 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ RZ, 5 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 8 (0x16) - 8 5 6 5
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ BZ, 0 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GY, 5 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ GZ, 5 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { GZ, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BZ, 1 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 9 (0x1a) - 8 5 5 6
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ BZ, 1 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ BY, 5 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BZ, 5 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { GZ, 4 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { BZ, 0 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { BZ, 2 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ BZ, 3 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 10 (0x1e) - 6 6 6 6
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ GZ, 4 },{ BZ, 0 },{ BZ, 1 },{ BY, 4 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GY, 5 },{ BY, 5 },{ BZ, 2 },{ GY, 4 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ GZ, 5 },{ BZ, 3 },{ BZ, 5 },{ BZ, 4 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ GY, 0 },{ GY, 1 },{ GY, 2 },{ GY, 3 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GZ, 0 },{ GZ, 1 },{ GZ, 2 },{ GZ, 3 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BY, 0 },{ BY, 1 },{ BY, 2 },{ BY, 3 },{ RY, 0 },{ RY, 1 },{ RY, 2 },{ RY, 3 },{ RY, 4 },
                    { RY, 5 },{ RZ, 0 },{ RZ, 1 },{ RZ, 2 },{ RZ, 3 },{ RZ, 4 },{ RZ, 5 },{ D, 0 },{ D, 1 },{ D, 2 },
                    { D, 3 },{ D, 4 },
                },

                {   // Mode 11 (0x03) - 10 10
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ RX, 6 },{ RX, 7 },{ RX, 8 },{ RX, 9 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GX, 6 },{ GX, 7 },{ GX, 8 },{ GX, 9 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BX, 6 },{ BX, 7 },{ BX, 8 },{ BX, 9 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },
                },

                {   // Mode 12 (0x07) - 11 9
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ RX, 6 },{ RX, 7 },{ RX, 8 },{ RW,10 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GX, 6 },{ GX, 7 },{ GX, 8 },{ GW,10 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BX, 6 },{ BX, 7 },{ BX, 8 },{ BW,10 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },
                },

                {   // Mode 13 (0x0b) - 12 8
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RX, 4 },
                    { RX, 5 },{ RX, 6 },{ RX, 7 },{ RW,11 },{ RW,10 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GX, 4 },
                    { GX, 5 },{ GX, 6 },{ GX, 7 },{ GW,11 },{ GW,10 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BX, 4 },
                    { BX, 5 },{ BX, 6 },{ BX, 7 },{ BW,11 },{ BW,10 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },
                },

                {   // Mode 14 (0x0f) - 16 4
                    { M, 0 },{ M, 1 },{ M, 2 },{ M, 3 },{ M, 4 },{ RW, 0 },{ RW, 1 },{ RW, 2 },{ RW, 3 },{ RW, 4 },
                    { RW, 5 },{ RW, 6 },{ RW, 7 },{ RW, 8 },{ RW, 9 },{ GW, 0 },{ GW, 1 },{ GW, 2 },{ GW, 3 },{ GW, 4 },
                    { GW, 5 },{ GW, 6 },{ GW, 7 },{ GW, 8 },{ GW, 9 },{ BW, 0 },{ BW, 1 },{ BW, 2 },{ BW, 3 },{ BW, 4 },
                    { BW, 5 },{ BW, 6 },{ BW, 7 },{ BW, 8 },{ BW, 9 },{ RX, 0 },{ RX, 1 },{ RX, 2 },{ RX, 3 },{ RW,15 },
                    { RW,14 },{ RW,13 },{ RW,12 },{ RW,11 },{ RW,10 },{ GX, 0 },{ GX, 1 },{ GX, 2 },{ GX, 3 },{ GW,15 },
                    { GW,14 },{ GW,13 },{ GW,12 },{ GW,11 },{ GW,10 },{ BX, 0 },{ BX, 1 },{ BX, 2 },{ BX, 3 },{ BW,15 },
                    { BW,14 },{ BW,13 },{ BW,12 },{ BW,11 },{ BW,10 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },{ NA, 0 },
                    { NA, 0 },{ NA, 0 },
                },
            };
        }

        struct PackingVector
        {
            uint32_t m_vector[4];
            int m_offset;

            void Init()
            {
                for (int i = 0; i < 4; i++)
                    m_vector[i] = 0;

                m_offset = 0;
            }

            inline void Pack(ParallelMath::ScalarUInt16 value, int bits)
            {
                int vOffset = m_offset >> 5;
                int bitOffset = m_offset & 0x1f;

                m_vector[vOffset] |= (static_cast<uint32_t>(value) << bitOffset) & static_cast<uint32_t>(0xffffffff);

                int overflowBits = bitOffset + bits - 32;
                if (overflowBits > 0)
                    m_vector[vOffset + 1] |= (static_cast<uint32_t>(value) >> (bits - overflowBits));

                m_offset += bits;
            }

            inline void Flush(uint8_t* output)
            {
                assert(m_offset == 128);

                for (int v = 0; v < 4; v++)
                {
                    uint32_t chunk = m_vector[v];
                    for (int b = 0; b < 4; b++)
                        output[v * 4 + b] = static_cast<uint8_t>((chunk >> (b * 8)) & 0xff);
                }
            }
        };


		struct UnpackingVector
		{
			uint32_t m_vector[4];

			void Init(const uint8_t *bytes)
			{
				for (int i = 0; i < 4; i++)
					m_vector[i] = 0;

				for (int b = 0; b < 16; b++)
					m_vector[b / 4] |= (bytes[b] << ((b % 4) * 8));
			}

			inline ParallelMath::ScalarUInt16 Unpack(int bits)
			{
				uint32_t bitMask = (1 << bits) - 1;

				ParallelMath::ScalarUInt16 result = static_cast<ParallelMath::ScalarUInt16>(m_vector[0] & bitMask);

				for (int i = 0; i < 4; i++)
				{
					m_vector[i] >>= bits;
					if (i != 3)
						m_vector[i] |= (m_vector[i + 1] & bitMask) << (32 - bits);
				}

				return result;
			}
		};

        void ComputeTweakFactors(int tweak, int range, float *outFactors)
        {
            int totalUnits = range - 1;
            int minOutsideUnits = ((tweak >> 1) & 1);
            int maxOutsideUnits = (tweak & 1);
            int insideUnits = totalUnits - minOutsideUnits - maxOutsideUnits;

            outFactors[0] = -static_cast<float>(minOutsideUnits) / static_cast<float>(insideUnits);
            outFactors[1] = static_cast<float>(maxOutsideUnits) / static_cast<float>(insideUnits) + 1.0f;
        }

        ParallelMath::Float ScaleHDRValue(const ParallelMath::Float &v, bool isSigned)
        {
            if (isSigned)
            {
                ParallelMath::Float offset = ParallelMath::Select(ParallelMath::Less(v, ParallelMath::MakeFloatZero()), ParallelMath::MakeFloat(-30.0f), ParallelMath::MakeFloat(30.0f));
                return (v * 32.0f + offset) / 31.0f;
            }
            else
                return (v * 64.0f + 30.0f) / 31.0f;
        }

        ParallelMath::SInt16 UnscaleHDRValueSigned(const ParallelMath::SInt16 &v)
        {
#ifdef CVTT_ENABLE_ASSERTS
            for (int i = 0; i < ParallelMath::ParallelSize; i++)
                assert(ParallelMath::Extract(v, i) != -32768)
#endif

            ParallelMath::Int16CompFlag negative = ParallelMath::Less(v, ParallelMath::MakeSInt16(0));
            ParallelMath::UInt15 absComp = ParallelMath::LosslessCast<ParallelMath::UInt15>::Cast(ParallelMath::Select(negative, ParallelMath::SInt16(ParallelMath::MakeSInt16(0) - v), v));

            ParallelMath::UInt31 multiplied = ParallelMath::XMultiply(absComp, ParallelMath::MakeUInt15(31));
            ParallelMath::UInt31 shifted = ParallelMath::RightShift(multiplied, 5);
            ParallelMath::UInt15 absCompScaled = ParallelMath::ToUInt15(shifted);
            ParallelMath::SInt16 signBits = ParallelMath::SelectOrZero(negative, ParallelMath::MakeSInt16(-32768));

            return ParallelMath::LosslessCast<ParallelMath::SInt16>::Cast(absCompScaled) | signBits;
        }

        ParallelMath::UInt15 UnscaleHDRValueUnsigned(const ParallelMath::UInt16 &v)
        {
            return ParallelMath::ToUInt15(ParallelMath::RightShift(ParallelMath::XMultiply(v, ParallelMath::MakeUInt15(31)), 6));
        }

        void UnscaleHDREndpoints(const ParallelMath::AInt16 inEP[2][3], ParallelMath::AInt16 outEP[2][3], bool isSigned)
        {
            for (int epi = 0; epi < 2; epi++)
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    if (isSigned)
                        outEP[epi][ch] = ParallelMath::LosslessCast<ParallelMath::AInt16>::Cast(UnscaleHDRValueSigned(ParallelMath::LosslessCast<ParallelMath::SInt16>::Cast(inEP[epi][ch])));
                    else
                        outEP[epi][ch] = ParallelMath::LosslessCast<ParallelMath::AInt16>::Cast(UnscaleHDRValueUnsigned(ParallelMath::LosslessCast<ParallelMath::UInt16>::Cast(inEP[epi][ch])));
                }
            }
        }

        template<int TVectorSize>
        class UnfinishedEndpoints
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;

            UnfinishedEndpoints()
            {
            }

            UnfinishedEndpoints(const MFloat *base, const MFloat *offset)
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_base[ch] = base[ch];
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_offset[ch] = offset[ch];
            }

            UnfinishedEndpoints(const UnfinishedEndpoints& other)
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_base[ch] = other.m_base[ch];
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_offset[ch] = other.m_offset[ch];
            }

            void FinishHDRUnsigned(int tweak, int range, MSInt16 *outEP0, MSInt16 *outEP1, ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                float tweakFactors[2];
                ComputeTweakFactors(tweak, range, tweakFactors);

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MUInt15 channelEPs[2];
                    for (int epi = 0; epi < 2; epi++)
                    {
                        MFloat f = ParallelMath::Clamp(m_base[ch] + m_offset[ch] * tweakFactors[epi], 0.0f, 31743.0f);
                        channelEPs[epi] = ParallelMath::RoundAndConvertToU15(f, roundingMode);
                    }

                    outEP0[ch] = ParallelMath::LosslessCast<MSInt16>::Cast(channelEPs[0]);
                    outEP1[ch] = ParallelMath::LosslessCast<MSInt16>::Cast(channelEPs[1]);
                }
            }

            void FinishHDRSigned(int tweak, int range, MSInt16* outEP0, MSInt16* outEP1, ParallelMath::RoundTowardNearestForScope* roundingMode)
            {
                float tweakFactors[2];
                ComputeTweakFactors(tweak, range, tweakFactors);

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MSInt16 channelEPs[2];
                    for (int epi = 0; epi < 2; epi++)
                    {
                        MFloat f = ParallelMath::Clamp(m_base[ch] + m_offset[ch] * tweakFactors[epi], -31743.0f, 31743.0f);
                        channelEPs[epi] = ParallelMath::RoundAndConvertToS16(f, roundingMode);
                    }

                    outEP0[ch] = channelEPs[0];
                    outEP1[ch] = channelEPs[1];
                }
            }

            void FinishLDR(int tweak, int range, MUInt15* outEP0, MUInt15* outEP1)
            {
                ParallelMath::RoundTowardNearestForScope roundingMode;

                float tweakFactors[2];
                ComputeTweakFactors(tweak, range, tweakFactors);

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MFloat ep0f = ParallelMath::Clamp(m_base[ch] + m_offset[ch] * tweakFactors[0], 0.0f, 255.0f);
                    MFloat ep1f = ParallelMath::Clamp(m_base[ch] + m_offset[ch] * tweakFactors[1], 0.0f, 255.0f);
                    outEP0[ch] = ParallelMath::RoundAndConvertToU15(ep0f, &roundingMode);
                    outEP1[ch] = ParallelMath::RoundAndConvertToU15(ep1f, &roundingMode);
                }
            }

            template<int TNewVectorSize>
            UnfinishedEndpoints<TNewVectorSize> ExpandTo(float filler)
            {
                MFloat newBase[TNewVectorSize];
                MFloat newOffset[TNewVectorSize];

                for (int ch = 0; ch < TNewVectorSize && ch < TVectorSize; ch++)
                {
                    newBase[ch] = m_base[ch];
                    newOffset[ch] = m_offset[ch];
                }

                MFloat fillerV = ParallelMath::MakeFloat(filler);

                for (int ch = TVectorSize; ch < TNewVectorSize; ch++)
                {
                    newBase[ch] = fillerV;
                    newOffset[ch] = ParallelMath::MakeFloatZero();
                }

                return UnfinishedEndpoints<TNewVectorSize>(newBase, newOffset);
            }

        private:
            MFloat m_base[TVectorSize];
            MFloat m_offset[TVectorSize];
        };

        template<int TMatrixSize>
        class PackedCovarianceMatrix
        {
        public:
            // 0: xx,
            // 1: xy, yy
            // 3: xz, yz, zz 
            // 6: xw, yw, zw, ww
            // ... etc.
            static const int PyramidSize = (TMatrixSize * (TMatrixSize + 1)) / 2;

            typedef ParallelMath::Float MFloat;

            PackedCovarianceMatrix()
            {
                for (int i = 0; i < PyramidSize; i++)
                    m_values[i] = ParallelMath::MakeFloatZero();
            }

            void Add(const ParallelMath::Float *vec, const ParallelMath::Float &weight)
            {
                int index = 0;
                for (int row = 0; row < TMatrixSize; row++)
                {
                    for (int col = 0; col <= row; col++)
                    {
                        m_values[index] = m_values[index] + vec[row] * vec[col] * weight;
                        index++;
                    }
                }
            }

            void Product(MFloat *outVec, const MFloat *inVec)
            {
                for (int row = 0; row < TMatrixSize; row++)
                {
                    MFloat sum = ParallelMath::MakeFloatZero();

                    int index = (row * (row + 1)) >> 1;
                    for (int col = 0; col < TMatrixSize; col++)
                    {
                        sum = sum + inVec[col] * m_values[index];
                        if (col >= row)
                            index += col + 1;
                        else
                            index++;
                    }

                    outVec[row] = sum;
                }
            }

        private:
            ParallelMath::Float m_values[PyramidSize];
        };

        static const int NumEndpointSelectorPasses = 3;

        template<int TVectorSize, int TIterationCount>
        class EndpointSelector
        {
        public:
            typedef ParallelMath::Float MFloat;

            EndpointSelector()
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_centroid[ch] = ParallelMath::MakeFloatZero();
                    m_direction[ch] = ParallelMath::MakeFloatZero();
                }
                m_weightTotal = ParallelMath::MakeFloatZero();
                m_minDist = ParallelMath::MakeFloat(FLT_MAX);
                m_maxDist = ParallelMath::MakeFloat(-FLT_MAX);
            }

            void ContributePass(const MFloat *value, int pass, const MFloat &weight)
            {
                if (pass == 0)
                    ContributeCentroid(value, weight);
                else if (pass == 1)
                    ContributeDirection(value, weight);
                else if (pass == 2)
                    ContributeMinMax(value);
            }

            void FinishPass(int pass)
            {
                if (pass == 0)
                    FinishCentroid();
                else if (pass == 1)
                    FinishDirection();
            }

            UnfinishedEndpoints<TVectorSize> GetEndpoints(const float channelWeights[TVectorSize]) const
            {
                MFloat unweightedBase[TVectorSize];
                MFloat unweightedOffset[TVectorSize];

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MFloat min = m_centroid[ch] + m_direction[ch] * m_minDist;
                    MFloat max = m_centroid[ch] + m_direction[ch] * m_maxDist;

                    float safeWeight = channelWeights[ch];
                    if (safeWeight == 0.f)
                        safeWeight = 1.0f;

                    unweightedBase[ch] = min / channelWeights[ch];
                    unweightedOffset[ch] = (max - min) / channelWeights[ch];
                }

                return UnfinishedEndpoints<TVectorSize>(unweightedBase, unweightedOffset);
            }

        private:
            void ContributeCentroid(const MFloat *value, const MFloat &weight)
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_centroid[ch] = m_centroid[ch] + value[ch] * weight;
                m_weightTotal = m_weightTotal + weight;
            }

            void FinishCentroid()
            {
                MFloat denom = m_weightTotal;
                ParallelMath::MakeSafeDenominator(denom);

                for (int ch = 0; ch < TVectorSize; ch++)
                    m_centroid[ch] = m_centroid[ch] / denom;
            }

            void ContributeDirection(const MFloat *value, const MFloat &weight)
            {
                MFloat diff[TVectorSize];
                for (int ch = 0; ch < TVectorSize; ch++)
                    diff[ch] = value[ch] - m_centroid[ch];

                m_covarianceMatrix.Add(diff, weight);
            }

            void FinishDirection()
            {
                MFloat approx[TVectorSize];
                for (int ch = 0; ch < TVectorSize; ch++)
                    approx[ch] = ParallelMath::MakeFloat(1.0f);

                for (int i = 0; i < TIterationCount; i++)
                {
                    MFloat product[TVectorSize];
                    m_covarianceMatrix.Product(product, approx);

                    MFloat largestComponent = product[0];
                    for (int ch = 1; ch < TVectorSize; ch++)
                        largestComponent = ParallelMath::Max(largestComponent, product[ch]);

                    // product = largestComponent*newApprox
                    ParallelMath::MakeSafeDenominator(largestComponent);
                    for (int ch = 0; ch < TVectorSize; ch++)
                        approx[ch] = product[ch] / largestComponent;
                }

                // Normalize
                MFloat approxLen = ParallelMath::MakeFloatZero();
                for (int ch = 0; ch < TVectorSize; ch++)
                    approxLen = approxLen + approx[ch] * approx[ch];

                approxLen = ParallelMath::Sqrt(approxLen);

                ParallelMath::MakeSafeDenominator(approxLen);

                for (int ch = 0; ch < TVectorSize; ch++)
                    m_direction[ch] = approx[ch] / approxLen;
            }

            void ContributeMinMax(const MFloat *value)
            {
                MFloat dist = ParallelMath::MakeFloatZero();
                for (int ch = 0; ch < TVectorSize; ch++)
                    dist = dist + m_direction[ch] * (value[ch] - m_centroid[ch]);

                m_minDist = ParallelMath::Min(m_minDist, dist);
                m_maxDist = ParallelMath::Max(m_maxDist, dist);
            }

            ParallelMath::Float m_centroid[TVectorSize];
            ParallelMath::Float m_direction[TVectorSize];
            PackedCovarianceMatrix<TVectorSize> m_covarianceMatrix;
            ParallelMath::Float m_weightTotal;

            ParallelMath::Float m_minDist;
            ParallelMath::Float m_maxDist;
        };

        static const ParallelMath::UInt16 g_weightReciprocals[] =
        {
            ParallelMath::MakeUInt16(0),        // -1 
            ParallelMath::MakeUInt16(0),        // 0
            ParallelMath::MakeUInt16(32768),    // 1
            ParallelMath::MakeUInt16(16384),    // 2
            ParallelMath::MakeUInt16(10923),    // 3
            ParallelMath::MakeUInt16(8192),     // 4
            ParallelMath::MakeUInt16(6554),     // 5
            ParallelMath::MakeUInt16(5461),     // 6
            ParallelMath::MakeUInt16(4681),     // 7
            ParallelMath::MakeUInt16(4096),     // 8
            ParallelMath::MakeUInt16(3641),     // 9
            ParallelMath::MakeUInt16(3277),     // 10
            ParallelMath::MakeUInt16(2979),     // 11
            ParallelMath::MakeUInt16(2731),     // 12
            ParallelMath::MakeUInt16(2521),     // 13
            ParallelMath::MakeUInt16(2341),     // 14
            ParallelMath::MakeUInt16(2185),     // 15
        };

        template<int TVectorSize>
        class IndexSelector
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::UInt31 MUInt31;

            template<class TInterpolationEPType, class TColorEPType>
            void Init(const float *channelWeights, const TInterpolationEPType interpolationEndPoints[2][TVectorSize], const TColorEPType colorSpaceEndpoints[2][TVectorSize], int range)
            {
                // In BC6H, the interpolation endpoints are higher-precision than the endpoints in color space.
                // We need to select indexes using the color-space endpoints.

                m_isUniform = true;
                for (int ch = 1; ch < TVectorSize; ch++)
                {
                    if (channelWeights[ch] != channelWeights[0])
                        m_isUniform = false;
                }

                // To work with channel weights, we need something where:
                // pxDiff = px - ep[0]
                // epDiff = ep[1] - ep[0]
                //
                // weightedEPDiff = epDiff * channelWeights
                // normalizedWeightedAxis = weightedEPDiff / len(weightedEPDiff)
                // normalizedIndex = dot(pxDiff * channelWeights, normalizedWeightedAxis) / len(weightedEPDiff)
                // index = normalizedIndex * maxValue
                //
                // Equivalent to:
                // axis = channelWeights * maxValue * epDiff * channelWeights / lenSquared(epDiff * channelWeights)
                // index = dot(axis, pxDiff)

                for (int ep = 0; ep < 2; ep++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        m_endPoint[ep][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(interpolationEndPoints[ep][ch]);

                m_range = range;
                m_maxValue = static_cast<float>(range - 1);

                MFloat epDiffWeighted[TVectorSize];
                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_origin[ch] = ParallelMath::ToFloat(colorSpaceEndpoints[0][ch]);
                    MFloat opposingOriginCh = ParallelMath::ToFloat(colorSpaceEndpoints[1][ch]);
                    epDiffWeighted[ch] = (opposingOriginCh - m_origin[ch]) * channelWeights[ch];
                }

                MFloat lenSquared = epDiffWeighted[0] * epDiffWeighted[0];
                for (int ch = 1; ch < TVectorSize; ch++)
                    lenSquared = lenSquared + epDiffWeighted[ch] * epDiffWeighted[ch];

                ParallelMath::MakeSafeDenominator(lenSquared);

                MFloat maxValueDividedByLengthSquared = ParallelMath::MakeFloat(m_maxValue) / lenSquared;

                for (int ch = 0; ch < TVectorSize; ch++)
                    m_axis[ch] = epDiffWeighted[ch] * channelWeights[ch] * maxValueDividedByLengthSquared;
            }

            template<bool TSigned>
            void Init(const float channelWeights[TVectorSize], const MUInt15 endPoints[2][TVectorSize], int range)
            {
                MAInt16 converted[2][TVectorSize];
                for (int epi = 0; epi < 2; epi++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        converted[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(endPoints[epi][ch]);

                Init<MUInt15, MUInt15>(channelWeights, endPoints, endPoints, range);
            }

            void ReconstructLDR_BC7(const MUInt15 &index, MUInt15* pixel, int numRealChannels)
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MUInt15 ep0f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply((ParallelMath::MakeUInt15(64) - weight), ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[0][ch])));
                    MUInt15 ep1f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(weight, ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[1][ch])));
                    pixel[ch] = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ep0f + ep1f + ParallelMath::MakeUInt15(32), 6));
                }
            }

            void ReconstructLDRPrecise(const MUInt15 &index, MUInt15* pixel, int numRealChannels)
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 64, 7));

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MUInt15 ep0f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply((ParallelMath::MakeUInt15(256) - weight), ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[0][ch])));
                    MUInt15 ep1f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(weight, ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[1][ch])));
                    pixel[ch] = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ep0f + ep1f + ParallelMath::MakeUInt15(128), 8));
                }
            }

            void ReconstructLDR_BC7(const MUInt15 &index, MUInt15* pixel)
            {
                ReconstructLDR_BC7(index, pixel, TVectorSize);
            }

            void ReconstructLDRPrecise(const MUInt15 &index, MUInt15* pixel)
            {
                ReconstructLDRPrecise(index, pixel, TVectorSize);
            }

            MUInt15 SelectIndexLDR(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope* rtn) const
            {
                MFloat dist = (pixel[0] - m_origin[0]) * m_axis[0];
                for (int ch = 1; ch < TVectorSize; ch++)
                    dist = dist + (pixel[ch] - m_origin[ch]) * m_axis[ch];

                return ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(dist, 0.0f, m_maxValue), rtn);
            }

        protected:
            MAInt16 m_endPoint[2][TVectorSize];

        private:
            MFloat m_origin[TVectorSize];
            MFloat m_axis[TVectorSize];
            int m_range;
            float m_maxValue;
            bool m_isUniform;
        };


        template<int TVectorSize>
        class IndexSelectorHDR : public IndexSelector<TVectorSize>
        {
        public:
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt31 MUInt31;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::Float MFloat;

        private:

            MUInt15 InvertSingle(const MUInt15& anIndex) const
            {
                MUInt15 inverted = m_maxValueMinusOne - anIndex;
                return ParallelMath::Select(m_isInverted, inverted, anIndex);
            }

            void ReconstructHDRSignedUninverted(const MUInt15 &index, MSInt16* pixel) const
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MSInt16 ep0 = ParallelMath::LosslessCast<MSInt16>::Cast(this->m_endPoint[0][ch]);
                    MSInt16 ep1 = ParallelMath::LosslessCast<MSInt16>::Cast(this->m_endPoint[1][ch]);

                    MSInt32 pixel32 = ParallelMath::XMultiply((ParallelMath::MakeUInt15(64) - weight), ep0) + ParallelMath::XMultiply(weight, ep1);

                    pixel32 = ParallelMath::RightShift(pixel32 + ParallelMath::MakeSInt32(32), 6);

                    pixel[ch] = UnscaleHDRValueSigned(ParallelMath::ToSInt16(pixel32));
                }
            }

            void ReconstructHDRUnsignedUninverted(const MUInt15 &index, MSInt16* pixel) const
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MUInt16 ep0 = ParallelMath::LosslessCast<MUInt16>::Cast(this->m_endPoint[0][ch]);
                    MUInt16 ep1 = ParallelMath::LosslessCast<MUInt16>::Cast(this->m_endPoint[1][ch]);

                    MUInt31 pixel31 = ParallelMath::XMultiply((ParallelMath::MakeUInt15(64) - weight), ep0) + ParallelMath::XMultiply(weight, ep1);

                    pixel31 = ParallelMath::RightShift(pixel31 + ParallelMath::MakeUInt31(32), 6);

                    pixel[ch] = ParallelMath::LosslessCast<MSInt16>::Cast(UnscaleHDRValueUnsigned(ParallelMath::ToUInt16(pixel31)));
                }
            }

            MFloat ErrorForInterpolatorComponent(int index, int ch, const MFloat *pixel) const
            {
                MFloat diff = pixel[ch] - m_reconstructedInterpolators[index][ch];
                return diff * diff;
            }

            MFloat ErrorForInterpolator(int index, const MFloat *pixel) const
            {
                MFloat error = ErrorForInterpolatorComponent(index, 0, pixel);
                for (int ch = 1; ch < TVectorSize; ch++)
                    error = error + ErrorForInterpolatorComponent(index, ch, pixel);
                return error;
            }

        public:

            void InitHDR(int range, bool isSigned, bool fastIndexing, const float *channelWeights)
            {
                assert(range <= 16);

                m_range = range;

                m_isInverted = ParallelMath::MakeBoolInt16(false);
                m_maxValueMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(range - 1));

                if (!fastIndexing)
                {
                    for (int i = 0; i < range; i++)
                    {
                        MSInt16 recon2CL[TVectorSize];

                        if (isSigned)
                            ReconstructHDRSignedUninverted(ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), recon2CL);
                        else
                            ReconstructHDRUnsignedUninverted(ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), recon2CL);

                        for (int ch = 0; ch < TVectorSize; ch++)
                            m_reconstructedInterpolators[i][ch] = ParallelMath::TwosCLHalfToFloat(recon2CL[ch]) * channelWeights[ch];
                    }
                }
            }

            void ReconstructHDRSigned(const MUInt15 &index, MSInt16* pixel) const
            {
                ReconstructHDRSignedUninverted(InvertSingle(index), pixel);
            }

            void ReconstructHDRUnsigned(const MUInt15 &index, MSInt16* pixel) const
            {
                ReconstructHDRUnsignedUninverted(InvertSingle(index), pixel);
            }

            void ConditionalInvert(const ParallelMath::Int16CompFlag &invert)
            {
                m_isInverted = invert;
            }

            MUInt15 SelectIndexHDRSlow(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope*) const
            {
                MUInt15 index = ParallelMath::MakeUInt15(0);

                MFloat bestError = ErrorForInterpolator(0, pixel);
                for (int i = 1; i < m_range; i++)
                {
                    MFloat error = ErrorForInterpolator(i, pixel);
                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                    ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt15(static_cast<uint16_t>(i)));
                    bestError = ParallelMath::Min(bestError, error);
                }

                return InvertSingle(index);
            }

            MUInt15 SelectIndexHDRFast(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope* rtn) const
            {
                return InvertSingle(this->SelectIndexLDR(pixel, rtn));
            }

        private:
            MFloat m_reconstructedInterpolators[16][TVectorSize];
            ParallelMath::Int16CompFlag m_isInverted;
            MUInt15 m_maxValueMinusOne;
            int m_range;
        };

        // Solve for a, b where v = a*t + b
        // This allows endpoints to be mapped to where T=0 and T=1
        // Least squares from totals:
        // a = (tv - t*v/w)/(tt - t*t/w)
        // b = (v - a*t)/w
        template<int TVectorSize>
        class EndpointRefiner
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;

            MFloat m_tv[TVectorSize];
            MFloat m_v[TVectorSize];
            MFloat m_tt;
            MFloat m_t;
            MFloat m_w;
            int m_wu;

            float m_rcpMaxIndex;
            float m_channelWeights[TVectorSize];
            float m_rcpChannelWeights[TVectorSize];

            void Init(int indexRange, const float channelWeights[TVectorSize])
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_tv[ch] = ParallelMath::MakeFloatZero();
                    m_v[ch] = ParallelMath::MakeFloatZero();
                }
                m_tt = ParallelMath::MakeFloatZero();
                m_t = ParallelMath::MakeFloatZero();
                m_w = ParallelMath::MakeFloatZero();

                m_rcpMaxIndex = 1.0f / static_cast<float>(indexRange - 1);

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_channelWeights[ch] = channelWeights[ch];
                    m_rcpChannelWeights[ch] = 1.0f;
                    if (m_channelWeights[ch] != 0.0f)
                        m_rcpChannelWeights[ch] = 1.0f / channelWeights[ch];
                }

                m_wu = 0;
            }

            void ContributePW(const MFloat *pwFloatPixel, const MUInt15 &index, const MFloat &weight)
            {
                MFloat t = ParallelMath::ToFloat(index) * m_rcpMaxIndex;

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MFloat v = pwFloatPixel[ch] * weight;

                    m_tv[ch] = m_tv[ch] + t * v;
                    m_v[ch] = m_v[ch] + v;
                }
                m_tt = m_tt + weight * t * t;
                m_t = m_t + weight * t;
                m_w = m_w + weight;
            }

            void ContributeUnweightedPW(const MFloat *pwFloatPixel, const MUInt15 &index, int numRealChannels)
            {
                MFloat t = ParallelMath::ToFloat(index) * m_rcpMaxIndex;

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MFloat v = pwFloatPixel[ch];

                    m_tv[ch] = m_tv[ch] + t * v;
                    m_v[ch] = m_v[ch] + v;
                }
                m_tt = m_tt + t * t;
                m_t = m_t + t;
                m_wu++;
            }

            void ContributeUnweightedPW(const MFloat *floatPixel, const MUInt15 &index)
            {
                ContributeUnweightedPW(floatPixel, index, TVectorSize);
            }

            void GetRefinedEndpoints(MFloat endPoint[2][TVectorSize])
            {
                // a = (tv - t*v/w)/(tt - t*t/w)
                // b = (v - a*t)/w
                MFloat w = m_w + ParallelMath::MakeFloat(static_cast<float>(m_wu));

                ParallelMath::MakeSafeDenominator(w);
                MFloat wRcp = ParallelMath::Reciprocal(w);

                MFloat adenom = (m_tt * w - m_t * m_t) * wRcp;

                ParallelMath::FloatCompFlag adenomZero = ParallelMath::Equal(adenom, ParallelMath::MakeFloatZero());
                ParallelMath::ConditionalSet(adenom, adenomZero, ParallelMath::MakeFloat(1.0f));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    /*
                    if (adenom == 0.0)
                        p1 = p2 = er.v / er.w;
                    else
                    {
                        float4 a = (er.tv - er.t*er.v / er.w) / adenom;
                        float4 b = (er.v - a * er.t) / er.w;
                        p1 = b;
                        p2 = a + b;
                    }
                    */

                    MFloat a = (m_tv[ch] - m_t * m_v[ch] * wRcp) / adenom;
                    MFloat b = (m_v[ch] - a * m_t) * wRcp;

                    MFloat p1 = b;
                    MFloat p2 = a + b;

                    ParallelMath::ConditionalSet(p1, adenomZero, (m_v[ch] * wRcp));
                    ParallelMath::ConditionalSet(p2, adenomZero, p1);

                    // Unweight
                    float inverseWeight = m_rcpChannelWeights[ch];

                    endPoint[0][ch] = p1 * inverseWeight;
                    endPoint[1][ch] = p2 * inverseWeight;
                }
            }

            void GetRefinedEndpointsLDR(MUInt15 endPoint[2][TVectorSize], int numRealChannels, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                MFloat floatEndPoint[2][TVectorSize];
                GetRefinedEndpoints(floatEndPoint);

                for (int epi = 0; epi < 2; epi++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        endPoint[epi][ch] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(floatEndPoint[epi][ch], 0.0f, 255.0f), roundingMode);
            }

            void GetRefinedEndpointsLDR(MUInt15 endPoint[2][TVectorSize], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                GetRefinedEndpointsLDR(endPoint, TVectorSize, roundingMode);
            }

            void GetRefinedEndpointsHDR(MSInt16 endPoint[2][TVectorSize], bool isSigned, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                MFloat floatEndPoint[2][TVectorSize];
                GetRefinedEndpoints(floatEndPoint);

                for (int epi = 0; epi < 2; epi++)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                    {
                        MFloat f = floatEndPoint[epi][ch];
                        if (isSigned)
                            endPoint[epi][ch] = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RoundAndConvertToS16(ParallelMath::Clamp(f, -31743.0f, 31743.0f), roundingMode));
                        else
                            endPoint[epi][ch] = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(f, 0.0f, 31743.0f), roundingMode));
                    }
                }
            }
        };

        template<int TVectorSize>
        class AggregatedError
        {
        public:
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt31 MUInt31;
            typedef ParallelMath::Float MFloat;

            AggregatedError()
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_errorUnweighted[ch] = ParallelMath::MakeUInt31(0);
            }

            void Add(const MUInt16 &channelErrorUnweighted, int ch)
            {
                m_errorUnweighted[ch] = m_errorUnweighted[ch] + ParallelMath::ToUInt31(channelErrorUnweighted);
            }

            MFloat Finalize(uint32_t flags, const float channelWeightsSq[TVectorSize]) const
            {
                if (flags & cvtt::Flags::Uniform)
                {
                    MUInt31 total = m_errorUnweighted[0];
                    for (int ch = 1; ch < TVectorSize; ch++)
                        total = total + m_errorUnweighted[ch];
                    return ParallelMath::ToFloat(total);
                }
                else
                {
                    MFloat total = ParallelMath::ToFloat(m_errorUnweighted[0]) * channelWeightsSq[0];
                    for (int ch = 1; ch < TVectorSize; ch++)
                        total = total + ParallelMath::ToFloat(m_errorUnweighted[ch]) * channelWeightsSq[ch];
                    return total;
                }
            }

        private:
            MUInt31 m_errorUnweighted[TVectorSize];
        };

        class BCCommon
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;

            static int TweakRoundsForRange(int range)
            {
                if (range == 3)
                    return 3;
                return 4;
            }

            template<int TVectorSize>
            static void ComputeErrorLDR(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], int numRealChannels, AggregatedError<TVectorSize> &aggError)
            {
                for (int ch = 0; ch < numRealChannels; ch++)
                    aggError.Add(ParallelMath::SqDiffUInt8(reconstructed[ch], original[ch]), ch);
            }

            template<int TVectorSize>
            static void ComputeErrorLDR(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], AggregatedError<TVectorSize> &aggError)
            {
                ComputeErrorLDR<TVectorSize>(flags, reconstructed, original, TVectorSize, aggError);
            }

            template<int TVectorSize>
            static MFloat ComputeErrorLDRSimple(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], int numRealChannels, const float *channelWeightsSq)
            {
                AggregatedError<TVectorSize> aggError;
                ComputeErrorLDR<TVectorSize>(flags, reconstructed, original, numRealChannels, aggError);
                return aggError.Finalize(flags, channelWeightsSq);
            }

            template<int TVectorSize>
            static MFloat ComputeErrorHDRFast(uint32_t flags, const MSInt16 reconstructed[TVectorSize], const MSInt16 original[TVectorSize], const float channelWeightsSq[TVectorSize])
            {
                MFloat error = ParallelMath::MakeFloatZero();
                if (flags & Flags::Uniform)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiffSInt16(reconstructed[ch], original[ch]);
                }
                else
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiffSInt16(reconstructed[ch], original[ch]) * ParallelMath::MakeFloat(channelWeightsSq[ch]);
                }

                return error;
            }

            template<int TVectorSize>
            static MFloat ComputeErrorHDRSlow(uint32_t flags, const MSInt16 reconstructed[TVectorSize], const MSInt16 original[TVectorSize], const float channelWeightsSq[TVectorSize])
            {
                MFloat error = ParallelMath::MakeFloatZero();
                if (flags & Flags::Uniform)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiff2CL(reconstructed[ch], original[ch]);
                }
                else
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiff2CL(reconstructed[ch], original[ch]) * ParallelMath::MakeFloat(channelWeightsSq[ch]);
                }

                return error;
            }

            template<int TChannelCount>
            static void PreWeightPixelsLDR(MFloat preWeightedPixels[16][TChannelCount], const MUInt15 pixels[16][TChannelCount], const float channelWeights[TChannelCount])
            {
                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < TChannelCount; ch++)
                        preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]) * channelWeights[ch];
                }
            }

            template<int TChannelCount>
            static void PreWeightPixelsHDR(MFloat preWeightedPixels[16][TChannelCount], const MSInt16 pixels[16][TChannelCount], const float channelWeights[TChannelCount])
            {
                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < TChannelCount; ch++)
                        preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]) * channelWeights[ch];
                }
            }
        };

        class BC7Computer
        {
        public:
            static const int MaxTweakRounds = 4;

            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::Float MFloat;

            struct WorkInfo
            {
                MUInt15 m_mode;
                MFloat m_error;
                MUInt15 m_ep[3][2][4];
                MUInt15 m_indexes[16];
                MUInt15 m_indexes2[16];

                union
                {
                    MUInt15 m_partition;
                    struct IndexSelectorAndRotation
                    {
                        MUInt15 m_indexSelector;
                        MUInt15 m_rotation;
                    } m_isr;
                } m_u;
            };

            static void TweakAlpha(const MUInt15 original[2], int tweak, int range, MUInt15 result[2])
            {
                ParallelMath::RoundTowardNearestForScope roundingMode;

                float tf[2];
                ComputeTweakFactors(tweak, range, tf);

                MFloat base = ParallelMath::ToFloat(original[0]);
                MFloat offs = ParallelMath::ToFloat(original[1]) - base;

                result[0] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(base + offs * tf[0], 0.0f, 255.0f), &roundingMode);
                result[1] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(base + offs * tf[1], 0.0f, 255.0f), &roundingMode);
            }

            static void Quantize(MUInt15* color, int bits, int channels, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                float maxColor = static_cast<float>((1 << bits) - 1);

                for (int i = 0; i < channels; i++)
                    color[i] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(ParallelMath::ToFloat(color[i]) * ParallelMath::MakeFloat(1.0f / 255.0f) * maxColor, 0.f, 255.f), roundingMode);
            }

            static void QuantizeP(MUInt15* color, int bits, uint16_t p, int channels, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                uint16_t pShift = static_cast<uint16_t>(1 << (7 - bits));
                MUInt15 pShiftV = ParallelMath::MakeUInt15(pShift);

                float maxColorF = static_cast<float>(255 - (1 << (7 - bits)));

                float maxQuantized = static_cast<float>((1 << bits) - 1);

                for (int ch = 0; ch < channels; ch++)
                {
                    MUInt15 clr = color[ch];
                    if (p)
                        clr = ParallelMath::Max(clr, pShiftV) - pShiftV;

                    MFloat rerangedColor = ParallelMath::ToFloat(clr) * maxQuantized / maxColorF;

                    clr = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(rerangedColor, 0.0f, maxQuantized), roundingMode) << 1;
                    if (p)
                        clr = clr | ParallelMath::MakeUInt15(1);

                    color[ch] = clr;
                }
            }

            static void Unquantize(MUInt15* color, int bits, int channels)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    MUInt15 clr = color[ch];
                    clr = clr << (8 - bits);
                    color[ch] = clr | ParallelMath::RightShift(clr, bits);
                }
            }

            static void CompressEndpoints0(MUInt15 ep[2][4], uint16_t p[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    QuantizeP(ep[j], 4, p[j], 3, roundingMode);
                    Unquantize(ep[j], 5, 3);
                    ep[j][3] = ParallelMath::MakeUInt15(255);
                }
            }

            static void CompressEndpoints1(MUInt15 ep[2][4], uint16_t p, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    QuantizeP(ep[j], 6, p, 3, roundingMode);
                    Unquantize(ep[j], 7, 3);
                    ep[j][3] = ParallelMath::MakeUInt15(255);
                }
            }

            static void CompressEndpoints2(MUInt15 ep[2][4], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    Quantize(ep[j], 5, 3, roundingMode);
                    Unquantize(ep[j], 5, 3);
                    ep[j][3] = ParallelMath::MakeUInt15(255);
                }
            }

            static void CompressEndpoints3(MUInt15 ep[2][4], uint16_t p[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    QuantizeP(ep[j], 7, p[j], 3, roundingMode);
                    ep[j][3] = ParallelMath::MakeUInt15(255);
                }
            }

            static void CompressEndpoints4(MUInt15 epRGB[2][3], MUInt15 epA[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    Quantize(epRGB[j], 5, 3, roundingMode);
                    Unquantize(epRGB[j], 5, 3);

                    Quantize(epA + j, 6, 1, roundingMode);
                    Unquantize(epA + j, 6, 1);
                }
            }

            static void CompressEndpoints5(MUInt15 epRGB[2][3], MUInt15 epA[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    Quantize(epRGB[j], 7, 3, roundingMode);
                    Unquantize(epRGB[j], 7, 3);
                }

                // Alpha is full precision
                (void)epA;
            }

            static void CompressEndpoints6(MUInt15 ep[2][4], uint16_t p[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                    QuantizeP(ep[j], 7, p[j], 4, roundingMode);
            }

            static void CompressEndpoints7(MUInt15 ep[2][4], uint16_t p[2], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                for (int j = 0; j < 2; j++)
                {
                    QuantizeP(ep[j], 5, p[j], 4, roundingMode);
                    Unquantize(ep[j], 6, 4);
                }
            }

            struct SinglePlaneTemporaries
            {
                UnfinishedEndpoints<3> unfinishedRGB[BC7Data::g_numShapesAll];
                UnfinishedEndpoints<4> unfinishedRGBA[BC7Data::g_numShapes12];

                MUInt15 fragmentBestIndexes[BC7Data::g_numFragments];
                MUInt15 shapeBestEP[BC7Data::g_maxFragmentsPerMode][2][4];
                MFloat shapeBestError[BC7Data::g_maxFragmentsPerMode];
            };

            static void TrySingleColorRGBAMultiTable(uint32_t flags, const MUInt15 pixels[16][4], const MFloat average[4], int numRealChannels, const uint8_t *fragmentStart, int shapeLength, const MFloat &staticAlphaError, const ParallelMath::Int16CompFlag punchThroughInvalid[4], MFloat& shapeBestError, MUInt15 shapeBestEP[2][4], MUInt15 *fragmentBestIndexes, const float *channelWeightsSq, const cvtt::Tables::BC7SC::Table*const* tables, int numTables, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                MFloat bestAverageError = ParallelMath::MakeFloat(FLT_MAX);

                MUInt15 intAverage[4];
                for (int ch = 0; ch < 4; ch++)
                    intAverage[ch] = ParallelMath::RoundAndConvertToU15(average[ch], rtn);

                MUInt15 eps[2][4];
                MUInt15 reconstructed[4];
                MUInt15 index = ParallelMath::MakeUInt15(0);

                for (int epi = 0; epi < 2; epi++)
                {
                    for (int ch = 0; ch < 3; ch++)
                        eps[epi][ch] = ParallelMath::MakeUInt15(0);
                    eps[epi][3] = ParallelMath::MakeUInt15(255);
                }

                for (int ch = 0; ch < 3; ch++)
                    reconstructed[ch] = ParallelMath::MakeUInt15(0);
                reconstructed[3] = ParallelMath::MakeUInt15(255);

                // Depending on the target index and parity bits, there are multiple valid solid colors.
                // We want to find the one closest to the actual average.
                MFloat epsAverageDiff = ParallelMath::MakeFloat(FLT_MAX);
                for (int t = 0; t < numTables; t++)
                {
                    const cvtt::Tables::BC7SC::Table& table = *(tables[t]);

                    ParallelMath::Int16CompFlag pti = punchThroughInvalid[table.m_pBits];

                    MUInt15 candidateReconstructed[4];
                    MUInt15 candidateEPs[2][4];

                    for (int i = 0; i < ParallelMath::ParallelSize; i++)
                    {
                        for (int ch = 0; ch < numRealChannels; ch++)
                        {
                            ParallelMath::ScalarUInt16 avgValue = ParallelMath::Extract(intAverage[ch], i);
                            assert(avgValue >= 0 && avgValue <= 255);

                            const cvtt::Tables::BC7SC::TableEntry &entry = table.m_entries[avgValue];

                            ParallelMath::PutUInt15(candidateEPs[0][ch], i, entry.m_min);
                            ParallelMath::PutUInt15(candidateEPs[1][ch], i, entry.m_max);
                            ParallelMath::PutUInt15(candidateReconstructed[ch], i, entry.m_actualColor);
                        }
                    }

                    MFloat avgError = ParallelMath::MakeFloatZero();
                    for (int ch = 0; ch < numRealChannels; ch++)
                    {
                        MFloat delta = ParallelMath::ToFloat(candidateReconstructed[ch]) - average[ch];
                        avgError = avgError + delta * delta * channelWeightsSq[ch];
                    }

                    ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(avgError, bestAverageError));
                    better = ParallelMath::AndNot(pti, better); // Mask out punch-through invalidations

                    if (ParallelMath::AnySet(better))
                    {
                        ParallelMath::ConditionalSet(bestAverageError, ParallelMath::Int16FlagToFloat(better), avgError);

                        MUInt15 candidateIndex = ParallelMath::MakeUInt15(table.m_index);

                        ParallelMath::ConditionalSet(index, better, candidateIndex);

                        for (int ch = 0; ch < numRealChannels; ch++)
                            ParallelMath::ConditionalSet(reconstructed[ch], better, candidateReconstructed[ch]);

                        for (int epi = 0; epi < 2; epi++)
                            for (int ch = 0; ch < numRealChannels; ch++)
                                ParallelMath::ConditionalSet(eps[epi][ch], better, candidateEPs[epi][ch]);
                    }
                }

                AggregatedError<4> aggError;
                for (int pxi = 0; pxi < shapeLength; pxi++)
                {
                    int px = fragmentStart[pxi];

                    BCCommon::ComputeErrorLDR<4>(flags, reconstructed, pixels[px], numRealChannels, aggError);
                }

                MFloat error = aggError.Finalize(flags, channelWeightsSq) + staticAlphaError;

                ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, shapeBestError));
                if (ParallelMath::AnySet(better))
                {
                    shapeBestError = ParallelMath::Min(shapeBestError, error);
                    for (int epi = 0; epi < 2; epi++)
                    {
                        for (int ch = 0; ch < numRealChannels; ch++)
                            ParallelMath::ConditionalSet(shapeBestEP[epi][ch], better, eps[epi][ch]);
                    }

                    for (int pxi = 0; pxi < shapeLength; pxi++)
                        ParallelMath::ConditionalSet(fragmentBestIndexes[pxi], better, index);
                }
            }


            static void TrySinglePlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], int numTweakRounds, int numRefineRounds, WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                if (numRefineRounds < 1)
                    numRefineRounds = 1;

                if (numTweakRounds < 1)
                    numTweakRounds = 1;
                else if (numTweakRounds > MaxTweakRounds)
                    numTweakRounds = MaxTweakRounds;

                float channelWeightsSq[4];

                for (int ch = 0; ch < 4; ch++)
                    channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

                SinglePlaneTemporaries temps;

                MUInt15 maxAlpha = ParallelMath::MakeUInt15(0);
                MUInt15 minAlpha = ParallelMath::MakeUInt15(255);
                ParallelMath::Int16CompFlag isPunchThrough = ParallelMath::MakeBoolInt16(true);
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 a = pixels[px][3];
                    maxAlpha = ParallelMath::Max(maxAlpha, a);
                    minAlpha = ParallelMath::Min(minAlpha, a);

                    isPunchThrough = (isPunchThrough & (ParallelMath::Equal(a, ParallelMath::MakeUInt15(0)) | ParallelMath::Equal(a, ParallelMath::MakeUInt15(255))));
                }

                ParallelMath::Int16CompFlag blockHasNonMaxAlpha = ParallelMath::Less(minAlpha, ParallelMath::MakeUInt15(255));
                ParallelMath::Int16CompFlag blockHasNonZeroAlpha = ParallelMath::Less(ParallelMath::MakeUInt15(0), maxAlpha);

                bool anyBlockHasAlpha = ParallelMath::AnySet(blockHasNonMaxAlpha);

                // Try RGB modes if any block has a min alpha 251 or higher
                bool allowRGBModes = ParallelMath::AnySet(ParallelMath::Less(ParallelMath::MakeUInt15(250), minAlpha));

                // Try mode 7 if any block has alpha.
                // Mode 7 is almost never selected for RGB blocks because mode 4 has very accurate 7.7.7.1 endpoints
                // and its parity bit doesn't affect alpha, meaning mode 7 can only be better in extremely specific
                // situations, and only by at most 1 unit of error per pixel.
                bool allowMode7 = anyBlockHasAlpha;

                MFloat preWeightedPixels[16][4];

                BCCommon::PreWeightPixelsLDR<4>(preWeightedPixels, pixels, channelWeights);

                const int *rgbInitialEPCollapseList = NULL;

                // Get initial RGB endpoints
                if (allowRGBModes)
                {
                    const int *shapeList;
                    int numShapesToEvaluate;

                    if (flags & Flags::BC7_EnablePartitioning)
                    {
                        if (flags & Flags::BC7_Enable3Subsets)
                        {
                            shapeList = BC7Data::g_shapeListAll;
                            rgbInitialEPCollapseList = BC7Data::g_shapeListAll;
                            numShapesToEvaluate = BC7Data::g_numShapesAll;
                        }
                        else
                        {
                            shapeList = BC7Data::g_shapeList12;
                            rgbInitialEPCollapseList = BC7Data::g_shapeList12Collapse;
                            numShapesToEvaluate = BC7Data::g_numShapes12;
                        }
                    }
                    else
                    {
                        shapeList = BC7Data::g_shapeList1;
                        rgbInitialEPCollapseList = BC7Data::g_shapeList1Collapse;
                        numShapesToEvaluate = BC7Data::g_numShapes1;
                    }

                    for (int shapeIter = 0; shapeIter < numShapesToEvaluate; shapeIter++)
                    {
                        int shape = shapeList[shapeIter];

                        int shapeStart = BC7Data::g_shapeRanges[shape][0];
                        int shapeSize = BC7Data::g_shapeRanges[shape][1];

                        EndpointSelector<3, 8> epSelector;

                        for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
                        {
                            for (int spx = 0; spx < shapeSize; spx++)
                            {
                                int px = BC7Data::g_fragments[shapeStart + spx];
                                epSelector.ContributePass(preWeightedPixels[px], epPass, ParallelMath::MakeFloat(1.0f));
                            }
                            epSelector.FinishPass(epPass);
                        }
                        temps.unfinishedRGB[shapeIter] = epSelector.GetEndpoints(channelWeights);
                    }
                }

                const int *rgbaInitialEPCollapseList = BC7Data::g_shapeList12Collapse;

                // Get initial RGBA endpoints
                {
                    const int *shapeList = BC7Data::g_shapeList12;
                    int numShapesToEvaluate = BC7Data::g_numShapes12;

                    for (int shapeIter = 0; shapeIter < numShapesToEvaluate; shapeIter++)
                    {
                        int shape = shapeList[shapeIter];

                        if (anyBlockHasAlpha || !allowRGBModes)
                        {
                            int shapeStart = BC7Data::g_shapeRanges[shape][0];
                            int shapeSize = BC7Data::g_shapeRanges[shape][1];

                            EndpointSelector<4, 8> epSelector;

                            for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
                            {
                                for (int spx = 0; spx < shapeSize; spx++)
                                {
                                    int px = BC7Data::g_fragments[shapeStart + spx];
                                    epSelector.ContributePass(preWeightedPixels[px], epPass, ParallelMath::MakeFloat(1.0f));
                                }
                                epSelector.FinishPass(epPass);
                            }
                            temps.unfinishedRGBA[shapeIter] = epSelector.GetEndpoints(channelWeights);
                        }
                        else
                        {
                            temps.unfinishedRGBA[shapeIter] = temps.unfinishedRGB[rgbInitialEPCollapseList[shape]].ExpandTo<4>(255);
                        }
                    }
                }

                for (uint16_t mode = 0; mode <= 7; mode++)
                {
                    if (!(flags & Flags::BC7_EnablePartitioning) && BC7Data::g_modes[mode].m_numSubsets != 1)
                        continue;

                    if (!(flags & Flags::BC7_Enable3Subsets) && BC7Data::g_modes[mode].m_numSubsets == 3)
                        continue;

                    if (mode == 4 || mode == 5)
                        continue;

                    if (mode < 4 && !allowRGBModes)
                        continue;

                    if (mode == 7 && !allowMode7)
                        continue;

                    bool isRGB = (mode < 4);

                    unsigned int numPartitions = 1 << BC7Data::g_modes[mode].m_partitionBits;
                    int numSubsets = BC7Data::g_modes[mode].m_numSubsets;
                    int indexPrec = BC7Data::g_modes[mode].m_indexBits;

                    int parityBitMax = 1;
                    if (BC7Data::g_modes[mode].m_pBitMode == BC7Data::PBitMode_PerEndpoint)
                        parityBitMax = 4;
                    else if (BC7Data::g_modes[mode].m_pBitMode == BC7Data::PBitMode_PerSubset)
                        parityBitMax = 2;

                    int numRealChannels = isRGB ? 3 : 4;

                    int numShapes;
                    const int *shapeList;
                    const int *shapeCollapseList;

                    if (numSubsets == 1)
                    {
                        numShapes = BC7Data::g_numShapes1;
                        shapeList = BC7Data::g_shapeList1;
                        shapeCollapseList = BC7Data::g_shapeList1Collapse;
                    }
                    else if (numSubsets == 2)
                    {
                        numShapes = BC7Data::g_numShapes2;
                        shapeList = BC7Data::g_shapeList2;
                        shapeCollapseList = BC7Data::g_shapeList2Collapse;
                    }
                    else
                    {
                        assert(numSubsets == 3);
                        if (numPartitions == 16)
                        {
                            numShapes = BC7Data::g_numShapes3Short;
                            shapeList = BC7Data::g_shapeList3Short;
                            shapeCollapseList = BC7Data::g_shapeList3ShortCollapse;
                        }
                        else
                        {
                            assert(numPartitions == 64);
                            numShapes = BC7Data::g_numShapes3;
                            shapeList = BC7Data::g_shapeList3;
                            shapeCollapseList = BC7Data::g_shapeList3Collapse;
                        }
                    }

                    for (int slot = 0; slot < BC7Data::g_maxFragmentsPerMode; slot++)
                        temps.shapeBestError[slot] = ParallelMath::MakeFloat(FLT_MAX);

                    for (int shapeIter = 0; shapeIter < numShapes; shapeIter++)
                    {
                        int shape = shapeList[shapeIter];
                        int shapeStart = BC7Data::g_shapeRanges[shape][0];
                        int shapeLength = BC7Data::g_shapeRanges[shape][1];
                        int shapeCollapsedEvalIndex = shapeCollapseList[shape];

                        AggregatedError<1> alphaAggError;
                        if (isRGB && anyBlockHasAlpha)
                        {
                            MUInt15 filledAlpha[1] = { ParallelMath::MakeUInt15(255) };

                            for (int pxi = 0; pxi < shapeLength; pxi++)
                            {
                                int px = BC7Data::g_fragments[shapeStart + pxi];
                                MUInt15 original[1] = { pixels[px][3] };
                                BCCommon::ComputeErrorLDR<1>(flags, filledAlpha, original, alphaAggError);
                            }
                        }

                        float alphaWeightsSq[1] = { channelWeightsSq[3] };
                        MFloat staticAlphaError = alphaAggError.Finalize(flags, alphaWeightsSq);

                        assert(shapeCollapsedEvalIndex >= 0);

                        MUInt15 tweakBaseEP[MaxTweakRounds][2][4];

                        for (int tweak = 0; tweak < numTweakRounds; tweak++)
                        {
                            if (isRGB)
                            {
                                temps.unfinishedRGB[rgbInitialEPCollapseList[shape]].FinishLDR(tweak, 1 << indexPrec, tweakBaseEP[tweak][0], tweakBaseEP[tweak][1]);
                                tweakBaseEP[tweak][0][3] = tweakBaseEP[tweak][1][3] = ParallelMath::MakeUInt15(255);
                            }
                            else
                            {
                                temps.unfinishedRGBA[rgbaInitialEPCollapseList[shape]].FinishLDR(tweak, 1 << indexPrec, tweakBaseEP[tweak][0], tweakBaseEP[tweak][1]);
                            }
                        }

                        ParallelMath::Int16CompFlag punchThroughInvalid[4];
                        for (int pIter = 0; pIter < parityBitMax; pIter++)
                        {
                            punchThroughInvalid[pIter] = ParallelMath::MakeBoolInt16(false);

                            if ((flags & Flags::BC7_RespectPunchThrough) && (mode == 6 || mode == 7))
                            {
                                // Modes 6 and 7 have parity bits that affect alpha
                                if (pIter == 0)
                                    punchThroughInvalid[pIter] = (isPunchThrough & blockHasNonZeroAlpha);
                                else if (pIter == parityBitMax - 1)
                                    punchThroughInvalid[pIter] = (isPunchThrough & blockHasNonMaxAlpha);
                                else
                                    punchThroughInvalid[pIter] = isPunchThrough;
                            }
                        }

                        for (int pIter = 0; pIter < parityBitMax; pIter++)
                        {
                            if (ParallelMath::AllSet(punchThroughInvalid[pIter]))
                                continue;

                            bool needPunchThroughCheck = ParallelMath::AnySet(punchThroughInvalid[pIter]);

                            for (int tweak = 0; tweak < numTweakRounds; tweak++)
                            {
                                uint16_t p[2];
                                p[0] = (pIter & 1);
                                p[1] = ((pIter >> 1) & 1);

                                MUInt15 ep[2][4];

                                for (int epi = 0; epi < 2; epi++)
                                    for (int ch = 0; ch < 4; ch++)
                                        ep[epi][ch] = tweakBaseEP[tweak][epi][ch];

                                for (int refine = 0; refine < numRefineRounds; refine++)
                                {
                                    switch (mode)
                                    {
                                    case 0:
                                        CompressEndpoints0(ep, p, rtn);
                                        break;
                                    case 1:
                                        CompressEndpoints1(ep, p[0], rtn);
                                        break;
                                    case 2:
                                        CompressEndpoints2(ep, rtn);
                                        break;
                                    case 3:
                                        CompressEndpoints3(ep, p, rtn);
                                        break;
                                    case 6:
                                        CompressEndpoints6(ep, p, rtn);
                                        break;
                                    case 7:
                                        CompressEndpoints7(ep, p, rtn);
                                        break;
                                    default:
                                        assert(false);
                                        break;
                                    };

                                    MFloat shapeError = ParallelMath::MakeFloatZero();

                                    IndexSelector<4> indexSelector;
                                    indexSelector.Init<false>(channelWeights, ep, 1 << indexPrec);

                                    EndpointRefiner<4> epRefiner;
                                    epRefiner.Init(1 << indexPrec, channelWeights);

                                    MUInt15 indexes[16];

                                    AggregatedError<4> aggError;
                                    for (int pxi = 0; pxi < shapeLength; pxi++)
                                    {
                                        int px = BC7Data::g_fragments[shapeStart + pxi];

                                        MUInt15 index;
                                        MUInt15 reconstructed[4];

                                        index = indexSelector.SelectIndexLDR(floatPixels[px], rtn);
                                        indexSelector.ReconstructLDR_BC7(index, reconstructed, numRealChannels);

                                        if (flags & cvtt::Flags::BC7_FastIndexing)
                                            BCCommon::ComputeErrorLDR<4>(flags, reconstructed, pixels[px], numRealChannels, aggError);
                                        else
                                        {
                                            MFloat error = BCCommon::ComputeErrorLDRSimple<4>(flags, reconstructed, pixels[px], numRealChannels, channelWeightsSq);

                                            MUInt15 altIndexes[2];
                                            altIndexes[0] = ParallelMath::Max(index, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                            altIndexes[1] = ParallelMath::Min(index + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << indexPrec) - 1)));

                                            for (int ii = 0; ii < 2; ii++)
                                            {
                                                indexSelector.ReconstructLDR_BC7(altIndexes[ii], reconstructed, numRealChannels);

                                                MFloat altError = BCCommon::ComputeErrorLDRSimple<4>(flags, reconstructed, pixels[px], numRealChannels, channelWeightsSq);
                                                ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altError, error));
                                                error = ParallelMath::Min(error, altError);
                                                ParallelMath::ConditionalSet(index, better, altIndexes[ii]);
                                            }

                                            shapeError = shapeError + error;
                                        }

                                        if (refine != numRefineRounds - 1)
                                            epRefiner.ContributeUnweightedPW(preWeightedPixels[px], index, numRealChannels);

                                        indexes[pxi] = index;
                                    }

                                    if (flags & cvtt::Flags::BC7_FastIndexing)
                                        shapeError = aggError.Finalize(flags, channelWeightsSq);

                                    if (isRGB)
                                        shapeError = shapeError + staticAlphaError;

                                    ParallelMath::FloatCompFlag shapeErrorBetter;
                                    ParallelMath::Int16CompFlag shapeErrorBetter16;

                                    shapeErrorBetter = ParallelMath::Less(shapeError, temps.shapeBestError[shapeCollapsedEvalIndex]);
                                    shapeErrorBetter16 = ParallelMath::FloatFlagToInt16(shapeErrorBetter);

                                    if (ParallelMath::AnySet(shapeErrorBetter16))
                                    {
                                        bool punchThroughOK = true;
                                        if (needPunchThroughCheck)
                                        {
                                            shapeErrorBetter16 = ParallelMath::AndNot(punchThroughInvalid[pIter], shapeErrorBetter16);
                                            shapeErrorBetter = ParallelMath::Int16FlagToFloat(shapeErrorBetter16);

                                            if (!ParallelMath::AnySet(shapeErrorBetter16))
                                                punchThroughOK = false;
                                        }

                                        if (punchThroughOK)
                                        {
                                            ParallelMath::ConditionalSet(temps.shapeBestError[shapeCollapsedEvalIndex], shapeErrorBetter, shapeError);
                                            for (int epi = 0; epi < 2; epi++)
                                                for (int ch = 0; ch < numRealChannels; ch++)
                                                    ParallelMath::ConditionalSet(temps.shapeBestEP[shapeCollapsedEvalIndex][epi][ch], shapeErrorBetter16, ep[epi][ch]);

                                            for (int pxi = 0; pxi < shapeLength; pxi++)
                                                ParallelMath::ConditionalSet(temps.fragmentBestIndexes[shapeStart + pxi], shapeErrorBetter16, indexes[pxi]);
                                        }
                                    }

                                    if (refine != numRefineRounds - 1)
                                        epRefiner.GetRefinedEndpointsLDR(ep, numRealChannels, rtn);
                                } // refine
                            } // tweak
                        } // p

                        if (flags & cvtt::Flags::BC7_TrySingleColor)
                        {
                            MUInt15 total[4];
                            for (int ch = 0; ch < 4; ch++)
                                total[ch] = ParallelMath::MakeUInt15(0);

                            for (int pxi = 0; pxi < shapeLength; pxi++)
                            {
                                int px = BC7Data::g_fragments[shapeStart + pxi];
                                for (int ch = 0; ch < 4; ch++)
                                    total[ch] = total[ch] + pixels[pxi][ch];
                            }

                            MFloat rcpShapeLength = ParallelMath::MakeFloat(1.0f / static_cast<float>(shapeLength));
                            MFloat average[4];
                            for (int ch = 0; ch < 4; ch++)
                                average[ch] = ParallelMath::ToFloat(total[ch]) * rcpShapeLength;

                            const uint8_t *fragment = BC7Data::g_fragments + shapeStart;
                            MFloat &shapeBestError = temps.shapeBestError[shapeCollapsedEvalIndex];
                            MUInt15(&shapeBestEP)[2][4] = temps.shapeBestEP[shapeCollapsedEvalIndex];
                            MUInt15 *fragmentBestIndexes = temps.fragmentBestIndexes + shapeStart;

                            const cvtt::Tables::BC7SC::Table **scTables = NULL;
                            int numSCTables = 0;

                            switch (mode)
                            {
                            case 0:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode0_p00_i1,
                                        &cvtt::Tables::BC7SC::g_mode0_p00_i2,
                                        &cvtt::Tables::BC7SC::g_mode0_p00_i3,
                                        &cvtt::Tables::BC7SC::g_mode0_p01_i1,
                                        &cvtt::Tables::BC7SC::g_mode0_p01_i2,
                                        &cvtt::Tables::BC7SC::g_mode0_p01_i3,
                                        &cvtt::Tables::BC7SC::g_mode0_p10_i1,
                                        &cvtt::Tables::BC7SC::g_mode0_p10_i2,
                                        &cvtt::Tables::BC7SC::g_mode0_p10_i3,
                                        &cvtt::Tables::BC7SC::g_mode0_p11_i1,
                                        &cvtt::Tables::BC7SC::g_mode0_p11_i2,
                                        &cvtt::Tables::BC7SC::g_mode0_p11_i3,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            case 1:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode1_p0_i1,
                                        &cvtt::Tables::BC7SC::g_mode1_p0_i2,
                                        &cvtt::Tables::BC7SC::g_mode1_p0_i3,
                                        &cvtt::Tables::BC7SC::g_mode1_p1_i1,
                                        &cvtt::Tables::BC7SC::g_mode1_p1_i2,
                                        &cvtt::Tables::BC7SC::g_mode1_p1_i3,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            case 2:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode2,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            case 3:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode3_p0,
                                        &cvtt::Tables::BC7SC::g_mode3_p1,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            case 6:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i1,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i2,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i3,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i4,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i5,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i6,
                                        &cvtt::Tables::BC7SC::g_mode6_p0_i7,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i1,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i2,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i3,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i4,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i5,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i6,
                                        &cvtt::Tables::BC7SC::g_mode6_p1_i7,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            case 7:
                                {
                                    const cvtt::Tables::BC7SC::Table *tables[] =
                                    {
                                        &cvtt::Tables::BC7SC::g_mode7_p00,
                                        &cvtt::Tables::BC7SC::g_mode7_p01,
                                        &cvtt::Tables::BC7SC::g_mode7_p10,
                                        &cvtt::Tables::BC7SC::g_mode7_p11,
                                    };
                                    scTables = tables;
                                    numSCTables = sizeof(tables) / sizeof(tables[0]);
                                }
                                break;
                            default:
                                assert(false);
                                break;
                            }

                            TrySingleColorRGBAMultiTable(flags, pixels, average, numRealChannels, fragment, shapeLength, staticAlphaError, punchThroughInvalid, shapeBestError, shapeBestEP, fragmentBestIndexes, channelWeightsSq, scTables, numSCTables, rtn);
                        }
                    } // shapeIter

                    for (uint16_t partition = 0; partition < numPartitions; partition++)
                    {
                        const int *partitionShapes;
                        if (numSubsets == 1)
                            partitionShapes = BC7Data::g_shapes1[partition];
                        else if (numSubsets == 2)
                            partitionShapes = BC7Data::g_shapes2[partition];
                        else
                        {
                            assert(numSubsets == 3);
                            partitionShapes = BC7Data::g_shapes3[partition];
                        }

                        MFloat totalError = ParallelMath::MakeFloatZero();
                        for (int subset = 0; subset < numSubsets; subset++)
                            totalError = totalError + temps.shapeBestError[shapeCollapseList[partitionShapes[subset]]];

                        ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(totalError, work.m_error);
                        ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                        if (ParallelMath::AnySet(errorBetter16))
                        {
                            for (int subset = 0; subset < numSubsets; subset++)
                            {
                                int shape = partitionShapes[subset];
                                int shapeStart = BC7Data::g_shapeRanges[shape][0];
                                int shapeLength = BC7Data::g_shapeRanges[shape][1];
                                int shapeCollapsedEvalIndex = shapeCollapseList[shape];

                                for (int epi = 0; epi < 2; epi++)
                                    for (int ch = 0; ch < 4; ch++)
                                        ParallelMath::ConditionalSet(work.m_ep[subset][epi][ch], errorBetter16, temps.shapeBestEP[shapeCollapsedEvalIndex][epi][ch]);

                                for (int pxi = 0; pxi < shapeLength; pxi++)
                                {
                                    int px = BC7Data::g_fragments[shapeStart + pxi];
                                    ParallelMath::ConditionalSet(work.m_indexes[px], errorBetter16, temps.fragmentBestIndexes[shapeStart + pxi]);
                                }
                            }

                            work.m_error = ParallelMath::Min(totalError, work.m_error);
                            ParallelMath::ConditionalSet(work.m_mode, errorBetter16, ParallelMath::MakeUInt15(mode));
                            ParallelMath::ConditionalSet(work.m_u.m_partition, errorBetter16, ParallelMath::MakeUInt15(partition));
                        }
                    }
                }
            }

            static void TryDualPlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], int numTweakRounds, int numRefineRounds, WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                // TODO: These error calculations are not optimal for weight-by-alpha, but this routine needs to be mostly rewritten for that.
                // The alpha/color solutions are co-dependent in that case, but a good way to solve it would probably be to
                // solve the alpha channel first, then solve the RGB channels, which in turn breaks down into two cases:
                // - Separate alpha channel, then weighted RGB
                // - Alpha+2 other channels, then the independent channel

                if (!(flags & Flags::BC7_EnableDualPlane))
                    return;

                if (numRefineRounds < 1)
                    numRefineRounds = 1;

                if (numTweakRounds < 1)
                    numTweakRounds = 1;
                else if (numTweakRounds > MaxTweakRounds)
                    numTweakRounds = MaxTweakRounds;

                float channelWeightsSq[4];
                for (int ch = 0; ch < 4; ch++)
                    channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

                for (uint16_t mode = 4; mode <= 5; mode++)
                {
                    for (uint16_t rotation = 0; rotation < 4; rotation++)
                    {
                        int alphaChannel = (rotation + 3) & 3;
                        int redChannel = (rotation == 1) ? 3 : 0;
                        int greenChannel = (rotation == 2) ? 3 : 1;
                        int blueChannel = (rotation == 3) ? 3 : 2;

                        MUInt15 rotatedRGB[16][3];
                        MFloat floatRotatedRGB[16][3];

                        for (int px = 0; px < 16; px++)
                        {
                            rotatedRGB[px][0] = pixels[px][redChannel];
                            rotatedRGB[px][1] = pixels[px][greenChannel];
                            rotatedRGB[px][2] = pixels[px][blueChannel];

                            for (int ch = 0; ch < 3; ch++)
                                floatRotatedRGB[px][ch] = ParallelMath::ToFloat(rotatedRGB[px][ch]);
                        }

                        uint16_t maxIndexSelector = (mode == 4) ? 2 : 1;

                        float rotatedRGBWeights[3] = { channelWeights[redChannel], channelWeights[greenChannel], channelWeights[blueChannel] };
                        float rotatedRGBWeightsSq[3] = { channelWeightsSq[redChannel], channelWeightsSq[greenChannel], channelWeightsSq[blueChannel] };
                        float rotatedAlphaWeight[1] = { channelWeights[alphaChannel] };
                        float rotatedAlphaWeightSq[1] = { channelWeightsSq[alphaChannel] };

                        float uniformWeight[1] = { 1.0f };   // Since the alpha channel is independent, there's no need to bother with weights when doing refinement or selection, only error

                        MFloat preWeightedRotatedRGB[16][3];
                        BCCommon::PreWeightPixelsLDR<3>(preWeightedRotatedRGB, rotatedRGB, rotatedRGBWeights);

                        for (uint16_t indexSelector = 0; indexSelector < maxIndexSelector; indexSelector++)
                        {
                            EndpointSelector<3, 8> rgbSelector;

                            for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
                            {
                                for (int px = 0; px < 16; px++)
                                    rgbSelector.ContributePass(preWeightedRotatedRGB[px], epPass, ParallelMath::MakeFloat(1.0f));

                                rgbSelector.FinishPass(epPass);
                            }

                            MUInt15 alphaRange[2];

                            alphaRange[0] = alphaRange[1] = pixels[0][alphaChannel];
                            for (int px = 1; px < 16; px++)
                            {
                                alphaRange[0] = ParallelMath::Min(pixels[px][alphaChannel], alphaRange[0]);
                                alphaRange[1] = ParallelMath::Max(pixels[px][alphaChannel], alphaRange[1]);
                            }

                            int rgbPrec = 0;
                            int alphaPrec = 0;

                            if (mode == 4)
                            {
                                rgbPrec = indexSelector ? 3 : 2;
                                alphaPrec = indexSelector ? 2 : 3;
                            }
                            else
                                rgbPrec = alphaPrec = 2;

                            UnfinishedEndpoints<3> unfinishedRGB = rgbSelector.GetEndpoints(rotatedRGBWeights);

                            MFloat bestRGBError = ParallelMath::MakeFloat(FLT_MAX);
                            MFloat bestAlphaError = ParallelMath::MakeFloat(FLT_MAX);

                            MUInt15 bestRGBIndexes[16];
                            MUInt15 bestAlphaIndexes[16];
                            MUInt15 bestEP[2][4];

                            for (int px = 0; px < 16; px++)
                                bestRGBIndexes[px] = bestAlphaIndexes[px] = ParallelMath::MakeUInt15(0);

                            for (int tweak = 0; tweak < numTweakRounds; tweak++)
                            {
                                MUInt15 rgbEP[2][3];
                                MUInt15 alphaEP[2];

                                unfinishedRGB.FinishLDR(tweak, 1 << rgbPrec, rgbEP[0], rgbEP[1]);

                                TweakAlpha(alphaRange, tweak, 1 << alphaPrec, alphaEP);

                                for (int refine = 0; refine < numRefineRounds; refine++)
                                {
                                    if (mode == 4)
                                        CompressEndpoints4(rgbEP, alphaEP, rtn);
                                    else
                                        CompressEndpoints5(rgbEP, alphaEP, rtn);


                                    IndexSelector<1> alphaIndexSelector;
                                    IndexSelector<3> rgbIndexSelector;

                                    {
                                        MUInt15 alphaEPTemp[2][1] = { { alphaEP[0] },{ alphaEP[1] } };
                                        alphaIndexSelector.Init<false>(uniformWeight, alphaEPTemp, 1 << alphaPrec);
                                    }
                                    rgbIndexSelector.Init<false>(rotatedRGBWeights, rgbEP, 1 << rgbPrec);

                                    EndpointRefiner<3> rgbRefiner;
                                    EndpointRefiner<1> alphaRefiner;

                                    rgbRefiner.Init(1 << rgbPrec, rotatedRGBWeights);
                                    alphaRefiner.Init(1 << alphaPrec, uniformWeight);

                                    MFloat errorRGB = ParallelMath::MakeFloatZero();
                                    MFloat errorA = ParallelMath::MakeFloatZero();

                                    MUInt15 rgbIndexes[16];
                                    MUInt15 alphaIndexes[16];

                                    AggregatedError<3> rgbAggError;
                                    AggregatedError<1> alphaAggError;

                                    for (int px = 0; px < 16; px++)
                                    {
                                        MUInt15 rgbIndex = rgbIndexSelector.SelectIndexLDR(floatRotatedRGB[px], rtn);
                                        MUInt15 alphaIndex = alphaIndexSelector.SelectIndexLDR(floatPixels[px] + alphaChannel, rtn);

                                        MUInt15 reconstructedRGB[3];
                                        MUInt15 reconstructedAlpha[1];

                                        rgbIndexSelector.ReconstructLDR_BC7(rgbIndex, reconstructedRGB);
                                        alphaIndexSelector.ReconstructLDR_BC7(alphaIndex, reconstructedAlpha);

                                        if (flags & cvtt::Flags::BC7_FastIndexing)
                                        {
                                            BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], rgbAggError);
                                            BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, alphaAggError);
                                        }
                                        else
                                        {
                                            AggregatedError<3> baseRGBAggError;
                                            AggregatedError<1> baseAlphaAggError;

                                            BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], baseRGBAggError);
                                            BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, baseAlphaAggError);

                                            MFloat rgbError = baseRGBAggError.Finalize(flags, rotatedRGBWeightsSq);
                                            MFloat alphaError = baseAlphaAggError.Finalize(flags, rotatedAlphaWeightSq);

                                            MUInt15 altRGBIndexes[2];
                                            MUInt15 altAlphaIndexes[2];

                                            altRGBIndexes[0] = ParallelMath::Max(rgbIndex, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                            altRGBIndexes[1] = ParallelMath::Min(rgbIndex + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << rgbPrec) - 1)));

                                            altAlphaIndexes[0] = ParallelMath::Max(alphaIndex, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                            altAlphaIndexes[1] = ParallelMath::Min(alphaIndex + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << alphaPrec) - 1)));

                                            for (int ii = 0; ii < 2; ii++)
                                            {
                                                rgbIndexSelector.ReconstructLDR_BC7(altRGBIndexes[ii], reconstructedRGB);
                                                alphaIndexSelector.ReconstructLDR_BC7(altAlphaIndexes[ii], reconstructedAlpha);

                                                AggregatedError<3> altRGBAggError;
                                                AggregatedError<1> altAlphaAggError;

                                                BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], altRGBAggError);
                                                BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, altAlphaAggError);

                                                MFloat altRGBError = altRGBAggError.Finalize(flags, rotatedRGBWeightsSq);
                                                MFloat altAlphaError = altAlphaAggError.Finalize(flags, rotatedAlphaWeightSq);

                                                ParallelMath::Int16CompFlag rgbBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altRGBError, rgbError));
                                                ParallelMath::Int16CompFlag alphaBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altAlphaError, alphaError));

                                                rgbError = ParallelMath::Min(altRGBError, rgbError);
                                                alphaError = ParallelMath::Min(altAlphaError, alphaError);

                                                ParallelMath::ConditionalSet(rgbIndex, rgbBetter, altRGBIndexes[ii]);
                                                ParallelMath::ConditionalSet(alphaIndex, alphaBetter, altAlphaIndexes[ii]);
                                            }

                                            errorRGB = errorRGB + rgbError;
                                            errorA = errorA + alphaError;
                                        }

                                        if (refine != numRefineRounds - 1)
                                        {
                                            rgbRefiner.ContributeUnweightedPW(preWeightedRotatedRGB[px], rgbIndex);
                                            alphaRefiner.ContributeUnweightedPW(floatPixels[px] + alphaChannel, alphaIndex);
                                        }

                                        if (flags & Flags::BC7_FastIndexing)
                                        {
                                            errorRGB = rgbAggError.Finalize(flags, rotatedRGBWeightsSq);
                                            errorA = rgbAggError.Finalize(flags, rotatedAlphaWeightSq);
                                        }

                                        rgbIndexes[px] = rgbIndex;
                                        alphaIndexes[px] = alphaIndex;
                                    }

                                    ParallelMath::FloatCompFlag rgbBetter = ParallelMath::Less(errorRGB, bestRGBError);
                                    ParallelMath::FloatCompFlag alphaBetter = ParallelMath::Less(errorA, bestAlphaError);

                                    ParallelMath::Int16CompFlag rgbBetterInt16 = ParallelMath::FloatFlagToInt16(rgbBetter);
                                    ParallelMath::Int16CompFlag alphaBetterInt16 = ParallelMath::FloatFlagToInt16(alphaBetter);

                                    if (ParallelMath::AnySet(rgbBetterInt16))
                                    {
                                        bestRGBError = ParallelMath::Min(errorRGB, bestRGBError);

                                        for (int px = 0; px < 16; px++)
                                            ParallelMath::ConditionalSet(bestRGBIndexes[px], rgbBetterInt16, rgbIndexes[px]);

                                        for (int ep = 0; ep < 2; ep++)
                                        {
                                            for (int ch = 0; ch < 3; ch++)
                                                ParallelMath::ConditionalSet(bestEP[ep][ch], rgbBetterInt16, rgbEP[ep][ch]);
                                        }
                                    }

                                    if (ParallelMath::AnySet(alphaBetterInt16))
                                    {
                                        bestAlphaError = ParallelMath::Min(errorA, bestAlphaError);

                                        for (int px = 0; px < 16; px++)
                                            ParallelMath::ConditionalSet(bestAlphaIndexes[px], alphaBetterInt16, alphaIndexes[px]);

                                        for (int ep = 0; ep < 2; ep++)
                                            ParallelMath::ConditionalSet(bestEP[ep][3], alphaBetterInt16, alphaEP[ep]);
                                    }

                                    if (refine != numRefineRounds - 1)
                                    {
                                        rgbRefiner.GetRefinedEndpointsLDR(rgbEP, rtn);

                                        MUInt15 alphaEPTemp[2][1];
                                        alphaRefiner.GetRefinedEndpointsLDR(alphaEPTemp, rtn);

                                        for (int i = 0; i < 2; i++)
                                            alphaEP[i] = alphaEPTemp[i][0];
                                    }
                                }	// refine
                            } // tweak

                            MFloat combinedError = bestRGBError + bestAlphaError;

                            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(combinedError, work.m_error);
                            ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                            work.m_error = ParallelMath::Min(combinedError, work.m_error);

                            ParallelMath::ConditionalSet(work.m_mode, errorBetter16, ParallelMath::MakeUInt15(mode));
                            ParallelMath::ConditionalSet(work.m_u.m_isr.m_rotation, errorBetter16, ParallelMath::MakeUInt15(rotation));
                            ParallelMath::ConditionalSet(work.m_u.m_isr.m_indexSelector, errorBetter16, ParallelMath::MakeUInt15(indexSelector));

                            for (int px = 0; px < 16; px++)
                            {
                                ParallelMath::ConditionalSet(work.m_indexes[px], errorBetter16, indexSelector ? bestAlphaIndexes[px] : bestRGBIndexes[px]);
                                ParallelMath::ConditionalSet(work.m_indexes2[px], errorBetter16, indexSelector ? bestRGBIndexes[px] : bestAlphaIndexes[px]);
                            }

                            for (int ep = 0; ep < 2; ep++)
                                for (int ch = 0; ch < 4; ch++)
                                    ParallelMath::ConditionalSet(work.m_ep[0][ep][ch], errorBetter16, bestEP[ep][ch]);
                        }
                    }
                }
            }

            template<class T>
            static void Swap(T& a, T& b)
            {
                T temp = a;
                a = b;
                b = temp;
            }

            static void Pack(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, const float channelWeights[4], int numTweakRounds, int numRefineRounds)
            {
                MUInt15 pixels[16][4];
                MFloat floatPixels[16][4];

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 4; ch++)
                        ParallelMath::ConvertLDRInputs(inputs, px, ch, pixels[px][ch]);
                }

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 4; ch++)
                        floatPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
                }

                WorkInfo work;
                memset(&work, 0, sizeof(work));

                work.m_error = ParallelMath::MakeFloat(FLT_MAX);

                {
                    ParallelMath::RoundTowardNearestForScope rtn;
                    TrySinglePlane(flags, pixels, floatPixels, channelWeights, numTweakRounds, numRefineRounds, work, &rtn);
                    TryDualPlane(flags, pixels, floatPixels, channelWeights, numTweakRounds, numRefineRounds, work, &rtn);
                }

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    PackingVector pv;
                    pv.Init();

                    ParallelMath::ScalarUInt16 mode = ParallelMath::Extract(work.m_mode, block);
                    ParallelMath::ScalarUInt16 partition = ParallelMath::Extract(work.m_u.m_partition, block);
                    ParallelMath::ScalarUInt16 indexSelector = ParallelMath::Extract(work.m_u.m_isr.m_indexSelector, block);

                    const BC7Data::BC7ModeInfo& modeInfo = BC7Data::g_modes[mode];

                    ParallelMath::ScalarUInt16 indexes[16];
                    ParallelMath::ScalarUInt16 indexes2[16];
                    ParallelMath::ScalarUInt16 endPoints[3][2][4];

                    for (int i = 0; i < 16; i++)
                    {
                        indexes[i] = ParallelMath::Extract(work.m_indexes[i], block);
                        if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                            indexes2[i] = ParallelMath::Extract(work.m_indexes2[i], block);
                    }

                    for (int subset = 0; subset < 3; subset++)
                    {
                        for (int ep = 0; ep < 2; ep++)
                        {
                            for (int ch = 0; ch < 4; ch++)
                                endPoints[subset][ep][ch] = ParallelMath::Extract(work.m_ep[subset][ep][ch], block);
                        }
                    }

                    int fixups[3] = { 0, 0, 0 };

                    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                    {
                        bool flipRGB = ((indexes[0] & (1 << (modeInfo.m_indexBits - 1))) != 0);
                        bool flipAlpha = ((indexes2[0] & (1 << (modeInfo.m_alphaIndexBits - 1))) != 0);

                        if (flipRGB)
                        {
                            uint16_t highIndex = (1 << modeInfo.m_indexBits) - 1;
                            for (int px = 0; px < 16; px++)
                                indexes[px] = highIndex - indexes[px];
                        }

                        if (flipAlpha)
                        {
                            uint16_t highIndex = (1 << modeInfo.m_alphaIndexBits) - 1;
                            for (int px = 0; px < 16; px++)
                                indexes2[px] = highIndex - indexes2[px];
                        }

                        if (indexSelector)
                            Swap(flipRGB, flipAlpha);

                        if (flipRGB)
                        {
                            for (int ch = 0; ch < 3; ch++)
                                Swap(endPoints[0][0][ch], endPoints[0][1][ch]);
                        }
                        if (flipAlpha)
                            Swap(endPoints[0][0][3], endPoints[0][1][3]);

                    }
                    else
                    {
                        if (modeInfo.m_numSubsets == 2)
                            fixups[1] = BC7Data::g_fixupIndexes2[partition];
                        else if (modeInfo.m_numSubsets == 3)
                        {
                            fixups[1] = BC7Data::g_fixupIndexes3[partition][0];
                            fixups[2] = BC7Data::g_fixupIndexes3[partition][1];
                        }

                        bool flip[3] = { false, false, false };
                        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                            flip[subset] = ((indexes[fixups[subset]] & (1 << (modeInfo.m_indexBits - 1))) != 0);

                        if (flip[0] || flip[1] || flip[2])
                        {
                            uint16_t highIndex = (1 << modeInfo.m_indexBits) - 1;
                            for (int px = 0; px < 16; px++)
                            {
                                int subset = 0;
                                if (modeInfo.m_numSubsets == 2)
                                    subset = (BC7Data::g_partitionMap[partition] >> px) & 1;
                                else if (modeInfo.m_numSubsets == 3)
                                    subset = (BC7Data::g_partitionMap2[partition] >> (px * 2)) & 3;

                                if (flip[subset])
                                    indexes[px] = highIndex - indexes[px];
                            }

                            int maxCH = (modeInfo.m_alphaMode == BC7Data::AlphaMode_Combined) ? 4 : 3;
                            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                            {
                                if (flip[subset])
                                    for (int ch = 0; ch < maxCH; ch++)
                                        Swap(endPoints[subset][0][ch], endPoints[subset][1][ch]);
                            }
                        }
                    }

                    pv.Pack(static_cast<uint8_t>(1 << mode), mode + 1);

                    if (modeInfo.m_partitionBits)
                        pv.Pack(partition, modeInfo.m_partitionBits);

                    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                    {
                        ParallelMath::ScalarUInt16 rotation = ParallelMath::Extract(work.m_u.m_isr.m_rotation, block);
                        pv.Pack(rotation, 2);
                    }

                    if (modeInfo.m_hasIndexSelector)
                        pv.Pack(indexSelector, 1);

                    // Encode RGB
                    for (int ch = 0; ch < 3; ch++)
                    {
                        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                        {
                            for (int ep = 0; ep < 2; ep++)
                            {
                                ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][ch];
                                epPart >>= (8 - modeInfo.m_rgbBits);

                                pv.Pack(epPart, modeInfo.m_rgbBits);
                            }
                        }
                    }

                    // Encode alpha
                    if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                    {
                        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                        {
                            for (int ep = 0; ep < 2; ep++)
                            {
                                ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][3];
                                epPart >>= (8 - modeInfo.m_alphaBits);

                                pv.Pack(epPart, modeInfo.m_alphaBits);
                            }
                        }
                    }

                    // Encode parity bits
                    if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerSubset)
                    {
                        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                        {
                            ParallelMath::ScalarUInt16 epPart = endPoints[subset][0][0];
                            epPart >>= (7 - modeInfo.m_rgbBits);
                            epPart &= 1;

                            pv.Pack(epPart, 1);
                        }
                    }
                    else if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerEndpoint)
                    {
                        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                        {
                            for (int ep = 0; ep < 2; ep++)
                            {
                                ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][0];
                                epPart >>= (7 - modeInfo.m_rgbBits);
                                epPart &= 1;

                                pv.Pack(epPart, 1);
                            }
                        }
                    }

                    // Encode indexes
                    for (int px = 0; px < 16; px++)
                    {
                        int bits = modeInfo.m_indexBits;
                        if ((px == 0) || (px == fixups[1]) || (px == fixups[2]))
                            bits--;

                        pv.Pack(indexes[px], bits);
                    }

                    // Encode secondary indexes
                    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                    {
                        for (int px = 0; px < 16; px++)
                        {
                            int bits = modeInfo.m_alphaIndexBits;
                            if (px == 0)
                                bits--;

                            pv.Pack(indexes2[px], bits);
                        }
                    }

                    pv.Flush(packedBlocks);

                    packedBlocks += 16;
                }
            }

            static void UnpackOne(PixelBlockU8 &output, const uint8_t* packedBlock)
            {
                UnpackingVector pv;
                pv.Init(packedBlock);

                int mode = 8;
                for (int i = 0; i < 8; i++)
                {
                    if (pv.Unpack(1) == 1)
                    {
                        mode = i;
                        break;
                    }
                }

                if (mode > 7)
                {
                    for (int px = 0; px < 16; px++)
                        for (int ch = 0; ch < 4; ch++)
                            output.m_pixels[px][ch] = 0;

                    return;
                }

                const BC7Data::BC7ModeInfo &modeInfo = BC7Data::g_modes[mode];

                int partition = 0;
                if (modeInfo.m_partitionBits)
                    partition = pv.Unpack(modeInfo.m_partitionBits);

                int rotation = 0;
                if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                    rotation = pv.Unpack(2);

                int indexSelector = 0;
                if (modeInfo.m_hasIndexSelector)
                    indexSelector = pv.Unpack(1);

                // Resolve fixups
                int fixups[3] = { 0, 0, 0 };

                if (modeInfo.m_alphaMode != BC7Data::AlphaMode_Separate)
                {
                    if (modeInfo.m_numSubsets == 2)
                        fixups[1] = BC7Data::g_fixupIndexes2[partition];
                    else if (modeInfo.m_numSubsets == 3)
                    {
                        fixups[1] = BC7Data::g_fixupIndexes3[partition][0];
                        fixups[2] = BC7Data::g_fixupIndexes3[partition][1];
                    }
                }

                int endPoints[3][2][4];

                // Decode RGB
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                    {
                        for (int ep = 0; ep < 2; ep++)
                            endPoints[subset][ep][ch] = (pv.Unpack(modeInfo.m_rgbBits) << (8 - modeInfo.m_rgbBits));
                    }
                }

                // Decode alpha
                if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                {
                    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                    {
                        for (int ep = 0; ep < 2; ep++)
                            endPoints[subset][ep][3] = (pv.Unpack(modeInfo.m_alphaBits) << (8 - modeInfo.m_alphaBits));
                    }
                }
                else
                {
                    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                    {
                        for (int ep = 0; ep < 2; ep++)
                            endPoints[subset][ep][3] = 255;
                    }
                }

                int parityBits = 0;

                // Decode parity bits
                if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerSubset)
                {
                    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                    {
                        int p = pv.Unpack(1);

                        for (int ep = 0; ep < 2; ep++)
                        {
                            for (int ch = 0; ch < 3; ch++)
                                endPoints[subset][ep][ch] |= p << (7 - modeInfo.m_rgbBits);

                            if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                                endPoints[subset][ep][3] |= p << (7 - modeInfo.m_alphaBits);
                        }
                    }

                    parityBits = 1;
                }
                else if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerEndpoint)
                {
                    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                    {
                        for (int ep = 0; ep < 2; ep++)
                        {
                            int p = pv.Unpack(1);

                            for (int ch = 0; ch < 3; ch++)
                                endPoints[subset][ep][ch] |= p << (7 - modeInfo.m_rgbBits);

                            if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                                endPoints[subset][ep][3] |= p << (7 - modeInfo.m_alphaBits);
                        }
                    }

                    parityBits = 1;
                }

                // Fill endpoint bits
                for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                {
                    for (int ep = 0; ep < 2; ep++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                            endPoints[subset][ep][ch] |= (endPoints[subset][ep][ch] >> (modeInfo.m_rgbBits + parityBits));

                        if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                            endPoints[subset][ep][3] |= (endPoints[subset][ep][3] >> (modeInfo.m_alphaBits + parityBits));
                    }
                }

                int indexes[16];
                int indexes2[16];

                // Decode indexes
                for (int px = 0; px < 16; px++)
                {
                    int bits = modeInfo.m_indexBits;
                    if ((px == 0) || (px == fixups[1]) || (px == fixups[2]))
                        bits--;

                    indexes[px] = pv.Unpack(bits);
                }

                // Decode secondary indexes
                if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                {
                    for (int px = 0; px < 16; px++)
                    {
                        int bits = modeInfo.m_alphaIndexBits;
                        if (px == 0)
                            bits--;

                        indexes2[px] = pv.Unpack(bits);
                    }
                }
                else
                {
                    for (int px = 0; px < 16; px++)
                        indexes2[px] = 0;
                }

                const int *alphaWeights = BC7Data::g_weightTables[modeInfo.m_alphaIndexBits];
                const int *rgbWeights = BC7Data::g_weightTables[modeInfo.m_indexBits];

                // Decode each pixel
                for (int px = 0; px < 16; px++)
                {
                    int rgbWeight = 0;
                    int alphaWeight = 0;

                    int rgbIndex = indexes[px];

                    rgbWeight = rgbWeights[indexes[px]];

                    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Combined)
                        alphaWeight = rgbWeight;
                    else if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                        alphaWeight = alphaWeights[indexes2[px]];

                    if (indexSelector == 1)
                    {
                        int temp = rgbWeight;
                        rgbWeight = alphaWeight;
                        alphaWeight = temp;
                    }

                    int pixel[4] = { 0, 0, 0, 255 };

                    int subset = 0;

                    if (modeInfo.m_numSubsets == 2)
                        subset = (BC7Data::g_partitionMap[partition] >> px) & 1;
                    else if (modeInfo.m_numSubsets == 3)
                        subset = (BC7Data::g_partitionMap2[partition] >> (px * 2)) & 3;

                    for (int ch = 0; ch < 3; ch++)
                        pixel[ch] = ((64 - rgbWeight) * endPoints[subset][0][ch] + rgbWeight * endPoints[subset][1][ch] + 32) >> 6;

                    if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                        pixel[3] = ((64 - alphaWeight) * endPoints[subset][0][3] + alphaWeight * endPoints[subset][1][3] + 32) >> 6;

                    if (rotation != 0)
                    {
                        int ch = rotation - 1;
                        int temp = pixel[ch];
                        pixel[ch] = pixel[3];
                        pixel[3] = temp;
                    }

                    for (int ch = 0; ch < 4; ch++)
                        output.m_pixels[px][ch] = static_cast<uint8_t>(pixel[ch]);
                }
            }
        };

        class BC6HComputer
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::UInt31 MUInt31;

            static const int MaxTweakRounds = 4;
            static const int MaxRefineRounds = 3;

            static MSInt16 QuantizeSingleEndpointElementSigned(const MSInt16 &elem2CL, int precision, const ParallelMath::RoundUpForScope* ru)
            {
                assert(ParallelMath::AllSet(ParallelMath::Less(elem2CL, ParallelMath::MakeSInt16(31744))));
                assert(ParallelMath::AllSet(ParallelMath::Less(ParallelMath::MakeSInt16(-31744), elem2CL)));

                // Expand to full range
                ParallelMath::Int16CompFlag isNegative = ParallelMath::Less(elem2CL, ParallelMath::MakeSInt16(0));
                MUInt15 absElem = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Select(isNegative, ParallelMath::MakeSInt16(0) - elem2CL, elem2CL));

                absElem = ParallelMath::RightShift(ParallelMath::RoundAndConvertToU15(ParallelMath::ToFloat(absElem) * 32.0f / 31.0f, ru), 16 - precision);

                MSInt16 absElemS16 = ParallelMath::LosslessCast<MSInt16>::Cast(absElem);

                return ParallelMath::Select(isNegative, ParallelMath::MakeSInt16(0) - absElemS16, absElemS16);
            }

            static MUInt15 QuantizeSingleEndpointElementUnsigned(const MUInt15 &elem, int precision, const ParallelMath::RoundUpForScope* ru)
            {
                MUInt16 expandedElem = ParallelMath::RoundAndConvertToU16(ParallelMath::Min(ParallelMath::ToFloat(elem) * 64.0f / 31.0f, ParallelMath::MakeFloat(65535.0f)), ru);
                return ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(expandedElem, 16 - precision));
            }

            static void UnquantizeSingleEndpointElementSigned(const MSInt16 &comp, int precision, MSInt16 &outUnquantized, MSInt16 &outUnquantizedFinished2CL)
            {
                MSInt16 zero = ParallelMath::MakeSInt16(0);

                ParallelMath::Int16CompFlag negative = ParallelMath::Less(comp, zero);
                MUInt15 absComp = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Select(negative, MSInt16(zero - comp), comp));

                MSInt16 unq;
                MUInt15 absUnq;

                if (precision >= 16)
                {
                    unq = comp;
                    absUnq = absComp;
                }
                else
                {
                    MSInt16 maxCompMinusOne = ParallelMath::MakeSInt16(static_cast<int16_t>((1 << (precision - 1)) - 2));
                    ParallelMath::Int16CompFlag isZero = ParallelMath::Equal(comp, zero);
                    ParallelMath::Int16CompFlag isMax = ParallelMath::Less(maxCompMinusOne, comp);

                    absUnq = (absComp << (16 - precision)) + ParallelMath::MakeUInt15(static_cast<uint16_t>(0x4000 >> (precision - 1)));
                    ParallelMath::ConditionalSet(absUnq, isZero, ParallelMath::MakeUInt15(0));
                    ParallelMath::ConditionalSet(absUnq, isMax, ParallelMath::MakeUInt15(0x7fff));

                    unq = ParallelMath::ConditionalNegate(negative, ParallelMath::LosslessCast<MSInt16>::Cast(absUnq));
                }

                outUnquantized = unq;

                MUInt15 funq = ParallelMath::ToUInt15(ParallelMath::RightShift(ParallelMath::XMultiply(absUnq, ParallelMath::MakeUInt15(31)), 5));

                outUnquantizedFinished2CL = ParallelMath::ConditionalNegate(negative, ParallelMath::LosslessCast<MSInt16>::Cast(funq));
            }

            static void UnquantizeSingleEndpointElementUnsigned(const MUInt15 &comp, int precision, MUInt16 &outUnquantized, MUInt16 &outUnquantizedFinished)
            {
                MUInt16 unq = ParallelMath::LosslessCast<MUInt16>::Cast(comp);
                if (precision < 15)
                {
                    MUInt15 zero = ParallelMath::MakeUInt15(0);
                    MUInt15 maxCompMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << precision) - 2));

                    ParallelMath::Int16CompFlag isZero = ParallelMath::Equal(comp, zero);
                    ParallelMath::Int16CompFlag isMax = ParallelMath::Less(maxCompMinusOne, comp);

                    unq = (ParallelMath::LosslessCast<MUInt16>::Cast(comp) << (16 - precision)) + ParallelMath::MakeUInt16(static_cast<uint16_t>(0x8000 >> precision));

                    ParallelMath::ConditionalSet(unq, isZero, ParallelMath::MakeUInt16(0));
                    ParallelMath::ConditionalSet(unq, isMax, ParallelMath::MakeUInt16(0xffff));
                }

                outUnquantized = unq;
                outUnquantizedFinished = ParallelMath::ToUInt16(ParallelMath::RightShift(ParallelMath::XMultiply(unq, ParallelMath::MakeUInt15(31)), 6));
            }

            static void QuantizeEndpointsSigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                MSInt16 unquantizedEP[2][3];
                MSInt16 finishedUnquantizedEP[2][3];

                {
                    ParallelMath::RoundUpForScope ru;

                    for (int epi = 0; epi < 2; epi++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                        {
                            MSInt16 qee = QuantizeSingleEndpointElementSigned(endPoints[epi][ch], precision, &ru);
                            UnquantizeSingleEndpointElementSigned(qee, precision, unquantizedEP[epi][ch], finishedUnquantizedEP[epi][ch]);
                            quantizedEndPoints[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(qee);
                        }
                    }
                }

                indexSelector.Init(channelWeights, unquantizedEP, finishedUnquantizedEP, indexRange);
                indexSelector.InitHDR(indexRange, true, fastIndexing, channelWeights);

                MUInt15 halfRangeMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange / 2) - 1);

                MUInt15 index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixelsColorSpace[fixupIndex], rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[fixupIndex], rtn);

                ParallelMath::Int16CompFlag invert = ParallelMath::Less(halfRangeMinusOne, index);

                if (ParallelMath::AnySet(invert))
                {
                    ParallelMath::ConditionalSet(index, invert, MUInt15(ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange - 1)) - index));

                    indexSelector.ConditionalInvert(invert);

                    for (int ch = 0; ch < 3; ch++)
                    {
                        MAInt16 firstEP = quantizedEndPoints[0][ch];
                        MAInt16 secondEP = quantizedEndPoints[1][ch];

                        quantizedEndPoints[0][ch] = ParallelMath::Select(invert, secondEP, firstEP);
                        quantizedEndPoints[1][ch] = ParallelMath::Select(invert, firstEP, secondEP);
                    }
                }

                indexes[fixupIndex] = index;
            }

            static void QuantizeEndpointsUnsigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                MUInt16 unquantizedEP[2][3];
                MUInt16 finishedUnquantizedEP[2][3];

                {
                    ParallelMath::RoundUpForScope ru;

                    for (int epi = 0; epi < 2; epi++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                        {
                            MUInt15 qee = QuantizeSingleEndpointElementUnsigned(ParallelMath::LosslessCast<MUInt15>::Cast(endPoints[epi][ch]), precision, &ru);
                            UnquantizeSingleEndpointElementUnsigned(qee, precision, unquantizedEP[epi][ch], finishedUnquantizedEP[epi][ch]);
                            quantizedEndPoints[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(qee);
                        }
                    }
                }

                indexSelector.Init(channelWeights, unquantizedEP, finishedUnquantizedEP, indexRange);
                indexSelector.InitHDR(indexRange, false, fastIndexing, channelWeights);

                MUInt15 halfRangeMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange / 2) - 1);

                MUInt15 index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixelsColorSpace[fixupIndex], rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[fixupIndex], rtn);

                ParallelMath::Int16CompFlag invert = ParallelMath::Less(halfRangeMinusOne, index);

                if (ParallelMath::AnySet(invert))
                {
                    ParallelMath::ConditionalSet(index, invert, MUInt15(ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange - 1)) - index));

                    indexSelector.ConditionalInvert(invert);

                    for (int ch = 0; ch < 3; ch++)
                    {
                        MAInt16 firstEP = quantizedEndPoints[0][ch];
                        MAInt16 secondEP = quantizedEndPoints[1][ch];

                        quantizedEndPoints[0][ch] = ParallelMath::Select(invert, secondEP, firstEP);
                        quantizedEndPoints[1][ch] = ParallelMath::Select(invert, firstEP, secondEP);
                    }
                }

                indexes[fixupIndex] = index;
            }

            static void EvaluatePartitionedLegality(const MAInt16 ep0[2][3], const MAInt16 ep1[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][2][3], ParallelMath::Int16CompFlag& outIsLegal)
            {
                ParallelMath::Int16CompFlag allLegal = ParallelMath::MakeBoolInt16(true);

                MAInt16 aSignificantMask = ParallelMath::MakeAInt16(static_cast<int16_t>((1 << aPrec) - 1));

                for (int ch = 0; ch < 3; ch++)
                {
                    outEncodedEPs[0][0][ch] = ep0[0][ch];
                    outEncodedEPs[0][1][ch] = ep0[1][ch];
                    outEncodedEPs[1][0][ch] = ep1[0][ch];
                    outEncodedEPs[1][1][ch] = ep1[1][ch];

                    if (isTransformed)
                    {
                        for (int subset = 0; subset < 2; subset++)
                        {
                            for (int epi = 0; epi < 2; epi++)
                            {
                                if (epi == 0 && subset == 0)
                                    continue;

                                MAInt16 bReduced = (outEncodedEPs[subset][epi][ch] & aSignificantMask);

                                MSInt16 delta = ParallelMath::TruncateToPrecisionSigned(ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::AbstractSubtract(outEncodedEPs[subset][epi][ch], outEncodedEPs[0][0][ch])), bPrec[ch]);

                                outEncodedEPs[subset][epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(delta);

                                MAInt16 reconstructed = (ParallelMath::AbstractAdd(outEncodedEPs[subset][epi][ch], outEncodedEPs[0][0][ch]) & aSignificantMask);
                                allLegal = allLegal & ParallelMath::Equal(reconstructed, bReduced);
                            }
                        }
                    }

                    if (!ParallelMath::AnySet(allLegal))
                        break;
                }

                outIsLegal = allLegal;
            }

            static void EvaluateSingleLegality(const MAInt16 ep[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][3], ParallelMath::Int16CompFlag& outIsLegal)
            {
                ParallelMath::Int16CompFlag allLegal = ParallelMath::MakeBoolInt16(true);

                MAInt16 aSignificantMask = ParallelMath::MakeAInt16(static_cast<int16_t>((1 << aPrec) - 1));

                for (int ch = 0; ch < 3; ch++)
                {
                    outEncodedEPs[0][ch] = ep[0][ch];
                    outEncodedEPs[1][ch] = ep[1][ch];

                    if (isTransformed)
                    {
                        MAInt16 bReduced = (outEncodedEPs[1][ch] & aSignificantMask);

                        MSInt16 delta = ParallelMath::TruncateToPrecisionSigned(ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::AbstractSubtract(outEncodedEPs[1][ch], outEncodedEPs[0][ch])), bPrec[ch]);

                        outEncodedEPs[1][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(delta);

                        MAInt16 reconstructed = (ParallelMath::AbstractAdd(outEncodedEPs[1][ch], outEncodedEPs[0][ch]) & aSignificantMask);
                        allLegal = allLegal & ParallelMath::Equal(reconstructed, bReduced);
                    }
                }

                outIsLegal = allLegal;
            }

            static void Pack(uint32_t flags, const PixelBlockF16* inputs, uint8_t* packedBlocks, const float channelWeights[4], bool isSigned, int numTweakRounds, int numRefineRounds)
            {
                if (numTweakRounds < 1)
                    numTweakRounds = 1;
                else if (numTweakRounds > MaxTweakRounds)
                    numTweakRounds = MaxTweakRounds;

                if (numRefineRounds < 1)
                    numRefineRounds = 1;
                else if (numRefineRounds > MaxRefineRounds)
                    numRefineRounds = MaxRefineRounds;

                bool fastIndexing = ((flags & cvtt::Flags::BC6H_FastIndexing) != 0);
                float channelWeightsSq[3];

                ParallelMath::RoundTowardNearestForScope rtn;

                MSInt16 pixels[16][3];
                MFloat floatPixels2CL[16][3];
                MFloat floatPixelsLinearWeighted[16][3];

                MSInt16 low15Bits = ParallelMath::MakeSInt16(32767);

                for (int ch = 0; ch < 3; ch++)
                    channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        MSInt16 pixelValue;
                        ParallelMath::ConvertHDRInputs(inputs, px, ch, pixelValue);

                        // Convert from sign+magnitude to 2CL
                        if (isSigned)
                        {
                            ParallelMath::Int16CompFlag negative = ParallelMath::Less(pixelValue, ParallelMath::MakeSInt16(0));
                            MSInt16 magnitude = (pixelValue & low15Bits);
                            ParallelMath::ConditionalSet(pixelValue, negative, ParallelMath::MakeSInt16(0) - magnitude);
                            pixelValue = ParallelMath::Max(pixelValue, ParallelMath::MakeSInt16(-31743));
                        }
                        else
                            pixelValue = ParallelMath::Max(pixelValue, ParallelMath::MakeSInt16(0));

                        pixelValue = ParallelMath::Min(pixelValue, ParallelMath::MakeSInt16(31743));

                        pixels[px][ch] = pixelValue;
                        floatPixels2CL[px][ch] = ParallelMath::ToFloat(pixelValue);
                        floatPixelsLinearWeighted[px][ch] = ParallelMath::TwosCLHalfToFloat(pixelValue) * channelWeights[ch];
                    }
                }

                MFloat preWeightedPixels[16][3];

                BCCommon::PreWeightPixelsHDR<3>(preWeightedPixels, pixels, channelWeights);

                MAInt16 bestEndPoints[2][2][3];
                MUInt15 bestIndexes[16];
                MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
                MUInt15 bestMode = ParallelMath::MakeUInt15(0);
                MUInt15 bestPartition = ParallelMath::MakeUInt15(0);

                for (int px = 0; px < 16; px++)
                    bestIndexes[px] = ParallelMath::MakeUInt15(0);

                for (int subset = 0; subset < 2; subset++)
                    for (int epi = 0; epi < 2; epi++)
                        for (int ch = 0; ch < 3; ch++)
                            bestEndPoints[subset][epi][ch] = ParallelMath::MakeAInt16(0);

                UnfinishedEndpoints<3> partitionedUFEP[32][2];
                UnfinishedEndpoints<3> singleUFEP;

                // Generate UFEP for partitions
                for (int p = 0; p < 32; p++)
                {
                    int partitionMask = BC7Data::g_partitionMap[p];

                    EndpointSelector<3, 8> epSelectors[2];

                    for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
                    {
                        for (int px = 0; px < 16; px++)
                        {
                            int subset = (partitionMask >> px) & 1;
                            epSelectors[subset].ContributePass(preWeightedPixels[px], pass, ParallelMath::MakeFloat(1.0f));
                        }

                        for (int subset = 0; subset < 2; subset++)
                            epSelectors[subset].FinishPass(pass);
                    }

                    for (int subset = 0; subset < 2; subset++)
                        partitionedUFEP[p][subset] = epSelectors[subset].GetEndpoints(channelWeights);
                }

                // Generate UFEP for single
                {
                    EndpointSelector<3, 8> epSelector;

                    for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
                    {
                        for (int px = 0; px < 16; px++)
                            epSelector.ContributePass(preWeightedPixels[px], pass, ParallelMath::MakeFloat(1.0f));

                        epSelector.FinishPass(pass);
                    }

                    singleUFEP = epSelector.GetEndpoints(channelWeights);
                }

                for (int partitionedInt = 0; partitionedInt < 2; partitionedInt++)
                {
                    bool partitioned = (partitionedInt == 1);

                    for (int aPrec = BC7Data::g_maxHDRPrecision; aPrec >= 0; aPrec--)
                    {
                        if (!BC7Data::g_hdrModesExistForPrecision[partitionedInt][aPrec])
                            continue;

                        int numPartitions = partitioned ? 32 : 1;
                        int numSubsets = partitioned ? 2 : 1;
                        int indexBits = partitioned ? 3 : 4;
                        int indexRange = (1 << indexBits);

                        for (int p = 0; p < numPartitions; p++)
                        {
                            int partitionMask = partitioned ? BC7Data::g_partitionMap[p] : 0;

                            const int MaxMetaRounds = MaxTweakRounds * MaxRefineRounds;

                            MAInt16 metaEndPointsQuantized[MaxMetaRounds][2][2][3];
                            MUInt15 metaIndexes[MaxMetaRounds][16];
                            MFloat metaError[MaxMetaRounds][2];

                            bool roundValid[MaxMetaRounds][2];

                            for (int r = 0; r < MaxMetaRounds; r++)
                                for (int subset = 0; subset < 2; subset++)
                                    roundValid[r][subset] = true;

                            for (int subset = 0; subset < numSubsets; subset++)
                            {
                                for (int tweak = 0; tweak < MaxTweakRounds; tweak++)
                                {
                                    EndpointRefiner<3> refiners[2];

                                    bool abortRemainingRefines = false;
                                    for (int refinePass = 0; refinePass < MaxRefineRounds; refinePass++)
                                    {
                                        int metaRound = tweak * MaxRefineRounds + refinePass;

                                        if (tweak >= numTweakRounds || refinePass >= numRefineRounds)
                                            abortRemainingRefines = true;

                                        if (abortRemainingRefines)
                                        {
                                            roundValid[metaRound][subset] = false;
                                            continue;
                                        }

                                        MAInt16(&mrQuantizedEndPoints)[2][2][3] = metaEndPointsQuantized[metaRound];
                                        MUInt15(&mrIndexes)[16] = metaIndexes[metaRound];

                                        MSInt16 endPointsColorSpace[2][3];

                                        if (refinePass == 0)
                                        {
                                            UnfinishedEndpoints<3> ufep = partitioned ? partitionedUFEP[p][subset] : singleUFEP;

                                            if (isSigned)
                                                ufep.FinishHDRSigned(tweak, indexRange, endPointsColorSpace[0], endPointsColorSpace[1], &rtn);
                                            else
                                                ufep.FinishHDRUnsigned(tweak, indexRange, endPointsColorSpace[0], endPointsColorSpace[1], &rtn);
                                        }
                                        else
                                            refiners[subset].GetRefinedEndpointsHDR(endPointsColorSpace, isSigned, &rtn);

                                        refiners[subset].Init(indexRange, channelWeights);

                                        int fixupIndex = (subset == 0) ? 0 : BC7Data::g_fixupIndexes2[p];

                                        IndexSelectorHDR<3> indexSelector;
                                        if (isSigned)
                                            QuantizeEndpointsSigned(endPointsColorSpace, floatPixels2CL, floatPixelsLinearWeighted, mrQuantizedEndPoints[subset], mrIndexes, indexSelector, fixupIndex, aPrec, indexRange, channelWeights, fastIndexing, &rtn);
                                        else
                                            QuantizeEndpointsUnsigned(endPointsColorSpace, floatPixels2CL, floatPixelsLinearWeighted, mrQuantizedEndPoints[subset], mrIndexes, indexSelector, fixupIndex, aPrec, indexRange, channelWeights, fastIndexing, &rtn);

                                        if (metaRound > 0)
                                        {
                                            ParallelMath::Int16CompFlag anySame = ParallelMath::MakeBoolInt16(false);

                                            for (int prevRound = 0; prevRound < metaRound; prevRound++)
                                            {
                                                MAInt16(&prevRoundEPs)[2][3] = metaEndPointsQuantized[prevRound][subset];

                                                ParallelMath::Int16CompFlag same = ParallelMath::MakeBoolInt16(true);

                                                for (int epi = 0; epi < 2; epi++)
                                                    for (int ch = 0; ch < 3; ch++)
                                                        same = (same & ParallelMath::Equal(prevRoundEPs[epi][ch], mrQuantizedEndPoints[subset][epi][ch]));

                                                anySame = (anySame | same);
                                                if (ParallelMath::AllSet(anySame))
                                                    break;
                                            }

                                            if (ParallelMath::AllSet(anySame))
                                            {
                                                roundValid[metaRound][subset] = false;
                                                continue;
                                            }
                                        }

                                        MFloat subsetError = ParallelMath::MakeFloatZero();

                                        {
                                            for (int px = 0; px < 16; px++)
                                            {
                                                if (subset != ((partitionMask >> px) & 1))
                                                    continue;

                                                MUInt15 index;
                                                if (px == fixupIndex)
                                                    index = mrIndexes[px];
                                                else
                                                {
                                                    index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixels2CL[px], &rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[px], &rtn);
                                                    mrIndexes[px] = index;
                                                }

                                                MSInt16 reconstructed[3];
                                                if (isSigned)
                                                    indexSelector.ReconstructHDRSigned(mrIndexes[px], reconstructed);
                                                else
                                                    indexSelector.ReconstructHDRUnsigned(mrIndexes[px], reconstructed);

                                                subsetError = subsetError + (fastIndexing ? BCCommon::ComputeErrorHDRFast<3>(flags, reconstructed, pixels[px], channelWeightsSq) : BCCommon::ComputeErrorHDRSlow<3>(flags, reconstructed, pixels[px], channelWeightsSq));

                                                if (refinePass != numRefineRounds - 1)
                                                    refiners[subset].ContributeUnweightedPW(preWeightedPixels[px], index);
                                            }
                                        }

                                        metaError[metaRound][subset] = subsetError;
                                    }
                                }
                            }

                            // Now we have a bunch of attempts, but not all of them will fit in the delta coding scheme
                            int numMeta1 = partitioned ? MaxMetaRounds : 1;
                            for (int meta0 = 0; meta0 < MaxMetaRounds; meta0++)
                            {
                                if (!roundValid[meta0][0])
                                    continue;

                                for (int meta1 = 0; meta1 < numMeta1; meta1++)
                                {
                                    MFloat combinedError = metaError[meta0][0];
                                    if (partitioned)
                                    {
                                        if (!roundValid[meta1][1])
                                            continue;

                                        combinedError = combinedError + metaError[meta1][1];
                                    }

                                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(combinedError, bestError);
                                    if (!ParallelMath::AnySet(errorBetter))
                                        continue;

                                    ParallelMath::Int16CompFlag needsCommit = ParallelMath::FloatFlagToInt16(errorBetter);

                                    // Figure out if this is encodable
                                    for (int mode = 0; mode < BC7Data::g_numHDRModes; mode++)
                                    {
                                        const BC7Data::BC6HModeInfo &modeInfo = BC7Data::g_hdrModes[mode];

                                        if (modeInfo.m_partitioned != partitioned || modeInfo.m_aPrec != aPrec)
                                            continue;

                                        MAInt16 encodedEPs[2][2][3];
                                        ParallelMath::Int16CompFlag isLegal;
                                        if (partitioned)
                                            EvaluatePartitionedLegality(metaEndPointsQuantized[meta0][0], metaEndPointsQuantized[meta1][1], modeInfo.m_aPrec, modeInfo.m_bPrec, modeInfo.m_transformed, encodedEPs, isLegal);
                                        else
                                            EvaluateSingleLegality(metaEndPointsQuantized[meta0][0], modeInfo.m_aPrec, modeInfo.m_bPrec, modeInfo.m_transformed, encodedEPs[0], isLegal);

                                        ParallelMath::Int16CompFlag isLegalAndBetter = (ParallelMath::FloatFlagToInt16(errorBetter) & isLegal);
                                        if (!ParallelMath::AnySet(isLegalAndBetter))
                                            continue;

                                        ParallelMath::FloatCompFlag isLegalAndBetterFloat = ParallelMath::Int16FlagToFloat(isLegalAndBetter);

                                        ParallelMath::ConditionalSet(bestError, isLegalAndBetterFloat, combinedError);
                                        ParallelMath::ConditionalSet(bestMode, isLegalAndBetter, ParallelMath::MakeUInt15(static_cast<uint16_t>(mode)));
                                        ParallelMath::ConditionalSet(bestPartition, isLegalAndBetter, ParallelMath::MakeUInt15(static_cast<uint16_t>(p)));

                                        for (int subset = 0; subset < numSubsets; subset++)
                                        {
                                            for (int epi = 0; epi < 2; epi++)
                                            {
                                                for (int ch = 0; ch < 3; ch++)
                                                    ParallelMath::ConditionalSet(bestEndPoints[subset][epi][ch], isLegalAndBetter, encodedEPs[subset][epi][ch]);
                                            }
                                        }

                                        for (int px = 0; px < 16; px++)
                                        {
                                            int subset = ((partitionMask >> px) & 1);
                                            if (subset == 0)
                                                ParallelMath::ConditionalSet(bestIndexes[px], isLegalAndBetter, metaIndexes[meta0][px]);
                                            else
                                                ParallelMath::ConditionalSet(bestIndexes[px], isLegalAndBetter, metaIndexes[meta1][px]);
                                        }

                                        needsCommit = ParallelMath::AndNot(needsCommit, isLegalAndBetter);
                                        if (!ParallelMath::AnySet(needsCommit))
                                            break;
                                    }
                                }
                            }
                        }
                    }
                }

                // At this point, everything should be set
                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    ParallelMath::ScalarUInt16 mode = ParallelMath::Extract(bestMode, block);
                    ParallelMath::ScalarUInt16 partition = ParallelMath::Extract(bestPartition, block);
                    int32_t eps[2][2][3];
                    ParallelMath::ScalarUInt16 indexes[16];

                    const BC7Data::BC6HModeInfo& modeInfo = BC7Data::g_hdrModes[mode];

                    const BC6HData::ModeDescriptor* desc = BC6HData::g_modeDescriptors[mode];

                    const size_t headerBits = modeInfo.m_partitioned ? 82 : 65;

                    for (int subset = 0; subset < 2; subset++)
                    {
                        for (int epi = 0; epi < 2; epi++)
                        {
                            for (int ch = 0; ch < 3; ch++)
                                eps[subset][epi][ch] = ParallelMath::Extract(bestEndPoints[subset][epi][ch], block);
                        }
                    }

                    for (int px = 0; px < 16; px++)
                        indexes[px] = ParallelMath::Extract(bestIndexes[px], block);

                    uint16_t modeID = modeInfo.m_modeID;

                    PackingVector pv;
                    pv.Init();

                    for (size_t i = 0; i < headerBits; i++)
                    {
                        int32_t codedValue = 0;
                        switch (desc[i].m_eField)
                        {
                        case BC6HData::M:  codedValue = modeID; break;
                        case BC6HData::D:  codedValue = partition; break;
                        case BC6HData::RW: codedValue = eps[0][0][0]; break;
                        case BC6HData::RX: codedValue = eps[0][1][0]; break;
                        case BC6HData::RY: codedValue = eps[1][0][0]; break;
                        case BC6HData::RZ: codedValue = eps[1][1][0]; break;
                        case BC6HData::GW: codedValue = eps[0][0][1]; break;
                        case BC6HData::GX: codedValue = eps[0][1][1]; break;
                        case BC6HData::GY: codedValue = eps[1][0][1]; break;
                        case BC6HData::GZ: codedValue = eps[1][1][1]; break;
                        case BC6HData::BW: codedValue = eps[0][0][2]; break;
                        case BC6HData::BX: codedValue = eps[0][1][2]; break;
                        case BC6HData::BY: codedValue = eps[1][0][2]; break;
                        case BC6HData::BZ: codedValue = eps[1][1][2]; break;
                        default: assert(false); break;
                        }

                        pv.Pack(static_cast<uint16_t>((codedValue >> desc[i].m_uBit) & 1), 1);
                    }

                    int fixupIndex1 = 0;
                    int indexBits = 4;
                    if (modeInfo.m_partitioned)
                    {
                        fixupIndex1 = BC7Data::g_fixupIndexes2[partition];
                        indexBits = 3;
                    }

                    for (int px = 0; px < 16; px++)
                    {
                        ParallelMath::ScalarUInt16 index = ParallelMath::Extract(bestIndexes[px], block);
                        if (px == 0 || px == fixupIndex1)
                            pv.Pack(index, indexBits - 1);
                        else
                            pv.Pack(index, indexBits);
                    }

                    pv.Flush(packedBlocks + 16 * block);
                }
            }

            static void SignExtendSingle(int &v, int bits)
            {
                if (v & (1 << (bits - 1)))
                    v |= -(1 << bits);
            }

            static void UnpackOne(PixelBlockF16 &output, const uint8_t *pBC, bool isSigned)
            {
                UnpackingVector pv;
                pv.Init(pBC);

                int numModeBits = 2;
                int modeBits = pv.Unpack(2);
                if (modeBits != 0 && modeBits != 1)
                {
                    modeBits |= pv.Unpack(3) << 2;
                    numModeBits += 3;
                }

                int mode = -1;
                for (int possibleMode = 0; possibleMode < BC7Data::g_numHDRModes; possibleMode++)
                {
                    if (BC7Data::g_hdrModes[possibleMode].m_modeID == modeBits)
                    {
                        mode = possibleMode;
                        break;
                    }
                }

                if (mode < 0)
                {
                    for (int px = 0; px < 16; px++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                            output.m_pixels[px][ch] = 0;
                        output.m_pixels[px][3] = 0x3c00;	// 1.0
                    }
                    return;
                }

                const BC7Data::BC6HModeInfo& modeInfo = BC7Data::g_hdrModes[mode];
                const size_t headerBits = modeInfo.m_partitioned ? 82 : 65;
                const BC6HData::ModeDescriptor* desc = BC6HData::g_modeDescriptors[mode];

                int32_t partition = 0;
                int32_t eps[2][2][3];

                for (int subset = 0; subset < 2; subset++)
                    for (int epi = 0; epi < 2; epi++)
                        for (int ch = 0; ch < 3; ch++)
                            eps[subset][epi][ch] = 0;

                for (size_t i = numModeBits; i < headerBits; i++)
                {
                    int32_t *pCodedValue = NULL;

                    switch (desc[i].m_eField)
                    {
                    case BC6HData::D:  pCodedValue = &partition; break;
                    case BC6HData::RW: pCodedValue = &eps[0][0][0]; break;
                    case BC6HData::RX: pCodedValue = &eps[0][1][0]; break;
                    case BC6HData::RY: pCodedValue = &eps[1][0][0]; break;
                    case BC6HData::RZ: pCodedValue = &eps[1][1][0]; break;
                    case BC6HData::GW: pCodedValue = &eps[0][0][1]; break;
                    case BC6HData::GX: pCodedValue = &eps[0][1][1]; break;
                    case BC6HData::GY: pCodedValue = &eps[1][0][1]; break;
                    case BC6HData::GZ: pCodedValue = &eps[1][1][1]; break;
                    case BC6HData::BW: pCodedValue = &eps[0][0][2]; break;
                    case BC6HData::BX: pCodedValue = &eps[0][1][2]; break;
                    case BC6HData::BY: pCodedValue = &eps[1][0][2]; break;
                    case BC6HData::BZ: pCodedValue = &eps[1][1][2]; break;
                    default: assert(false); break;
                    }

                    (*pCodedValue) |= pv.Unpack(1) << desc[i].m_uBit;
                }


                uint16_t modeID = modeInfo.m_modeID;

                int fixupIndex1 = 0;
                int indexBits = 4;
                int numSubsets = 1;
                if (modeInfo.m_partitioned)
                {
                    fixupIndex1 = BC7Data::g_fixupIndexes2[partition];
                    indexBits = 3;
                    numSubsets = 2;
                }

                int indexes[16];
                for (int px = 0; px < 16; px++)
                {
                    if (px == 0 || px == fixupIndex1)
                        indexes[px] = pv.Unpack(indexBits - 1);
                    else
                        indexes[px] = pv.Unpack(indexBits);
                }

                if (modeInfo.m_partitioned)
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        if (isSigned)
                            SignExtendSingle(eps[0][0][ch], modeInfo.m_aPrec);
                        if (modeInfo.m_transformed || isSigned)
                        {
                            SignExtendSingle(eps[0][1][ch], modeInfo.m_bPrec[ch]);
                            SignExtendSingle(eps[1][0][ch], modeInfo.m_bPrec[ch]);
                            SignExtendSingle(eps[1][1][ch], modeInfo.m_bPrec[ch]);
                        }
                    }
                }
                else
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        if (isSigned)
                            SignExtendSingle(eps[0][0][ch], modeInfo.m_aPrec);
                        if (modeInfo.m_transformed || isSigned)
                            SignExtendSingle(eps[0][1][ch], modeInfo.m_bPrec[ch]);
                    }
                }

                int aPrec = modeInfo.m_aPrec;

                if (modeInfo.m_transformed)
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        int wrapMask = (1 << aPrec) - 1;

                        eps[0][1][ch] = ((eps[0][0][ch] + eps[0][1][ch]) & wrapMask);
                        if (isSigned)
                            SignExtendSingle(eps[0][1][ch], aPrec);

                        if (modeInfo.m_partitioned)
                        {
                            eps[1][0][ch] = ((eps[0][0][ch] + eps[1][0][ch]) & wrapMask);
                            eps[1][1][ch] = ((eps[0][0][ch] + eps[1][1][ch]) & wrapMask);

                            if (isSigned)
                            {
                                SignExtendSingle(eps[1][0][ch], aPrec);
                                SignExtendSingle(eps[1][1][ch], aPrec);
                            }
                        }
                    }
                }

                // Unquantize endpoints
                for (int subset = 0; subset < numSubsets; subset++)
                {
                    for (int epi = 0; epi < 2; epi++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                        {
                            int &v = eps[subset][epi][ch];

                            if (isSigned)
                            {
                                if (aPrec >= 16)
                                {
                                    // Nothing
                                }
                                else
                                {
                                    bool s = false;
                                    int comp = v;
                                    if (v < 0)
                                    {
                                        s = true;
                                        comp = -comp;
                                    }

                                    int unq = 0;
                                    if (comp == 0)
                                        unq = 0;
                                    else if (comp >= ((1 << (aPrec - 1)) - 1))
                                        unq = 0x7fff;
                                    else
                                        unq = ((comp << 15) + 0x4000) >> (aPrec - 1);

                                    if (s)
                                        unq = -unq;

                                    v = unq;
                                }
                            }
                            else
                            {
                                if (aPrec >= 15)
                                {
                                    // Nothing
                                }
                                else if (v == 0)
                                {
                                    // Nothing
                                }
                                else if (v == ((1 << aPrec) - 1))
                                    v = 0xffff;
                                else
                                    v = ((v << 16) + 0x8000) >> aPrec;
                            }
                        }
                    }
                }

                const int *weights = BC7Data::g_weightTables[indexBits];

                for (int px = 0; px < 16; px++)
                {
                    int subset = 0;
                    if (modeInfo.m_partitioned)
                        subset = (BC7Data::g_partitionMap[partition] >> px) & 1;

                    int w = weights[indexes[px]];
                    for (int ch = 0; ch < 3; ch++)
                    {
                        int comp = ((64 - w) * eps[subset][0][ch] + w * eps[subset][1][ch] + 32) >> 6;

                        if (isSigned)
                        {
                            if (comp < 0)
                                comp = -(((-comp) * 31) >> 5);
                            else
                                comp = (comp * 31) >> 5;

                            int s = 0;
                            if (comp < 0)
                            {
                                s = 0x8000;
                                comp = -comp;
                            }

                            output.m_pixels[px][ch] = static_cast<uint16_t>(s | comp);
                        }
                        else
                        {
                            comp = (comp * 31) >> 6;
                            output.m_pixels[px][ch] = static_cast<uint16_t>(comp);
                        }
                    }
                    output.m_pixels[px][3] = 0x3c00;	// 1.0
                }
            }
        };

        namespace S3TCSingleColorTables
        {
            struct SingleColorTableEntry
            {
                uint8_t m_min;
                uint8_t m_max;
                uint8_t m_actualColor;
                uint8_t m_span;
            };

            SingleColorTableEntry g_singleColor5_3[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 8, 0, 2, 8 }, { 8, 0, 2, 8 }, { 0, 8, 5, 8 }, { 0, 8, 5, 8 }, { 0, 8, 5, 8 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 16, 8, 10, 8 }, { 33, 0, 11, 33 }, { 8, 16, 13, 8 }, { 8, 16, 13, 8 }, { 8, 16, 13, 8 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 24, 16, 18, 8 }, { 41, 8, 19, 33 }, { 16, 24, 21, 8 }, { 16, 24, 21, 8 }, { 0, 33, 22, 33 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 33, 24, 27, 9 }, { 33, 24, 27, 9 }, { 33, 24, 27, 9 }, { 41, 24, 29, 17 }, { 24, 33, 30, 9 }, { 24, 33, 30, 9 },
                { 16, 41, 32, 25 }, { 33, 33, 33, 0 }, { 33, 33, 33, 0 }, { 41, 33, 35, 8 }, { 41, 33, 35, 8 }, { 33, 41, 38, 8 }, { 33, 41, 38, 8 }, { 33, 41, 38, 8 },
                { 24, 49, 40, 25 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 49, 41, 43, 8 }, { 66, 33, 44, 33 }, { 41, 49, 46, 8 }, { 41, 49, 46, 8 }, { 41, 49, 46, 8 },
                { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 57, 49, 51, 8 }, { 74, 41, 52, 33 }, { 49, 57, 54, 8 }, { 49, 57, 54, 8 }, { 33, 66, 55, 33 },
                { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 66, 57, 60, 9 }, { 66, 57, 60, 9 }, { 66, 57, 60, 9 }, { 74, 57, 62, 17 }, { 57, 66, 63, 9 },
                { 57, 66, 63, 9 }, { 49, 74, 65, 25 }, { 66, 66, 66, 0 }, { 66, 66, 66, 0 }, { 74, 66, 68, 8 }, { 74, 66, 68, 8 }, { 66, 74, 71, 8 }, { 66, 74, 71, 8 },
                { 66, 74, 71, 8 }, { 57, 82, 73, 25 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 82, 74, 76, 8 }, { 99, 66, 77, 33 }, { 74, 82, 79, 8 }, { 74, 82, 79, 8 },
                { 74, 82, 79, 8 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 90, 82, 84, 8 }, { 107, 74, 85, 33 }, { 82, 90, 87, 8 }, { 82, 90, 87, 8 },
                { 66, 99, 88, 33 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 99, 90, 93, 9 }, { 99, 90, 93, 9 }, { 99, 90, 93, 9 }, { 107, 90, 95, 17 },
                { 90, 99, 96, 9 }, { 90, 99, 96, 9 }, { 82, 107, 98, 25 }, { 99, 99, 99, 0 }, { 99, 99, 99, 0 }, { 107, 99, 101, 8 }, { 107, 99, 101, 8 }, { 99, 107, 104, 8 },
                { 99, 107, 104, 8 }, { 99, 107, 104, 8 }, { 90, 115, 106, 25 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 115, 107, 109, 8 }, { 132, 99, 110, 33 }, { 107, 115, 112, 8 },
                { 107, 115, 112, 8 }, { 107, 115, 112, 8 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 123, 115, 117, 8 }, { 140, 107, 118, 33 }, { 115, 123, 120, 8 },
                { 115, 123, 120, 8 }, { 99, 132, 121, 33 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 132, 123, 126, 9 }, { 132, 123, 126, 9 }, { 132, 123, 126, 9 },
                { 140, 123, 128, 17 }, { 123, 132, 129, 9 }, { 123, 132, 129, 9 }, { 115, 140, 131, 25 }, { 132, 132, 132, 0 }, { 132, 132, 132, 0 }, { 140, 132, 134, 8 }, { 140, 132, 134, 8 },
                { 132, 140, 137, 8 }, { 132, 140, 137, 8 }, { 132, 140, 137, 8 }, { 123, 148, 139, 25 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 148, 140, 142, 8 }, { 165, 132, 143, 33 },
                { 140, 148, 145, 8 }, { 140, 148, 145, 8 }, { 140, 148, 145, 8 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 156, 148, 150, 8 }, { 173, 140, 151, 33 },
                { 148, 156, 153, 8 }, { 148, 156, 153, 8 }, { 132, 165, 154, 33 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 165, 156, 159, 9 }, { 165, 156, 159, 9 },
                { 165, 156, 159, 9 }, { 173, 156, 161, 17 }, { 156, 165, 162, 9 }, { 156, 165, 162, 9 }, { 148, 173, 164, 25 }, { 165, 165, 165, 0 }, { 165, 165, 165, 0 }, { 173, 165, 167, 8 },
                { 173, 165, 167, 8 }, { 165, 173, 170, 8 }, { 165, 173, 170, 8 }, { 165, 173, 170, 8 }, { 156, 181, 172, 25 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 }, { 181, 173, 175, 8 },
                { 198, 165, 176, 33 }, { 173, 181, 178, 8 }, { 173, 181, 178, 8 }, { 173, 181, 178, 8 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 189, 181, 183, 8 },
                { 206, 173, 184, 33 }, { 181, 189, 186, 8 }, { 181, 189, 186, 8 }, { 165, 198, 187, 33 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 198, 189, 192, 9 },
                { 198, 189, 192, 9 }, { 198, 189, 192, 9 }, { 206, 189, 194, 17 }, { 189, 198, 195, 9 }, { 189, 198, 195, 9 }, { 181, 206, 197, 25 }, { 198, 198, 198, 0 }, { 198, 198, 198, 0 },
                { 206, 198, 200, 8 }, { 206, 198, 200, 8 }, { 198, 206, 203, 8 }, { 198, 206, 203, 8 }, { 198, 206, 203, 8 }, { 189, 214, 205, 25 }, { 206, 206, 206, 0 }, { 206, 206, 206, 0 },
                { 214, 206, 208, 8 }, { 231, 198, 209, 33 }, { 206, 214, 211, 8 }, { 206, 214, 211, 8 }, { 206, 214, 211, 8 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 },
                { 222, 214, 216, 8 }, { 239, 206, 217, 33 }, { 214, 222, 219, 8 }, { 214, 222, 219, 8 }, { 198, 231, 220, 33 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 },
                { 231, 222, 225, 9 }, { 231, 222, 225, 9 }, { 231, 222, 225, 9 }, { 239, 222, 227, 17 }, { 222, 231, 228, 9 }, { 222, 231, 228, 9 }, { 214, 239, 230, 25 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 239, 231, 233, 8 }, { 239, 231, 233, 8 }, { 231, 239, 236, 8 }, { 231, 239, 236, 8 }, { 231, 239, 236, 8 }, { 222, 247, 238, 25 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 247, 239, 241, 8 }, { 247, 239, 241, 8 }, { 239, 247, 244, 8 }, { 239, 247, 244, 8 }, { 239, 247, 244, 8 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 255, 247, 249, 8 }, { 255, 247, 249, 8 }, { 247, 255, 252, 8 }, { 247, 255, 252, 8 }, { 247, 255, 252, 8 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor6_3[256] =
            {
                { 0, 0, 0, 0 }, { 4, 0, 1, 4 }, { 0, 4, 2, 4 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 8, 4, 5, 4 }, { 4, 8, 6, 4 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 12, 8, 9, 4 }, { 8, 12, 10, 4 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 16, 12, 13, 4 }, { 12, 16, 14, 4 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 20, 16, 17, 4 }, { 16, 20, 18, 4 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 24, 20, 21, 4 }, { 20, 24, 22, 4 }, { 69, 0, 23, 69 },
                { 24, 24, 24, 0 }, { 28, 24, 25, 4 }, { 24, 28, 26, 4 }, { 65, 8, 27, 57 }, { 28, 28, 28, 0 }, { 32, 28, 29, 4 }, { 28, 32, 30, 4 }, { 69, 12, 31, 57 },
                { 32, 32, 32, 0 }, { 36, 32, 33, 4 }, { 32, 36, 34, 4 }, { 65, 20, 35, 45 }, { 36, 36, 36, 0 }, { 40, 36, 37, 4 }, { 36, 40, 38, 4 }, { 69, 24, 39, 45 },
                { 40, 40, 40, 0 }, { 44, 40, 41, 4 }, { 40, 44, 42, 4 }, { 65, 32, 43, 33 }, { 44, 44, 44, 0 }, { 48, 44, 45, 4 }, { 44, 48, 46, 4 }, { 69, 36, 47, 33 },
                { 48, 48, 48, 0 }, { 52, 48, 49, 4 }, { 48, 52, 50, 4 }, { 65, 44, 51, 21 }, { 52, 52, 52, 0 }, { 56, 52, 53, 4 }, { 52, 56, 54, 4 }, { 69, 48, 55, 21 },
                { 56, 56, 56, 0 }, { 60, 56, 57, 4 }, { 56, 60, 58, 4 }, { 65, 56, 59, 9 }, { 60, 60, 60, 0 }, { 65, 60, 61, 5 }, { 56, 65, 62, 9 }, { 60, 65, 63, 5 },
                { 56, 69, 64, 13 }, { 65, 65, 65, 0 }, { 69, 65, 66, 4 }, { 65, 69, 67, 4 }, { 60, 73, 68, 13 }, { 69, 69, 69, 0 }, { 73, 69, 70, 4 }, { 69, 73, 71, 4 },
                { 56, 81, 72, 25 }, { 73, 73, 73, 0 }, { 77, 73, 74, 4 }, { 73, 77, 75, 4 }, { 60, 85, 76, 25 }, { 77, 77, 77, 0 }, { 81, 77, 78, 4 }, { 77, 81, 79, 4 },
                { 56, 93, 80, 37 }, { 81, 81, 81, 0 }, { 85, 81, 82, 4 }, { 81, 85, 83, 4 }, { 60, 97, 84, 37 }, { 85, 85, 85, 0 }, { 89, 85, 86, 4 }, { 85, 89, 87, 4 },
                { 56, 105, 88, 49 }, { 89, 89, 89, 0 }, { 93, 89, 90, 4 }, { 89, 93, 91, 4 }, { 60, 109, 92, 49 }, { 93, 93, 93, 0 }, { 97, 93, 94, 4 }, { 93, 97, 95, 4 },
                { 134, 77, 96, 57 }, { 97, 97, 97, 0 }, { 101, 97, 98, 4 }, { 97, 101, 99, 4 }, { 130, 85, 100, 45 }, { 101, 101, 101, 0 }, { 105, 101, 102, 4 }, { 101, 105, 103, 4 },
                { 134, 89, 104, 45 }, { 105, 105, 105, 0 }, { 109, 105, 106, 4 }, { 105, 109, 107, 4 }, { 130, 97, 108, 33 }, { 109, 109, 109, 0 }, { 113, 109, 110, 4 }, { 109, 113, 111, 4 },
                { 134, 101, 112, 33 }, { 113, 113, 113, 0 }, { 117, 113, 114, 4 }, { 113, 117, 115, 4 }, { 130, 109, 116, 21 }, { 117, 117, 117, 0 }, { 121, 117, 118, 4 }, { 117, 121, 119, 4 },
                { 134, 113, 120, 21 }, { 121, 121, 121, 0 }, { 125, 121, 122, 4 }, { 121, 125, 123, 4 }, { 130, 121, 124, 9 }, { 125, 125, 125, 0 }, { 130, 125, 126, 5 }, { 121, 130, 127, 9 },
                { 125, 130, 128, 5 }, { 121, 134, 129, 13 }, { 130, 130, 130, 0 }, { 134, 130, 131, 4 }, { 130, 134, 132, 4 }, { 125, 138, 133, 13 }, { 134, 134, 134, 0 }, { 138, 134, 135, 4 },
                { 134, 138, 136, 4 }, { 121, 146, 137, 25 }, { 138, 138, 138, 0 }, { 142, 138, 139, 4 }, { 138, 142, 140, 4 }, { 125, 150, 141, 25 }, { 142, 142, 142, 0 }, { 146, 142, 143, 4 },
                { 142, 146, 144, 4 }, { 121, 158, 145, 37 }, { 146, 146, 146, 0 }, { 150, 146, 147, 4 }, { 146, 150, 148, 4 }, { 125, 162, 149, 37 }, { 150, 150, 150, 0 }, { 154, 150, 151, 4 },
                { 150, 154, 152, 4 }, { 121, 170, 153, 49 }, { 154, 154, 154, 0 }, { 158, 154, 155, 4 }, { 154, 158, 156, 4 }, { 125, 174, 157, 49 }, { 158, 158, 158, 0 }, { 162, 158, 159, 4 },
                { 158, 162, 160, 4 }, { 199, 142, 161, 57 }, { 162, 162, 162, 0 }, { 166, 162, 163, 4 }, { 162, 166, 164, 4 }, { 195, 150, 165, 45 }, { 166, 166, 166, 0 }, { 170, 166, 167, 4 },
                { 166, 170, 168, 4 }, { 199, 154, 169, 45 }, { 170, 170, 170, 0 }, { 174, 170, 171, 4 }, { 170, 174, 172, 4 }, { 195, 162, 173, 33 }, { 174, 174, 174, 0 }, { 178, 174, 175, 4 },
                { 174, 178, 176, 4 }, { 199, 166, 177, 33 }, { 178, 178, 178, 0 }, { 182, 178, 179, 4 }, { 178, 182, 180, 4 }, { 195, 174, 181, 21 }, { 182, 182, 182, 0 }, { 186, 182, 183, 4 },
                { 182, 186, 184, 4 }, { 199, 178, 185, 21 }, { 186, 186, 186, 0 }, { 190, 186, 187, 4 }, { 186, 190, 188, 4 }, { 195, 186, 189, 9 }, { 190, 190, 190, 0 }, { 195, 190, 191, 5 },
                { 186, 195, 192, 9 }, { 190, 195, 193, 5 }, { 186, 199, 194, 13 }, { 195, 195, 195, 0 }, { 199, 195, 196, 4 }, { 195, 199, 197, 4 }, { 190, 203, 198, 13 }, { 199, 199, 199, 0 },
                { 203, 199, 200, 4 }, { 199, 203, 201, 4 }, { 186, 211, 202, 25 }, { 203, 203, 203, 0 }, { 207, 203, 204, 4 }, { 203, 207, 205, 4 }, { 190, 215, 206, 25 }, { 207, 207, 207, 0 },
                { 211, 207, 208, 4 }, { 207, 211, 209, 4 }, { 186, 223, 210, 37 }, { 211, 211, 211, 0 }, { 215, 211, 212, 4 }, { 211, 215, 213, 4 }, { 190, 227, 214, 37 }, { 215, 215, 215, 0 },
                { 219, 215, 216, 4 }, { 215, 219, 217, 4 }, { 186, 235, 218, 49 }, { 219, 219, 219, 0 }, { 223, 219, 220, 4 }, { 219, 223, 221, 4 }, { 190, 239, 222, 49 }, { 223, 223, 223, 0 },
                { 227, 223, 224, 4 }, { 223, 227, 225, 4 }, { 186, 247, 226, 61 }, { 227, 227, 227, 0 }, { 231, 227, 228, 4 }, { 227, 231, 229, 4 }, { 190, 251, 230, 61 }, { 231, 231, 231, 0 },
                { 235, 231, 232, 4 }, { 231, 235, 233, 4 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 239, 235, 236, 4 }, { 235, 239, 237, 4 }, { 239, 239, 239, 0 }, { 239, 239, 239, 0 },
                { 243, 239, 240, 4 }, { 239, 243, 241, 4 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 247, 243, 244, 4 }, { 243, 247, 245, 4 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 251, 247, 248, 4 }, { 247, 251, 249, 4 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 255, 251, 252, 4 }, { 251, 255, 253, 4 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor5_2[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 8, 4, 8 }, { 0, 8, 4, 8 }, { 0, 8, 4, 8 }, { 8, 8, 8, 0 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 16, 12, 8 }, { 8, 16, 12, 8 }, { 8, 16, 12, 8 }, { 16, 16, 16, 0 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 24, 20, 8 }, { 16, 24, 20, 8 }, { 16, 24, 20, 8 }, { 24, 24, 24, 0 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 41, 32, 17 },
                { 24, 41, 32, 17 }, { 33, 33, 33, 0 }, { 33, 33, 33, 0 }, { 24, 49, 36, 25 }, { 24, 49, 36, 25 }, { 33, 41, 37, 8 }, { 33, 41, 37, 8 }, { 24, 57, 40, 33 },
                { 24, 57, 40, 33 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 41, 49, 45, 8 }, { 41, 49, 45, 8 }, { 41, 49, 45, 8 }, { 49, 49, 49, 0 },
                { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 57, 53, 8 }, { 49, 57, 53, 8 }, { 49, 57, 53, 8 }, { 57, 57, 57, 0 },
                { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 },
                { 57, 74, 65, 17 }, { 57, 74, 65, 17 }, { 66, 66, 66, 0 }, { 66, 66, 66, 0 }, { 57, 82, 69, 25 }, { 57, 82, 69, 25 }, { 66, 74, 70, 8 }, { 66, 74, 70, 8 },
                { 57, 90, 73, 33 }, { 57, 90, 73, 33 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 74, 82, 78, 8 }, { 74, 82, 78, 8 }, { 74, 82, 78, 8 },
                { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 90, 86, 8 }, { 82, 90, 86, 8 }, { 82, 90, 86, 8 },
                { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 99, 94, 9 }, { 90, 99, 94, 9 }, { 90, 99, 94, 9 },
                { 90, 99, 94, 9 }, { 90, 107, 98, 17 }, { 90, 107, 98, 17 }, { 99, 99, 99, 0 }, { 99, 99, 99, 0 }, { 90, 115, 102, 25 }, { 90, 115, 102, 25 }, { 99, 107, 103, 8 },
                { 99, 107, 103, 8 }, { 90, 123, 106, 33 }, { 90, 123, 106, 33 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 107, 115, 111, 8 }, { 107, 115, 111, 8 },
                { 107, 115, 111, 8 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 123, 119, 8 }, { 115, 123, 119, 8 },
                { 115, 123, 119, 8 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 132, 127, 9 }, { 123, 132, 127, 9 },
                { 123, 132, 127, 9 }, { 123, 132, 127, 9 }, { 123, 140, 131, 17 }, { 123, 140, 131, 17 }, { 132, 132, 132, 0 }, { 132, 132, 132, 0 }, { 123, 148, 135, 25 }, { 123, 148, 135, 25 },
                { 132, 140, 136, 8 }, { 132, 140, 136, 8 }, { 123, 156, 139, 33 }, { 123, 156, 139, 33 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 140, 148, 144, 8 },
                { 140, 148, 144, 8 }, { 140, 148, 144, 8 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 156, 152, 8 },
                { 148, 156, 152, 8 }, { 148, 156, 152, 8 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 165, 160, 9 },
                { 156, 165, 160, 9 }, { 156, 165, 160, 9 }, { 156, 165, 160, 9 }, { 156, 173, 164, 17 }, { 156, 173, 164, 17 }, { 165, 165, 165, 0 }, { 165, 165, 165, 0 }, { 156, 181, 168, 25 },
                { 156, 181, 168, 25 }, { 165, 173, 169, 8 }, { 165, 173, 169, 8 }, { 156, 189, 172, 33 }, { 156, 189, 172, 33 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 },
                { 173, 181, 177, 8 }, { 173, 181, 177, 8 }, { 173, 181, 177, 8 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 },
                { 181, 189, 185, 8 }, { 181, 189, 185, 8 }, { 181, 189, 185, 8 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 },
                { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 206, 197, 17 }, { 189, 206, 197, 17 }, { 198, 198, 198, 0 }, { 198, 198, 198, 0 },
                { 189, 214, 201, 25 }, { 189, 214, 201, 25 }, { 198, 206, 202, 8 }, { 198, 206, 202, 8 }, { 189, 222, 205, 33 }, { 189, 222, 205, 33 }, { 206, 206, 206, 0 }, { 206, 206, 206, 0 },
                { 206, 206, 206, 0 }, { 206, 214, 210, 8 }, { 206, 214, 210, 8 }, { 206, 214, 210, 8 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 },
                { 214, 214, 214, 0 }, { 214, 222, 218, 8 }, { 214, 222, 218, 8 }, { 214, 222, 218, 8 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 },
                { 222, 222, 222, 0 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 239, 230, 17 }, { 222, 239, 230, 17 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 222, 247, 234, 25 }, { 222, 247, 234, 25 }, { 231, 239, 235, 8 }, { 231, 239, 235, 8 }, { 222, 255, 238, 33 }, { 222, 255, 238, 33 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 239, 239, 239, 0 }, { 239, 247, 243, 8 }, { 239, 247, 243, 8 }, { 239, 247, 243, 8 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 247, 247, 247, 0 }, { 247, 255, 251, 8 }, { 247, 255, 251, 8 }, { 247, 255, 251, 8 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor6_2[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 4, 2, 4 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 4, 8, 6, 4 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 12, 10, 4 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 12, 16, 14, 4 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 20, 18, 4 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 20, 24, 22, 4 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 28, 26, 4 }, { 28, 28, 28, 0 }, { 28, 28, 28, 0 }, { 28, 28, 28, 0 }, { 28, 32, 30, 4 }, { 32, 32, 32, 0 },
                { 32, 32, 32, 0 }, { 32, 32, 32, 0 }, { 32, 36, 34, 4 }, { 36, 36, 36, 0 }, { 36, 36, 36, 0 }, { 36, 36, 36, 0 }, { 36, 40, 38, 4 }, { 40, 40, 40, 0 },
                { 40, 40, 40, 0 }, { 40, 40, 40, 0 }, { 40, 44, 42, 4 }, { 44, 44, 44, 0 }, { 44, 44, 44, 0 }, { 44, 44, 44, 0 }, { 44, 48, 46, 4 }, { 48, 48, 48, 0 },
                { 48, 48, 48, 0 }, { 48, 48, 48, 0 }, { 48, 52, 50, 4 }, { 52, 52, 52, 0 }, { 52, 52, 52, 0 }, { 52, 52, 52, 0 }, { 52, 56, 54, 4 }, { 56, 56, 56, 0 },
                { 56, 56, 56, 0 }, { 56, 56, 56, 0 }, { 56, 60, 58, 4 }, { 60, 60, 60, 0 }, { 60, 60, 60, 0 }, { 60, 60, 60, 0 }, { 60, 65, 62, 5 }, { 60, 65, 62, 5 },
                { 60, 69, 64, 9 }, { 65, 65, 65, 0 }, { 60, 73, 66, 13 }, { 65, 69, 67, 4 }, { 60, 77, 68, 17 }, { 69, 69, 69, 0 }, { 60, 81, 70, 21 }, { 69, 73, 71, 4 },
                { 60, 85, 72, 25 }, { 73, 73, 73, 0 }, { 60, 89, 74, 29 }, { 73, 77, 75, 4 }, { 60, 93, 76, 33 }, { 77, 77, 77, 0 }, { 60, 97, 78, 37 }, { 77, 81, 79, 4 },
                { 60, 101, 80, 41 }, { 81, 81, 81, 0 }, { 60, 105, 82, 45 }, { 81, 85, 83, 4 }, { 60, 109, 84, 49 }, { 85, 85, 85, 0 }, { 60, 113, 86, 53 }, { 85, 89, 87, 4 },
                { 60, 117, 88, 57 }, { 89, 89, 89, 0 }, { 60, 121, 90, 61 }, { 89, 93, 91, 4 }, { 60, 125, 92, 65 }, { 93, 93, 93, 0 }, { 93, 93, 93, 0 }, { 93, 97, 95, 4 },
                { 97, 97, 97, 0 }, { 97, 97, 97, 0 }, { 97, 97, 97, 0 }, { 97, 101, 99, 4 }, { 101, 101, 101, 0 }, { 101, 101, 101, 0 }, { 101, 101, 101, 0 }, { 101, 105, 103, 4 },
                { 105, 105, 105, 0 }, { 105, 105, 105, 0 }, { 105, 105, 105, 0 }, { 105, 109, 107, 4 }, { 109, 109, 109, 0 }, { 109, 109, 109, 0 }, { 109, 109, 109, 0 }, { 109, 113, 111, 4 },
                { 113, 113, 113, 0 }, { 113, 113, 113, 0 }, { 113, 113, 113, 0 }, { 113, 117, 115, 4 }, { 117, 117, 117, 0 }, { 117, 117, 117, 0 }, { 117, 117, 117, 0 }, { 117, 121, 119, 4 },
                { 121, 121, 121, 0 }, { 121, 121, 121, 0 }, { 121, 121, 121, 0 }, { 121, 125, 123, 4 }, { 125, 125, 125, 0 }, { 125, 125, 125, 0 }, { 125, 125, 125, 0 }, { 125, 130, 127, 5 },
                { 125, 130, 127, 5 }, { 125, 134, 129, 9 }, { 130, 130, 130, 0 }, { 125, 138, 131, 13 }, { 130, 134, 132, 4 }, { 125, 142, 133, 17 }, { 134, 134, 134, 0 }, { 125, 146, 135, 21 },
                { 134, 138, 136, 4 }, { 125, 150, 137, 25 }, { 138, 138, 138, 0 }, { 125, 154, 139, 29 }, { 138, 142, 140, 4 }, { 125, 158, 141, 33 }, { 142, 142, 142, 0 }, { 125, 162, 143, 37 },
                { 142, 146, 144, 4 }, { 125, 166, 145, 41 }, { 146, 146, 146, 0 }, { 125, 170, 147, 45 }, { 146, 150, 148, 4 }, { 125, 174, 149, 49 }, { 150, 150, 150, 0 }, { 125, 178, 151, 53 },
                { 150, 154, 152, 4 }, { 125, 182, 153, 57 }, { 154, 154, 154, 0 }, { 125, 186, 155, 61 }, { 154, 158, 156, 4 }, { 125, 190, 157, 65 }, { 158, 158, 158, 0 }, { 158, 158, 158, 0 },
                { 158, 162, 160, 4 }, { 162, 162, 162, 0 }, { 162, 162, 162, 0 }, { 162, 162, 162, 0 }, { 162, 166, 164, 4 }, { 166, 166, 166, 0 }, { 166, 166, 166, 0 }, { 166, 166, 166, 0 },
                { 166, 170, 168, 4 }, { 170, 170, 170, 0 }, { 170, 170, 170, 0 }, { 170, 170, 170, 0 }, { 170, 174, 172, 4 }, { 174, 174, 174, 0 }, { 174, 174, 174, 0 }, { 174, 174, 174, 0 },
                { 174, 178, 176, 4 }, { 178, 178, 178, 0 }, { 178, 178, 178, 0 }, { 178, 178, 178, 0 }, { 178, 182, 180, 4 }, { 182, 182, 182, 0 }, { 182, 182, 182, 0 }, { 182, 182, 182, 0 },
                { 182, 186, 184, 4 }, { 186, 186, 186, 0 }, { 186, 186, 186, 0 }, { 186, 186, 186, 0 }, { 186, 190, 188, 4 }, { 190, 190, 190, 0 }, { 190, 190, 190, 0 }, { 190, 190, 190, 0 },
                { 190, 195, 192, 5 }, { 190, 195, 192, 5 }, { 190, 199, 194, 9 }, { 195, 195, 195, 0 }, { 190, 203, 196, 13 }, { 195, 199, 197, 4 }, { 190, 207, 198, 17 }, { 199, 199, 199, 0 },
                { 190, 211, 200, 21 }, { 199, 203, 201, 4 }, { 190, 215, 202, 25 }, { 203, 203, 203, 0 }, { 190, 219, 204, 29 }, { 203, 207, 205, 4 }, { 190, 223, 206, 33 }, { 207, 207, 207, 0 },
                { 190, 227, 208, 37 }, { 207, 211, 209, 4 }, { 190, 231, 210, 41 }, { 211, 211, 211, 0 }, { 190, 235, 212, 45 }, { 211, 215, 213, 4 }, { 190, 239, 214, 49 }, { 215, 215, 215, 0 },
                { 190, 243, 216, 53 }, { 215, 219, 217, 4 }, { 190, 247, 218, 57 }, { 219, 219, 219, 0 }, { 190, 251, 220, 61 }, { 219, 223, 221, 4 }, { 190, 255, 222, 65 }, { 223, 223, 223, 0 },
                { 223, 223, 223, 0 }, { 223, 227, 225, 4 }, { 227, 227, 227, 0 }, { 227, 227, 227, 0 }, { 227, 227, 227, 0 }, { 227, 231, 229, 4 }, { 231, 231, 231, 0 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 231, 235, 233, 4 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 235, 239, 237, 4 }, { 239, 239, 239, 0 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 239, 243, 241, 4 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 243, 247, 245, 4 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 247, 251, 249, 4 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 251, 255, 253, 4 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor5_3_p[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 8, 0, 2, 8 }, { 8, 0, 2, 8 }, { 0, 8, 5, 8 }, { 0, 8, 5, 8 }, { 0, 8, 5, 8 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 16, 8, 10, 8 }, { 33, 0, 11, 33 }, { 8, 16, 13, 8 }, { 8, 16, 13, 8 }, { 8, 16, 13, 8 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 24, 16, 18, 8 }, { 41, 8, 19, 33 }, { 16, 24, 21, 8 }, { 16, 24, 21, 8 }, { 0, 33, 22, 33 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 33, 24, 27, 9 }, { 33, 24, 27, 9 }, { 33, 24, 27, 9 }, { 41, 24, 29, 17 }, { 24, 33, 30, 9 }, { 24, 33, 30, 9 },
                { 16, 41, 32, 25 }, { 33, 33, 33, 0 }, { 33, 33, 33, 0 }, { 41, 33, 35, 8 }, { 41, 33, 35, 8 }, { 33, 41, 38, 8 }, { 33, 41, 38, 8 }, { 33, 41, 38, 8 },
                { 24, 49, 40, 25 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 49, 41, 43, 8 }, { 66, 33, 44, 33 }, { 41, 49, 46, 8 }, { 41, 49, 46, 8 }, { 41, 49, 46, 8 },
                { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 57, 49, 51, 8 }, { 74, 41, 52, 33 }, { 49, 57, 54, 8 }, { 49, 57, 54, 8 }, { 33, 66, 55, 33 },
                { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 66, 57, 60, 9 }, { 66, 57, 60, 9 }, { 66, 57, 60, 9 }, { 74, 57, 62, 17 }, { 57, 66, 63, 9 },
                { 57, 66, 63, 9 }, { 49, 74, 65, 25 }, { 66, 66, 66, 0 }, { 66, 66, 66, 0 }, { 74, 66, 68, 8 }, { 74, 66, 68, 8 }, { 66, 74, 71, 8 }, { 66, 74, 71, 8 },
                { 66, 74, 71, 8 }, { 57, 82, 73, 25 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 82, 74, 76, 8 }, { 99, 66, 77, 33 }, { 74, 82, 79, 8 }, { 74, 82, 79, 8 },
                { 74, 82, 79, 8 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 90, 82, 84, 8 }, { 107, 74, 85, 33 }, { 82, 90, 87, 8 }, { 82, 90, 87, 8 },
                { 66, 99, 88, 33 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 99, 90, 93, 9 }, { 99, 90, 93, 9 }, { 99, 90, 93, 9 }, { 107, 90, 95, 17 },
                { 90, 99, 96, 9 }, { 90, 99, 96, 9 }, { 82, 107, 98, 25 }, { 99, 99, 99, 0 }, { 99, 99, 99, 0 }, { 107, 99, 101, 8 }, { 107, 99, 101, 8 }, { 99, 107, 104, 8 },
                { 99, 107, 104, 8 }, { 99, 107, 104, 8 }, { 90, 115, 106, 25 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 115, 107, 109, 8 }, { 132, 99, 110, 33 }, { 107, 115, 112, 8 },
                { 107, 115, 112, 8 }, { 107, 115, 112, 8 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 123, 115, 117, 8 }, { 140, 107, 118, 33 }, { 115, 123, 120, 8 },
                { 115, 123, 120, 8 }, { 99, 132, 121, 33 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 132, 123, 126, 9 }, { 132, 123, 126, 9 }, { 132, 123, 126, 9 },
                { 140, 123, 128, 17 }, { 123, 132, 129, 9 }, { 123, 132, 129, 9 }, { 115, 140, 131, 25 }, { 132, 132, 132, 0 }, { 132, 132, 132, 0 }, { 140, 132, 134, 8 }, { 140, 132, 134, 8 },
                { 132, 140, 137, 8 }, { 132, 140, 137, 8 }, { 132, 140, 137, 8 }, { 123, 148, 139, 25 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 148, 140, 142, 8 }, { 165, 132, 143, 33 },
                { 140, 148, 145, 8 }, { 140, 148, 145, 8 }, { 140, 148, 145, 8 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 156, 148, 150, 8 }, { 173, 140, 151, 33 },
                { 148, 156, 153, 8 }, { 148, 156, 153, 8 }, { 132, 165, 154, 33 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 165, 156, 159, 9 }, { 165, 156, 159, 9 },
                { 165, 156, 159, 9 }, { 173, 156, 161, 17 }, { 156, 165, 162, 9 }, { 156, 165, 162, 9 }, { 148, 173, 164, 25 }, { 165, 165, 165, 0 }, { 165, 165, 165, 0 }, { 173, 165, 167, 8 },
                { 173, 165, 167, 8 }, { 165, 173, 170, 8 }, { 165, 173, 170, 8 }, { 165, 173, 170, 8 }, { 156, 181, 172, 25 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 }, { 181, 173, 175, 8 },
                { 198, 165, 176, 33 }, { 173, 181, 178, 8 }, { 173, 181, 178, 8 }, { 173, 181, 178, 8 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 189, 181, 183, 8 },
                { 206, 173, 184, 33 }, { 181, 189, 186, 8 }, { 181, 189, 186, 8 }, { 165, 198, 187, 33 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 198, 189, 192, 9 },
                { 198, 189, 192, 9 }, { 198, 189, 192, 9 }, { 206, 189, 194, 17 }, { 189, 198, 195, 9 }, { 189, 198, 195, 9 }, { 181, 206, 197, 25 }, { 198, 198, 198, 0 }, { 198, 198, 198, 0 },
                { 206, 198, 200, 8 }, { 206, 198, 200, 8 }, { 198, 206, 203, 8 }, { 198, 206, 203, 8 }, { 198, 206, 203, 8 }, { 189, 214, 205, 25 }, { 206, 206, 206, 0 }, { 206, 206, 206, 0 },
                { 214, 206, 208, 8 }, { 231, 198, 209, 33 }, { 206, 214, 211, 8 }, { 206, 214, 211, 8 }, { 206, 214, 211, 8 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 },
                { 222, 214, 216, 8 }, { 239, 206, 217, 33 }, { 214, 222, 219, 8 }, { 214, 222, 219, 8 }, { 198, 231, 220, 33 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 },
                { 231, 222, 225, 9 }, { 231, 222, 225, 9 }, { 231, 222, 225, 9 }, { 239, 222, 227, 17 }, { 222, 231, 228, 9 }, { 222, 231, 228, 9 }, { 214, 239, 230, 25 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 239, 231, 233, 8 }, { 239, 231, 233, 8 }, { 231, 239, 236, 8 }, { 231, 239, 236, 8 }, { 231, 239, 236, 8 }, { 222, 247, 238, 25 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 247, 239, 241, 8 }, { 247, 239, 241, 8 }, { 239, 247, 244, 8 }, { 239, 247, 244, 8 }, { 239, 247, 244, 8 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 255, 247, 249, 8 }, { 255, 247, 249, 8 }, { 247, 255, 252, 8 }, { 247, 255, 252, 8 }, { 247, 255, 252, 8 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor6_3_p[256] =
            {
                { 0, 0, 0, 0 }, { 4, 0, 1, 4 }, { 0, 4, 2, 4 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 8, 4, 5, 4 }, { 4, 8, 6, 4 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 12, 8, 9, 4 }, { 8, 12, 10, 4 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 16, 12, 13, 4 }, { 12, 16, 14, 4 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 20, 16, 17, 4 }, { 16, 20, 18, 4 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 24, 20, 21, 4 }, { 20, 24, 22, 4 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 28, 24, 25, 4 }, { 24, 28, 26, 4 }, { 28, 28, 28, 0 }, { 28, 28, 28, 0 }, { 32, 28, 29, 4 }, { 28, 32, 30, 4 }, { 32, 32, 32, 0 },
                { 32, 32, 32, 0 }, { 36, 32, 33, 4 }, { 32, 36, 34, 4 }, { 36, 36, 36, 0 }, { 36, 36, 36, 0 }, { 40, 36, 37, 4 }, { 36, 40, 38, 4 }, { 40, 40, 40, 0 },
                { 40, 40, 40, 0 }, { 44, 40, 41, 4 }, { 40, 44, 42, 4 }, { 65, 32, 43, 33 }, { 44, 44, 44, 0 }, { 48, 44, 45, 4 }, { 44, 48, 46, 4 }, { 69, 36, 47, 33 },
                { 48, 48, 48, 0 }, { 52, 48, 49, 4 }, { 48, 52, 50, 4 }, { 65, 44, 51, 21 }, { 52, 52, 52, 0 }, { 56, 52, 53, 4 }, { 52, 56, 54, 4 }, { 69, 48, 55, 21 },
                { 56, 56, 56, 0 }, { 60, 56, 57, 4 }, { 56, 60, 58, 4 }, { 65, 56, 59, 9 }, { 60, 60, 60, 0 }, { 65, 60, 61, 5 }, { 56, 65, 62, 9 }, { 60, 65, 63, 5 },
                { 56, 69, 64, 13 }, { 65, 65, 65, 0 }, { 69, 65, 66, 4 }, { 65, 69, 67, 4 }, { 60, 73, 68, 13 }, { 69, 69, 69, 0 }, { 73, 69, 70, 4 }, { 69, 73, 71, 4 },
                { 56, 81, 72, 25 }, { 73, 73, 73, 0 }, { 77, 73, 74, 4 }, { 73, 77, 75, 4 }, { 60, 85, 76, 25 }, { 77, 77, 77, 0 }, { 81, 77, 78, 4 }, { 77, 81, 79, 4 },
                { 81, 81, 81, 0 }, { 81, 81, 81, 0 }, { 85, 81, 82, 4 }, { 81, 85, 83, 4 }, { 85, 85, 85, 0 }, { 85, 85, 85, 0 }, { 89, 85, 86, 4 }, { 85, 89, 87, 4 },
                { 89, 89, 89, 0 }, { 89, 89, 89, 0 }, { 93, 89, 90, 4 }, { 89, 93, 91, 4 }, { 93, 93, 93, 0 }, { 93, 93, 93, 0 }, { 97, 93, 94, 4 }, { 93, 97, 95, 4 },
                { 97, 97, 97, 0 }, { 97, 97, 97, 0 }, { 101, 97, 98, 4 }, { 97, 101, 99, 4 }, { 101, 101, 101, 0 }, { 101, 101, 101, 0 }, { 105, 101, 102, 4 }, { 101, 105, 103, 4 },
                { 105, 105, 105, 0 }, { 105, 105, 105, 0 }, { 109, 105, 106, 4 }, { 105, 109, 107, 4 }, { 130, 97, 108, 33 }, { 109, 109, 109, 0 }, { 113, 109, 110, 4 }, { 109, 113, 111, 4 },
                { 134, 101, 112, 33 }, { 113, 113, 113, 0 }, { 117, 113, 114, 4 }, { 113, 117, 115, 4 }, { 130, 109, 116, 21 }, { 117, 117, 117, 0 }, { 121, 117, 118, 4 }, { 117, 121, 119, 4 },
                { 134, 113, 120, 21 }, { 121, 121, 121, 0 }, { 125, 121, 122, 4 }, { 121, 125, 123, 4 }, { 130, 121, 124, 9 }, { 125, 125, 125, 0 }, { 130, 125, 126, 5 }, { 121, 130, 127, 9 },
                { 125, 130, 128, 5 }, { 121, 134, 129, 13 }, { 130, 130, 130, 0 }, { 134, 130, 131, 4 }, { 130, 134, 132, 4 }, { 125, 138, 133, 13 }, { 134, 134, 134, 0 }, { 138, 134, 135, 4 },
                { 134, 138, 136, 4 }, { 121, 146, 137, 25 }, { 138, 138, 138, 0 }, { 142, 138, 139, 4 }, { 138, 142, 140, 4 }, { 125, 150, 141, 25 }, { 142, 142, 142, 0 }, { 146, 142, 143, 4 },
                { 142, 146, 144, 4 }, { 146, 146, 146, 0 }, { 146, 146, 146, 0 }, { 150, 146, 147, 4 }, { 146, 150, 148, 4 }, { 150, 150, 150, 0 }, { 150, 150, 150, 0 }, { 154, 150, 151, 4 },
                { 150, 154, 152, 4 }, { 154, 154, 154, 0 }, { 154, 154, 154, 0 }, { 158, 154, 155, 4 }, { 154, 158, 156, 4 }, { 158, 158, 158, 0 }, { 158, 158, 158, 0 }, { 162, 158, 159, 4 },
                { 158, 162, 160, 4 }, { 162, 162, 162, 0 }, { 162, 162, 162, 0 }, { 166, 162, 163, 4 }, { 162, 166, 164, 4 }, { 166, 166, 166, 0 }, { 166, 166, 166, 0 }, { 170, 166, 167, 4 },
                { 166, 170, 168, 4 }, { 170, 170, 170, 0 }, { 170, 170, 170, 0 }, { 174, 170, 171, 4 }, { 170, 174, 172, 4 }, { 195, 162, 173, 33 }, { 174, 174, 174, 0 }, { 178, 174, 175, 4 },
                { 174, 178, 176, 4 }, { 199, 166, 177, 33 }, { 178, 178, 178, 0 }, { 182, 178, 179, 4 }, { 178, 182, 180, 4 }, { 195, 174, 181, 21 }, { 182, 182, 182, 0 }, { 186, 182, 183, 4 },
                { 182, 186, 184, 4 }, { 199, 178, 185, 21 }, { 186, 186, 186, 0 }, { 190, 186, 187, 4 }, { 186, 190, 188, 4 }, { 195, 186, 189, 9 }, { 190, 190, 190, 0 }, { 195, 190, 191, 5 },
                { 186, 195, 192, 9 }, { 190, 195, 193, 5 }, { 186, 199, 194, 13 }, { 195, 195, 195, 0 }, { 199, 195, 196, 4 }, { 195, 199, 197, 4 }, { 190, 203, 198, 13 }, { 199, 199, 199, 0 },
                { 203, 199, 200, 4 }, { 199, 203, 201, 4 }, { 186, 211, 202, 25 }, { 203, 203, 203, 0 }, { 207, 203, 204, 4 }, { 203, 207, 205, 4 }, { 190, 215, 206, 25 }, { 207, 207, 207, 0 },
                { 211, 207, 208, 4 }, { 207, 211, 209, 4 }, { 211, 211, 211, 0 }, { 211, 211, 211, 0 }, { 215, 211, 212, 4 }, { 211, 215, 213, 4 }, { 215, 215, 215, 0 }, { 215, 215, 215, 0 },
                { 219, 215, 216, 4 }, { 215, 219, 217, 4 }, { 219, 219, 219, 0 }, { 219, 219, 219, 0 }, { 223, 219, 220, 4 }, { 219, 223, 221, 4 }, { 223, 223, 223, 0 }, { 223, 223, 223, 0 },
                { 227, 223, 224, 4 }, { 223, 227, 225, 4 }, { 227, 227, 227, 0 }, { 227, 227, 227, 0 }, { 231, 227, 228, 4 }, { 227, 231, 229, 4 }, { 231, 231, 231, 0 }, { 231, 231, 231, 0 },
                { 235, 231, 232, 4 }, { 231, 235, 233, 4 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 239, 235, 236, 4 }, { 235, 239, 237, 4 }, { 239, 239, 239, 0 }, { 239, 239, 239, 0 },
                { 243, 239, 240, 4 }, { 239, 243, 241, 4 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 247, 243, 244, 4 }, { 243, 247, 245, 4 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 251, 247, 248, 4 }, { 247, 251, 249, 4 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 255, 251, 252, 4 }, { 251, 255, 253, 4 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor5_2_p[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 8, 4, 8 }, { 0, 8, 4, 8 }, { 0, 8, 4, 8 }, { 8, 8, 8, 0 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 16, 12, 8 }, { 8, 16, 12, 8 }, { 8, 16, 12, 8 }, { 16, 16, 16, 0 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 24, 20, 8 }, { 16, 24, 20, 8 }, { 16, 24, 20, 8 }, { 24, 24, 24, 0 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 33, 28, 9 }, { 24, 41, 32, 17 },
                { 24, 41, 32, 17 }, { 33, 33, 33, 0 }, { 33, 33, 33, 0 }, { 24, 49, 36, 25 }, { 24, 49, 36, 25 }, { 33, 41, 37, 8 }, { 33, 41, 37, 8 }, { 24, 57, 40, 33 },
                { 24, 57, 40, 33 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 41, 41, 41, 0 }, { 41, 49, 45, 8 }, { 41, 49, 45, 8 }, { 41, 49, 45, 8 }, { 49, 49, 49, 0 },
                { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 49, 49, 0 }, { 49, 57, 53, 8 }, { 49, 57, 53, 8 }, { 49, 57, 53, 8 }, { 57, 57, 57, 0 },
                { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 57, 57, 0 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 }, { 57, 66, 61, 9 },
                { 57, 74, 65, 17 }, { 57, 74, 65, 17 }, { 66, 66, 66, 0 }, { 66, 66, 66, 0 }, { 57, 82, 69, 25 }, { 57, 82, 69, 25 }, { 66, 74, 70, 8 }, { 66, 74, 70, 8 },
                { 57, 90, 73, 33 }, { 57, 90, 73, 33 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 74, 74, 74, 0 }, { 74, 82, 78, 8 }, { 74, 82, 78, 8 }, { 74, 82, 78, 8 },
                { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 82, 82, 0 }, { 82, 90, 86, 8 }, { 82, 90, 86, 8 }, { 82, 90, 86, 8 },
                { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 90, 90, 0 }, { 90, 99, 94, 9 }, { 90, 99, 94, 9 }, { 90, 99, 94, 9 },
                { 90, 99, 94, 9 }, { 90, 107, 98, 17 }, { 90, 107, 98, 17 }, { 99, 99, 99, 0 }, { 99, 99, 99, 0 }, { 90, 115, 102, 25 }, { 90, 115, 102, 25 }, { 99, 107, 103, 8 },
                { 99, 107, 103, 8 }, { 90, 123, 106, 33 }, { 90, 123, 106, 33 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 107, 107, 107, 0 }, { 107, 115, 111, 8 }, { 107, 115, 111, 8 },
                { 107, 115, 111, 8 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 115, 115, 0 }, { 115, 123, 119, 8 }, { 115, 123, 119, 8 },
                { 115, 123, 119, 8 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 123, 123, 0 }, { 123, 132, 127, 9 }, { 123, 132, 127, 9 },
                { 123, 132, 127, 9 }, { 123, 132, 127, 9 }, { 123, 140, 131, 17 }, { 123, 140, 131, 17 }, { 132, 132, 132, 0 }, { 132, 132, 132, 0 }, { 123, 148, 135, 25 }, { 123, 148, 135, 25 },
                { 132, 140, 136, 8 }, { 132, 140, 136, 8 }, { 123, 156, 139, 33 }, { 123, 156, 139, 33 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 140, 140, 140, 0 }, { 140, 148, 144, 8 },
                { 140, 148, 144, 8 }, { 140, 148, 144, 8 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 148, 148, 0 }, { 148, 156, 152, 8 },
                { 148, 156, 152, 8 }, { 148, 156, 152, 8 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 156, 156, 0 }, { 156, 165, 160, 9 },
                { 156, 165, 160, 9 }, { 156, 165, 160, 9 }, { 156, 165, 160, 9 }, { 156, 173, 164, 17 }, { 156, 173, 164, 17 }, { 165, 165, 165, 0 }, { 165, 165, 165, 0 }, { 156, 181, 168, 25 },
                { 156, 181, 168, 25 }, { 165, 173, 169, 8 }, { 165, 173, 169, 8 }, { 156, 189, 172, 33 }, { 156, 189, 172, 33 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 }, { 173, 173, 173, 0 },
                { 173, 181, 177, 8 }, { 173, 181, 177, 8 }, { 173, 181, 177, 8 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 }, { 181, 181, 181, 0 },
                { 181, 189, 185, 8 }, { 181, 189, 185, 8 }, { 181, 189, 185, 8 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 }, { 189, 189, 189, 0 },
                { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 198, 193, 9 }, { 189, 206, 197, 17 }, { 189, 206, 197, 17 }, { 198, 198, 198, 0 }, { 198, 198, 198, 0 },
                { 189, 214, 201, 25 }, { 189, 214, 201, 25 }, { 198, 206, 202, 8 }, { 198, 206, 202, 8 }, { 189, 222, 205, 33 }, { 189, 222, 205, 33 }, { 206, 206, 206, 0 }, { 206, 206, 206, 0 },
                { 206, 206, 206, 0 }, { 206, 214, 210, 8 }, { 206, 214, 210, 8 }, { 206, 214, 210, 8 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 }, { 214, 214, 214, 0 },
                { 214, 214, 214, 0 }, { 214, 222, 218, 8 }, { 214, 222, 218, 8 }, { 214, 222, 218, 8 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 }, { 222, 222, 222, 0 },
                { 222, 222, 222, 0 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 231, 226, 9 }, { 222, 239, 230, 17 }, { 222, 239, 230, 17 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 222, 247, 234, 25 }, { 222, 247, 234, 25 }, { 231, 239, 235, 8 }, { 231, 239, 235, 8 }, { 222, 255, 238, 33 }, { 222, 255, 238, 33 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 239, 239, 239, 0 }, { 239, 247, 243, 8 }, { 239, 247, 243, 8 }, { 239, 247, 243, 8 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 247, 247, 247, 0 }, { 247, 255, 251, 8 }, { 247, 255, 251, 8 }, { 247, 255, 251, 8 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };

            SingleColorTableEntry g_singleColor6_2_p[256] =
            {
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 4, 2, 4 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 4, 4, 4, 0 }, { 4, 8, 6, 4 }, { 8, 8, 8, 0 },
                { 8, 8, 8, 0 }, { 8, 8, 8, 0 }, { 8, 12, 10, 4 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 12, 12, 12, 0 }, { 12, 16, 14, 4 }, { 16, 16, 16, 0 },
                { 16, 16, 16, 0 }, { 16, 16, 16, 0 }, { 16, 20, 18, 4 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 20, 20, 20, 0 }, { 20, 24, 22, 4 }, { 24, 24, 24, 0 },
                { 24, 24, 24, 0 }, { 24, 24, 24, 0 }, { 24, 28, 26, 4 }, { 28, 28, 28, 0 }, { 28, 28, 28, 0 }, { 28, 28, 28, 0 }, { 28, 32, 30, 4 }, { 32, 32, 32, 0 },
                { 32, 32, 32, 0 }, { 32, 32, 32, 0 }, { 32, 36, 34, 4 }, { 36, 36, 36, 0 }, { 36, 36, 36, 0 }, { 36, 36, 36, 0 }, { 36, 40, 38, 4 }, { 40, 40, 40, 0 },
                { 40, 40, 40, 0 }, { 40, 40, 40, 0 }, { 40, 44, 42, 4 }, { 44, 44, 44, 0 }, { 44, 44, 44, 0 }, { 44, 44, 44, 0 }, { 44, 48, 46, 4 }, { 48, 48, 48, 0 },
                { 48, 48, 48, 0 }, { 48, 48, 48, 0 }, { 48, 52, 50, 4 }, { 52, 52, 52, 0 }, { 52, 52, 52, 0 }, { 52, 52, 52, 0 }, { 52, 56, 54, 4 }, { 56, 56, 56, 0 },
                { 56, 56, 56, 0 }, { 56, 56, 56, 0 }, { 56, 60, 58, 4 }, { 60, 60, 60, 0 }, { 60, 60, 60, 0 }, { 60, 60, 60, 0 }, { 60, 65, 62, 5 }, { 60, 65, 62, 5 },
                { 60, 69, 64, 9 }, { 65, 65, 65, 0 }, { 60, 73, 66, 13 }, { 65, 69, 67, 4 }, { 60, 77, 68, 17 }, { 69, 69, 69, 0 }, { 60, 81, 70, 21 }, { 69, 73, 71, 4 },
                { 60, 85, 72, 25 }, { 73, 73, 73, 0 }, { 60, 89, 74, 29 }, { 73, 77, 75, 4 }, { 60, 93, 76, 33 }, { 77, 77, 77, 0 }, { 77, 77, 77, 0 }, { 77, 81, 79, 4 },
                { 81, 81, 81, 0 }, { 81, 81, 81, 0 }, { 81, 81, 81, 0 }, { 81, 85, 83, 4 }, { 85, 85, 85, 0 }, { 85, 85, 85, 0 }, { 85, 85, 85, 0 }, { 85, 89, 87, 4 },
                { 89, 89, 89, 0 }, { 89, 89, 89, 0 }, { 89, 89, 89, 0 }, { 89, 93, 91, 4 }, { 93, 93, 93, 0 }, { 93, 93, 93, 0 }, { 93, 93, 93, 0 }, { 93, 97, 95, 4 },
                { 97, 97, 97, 0 }, { 97, 97, 97, 0 }, { 97, 97, 97, 0 }, { 97, 101, 99, 4 }, { 101, 101, 101, 0 }, { 101, 101, 101, 0 }, { 101, 101, 101, 0 }, { 101, 105, 103, 4 },
                { 105, 105, 105, 0 }, { 105, 105, 105, 0 }, { 105, 105, 105, 0 }, { 105, 109, 107, 4 }, { 109, 109, 109, 0 }, { 109, 109, 109, 0 }, { 109, 109, 109, 0 }, { 109, 113, 111, 4 },
                { 113, 113, 113, 0 }, { 113, 113, 113, 0 }, { 113, 113, 113, 0 }, { 113, 117, 115, 4 }, { 117, 117, 117, 0 }, { 117, 117, 117, 0 }, { 117, 117, 117, 0 }, { 117, 121, 119, 4 },
                { 121, 121, 121, 0 }, { 121, 121, 121, 0 }, { 121, 121, 121, 0 }, { 121, 125, 123, 4 }, { 125, 125, 125, 0 }, { 125, 125, 125, 0 }, { 125, 125, 125, 0 }, { 125, 130, 127, 5 },
                { 125, 130, 127, 5 }, { 125, 134, 129, 9 }, { 130, 130, 130, 0 }, { 125, 138, 131, 13 }, { 130, 134, 132, 4 }, { 125, 142, 133, 17 }, { 134, 134, 134, 0 }, { 125, 146, 135, 21 },
                { 134, 138, 136, 4 }, { 125, 150, 137, 25 }, { 138, 138, 138, 0 }, { 125, 154, 139, 29 }, { 138, 142, 140, 4 }, { 125, 158, 141, 33 }, { 142, 142, 142, 0 }, { 142, 142, 142, 0 },
                { 142, 146, 144, 4 }, { 146, 146, 146, 0 }, { 146, 146, 146, 0 }, { 146, 146, 146, 0 }, { 146, 150, 148, 4 }, { 150, 150, 150, 0 }, { 150, 150, 150, 0 }, { 150, 150, 150, 0 },
                { 150, 154, 152, 4 }, { 154, 154, 154, 0 }, { 154, 154, 154, 0 }, { 154, 154, 154, 0 }, { 154, 158, 156, 4 }, { 158, 158, 158, 0 }, { 158, 158, 158, 0 }, { 158, 158, 158, 0 },
                { 158, 162, 160, 4 }, { 162, 162, 162, 0 }, { 162, 162, 162, 0 }, { 162, 162, 162, 0 }, { 162, 166, 164, 4 }, { 166, 166, 166, 0 }, { 166, 166, 166, 0 }, { 166, 166, 166, 0 },
                { 166, 170, 168, 4 }, { 170, 170, 170, 0 }, { 170, 170, 170, 0 }, { 170, 170, 170, 0 }, { 170, 174, 172, 4 }, { 174, 174, 174, 0 }, { 174, 174, 174, 0 }, { 174, 174, 174, 0 },
                { 174, 178, 176, 4 }, { 178, 178, 178, 0 }, { 178, 178, 178, 0 }, { 178, 178, 178, 0 }, { 178, 182, 180, 4 }, { 182, 182, 182, 0 }, { 182, 182, 182, 0 }, { 182, 182, 182, 0 },
                { 182, 186, 184, 4 }, { 186, 186, 186, 0 }, { 186, 186, 186, 0 }, { 186, 186, 186, 0 }, { 186, 190, 188, 4 }, { 190, 190, 190, 0 }, { 190, 190, 190, 0 }, { 190, 190, 190, 0 },
                { 190, 195, 192, 5 }, { 190, 195, 192, 5 }, { 190, 199, 194, 9 }, { 195, 195, 195, 0 }, { 190, 203, 196, 13 }, { 195, 199, 197, 4 }, { 190, 207, 198, 17 }, { 199, 199, 199, 0 },
                { 190, 211, 200, 21 }, { 199, 203, 201, 4 }, { 190, 215, 202, 25 }, { 203, 203, 203, 0 }, { 190, 219, 204, 29 }, { 203, 207, 205, 4 }, { 190, 223, 206, 33 }, { 207, 207, 207, 0 },
                { 207, 207, 207, 0 }, { 207, 211, 209, 4 }, { 211, 211, 211, 0 }, { 211, 211, 211, 0 }, { 211, 211, 211, 0 }, { 211, 215, 213, 4 }, { 215, 215, 215, 0 }, { 215, 215, 215, 0 },
                { 215, 215, 215, 0 }, { 215, 219, 217, 4 }, { 219, 219, 219, 0 }, { 219, 219, 219, 0 }, { 219, 219, 219, 0 }, { 219, 223, 221, 4 }, { 223, 223, 223, 0 }, { 223, 223, 223, 0 },
                { 223, 223, 223, 0 }, { 223, 227, 225, 4 }, { 227, 227, 227, 0 }, { 227, 227, 227, 0 }, { 227, 227, 227, 0 }, { 227, 231, 229, 4 }, { 231, 231, 231, 0 }, { 231, 231, 231, 0 },
                { 231, 231, 231, 0 }, { 231, 235, 233, 4 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 235, 235, 235, 0 }, { 235, 239, 237, 4 }, { 239, 239, 239, 0 }, { 239, 239, 239, 0 },
                { 239, 239, 239, 0 }, { 239, 243, 241, 4 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 243, 243, 243, 0 }, { 243, 247, 245, 4 }, { 247, 247, 247, 0 }, { 247, 247, 247, 0 },
                { 247, 247, 247, 0 }, { 247, 251, 249, 4 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 251, 251, 251, 0 }, { 251, 255, 253, 4 }, { 255, 255, 255, 0 }, { 255, 255, 255, 0 },
            };
        }

        class S3TCComputer
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;

            static void Init(MFloat& error)
            {
                error = ParallelMath::MakeFloat(FLT_MAX);
            }

            static void QuantizeTo6Bits(MUInt15& v)
            {
                MUInt15 reduced = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(v, ParallelMath::MakeUInt15(253)) + ParallelMath::MakeUInt16(512), 10));
                v = (reduced << 2) | ParallelMath::RightShift(reduced, 4);
            }

            static void QuantizeTo5Bits(MUInt15& v)
            {
                MUInt15 reduced = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(v, ParallelMath::MakeUInt15(249)) + ParallelMath::MakeUInt16(1024), 11));
                v = (reduced << 3) | ParallelMath::RightShift(reduced, 2);
            }

            static void QuantizeTo565(MUInt15 endPoint[3])
            {
                QuantizeTo5Bits(endPoint[0]);
                QuantizeTo6Bits(endPoint[1]);
                QuantizeTo5Bits(endPoint[2]);
            }

            static MFloat ParanoidFactorForSpan(const MSInt16& span)
            {
                return ParallelMath::Abs(ParallelMath::ToFloat(span)) * 0.03f;
            }

            static MFloat ParanoidDiff(const MUInt15& a, const MUInt15& b, const MFloat& d)
            {
                MFloat absDiff = ParallelMath::Abs(ParallelMath::ToFloat(ParallelMath::LosslessCast<MSInt16>::Cast(a) - ParallelMath::LosslessCast<MSInt16>::Cast(b)));
                absDiff = absDiff + d;
                return absDiff * absDiff;
            }

            static void TestSingleColor(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], int range, const float* channelWeights,
                MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                float channelWeightsSq[3];

                for (int ch = 0; ch < 3; ch++)
                    channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

                MUInt15 totals[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 3; ch++)
                        totals[ch] = totals[ch] + pixels[px][ch];
                }

                MUInt15 average[3];
                for (int ch = 0; ch < 3; ch++)
                    average[ch] = ParallelMath::RightShift(totals[ch] + ParallelMath::MakeUInt15(8), 4);

                const S3TCSingleColorTables::SingleColorTableEntry* rbTable = NULL;
                const S3TCSingleColorTables::SingleColorTableEntry* gTable = NULL;
                if (flags & cvtt::Flags::S3TC_Paranoid)
                {
                    if (range == 4)
                    {
                        rbTable = S3TCSingleColorTables::g_singleColor5_3_p;
                        gTable = S3TCSingleColorTables::g_singleColor6_3_p;
                    }
                    else
                    {
                        assert(range == 3);
                        rbTable = S3TCSingleColorTables::g_singleColor5_2_p;
                        gTable = S3TCSingleColorTables::g_singleColor6_2_p;
                    }
                }
                else
                {
                    if (range == 4)
                    {
                        rbTable = S3TCSingleColorTables::g_singleColor5_3;
                        gTable = S3TCSingleColorTables::g_singleColor6_3;
                    }
                    else
                    {
                        assert(range == 3);
                        rbTable = S3TCSingleColorTables::g_singleColor5_2;
                        gTable = S3TCSingleColorTables::g_singleColor6_2;
                    }
                }

                MUInt15 interpolated[3];
                MUInt15 eps[2][3];
                MSInt16 spans[3];
                for (int i = 0; i < ParallelMath::ParallelSize; i++)
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        uint16_t avg = ParallelMath::Extract(average[ch], i);
                        const S3TCSingleColorTables::SingleColorTableEntry& tableEntry = ((ch == 1) ? gTable[avg] : rbTable[avg]);
                        ParallelMath::PutUInt15(eps[0][ch], i, tableEntry.m_min);
                        ParallelMath::PutUInt15(eps[1][ch], i, tableEntry.m_max);
                        ParallelMath::PutUInt15(interpolated[ch], i, tableEntry.m_actualColor);
                        ParallelMath::PutSInt16(spans[ch], i, tableEntry.m_span);
                    }
                }

                MFloat error = ParallelMath::MakeFloatZero();
                if (flags & cvtt::Flags::S3TC_Paranoid)
                {
                    MFloat spanParanoidFactors[3];
                    for (int ch = 0; ch < 3; ch++)
                        spanParanoidFactors[ch] = ParanoidFactorForSpan(spans[ch]);

                    for (int px = 0; px < 16; px++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                            error = error + ParanoidDiff(interpolated[ch], pixels[px][ch], spanParanoidFactors[ch]) * channelWeightsSq[ch];
                    }
                }
                else
                {
                    for (int px = 0; px < 16; px++)
                    {
                        for (int ch = 0; ch < 3; ch++)
                            error = error + ParallelMath::ToFloat(ParallelMath::SqDiffUInt8(interpolated[ch], pixels[px][ch])) * channelWeightsSq[ch];
                    }
                }

                ParallelMath::FloatCompFlag better = ParallelMath::Less(error, bestError);
                ParallelMath::Int16CompFlag better16 = ParallelMath::FloatFlagToInt16(better);

                if (ParallelMath::AnySet(better16))
                {
                    bestError = ParallelMath::Min(bestError, error);
                    for (int epi = 0; epi < 2; epi++)
                        for (int ch = 0; ch < 3; ch++)
                            ParallelMath::ConditionalSet(bestEndpoints[epi][ch], better16, eps[epi][ch]);

                    MUInt15 vindexes = ParallelMath::MakeUInt15(1);
                    for (int px = 0; px < 16; px++)
                        ParallelMath::ConditionalSet(bestIndexes[px], better16, vindexes);

                    ParallelMath::ConditionalSet(bestRange, better16, ParallelMath::MakeUInt15(range));
                }
            }

            static void TestEndpoints(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], const MUInt15 unquantizedEndPoints[2][3], int range, const float* channelWeights,
                MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, EndpointRefiner<3> *refiner, const ParallelMath::RoundTowardNearestForScope *rtn)
            {
                float channelWeightsSq[3];

                for (int ch = 0; ch < 3; ch++)
                    channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

                MUInt15 endPoints[2][3];

                for (int ep = 0; ep < 2; ep++)
                    for (int ch = 0; ch < 3; ch++)
                        endPoints[ep][ch] = unquantizedEndPoints[ep][ch];

                QuantizeTo565(endPoints[0]);
                QuantizeTo565(endPoints[1]);

                IndexSelector<3> selector;
                selector.Init<false>(channelWeights, endPoints, range);

                MUInt15 indexes[16];

                MFloat paranoidFactors[3];
                for (int ch = 0; ch < 3; ch++)
                    paranoidFactors[ch] = ParanoidFactorForSpan(ParallelMath::LosslessCast<MSInt16>::Cast(endPoints[0][ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(endPoints[1][ch]));

                MFloat error = ParallelMath::MakeFloatZero();
                AggregatedError<3> aggError;
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 index = selector.SelectIndexLDR(floatPixels[px], rtn);
                    indexes[px] = index;

                    if (refiner)
                        refiner->ContributeUnweightedPW(preWeightedPixels[px], index);

                    MUInt15 reconstructed[3];
                    selector.ReconstructLDRPrecise(index, reconstructed);

                    if (flags & Flags::S3TC_Paranoid)
                    {
                        for (int ch = 0; ch < 3; ch++)
                            error = error + ParanoidDiff(reconstructed[ch], pixels[px][ch], paranoidFactors[ch]) * channelWeightsSq[ch];
                    }
                    else
                        BCCommon::ComputeErrorLDR<3>(flags, reconstructed, pixels[px], aggError);
                }

                if (!(flags & Flags::S3TC_Paranoid))
                    error = aggError.Finalize(flags, channelWeightsSq);

                ParallelMath::FloatCompFlag better = ParallelMath::Less(error, bestError);

                if (ParallelMath::AnySet(better))
                {
                    ParallelMath::Int16CompFlag betterInt16 = ParallelMath::FloatFlagToInt16(better);

                    ParallelMath::ConditionalSet(bestError, better, error);

                    for (int ep = 0; ep < 2; ep++)
                        for (int ch = 0; ch < 3; ch++)
                            ParallelMath::ConditionalSet(bestEndpoints[ep][ch], betterInt16, endPoints[ep][ch]);

                    for (int px = 0; px < 16; px++)
                        ParallelMath::ConditionalSet(bestIndexes[px], betterInt16, indexes[px]);

                    ParallelMath::ConditionalSet(bestRange, betterInt16, ParallelMath::MakeUInt15(static_cast<uint16_t>(range)));
                }
            }

            static void TestCounts(uint32_t flags, const int *counts, int nCounts, const MUInt15 &numElements, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], bool alphaTest,
                const MFloat floatSortedInputs[16][4], const MFloat preWeightedFloatSortedInputs[16][4], const float *channelWeights, MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange,
                const ParallelMath::RoundTowardNearestForScope* rtn)
            {
                UNREFERENCED_PARAMETER(alphaTest);
                UNREFERENCED_PARAMETER(flags);

                EndpointRefiner<3> refiner;

                refiner.Init(nCounts, channelWeights);

                bool escape = false;
                int e = 0;
                for (int i = 0; i < nCounts; i++)
                {
                    for (int n = 0; n < counts[i]; n++)
                    {
                        ParallelMath::Int16CompFlag valid = ParallelMath::Less(ParallelMath::MakeUInt15(static_cast<uint16_t>(n)), numElements);
                        if (!ParallelMath::AnySet(valid))
                        {
                            escape = true;
                            break;
                        }

                        if (ParallelMath::AllSet(valid))
                            refiner.ContributeUnweightedPW(preWeightedFloatSortedInputs[e++], ParallelMath::MakeUInt15(static_cast<uint16_t>(i)));
                        else
                        {
                            MFloat weight = ParallelMath::Select(ParallelMath::Int16FlagToFloat(valid), ParallelMath::MakeFloat(1.0f), ParallelMath::MakeFloat(0.0f));
                            refiner.ContributePW(preWeightedFloatSortedInputs[e++], ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), weight);
                        }
                    }

                    if (escape)
                        break;
                }

                MUInt15 endPoints[2][3];
                refiner.GetRefinedEndpointsLDR(endPoints, rtn);

                TestEndpoints(flags, pixels, floatPixels, preWeightedPixels, endPoints, nCounts, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, NULL, rtn);
            }

            static void PackExplicitAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride)
            {
                UNREFERENCED_PARAMETER(flags);
                ParallelMath::RoundTowardNearestForScope rtn;

                float weights[1] = { 1.0f };

                MUInt15 pixels[16];
                MFloat floatPixels[16];

                for (int px = 0; px < 16; px++)
                {
                    ParallelMath::ConvertLDRInputs(inputs, px, inputChannel, pixels[px]);
                    floatPixels[px] = ParallelMath::ToFloat(pixels[px]);
                }

                MUInt15 ep[2][1] = { { ParallelMath::MakeUInt15(0) },{ ParallelMath::MakeUInt15(255) } };

                IndexSelector<1> selector;
                selector.Init<false>(weights, ep, 16);

                MUInt15 indexes[16];

                for (int px = 0; px < 16; px++)
                    indexes[px] = selector.SelectIndexLDR(&floatPixels[px], &rtn);

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    for (int px = 0; px < 16; px += 8)
                    {
                        int index0 = ParallelMath::Extract(indexes[px], block);
                        int index1 = ParallelMath::Extract(indexes[px], block);

                        packedBlocks[px / 2] = static_cast<uint8_t>(index0 | (index1 << 4));
                    }

                    packedBlocks += packedBlockStride;
                }
            }

            static void PackInterpolatedAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride, bool isSigned, int maxTweakRounds, int numRefineRounds)
            {
                if (maxTweakRounds < 1)
                    maxTweakRounds = 1;

                if (numRefineRounds < 1)
                    numRefineRounds = 1;

                ParallelMath::RoundTowardNearestForScope rtn;

                float oneWeight[1] = { 1.0f };

                MUInt15 pixels[16];
                MFloat floatPixels[16];

                MUInt15 highTerminal = isSigned ? ParallelMath::MakeUInt15(254) : ParallelMath::MakeUInt15(255);
                MUInt15 highTerminalMinusOne = highTerminal - ParallelMath::MakeUInt15(1);

                for (int px = 0; px < 16; px++)
                {
                    ParallelMath::ConvertLDRInputs(inputs, px, inputChannel, pixels[px]);

                    if (isSigned)
                        pixels[px] = ParallelMath::Min(pixels[px], highTerminal);

                    floatPixels[px] = ParallelMath::ToFloat(pixels[px]);
                }

                MUInt15 sortedPixels[16];
                for (int px = 0; px < 16; px++)
                    sortedPixels[px] = pixels[px];

                for (int sortEnd = 15; sortEnd > 0; sortEnd--)
                {
                    for (int sortOffset = 0; sortOffset < sortEnd; sortOffset++)
                    {
                        MUInt15 a = sortedPixels[sortOffset];
                        MUInt15 b = sortedPixels[sortOffset + 1];

                        sortedPixels[sortOffset] = ParallelMath::Min(a, b);
                        sortedPixels[sortOffset + 1] = ParallelMath::Max(a, b);
                    }
                }

                MUInt15 zero = ParallelMath::MakeUInt15(0);
                MUInt15 one = ParallelMath::MakeUInt15(1);

                MUInt15 bestIsFullRange = zero;
                MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
                MUInt15 bestEP[2] = { zero, zero };
                MUInt15 bestIndexes[16] = {
                    zero, zero, zero, zero,
                    zero, zero, zero, zero,
                    zero, zero, zero, zero,
                    zero, zero, zero, zero
                };

                // Full-precision
                {
                    MUInt15 minEP = sortedPixels[0];
                    MUInt15 maxEP = sortedPixels[15];

                    MFloat base[1] = { ParallelMath::ToFloat(minEP) };
                    MFloat offset[1] = { ParallelMath::ToFloat(maxEP - minEP) };

                    UnfinishedEndpoints<1> ufep = UnfinishedEndpoints<1>(base, offset);

                    int numTweakRounds = BCCommon::TweakRoundsForRange(8);
                    if (numTweakRounds > maxTweakRounds)
                        numTweakRounds = maxTweakRounds;

                    for (int tweak = 0; tweak < numTweakRounds; tweak++)
                    {
                        MUInt15 ep[2][1];

                        ufep.FinishLDR(tweak, 8, ep[0], ep[1]);

                        for (int refinePass = 0; refinePass < numRefineRounds; refinePass++)
                        {
                            EndpointRefiner<1> refiner;
                            refiner.Init(8, oneWeight);

                            if (isSigned)
                                for (int epi = 0; epi < 2; epi++)
                                    ep[epi][0] = ParallelMath::Min(ep[epi][0], highTerminal);

                            IndexSelector<1> indexSelector;
                            indexSelector.Init<false>(oneWeight, ep, 8);

                            MUInt15 indexes[16];

                            AggregatedError<1> aggError;
                            for (int px = 0; px < 16; px++)
                            {
                                MUInt15 index = indexSelector.SelectIndexLDR(&floatPixels[px], &rtn);

                                MUInt15 reconstructedPixel;

                                indexSelector.ReconstructLDRPrecise(index, &reconstructedPixel);
                                BCCommon::ComputeErrorLDR<1>(flags, &reconstructedPixel, &pixels[px], aggError);

                                if (refinePass != numRefineRounds - 1)
                                    refiner.ContributeUnweightedPW(&floatPixels[px], index);

                                indexes[px] = index;
                            }
                            MFloat error = aggError.Finalize(flags | Flags::Uniform, oneWeight);

                            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                            ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                            if (ParallelMath::AnySet(errorBetter16))
                            {
                                bestError = ParallelMath::Min(error, bestError);
                                ParallelMath::ConditionalSet(bestIsFullRange, errorBetter16, one);
                                for (int px = 0; px < 16; px++)
                                    ParallelMath::ConditionalSet(bestIndexes[px], errorBetter16, indexes[px]);

                                for (int epi = 0; epi < 2; epi++)
                                    ParallelMath::ConditionalSet(bestEP[epi], errorBetter16, ep[epi][0]);
                            }

                            if (refinePass != numRefineRounds - 1)
                                refiner.GetRefinedEndpointsLDR(ep, &rtn);
                        }
                    }
                }

                // Reduced precision with special endpoints
                {
                    MUInt15 bestHeuristicMin = sortedPixels[0];
                    MUInt15 bestHeuristicMax = sortedPixels[15];

                    ParallelMath::Int16CompFlag canTryClipping;

                    // In reduced precision, we want try putting endpoints at the reserved indexes at the ends.
                    // The heuristic we use is to assign indexes to the end as long as they aren't off by more than half of the index range.
                    // This will usually not find anything, but it's cheap to check.

                    {
                        MUInt15 largestPossibleRange = bestHeuristicMax - bestHeuristicMin; // Max: 255
                        MUInt15 lowestPossibleClearance = ParallelMath::Min(bestHeuristicMin, static_cast<MUInt15>(highTerminal - bestHeuristicMax));

                        MUInt15 lowestPossibleClearanceTimes10 = (lowestPossibleClearance << 2) + (lowestPossibleClearance << 4);
                        canTryClipping = ParallelMath::LessOrEqual(lowestPossibleClearanceTimes10, largestPossibleRange);
                    }

                    if (ParallelMath::AnySet(canTryClipping))
                    {
                        MUInt15 lowClearances[16];
                        MUInt15 highClearances[16];
                        MUInt15 bestSkipCount = ParallelMath::MakeUInt15(0);

                        lowClearances[0] = highClearances[0] = ParallelMath::MakeUInt15(0);

                        for (int px = 1; px < 16; px++)
                        {
                            lowClearances[px] = sortedPixels[px - 1];
                            highClearances[px] = highTerminal - sortedPixels[16 - px];
                        }

                        for (uint16_t firstIndex = 0; firstIndex < 16; firstIndex++)
                        {
                            uint16_t numSkippedLow = firstIndex;

                            MUInt15 lowClearance = lowClearances[firstIndex];

                            for (uint16_t lastIndex = firstIndex; lastIndex < 16; lastIndex++)
                            {
                                uint16_t numSkippedHigh = 15 - lastIndex;
                                uint16_t numSkipped = numSkippedLow + numSkippedHigh;

                                MUInt15 numSkippedV = ParallelMath::MakeUInt15(numSkipped);

                                ParallelMath::Int16CompFlag areMoreSkipped = ParallelMath::Less(bestSkipCount, numSkippedV);

                                if (!ParallelMath::AnySet(areMoreSkipped))
                                    continue;

                                MUInt15 clearance = ParallelMath::Max(highClearances[numSkippedHigh], lowClearance);
                                MUInt15 clearanceTimes10 = (clearance << 2) + (clearance << 4);

                                MUInt15 range = sortedPixels[lastIndex] - sortedPixels[firstIndex];

                                ParallelMath::Int16CompFlag isBetter = (areMoreSkipped & ParallelMath::LessOrEqual(clearanceTimes10, range));
                                ParallelMath::ConditionalSet(bestHeuristicMin, isBetter, sortedPixels[firstIndex]);
                                ParallelMath::ConditionalSet(bestHeuristicMax, isBetter, sortedPixels[lastIndex]);
                            }
                        }
                    }

                    MUInt15 bestSimpleMin = one;
                    MUInt15 bestSimpleMax = highTerminalMinusOne;

                    for (int px = 0; px < 16; px++)
                    {
                        ParallelMath::ConditionalSet(bestSimpleMin, ParallelMath::Less(zero, sortedPixels[15 - px]), sortedPixels[15 - px]);
                        ParallelMath::ConditionalSet(bestSimpleMax, ParallelMath::Less(sortedPixels[px], highTerminal), sortedPixels[px]);
                    }

                    MUInt15 minEPs[2] = { bestSimpleMin, bestHeuristicMin };
                    MUInt15 maxEPs[2] = { bestSimpleMax, bestHeuristicMax };

                    int minEPRange = 2;
                    if (ParallelMath::AllSet(ParallelMath::Equal(minEPs[0], minEPs[1])))
                        minEPRange = 1;

                    int maxEPRange = 2;
                    if (ParallelMath::AllSet(ParallelMath::Equal(maxEPs[0], maxEPs[1])))
                        maxEPRange = 1;

                    for (int minEPIndex = 0; minEPIndex < minEPRange; minEPIndex++)
                    {
                        for (int maxEPIndex = 0; maxEPIndex < maxEPRange; maxEPIndex++)
                        {
                            MFloat base[1] = { ParallelMath::ToFloat(minEPs[minEPIndex]) };
                            MFloat offset[1] = { ParallelMath::ToFloat(maxEPs[maxEPIndex] - minEPs[minEPIndex]) };

                            UnfinishedEndpoints<1> ufep = UnfinishedEndpoints<1>(base, offset);

                            int numTweakRounds = BCCommon::TweakRoundsForRange(6);
                            if (numTweakRounds > maxTweakRounds)
                                numTweakRounds = maxTweakRounds;

                            for (int tweak = 0; tweak < numTweakRounds; tweak++)
                            {
                                MUInt15 ep[2][1];

                                ufep.FinishLDR(tweak, 8, ep[0], ep[1]);

                                for (int refinePass = 0; refinePass < numRefineRounds; refinePass++)
                                {
                                    EndpointRefiner<1> refiner;
                                    refiner.Init(6, oneWeight);

                                    if (isSigned)
                                        for (int epi = 0; epi < 2; epi++)
                                            ep[epi][0] = ParallelMath::Min(ep[epi][0], highTerminal);

                                    IndexSelector<1> indexSelector;
                                    indexSelector.Init<false>(oneWeight, ep, 6);

                                    MUInt15 indexes[16];
                                    MFloat error = ParallelMath::MakeFloatZero();

                                    for (int px = 0; px < 16; px++)
                                    {
                                        MUInt15 selectedIndex = indexSelector.SelectIndexLDR(&floatPixels[px], &rtn);

                                        MUInt15 reconstructedPixel;

                                        indexSelector.ReconstructLDRPrecise(selectedIndex, &reconstructedPixel);

                                        MFloat zeroError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &zero, &pixels[px], 1, oneWeight);
                                        MFloat highTerminalError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &highTerminal, &pixels[px], 1, oneWeight);
                                        MFloat selectedIndexError = BCCommon::ComputeErrorLDRSimple<1>(flags | Flags::Uniform, &reconstructedPixel, &pixels[px], 1, oneWeight);

                                        MFloat bestPixelError = zeroError;
                                        MUInt15 index = ParallelMath::MakeUInt15(6);

                                        ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(ParallelMath::Less(highTerminalError, bestPixelError)), ParallelMath::MakeUInt15(7));
                                        bestPixelError = ParallelMath::Min(bestPixelError, highTerminalError);

                                        ParallelMath::FloatCompFlag selectedIndexBetter = ParallelMath::Less(selectedIndexError, bestPixelError);

                                        if (ParallelMath::AllSet(selectedIndexBetter))
                                        {
                                            if (refinePass != numRefineRounds - 1)
                                                refiner.ContributeUnweightedPW(&floatPixels[px], selectedIndex);
                                        }
                                        else
                                        {
                                            MFloat refineWeight = ParallelMath::Select(selectedIndexBetter, ParallelMath::MakeFloat(1.0f), ParallelMath::MakeFloatZero());

                                            if (refinePass != numRefineRounds - 1)
                                                refiner.ContributePW(&floatPixels[px], selectedIndex, refineWeight);
                                        }

                                        ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(selectedIndexBetter), selectedIndex);
                                        bestPixelError = ParallelMath::Min(bestPixelError, selectedIndexError);

                                        error = error + bestPixelError;

                                        indexes[px] = index;
                                    }

                                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                                    ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                                    if (ParallelMath::AnySet(errorBetter16))
                                    {
                                        bestError = ParallelMath::Min(error, bestError);
                                        ParallelMath::ConditionalSet(bestIsFullRange, errorBetter16, zero);
                                        for (int px = 0; px < 16; px++)
                                            ParallelMath::ConditionalSet(bestIndexes[px], errorBetter16, indexes[px]);

                                        for (int epi = 0; epi < 2; epi++)
                                            ParallelMath::ConditionalSet(bestEP[epi], errorBetter16, ep[epi][0]);
                                    }

                                    if (refinePass != numRefineRounds - 1)
                                        refiner.GetRefinedEndpointsLDR(ep, &rtn);
                                }
                            }
                        }
                    }
                }

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    int ep0 = ParallelMath::Extract(bestEP[0], block);
                    int ep1 = ParallelMath::Extract(bestEP[1], block);
                    int isFullRange = ParallelMath::Extract(bestIsFullRange, block);

                    if (isSigned)
                    {
                        ep0 -= 127;
                        ep1 -= 127;

                        assert(ep0 >= -127 && ep0 <= 127);
                        assert(ep1 >= -127 && ep1 <= 127);
                    }


                    bool swapEndpoints = (isFullRange != 0) != (ep0 > ep1);

                    if (swapEndpoints)
                        std::swap(ep0, ep1);

                    uint16_t dumpBits = 0;
                    int dumpBitsOffset = 0;
                    int dumpByteOffset = 2;
                    packedBlocks[0] = static_cast<uint8_t>(ep0 & 0xff);
                    packedBlocks[1] = static_cast<uint8_t>(ep1 & 0xff);

                    int maxValue = (isFullRange != 0) ? 7 : 5;

                    for (int px = 0; px < 16; px++)
                    {
                        int index = ParallelMath::Extract(bestIndexes[px], block);

                        if (swapEndpoints && index <= maxValue)
                            index = maxValue - index;

                        if (index != 0)
                        {
                            if (index == maxValue)
                                index = 1;
                            else if (index < maxValue)
                                index++;
                        }

                        assert(index >= 0 && index < 8);

                        dumpBits |= static_cast<uint16_t>(index << dumpBitsOffset);
                        dumpBitsOffset += 3;

                        if (dumpBitsOffset >= 8)
                        {
                            assert(dumpByteOffset < 8);
                            packedBlocks[dumpByteOffset] = static_cast<uint8_t>(dumpBits & 0xff);
                            dumpBits >>= 8;
                            dumpBitsOffset -= 8;
                            dumpByteOffset++;
                        }
                    }

                    assert(dumpBitsOffset == 0);
                    assert(dumpByteOffset == 8);

                    packedBlocks += packedBlockStride;
                }
            }

            static void PackRGB(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, size_t packedBlockStride, const float channelWeights[4], bool alphaTest, float alphaThreshold, bool exhaustive, int maxTweakRounds, int numRefineRounds)
            {
                ParallelMath::RoundTowardNearestForScope rtn;

                if (numRefineRounds < 1)
                    numRefineRounds = 1;

                if (maxTweakRounds < 1)
                    maxTweakRounds = 1;

                EndpointSelector<3, 8> endpointSelector;

                MUInt15 pixels[16][4];
                MFloat floatPixels[16][4];

                MFloat preWeightedPixels[16][4];

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 4; ch++)
                        ParallelMath::ConvertLDRInputs(inputs, px, ch, pixels[px][ch]);
                }

                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < 4; ch++)
                        floatPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
                }

                if (alphaTest)
                {
                    MUInt15 threshold = ParallelMath::MakeUInt15(static_cast<uint16_t>(floor(alphaThreshold * 255.0f + 0.5f)));

                    for (int px = 0; px < 16; px++)
                    {
                        ParallelMath::Int16CompFlag belowThreshold = ParallelMath::Less(pixels[px][3], threshold);
                        pixels[px][3] = ParallelMath::Select(belowThreshold, ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(255));
                    }
                }

                BCCommon::PreWeightPixelsLDR<4>(preWeightedPixels, pixels, channelWeights);

                MUInt15 minAlpha = ParallelMath::MakeUInt15(255);

                for (int px = 0; px < 16; px++)
                    minAlpha = ParallelMath::Min(minAlpha, pixels[px][3]);

                MFloat pixelWeights[16];
                for (int px = 0; px < 16; px++)
                {
                    pixelWeights[px] = ParallelMath::MakeFloat(1.0f);
                    if (alphaTest)
                    {
                        ParallelMath::Int16CompFlag isTransparent = ParallelMath::Less(pixels[px][3], ParallelMath::MakeUInt15(255));

                        ParallelMath::ConditionalSet(pixelWeights[px], ParallelMath::Int16FlagToFloat(isTransparent), ParallelMath::MakeFloatZero());
                    }
                }

                for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
                {
                    for (int px = 0; px < 16; px++)
                        endpointSelector.ContributePass(preWeightedPixels[px], pass, pixelWeights[px]);

                    endpointSelector.FinishPass(pass);
                }

                UnfinishedEndpoints<3> ufep = endpointSelector.GetEndpoints(channelWeights);

                MUInt15 bestEndpoints[2][3];
                MUInt15 bestIndexes[16];
                MUInt15 bestRange = ParallelMath::MakeUInt15(0);
                MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);

                for (int px = 0; px < 16; px++)
                    bestIndexes[px] = ParallelMath::MakeUInt15(0);

                for (int ep = 0; ep < 2; ep++)
                    for (int ch = 0; ch < 3; ch++)
                        bestEndpoints[ep][ch] = ParallelMath::MakeUInt15(0);

                if (exhaustive)
                {
                    MSInt16 sortBins[16];

                    {
                        // Compute an 11-bit index, change it to signed, stuff it in the high bits of the sort bins,
                        // and pack the original indexes into the low bits.

                        MUInt15 sortEP[2][3];
                        ufep.FinishLDR(0, 11, sortEP[0], sortEP[1]);

                        IndexSelector<3> sortSelector;
                        sortSelector.Init<false>(channelWeights, sortEP, 1 << 11);

                        for (int16_t px = 0; px < 16; px++)
                        {
                            MSInt16 sortBin = ParallelMath::LosslessCast<MSInt16>::Cast(sortSelector.SelectIndexLDR(floatPixels[px], &rtn) << 4);

                            if (alphaTest)
                            {
                                ParallelMath::Int16CompFlag isTransparent = ParallelMath::Less(pixels[px][3], ParallelMath::MakeUInt15(255));

                                ParallelMath::ConditionalSet(sortBin, isTransparent, ParallelMath::MakeSInt16(-16)); // 0xfff0
                            }

                            sortBin = sortBin + ParallelMath::MakeSInt16(px);

                            sortBins[px] = sortBin;
                        }
                    }

                    // Sort bins
                    for (int sortEnd = 1; sortEnd < 16; sortEnd++)
                    {
                        for (int sortLoc = sortEnd; sortLoc > 0; sortLoc--)
                        {
                            MSInt16 a = sortBins[sortLoc];
                            MSInt16 b = sortBins[sortLoc - 1];

                            sortBins[sortLoc] = ParallelMath::Max(a, b);
                            sortBins[sortLoc - 1] = ParallelMath::Min(a, b);
                        }
                    }

                    MUInt15 firstElement = ParallelMath::MakeUInt15(0);
                    for (uint16_t e = 0; e < 16; e++)
                    {
                        ParallelMath::Int16CompFlag isInvalid = ParallelMath::Less(sortBins[e], ParallelMath::MakeSInt16(0));
                        ParallelMath::ConditionalSet(firstElement, isInvalid, ParallelMath::MakeUInt15(e + 1));
                        if (!ParallelMath::AnySet(isInvalid))
                            break;
                    }

                    MUInt15 numElements = ParallelMath::MakeUInt15(16) - firstElement;

                    MUInt15 sortedInputs[16][4];
                    MFloat floatSortedInputs[16][4];
                    MFloat pwFloatSortedInputs[16][4];

                    for (int e = 0; e < 16; e++)
                    {
                        for (int ch = 0; ch < 4; ch++)
                            sortedInputs[e][ch] = ParallelMath::MakeUInt15(0);
                    }

                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        for (int e = ParallelMath::Extract(firstElement, block); e < 16; e++)
                        {
                            ParallelMath::ScalarUInt16 sortBin = ParallelMath::Extract(sortBins[e], block);
                            int originalIndex = (sortBin & 15);

                            for (int ch = 0; ch < 4; ch++)
                                ParallelMath::PutUInt15(sortedInputs[15 - e][ch], block, ParallelMath::Extract(pixels[originalIndex][ch], block));
                        }
                    }

                    for (int e = 0; e < 16; e++)
                    {
                        for (int ch = 0; ch < 4; ch++)
                        {
                            MFloat f = ParallelMath::ToFloat(sortedInputs[e][ch]);
                            floatSortedInputs[e][ch] = f;
                            pwFloatSortedInputs[e][ch] = f * channelWeights[ch];
                        }
                    }

                    for (int n0 = 0; n0 <= 15; n0++)
                    {
                        int remainingFor1 = 16 - n0;
                        if (remainingFor1 == 16)
                            remainingFor1 = 15;

                        for (int n1 = 0; n1 <= remainingFor1; n1++)
                        {
                            int remainingFor2 = 16 - n1 - n0;
                            if (remainingFor2 == 16)
                                remainingFor2 = 15;

                            for (int n2 = 0; n2 <= remainingFor2; n2++)
                            {
                                int n3 = 16 - n2 - n1 - n0;

                                if (n3 == 16)
                                    continue;

                                int counts[4] = { n0, n1, n2, n3 };

                                TestCounts(flags, counts, 4, numElements, pixels, floatPixels, preWeightedPixels, alphaTest, floatSortedInputs, pwFloatSortedInputs, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
                            }
                        }
                    }

                    TestSingleColor(flags, pixels, floatPixels, 4, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);

                    if (alphaTest)
                    {
                        for (int n0 = 0; n0 <= 15; n0++)
                        {
                            int remainingFor1 = 16 - n0;
                            if (remainingFor1 == 16)
                                remainingFor1 = 15;

                            for (int n1 = 0; n1 <= remainingFor1; n1++)
                            {
                                int n2 = 16 - n1 - n0;

                                if (n2 == 16)
                                    continue;

                                int counts[3] = { n0, n1, n2 };

                                TestCounts(flags, counts, 3, numElements, pixels, floatPixels, preWeightedPixels, alphaTest, floatSortedInputs, pwFloatSortedInputs, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
                            }
                        }

                        TestSingleColor(flags, pixels, floatPixels, 3, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &rtn);
                    }
                }
                else
                {
                    int minRange = alphaTest ? 3 : 4;

                    for (int range = minRange; range <= 4; range++)
                    {
                        int tweakRounds = BCCommon::TweakRoundsForRange(range);
                        if (tweakRounds > maxTweakRounds)
                            tweakRounds = maxTweakRounds;

                        for (int tweak = 0; tweak < tweakRounds; tweak++)
                        {
                            MUInt15 endPoints[2][3];

                            ufep.FinishLDR(tweak, range, endPoints[0], endPoints[1]);

                            for (int refine = 0; refine < numRefineRounds; refine++)
                            {
                                EndpointRefiner<3> refiner;
                                refiner.Init(range, channelWeights);

                                TestEndpoints(flags, pixels, floatPixels, preWeightedPixels, endPoints, range, channelWeights, bestError, bestEndpoints, bestIndexes, bestRange, &refiner, &rtn);

                                if (refine != numRefineRounds - 1)
                                    refiner.GetRefinedEndpointsLDR(endPoints, &rtn);
                            }
                        }
                    }
                }

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    ParallelMath::ScalarUInt16 range = ParallelMath::Extract(bestRange, block);
                    assert(range == 3 || range == 4);

                    ParallelMath::ScalarUInt16 compressedEP[2];
                    for (int ep = 0; ep < 2; ep++)
                    {
                        ParallelMath::ScalarUInt16 endPoint[3];
                        for (int ch = 0; ch < 3; ch++)
                            endPoint[ch] = ParallelMath::Extract(bestEndpoints[ep][ch], block);

                        int compressed = (endPoint[0] & 0xf8) << 8;
                        compressed |= (endPoint[1] & 0xfc) << 3;
                        compressed |= (endPoint[2] & 0xf8) >> 3;

                        compressedEP[ep] = static_cast<ParallelMath::ScalarUInt16>(compressed);
                    }

                    int indexOrder[4];

                    if (range == 4)
                    {
                        if (compressedEP[0] == compressedEP[1])
                        {
                            indexOrder[0] = 0;
                            indexOrder[1] = 0;
                            indexOrder[2] = 0;
                            indexOrder[3] = 0;
                        }
                        else if (compressedEP[0] < compressedEP[1])
                        {
                            std::swap(compressedEP[0], compressedEP[1]);
                            indexOrder[0] = 1;
                            indexOrder[1] = 3;
                            indexOrder[2] = 2;
                            indexOrder[3] = 0;
                        }
                        else
                        {
                            indexOrder[0] = 0;
                            indexOrder[1] = 2;
                            indexOrder[2] = 3;
                            indexOrder[3] = 1;
                        }
                    }
                    else
                    {
                        assert(range == 3);

                        if (compressedEP[0] > compressedEP[1])
                        {
                            std::swap(compressedEP[0], compressedEP[1]);
                            indexOrder[0] = 1;
                            indexOrder[1] = 2;
                            indexOrder[2] = 0;
                        }
                        else
                        {
                            indexOrder[0] = 0;
                            indexOrder[1] = 2;
                            indexOrder[2] = 1;
                        }
                        indexOrder[3] = 3;
                    }

                    packedBlocks[0] = static_cast<uint8_t>(compressedEP[0] & 0xff);
                    packedBlocks[1] = static_cast<uint8_t>((compressedEP[0] >> 8) & 0xff);
                    packedBlocks[2] = static_cast<uint8_t>(compressedEP[1] & 0xff);
                    packedBlocks[3] = static_cast<uint8_t>((compressedEP[1] >> 8) & 0xff);

                    for (int i = 0; i < 16; i += 4)
                    {
                        int packedIndexes = 0;
                        for (int subi = 0; subi < 4; subi++)
                        {
                            ParallelMath::ScalarUInt16 index = ParallelMath::Extract(bestIndexes[i + subi], block);
                            packedIndexes |= (indexOrder[index] << (subi * 2));
                        }

                        packedBlocks[4 + i / 4] = static_cast<uint8_t>(packedIndexes);
                    }

                    packedBlocks += packedBlockStride;
                }
            }
        };

        // Signed input blocks are converted into unsigned space, with the maximum value being 254
        void BiasSignedInput(PixelBlockU8 inputNormalized[ParallelMath::ParallelSize], const PixelBlockS8 inputSigned[ParallelMath::ParallelSize])
        {
            for (size_t block = 0; block < ParallelMath::ParallelSize; block++)
            {
                const PixelBlockS8& inputSignedBlock = inputSigned[block];
                PixelBlockU8& inputNormalizedBlock = inputNormalized[block];

                for (size_t px = 0; px < 16; px++)
                {
                    for (size_t ch = 0; ch < 4; ch++)
                        inputNormalizedBlock.m_pixels[px][ch] = static_cast<uint8_t>(std::max<int>(inputSignedBlock.m_pixels[px][ch], -127) + 127);
                }
            }
        }

        void FillWeights(const Options &options, float channelWeights[4])
        {
            if (options.flags & Flags::Uniform)
                channelWeights[0] = channelWeights[1] = channelWeights[2] = channelWeights[3] = 1.0f;
            else
            {
                channelWeights[0] = options.redWeight;
                channelWeights[1] = options.greenWeight;
                channelWeights[2] = options.blueWeight;
                channelWeights[3] = options.alphaWeight;
            }
        }
    }

    namespace Kernels
    {
        void EncodeBC7(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC7Computer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, options.seedPoints, options.refineRoundsBC7);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC6HU(uint8_t *pBC, const PixelBlockF16 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC6HComputer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, false, options.seedPoints, options.refineRoundsBC6H);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC6HS(uint8_t *pBC, const PixelBlockF16 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC6HComputer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, true, options.seedPoints, options.refineRoundsBC6H);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC1(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC, 8, channelWeights, true, options.threshold, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC2(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC + 8, 16, channelWeights, false, 1.0f, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                Internal::S3TCComputer::PackExplicitAlpha(options.flags, pBlocks + blockBase, 3, pBC, 16);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC3(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC + 8, 16, channelWeights, false, 1.0f, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 3, pBC, 16, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC4U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 0, pBC, 8, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC4S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                PixelBlockU8 inputBlocks[ParallelMath::ParallelSize];
                Internal::BiasSignedInput(inputBlocks, pBlocks + blockBase);

                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 0, pBC, 8, true, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC5U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 0, pBC, 16, false, options.seedPoints, options.refineRoundsIIC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 1, pBC + 8, 16, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC5S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Internal::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                PixelBlockU8 inputBlocks[ParallelMath::ParallelSize];
                Internal::BiasSignedInput(inputBlocks, pBlocks + blockBase);

                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 0, pBC, 16, true, options.seedPoints, options.refineRoundsIIC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 1, pBC + 8, 16, true, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void DecodeBC7(PixelBlockU8 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC7Computer::UnpackOne(pBlocks[blockBase], pBC);
                pBC += 16;
            }
        }

        void DecodeBC6HU(PixelBlockF16 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC6HComputer::UnpackOne(pBlocks[blockBase], pBC, false);
                pBC += 16;
            }
        }

        void DecodeBC6HS(PixelBlockF16 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC6HComputer::UnpackOne(pBlocks[blockBase], pBC, true);
                pBC += 16;
            }
        }
    }
}
