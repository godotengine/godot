#include "../simd/integer.h"

#if GLM_ARCH & GLM_ARCH_SSE2_BIT

namespace glm{
namespace detail
{
	template<qualifier Q>
	struct compute_bitfieldReverseStep<4, uint, Q, true, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint, Q> call(vec<4, uint, Q> const& v, uint Mask, uint Shift)
		{
			__m128i const set0 = v.data;

			__m128i const set1 = _mm_set1_epi32(static_cast<int>(Mask));
			__m128i const and1 = _mm_and_si128(set0, set1);
			__m128i const sft1 = _mm_slli_epi32(and1, Shift);

			__m128i const set2 = _mm_andnot_si128(set0, _mm_set1_epi32(-1));
			__m128i const and2 = _mm_and_si128(set0, set2);
			__m128i const sft2 = _mm_srai_epi32(and2, Shift);

			__m128i const or0 = _mm_or_si128(sft1, sft2);

			return or0;
		}
	};

	template<qualifier Q>
	struct compute_bitfieldBitCountStep<4, uint, Q, true, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint, Q> call(vec<4, uint, Q> const& v, uint Mask, uint Shift)
		{
			__m128i const set0 = v.data;

			__m128i const set1 = _mm_set1_epi32(static_cast<int>(Mask));
			__m128i const and0 = _mm_and_si128(set0, set1);
			__m128i const sft0 = _mm_slli_epi32(set0, Shift);
			__m128i const and1 = _mm_and_si128(sft0, set1);
			__m128i const add0 = _mm_add_epi32(and0, and1);

			return add0;
		}
	};
}//namespace detail

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<>
	GLM_FUNC_QUALIFIER int bitCount(uint x)
	{
		return _mm_popcnt_u32(x);
	}

#	if(GLM_MODEL == GLM_MODEL_64)
	template<>
	GLM_FUNC_QUALIFIER int bitCount(detail::uint64 x)
	{
		return static_cast<int>(_mm_popcnt_u64(x));
	}
#	endif//GLM_MODEL
#	endif//GLM_ARCH

}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT
