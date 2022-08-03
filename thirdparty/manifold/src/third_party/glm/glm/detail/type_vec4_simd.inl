#if GLM_ARCH & GLM_ARCH_SSE2_BIT

namespace glm{
namespace detail
{
#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	template<qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<4, float, Q, E0,E1,E2,E3, true> : public _swizzle_base0<float, 4>
	{
		GLM_FUNC_QUALIFIER vec<4, float, Q> operator ()()  const
		{
			__m128 data = *reinterpret_cast<__m128 const*>(&this->_buffer);

			vec<4, float, Q> Result;
#			if GLM_ARCH & GLM_ARCH_AVX_BIT
				Result.data = _mm_permute_ps(data, _MM_SHUFFLE(E3, E2, E1, E0));
#			else
				Result.data = _mm_shuffle_ps(data, data, _MM_SHUFFLE(E3, E2, E1, E0));
#			endif
			return Result;
		}
	};

	template<qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<4, int, Q, E0,E1,E2,E3, true> : public _swizzle_base0<int, 4>
	{
		GLM_FUNC_QUALIFIER vec<4, int, Q> operator ()()  const
		{
			__m128i data = *reinterpret_cast<__m128i const*>(&this->_buffer);

			vec<4, int, Q> Result;
			Result.data = _mm_shuffle_epi32(data, _MM_SHUFFLE(E3, E2, E1, E0));
			return Result;
		}
	};

	template<qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<4, uint, Q, E0,E1,E2,E3, true> : public _swizzle_base0<uint, 4>
	{
		GLM_FUNC_QUALIFIER vec<4, uint, Q> operator ()()  const
		{
			__m128i data = *reinterpret_cast<__m128i const*>(&this->_buffer);

			vec<4, uint, Q> Result;
			Result.data = _mm_shuffle_epi32(data, _MM_SHUFFLE(E3, E2, E1, E0));
			return Result;
		}
	};
#	endif// GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

	template<qualifier Q>
	struct compute_vec4_add<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_add_ps(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_vec4_add<double, Q, true>
	{
		static vec<4, double, Q> call(vec<4, double, Q> const& a, vec<4, double, Q> const& b)
		{
			vec<4, double, Q> Result;
			Result.data = _mm256_add_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_vec4_sub<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_sub_ps(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_vec4_sub<double, Q, true>
	{
		static vec<4, double, Q> call(vec<4, double, Q> const& a, vec<4, double, Q> const& b)
		{
			vec<4, double, Q> Result;
			Result.data = _mm256_sub_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_vec4_mul<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_mul_ps(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_vec4_mul<double, Q, true>
	{
		static vec<4, double, Q> call(vec<4, double, Q> const& a, vec<4, double, Q> const& b)
		{
			vec<4, double, Q> Result;
			Result.data = _mm256_mul_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_vec4_div<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_div_ps(a.data, b.data);
			return Result;
		}
	};

	#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_vec4_div<double, Q, true>
	{
		static vec<4, double, Q> call(vec<4, double, Q> const& a, vec<4, double, Q> const& b)
		{
			vec<4, double, Q> Result;
			Result.data = _mm256_div_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<>
	struct compute_vec4_div<float, aligned_lowp, true>
	{
		static vec<4, float, aligned_lowp> call(vec<4, float, aligned_lowp> const& a, vec<4, float, aligned_lowp> const& b)
		{
			vec<4, float, aligned_lowp> Result;
			Result.data = _mm_mul_ps(a.data, _mm_rcp_ps(b.data));
			return Result;
		}
	};

	template<typename T, qualifier Q>
	struct compute_vec4_and<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_and_si128(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_and<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_and_si256(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<typename T, qualifier Q>
	struct compute_vec4_or<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_or_si128(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_or<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_or_si256(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<typename T, qualifier Q>
	struct compute_vec4_xor<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_xor_si128(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_xor<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_xor_si256(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<typename T, qualifier Q>
	struct compute_vec4_shift_left<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_sll_epi32(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_shift_left<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_sll_epi64(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<typename T, qualifier Q>
	struct compute_vec4_shift_right<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_srl_epi32(a.data, b.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_shift_right<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& a, vec<4, T, Q> const& b)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_srl_epi64(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<typename T, qualifier Q>
	struct compute_vec4_bitwise_not<T, Q, true, 32, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& v)
		{
			vec<4, T, Q> Result;
			Result.data = _mm_xor_si128(v.data, _mm_set1_epi32(-1));
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<typename T, qualifier Q>
	struct compute_vec4_bitwise_not<T, Q, true, 64, true>
	{
		static vec<4, T, Q> call(vec<4, T, Q> const& v)
		{
			vec<4, T, Q> Result;
			Result.data = _mm256_xor_si256(v.data, _mm_set1_epi32(-1));
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_vec4_equal<float, Q, false, 32, true>
	{
		static bool call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			return _mm_movemask_ps(_mm_cmpneq_ps(v1.data, v2.data)) == 0;
		}
	};

#	if GLM_ARCH & GLM_ARCH_SSE41_BIT
	template<qualifier Q>
	struct compute_vec4_equal<int, Q, true, 32, true>
	{
		static bool call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			//return _mm_movemask_epi8(_mm_cmpeq_epi32(v1.data, v2.data)) != 0;
			__m128i neq = _mm_xor_si128(v1.data, v2.data);
			return _mm_test_all_zeros(neq, neq) == 0;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_vec4_nequal<float, Q, false, 32, true>
	{
		static bool call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			return _mm_movemask_ps(_mm_cmpneq_ps(v1.data, v2.data)) != 0;
		}
	};

#	if GLM_ARCH & GLM_ARCH_SSE41_BIT
	template<qualifier Q>
	struct compute_vec4_nequal<int, Q, true, 32, true>
	{
		static bool call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			//return _mm_movemask_epi8(_mm_cmpneq_epi32(v1.data, v2.data)) != 0;
			__m128i neq = _mm_xor_si128(v1.data, v2.data);
			return _mm_test_all_zeros(neq, neq) != 0;
		}
	};
#	endif
}//namespace detail

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(float _s) :
		data(_mm_set1_ps(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(float _s) :
		data(_mm_set1_ps(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(float _s) :
		data(_mm_set1_ps(_s))
	{}

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, double, aligned_lowp>::vec(double _s) :
		data(_mm256_set1_pd(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, double, aligned_mediump>::vec(double _s) :
		data(_mm256_set1_pd(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, double, aligned_highp>::vec(double _s) :
		data(_mm256_set1_pd(_s))
	{}
#	endif

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_lowp>::vec(int _s) :
		data(_mm_set1_epi32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_mediump>::vec(int _s) :
		data(_mm_set1_epi32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_highp>::vec(int _s) :
		data(_mm_set1_epi32(_s))
	{}

#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, detail::int64, aligned_lowp>::vec(detail::int64 _s) :
		data(_mm256_set1_epi64x(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, detail::int64, aligned_mediump>::vec(detail::int64 _s) :
		data(_mm256_set1_epi64x(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, detail::int64, aligned_highp>::vec(detail::int64 _s) :
		data(_mm256_set1_epi64x(_s))
	{}
#	endif

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(float _x, float _y, float _z, float _w) :
		data(_mm_set_ps(_w, _z, _y, _x))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(float _x, float _y, float _z, float _w) :
		data(_mm_set_ps(_w, _z, _y, _x))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(float _x, float _y, float _z, float _w) :
		data(_mm_set_ps(_w, _z, _y, _x))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_lowp>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_set_epi32(_w, _z, _y, _x))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_mediump>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_set_epi32(_w, _z, _y, _x))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_highp>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_set_epi32(_w, _z, _y, _x))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_cvtepi32_ps(_mm_set_epi32(_w, _z, _y, _x)))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_cvtepi32_ps(_mm_set_epi32(_w, _z, _y, _x)))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(int _x, int _y, int _z, int _w) :
		data(_mm_cvtepi32_ps(_mm_set_epi32(_w, _z, _y, _x)))
	{}
}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

#if GLM_ARCH & GLM_ARCH_NEON_BIT
namespace glm {
namespace detail {

	template<qualifier Q>
	struct compute_vec4_add<float, Q, true>
	{
		static
		vec<4, float, Q>
		call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = vaddq_f32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_add<uint, Q, true>
	{
		static
		vec<4, uint, Q>
		call(vec<4, uint, Q> const& a, vec<4, uint, Q> const& b)
		{
			vec<4, uint, Q> Result;
			Result.data = vaddq_u32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_add<int, Q, true>
	{
		static
		vec<4, int, Q>
		call(vec<4, int, Q> const& a, vec<4, int, Q> const& b)
		{
			vec<4, int, Q> Result;
			Result.data = vaddq_s32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_sub<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = vsubq_f32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_sub<uint, Q, true>
	{
		static vec<4, uint, Q> call(vec<4, uint, Q> const& a, vec<4, uint, Q> const& b)
		{
			vec<4, uint, Q> Result;
			Result.data = vsubq_u32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_sub<int, Q, true>
	{
		static vec<4, int, Q> call(vec<4, int, Q> const& a, vec<4, int, Q> const& b)
		{
			vec<4, int, Q> Result;
			Result.data = vsubq_s32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_mul<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = vmulq_f32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_mul<uint, Q, true>
	{
		static vec<4, uint, Q> call(vec<4, uint, Q> const& a, vec<4, uint, Q> const& b)
		{
			vec<4, uint, Q> Result;
			Result.data = vmulq_u32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_mul<int, Q, true>
	{
		static vec<4, int, Q> call(vec<4, int, Q> const& a, vec<4, int, Q> const& b)
		{
			vec<4, int, Q> Result;
			Result.data = vmulq_s32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_div<float, Q, true>
	{
		static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b)
		{
			vec<4, float, Q> Result;
			Result.data = vdivq_f32(a.data, b.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_vec4_equal<float, Q, false, 32, true>
	{
		static bool call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_f32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<qualifier Q>
	struct compute_vec4_equal<uint, Q, false, 32, true>
	{
		static bool call(vec<4, uint, Q> const& v1, vec<4, uint, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_u32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<qualifier Q>
	struct compute_vec4_equal<int, Q, false, 32, true>
	{
		static bool call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_s32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<qualifier Q>
	struct compute_vec4_nequal<float, Q, false, 32, true>
	{
		static bool call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			return !compute_vec4_equal<float, Q, false, 32, true>::call(v1, v2);
		}
	};

	template<qualifier Q>
	struct compute_vec4_nequal<uint, Q, false, 32, true>
	{
		static bool call(vec<4, uint, Q> const& v1, vec<4, uint, Q> const& v2)
		{
			return !compute_vec4_equal<uint, Q, false, 32, true>::call(v1, v2);
		}
	};

	template<qualifier Q>
	struct compute_vec4_nequal<int, Q, false, 32, true>
	{
		static bool call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			return !compute_vec4_equal<int, Q, false, 32, true>::call(v1, v2);
		}
	};

}//namespace detail

#if !GLM_CONFIG_XYZW_ONLY
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(float _s) :
		data(vdupq_n_f32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(float _s) :
		data(vdupq_n_f32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(float _s) :
		data(vdupq_n_f32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_lowp>::vec(int _s) :
		data(vdupq_n_s32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_mediump>::vec(int _s) :
		data(vdupq_n_s32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, int, aligned_highp>::vec(int _s) :
		data(vdupq_n_s32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, uint, aligned_lowp>::vec(uint _s) :
		data(vdupq_n_u32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, uint, aligned_mediump>::vec(uint _s) :
		data(vdupq_n_u32(_s))
	{}

	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, uint, aligned_highp>::vec(uint _s) :
		data(vdupq_n_u32(_s))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(const vec<4, float, aligned_highp>& rhs) :
		data(rhs.data)
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(const vec<4, int, aligned_highp>& rhs) :
		data(vcvtq_f32_s32(rhs.data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(const vec<4, uint, aligned_highp>& rhs) :
		data(vcvtq_f32_u32(rhs.data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(int _x, int _y, int _z, int _w) :
		data(vcvtq_f32_s32(vec<4, int, aligned_lowp>(_x, _y, _z, _w).data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(int _x, int _y, int _z, int _w) :
		data(vcvtq_f32_s32(vec<4, int, aligned_mediump>(_x, _y, _z, _w).data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(int _x, int _y, int _z, int _w) :
		data(vcvtq_f32_s32(vec<4, int, aligned_highp>(_x, _y, _z, _w).data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_lowp>::vec(uint _x, uint _y, uint _z, uint _w) :
		data(vcvtq_f32_u32(vec<4, uint, aligned_lowp>(_x, _y, _z, _w).data))
	{}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_mediump>::vec(uint _x, uint _y, uint _z, uint _w) :
		data(vcvtq_f32_u32(vec<4, uint, aligned_mediump>(_x, _y, _z, _w).data))
	{}


	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, float, aligned_highp>::vec(uint _x, uint _y, uint _z, uint _w) :
		data(vcvtq_f32_u32(vec<4, uint, aligned_highp>(_x, _y, _z, _w).data))
	{}

#endif
}//namespace glm

#endif
