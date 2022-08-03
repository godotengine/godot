/// @ref core

#if GLM_ARCH & GLM_ARCH_SSE2_BIT

namespace glm{
namespace detail
{
/*
	template<qualifier Q>
	struct compute_quat_mul<float, Q, true>
	{
		static qua<float, Q> call(qua<float, Q> const& q1, qua<float, Q> const& q2)
		{
			// SSE2 STATS: 11 shuffle, 8 mul, 8 add
			// SSE4 STATS: 3 shuffle, 4 mul, 4 dpps

			__m128 const mul0 = _mm_mul_ps(q1.data, _mm_shuffle_ps(q2.data, q2.data, _MM_SHUFFLE(0, 1, 2, 3)));
			__m128 const mul1 = _mm_mul_ps(q1.data, _mm_shuffle_ps(q2.data, q2.data, _MM_SHUFFLE(1, 0, 3, 2)));
			__m128 const mul2 = _mm_mul_ps(q1.data, _mm_shuffle_ps(q2.data, q2.data, _MM_SHUFFLE(2, 3, 0, 1)));
			__m128 const mul3 = _mm_mul_ps(q1.data, q2.data);

#			if GLM_ARCH & GLM_ARCH_SSE41_BIT
				__m128 const add0 = _mm_dp_ps(mul0, _mm_set_ps(1.0f, -1.0f,  1.0f,  1.0f), 0xff);
				__m128 const add1 = _mm_dp_ps(mul1, _mm_set_ps(1.0f,  1.0f,  1.0f, -1.0f), 0xff);
				__m128 const add2 = _mm_dp_ps(mul2, _mm_set_ps(1.0f,  1.0f, -1.0f,  1.0f), 0xff);
				__m128 const add3 = _mm_dp_ps(mul3, _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f), 0xff);
#			else
				__m128 const mul4 = _mm_mul_ps(mul0, _mm_set_ps(1.0f, -1.0f,  1.0f,  1.0f));
				__m128 const add0 = _mm_add_ps(mul0, _mm_movehl_ps(mul4, mul4));
				__m128 const add4 = _mm_add_ss(add0, _mm_shuffle_ps(add0, add0, 1));

				__m128 const mul5 = _mm_mul_ps(mul1, _mm_set_ps(1.0f,  1.0f,  1.0f, -1.0f));
				__m128 const add1 = _mm_add_ps(mul1, _mm_movehl_ps(mul5, mul5));
				__m128 const add5 = _mm_add_ss(add1, _mm_shuffle_ps(add1, add1, 1));

				__m128 const mul6 = _mm_mul_ps(mul2, _mm_set_ps(1.0f,  1.0f, -1.0f,  1.0f));
				__m128 const add2 = _mm_add_ps(mul6, _mm_movehl_ps(mul6, mul6));
				__m128 const add6 = _mm_add_ss(add2, _mm_shuffle_ps(add2, add2, 1));

				__m128 const mul7 = _mm_mul_ps(mul3, _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f));
				__m128 const add3 = _mm_add_ps(mul3, _mm_movehl_ps(mul7, mul7));
				__m128 const add7 = _mm_add_ss(add3, _mm_shuffle_ps(add3, add3, 1));
		#endif

			// This SIMD code is a politically correct way of doing this, but in every test I've tried it has been slower than
			// the final code below. I'll keep this here for reference - maybe somebody else can do something better...
			//
			//__m128 xxyy = _mm_shuffle_ps(add4, add5, _MM_SHUFFLE(0, 0, 0, 0));
			//__m128 zzww = _mm_shuffle_ps(add6, add7, _MM_SHUFFLE(0, 0, 0, 0));
			//
			//return _mm_shuffle_ps(xxyy, zzww, _MM_SHUFFLE(2, 0, 2, 0));

			qua<float, Q> Result;
			_mm_store_ss(&Result.x, add4);
			_mm_store_ss(&Result.y, add5);
			_mm_store_ss(&Result.z, add6);
			_mm_store_ss(&Result.w, add7);
			return Result;
		}
	};
*/

	template<qualifier Q>
	struct compute_quat_add<float, Q, true>
	{
		static qua<float, Q> call(qua<float, Q> const& q, qua<float, Q> const& p)
		{
			qua<float, Q> Result;
			Result.data = _mm_add_ps(q.data, p.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_quat_add<double, Q, true>
	{
		static qua<double, Q> call(qua<double, Q> const& a, qua<double, Q> const& b)
		{
			qua<double, Q> Result;
			Result.data = _mm256_add_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_quat_sub<float, Q, true>
	{
		static qua<float, Q> call(qua<float, Q> const& q, qua<float, Q> const& p)
		{
			qua<float, Q> Result;
			Result.data = _mm_sub_ps(q.data, p.data);
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_quat_sub<double, Q, true>
	{
		static qua<double, Q> call(qua<double, Q> const& a, qua<double, Q> const& b)
		{
			qua<double, Q> Result;
			Result.data = _mm256_sub_pd(a.data, b.data);
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_quat_mul_scalar<float, Q, true>
	{
		static qua<float, Q> call(qua<float, Q> const& q, float s)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_mul_ps(q.data, _mm_set_ps1(s));
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_quat_mul_scalar<double, Q, true>
	{
		static qua<double, Q> call(qua<double, Q> const& q, double s)
		{
			qua<double, Q> Result;
			Result.data = _mm256_mul_pd(q.data, _mm_set_ps1(s));
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_quat_div_scalar<float, Q, true>
	{
		static qua<float, Q> call(qua<float, Q> const& q, float s)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_div_ps(q.data, _mm_set_ps1(s));
			return Result;
		}
	};

#	if GLM_ARCH & GLM_ARCH_AVX_BIT
	template<qualifier Q>
	struct compute_quat_div_scalar<double, Q, true>
	{
		static qua<double, Q> call(qua<double, Q> const& q, double s)
		{
			qua<double, Q> Result;
			Result.data = _mm256_div_pd(q.data, _mm_set_ps1(s));
			return Result;
		}
	};
#	endif

	template<qualifier Q>
	struct compute_quat_mul_vec4<float, Q, true>
	{
		static vec<4, float, Q> call(qua<float, Q> const& q, vec<4, float, Q> const& v)
		{
			__m128 const q_wwww = _mm_shuffle_ps(q.data, q.data, _MM_SHUFFLE(3, 3, 3, 3));
			__m128 const q_swp0 = _mm_shuffle_ps(q.data, q.data, _MM_SHUFFLE(3, 0, 2, 1));
			__m128 const q_swp1 = _mm_shuffle_ps(q.data, q.data, _MM_SHUFFLE(3, 1, 0, 2));
			__m128 const v_swp0 = _mm_shuffle_ps(v.data, v.data, _MM_SHUFFLE(3, 0, 2, 1));
			__m128 const v_swp1 = _mm_shuffle_ps(v.data, v.data, _MM_SHUFFLE(3, 1, 0, 2));

			__m128 uv      = _mm_sub_ps(_mm_mul_ps(q_swp0, v_swp1), _mm_mul_ps(q_swp1, v_swp0));
			__m128 uv_swp0 = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 0, 2, 1));
			__m128 uv_swp1 = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 1, 0, 2));
			__m128 uuv     = _mm_sub_ps(_mm_mul_ps(q_swp0, uv_swp1), _mm_mul_ps(q_swp1, uv_swp0));

			__m128 const two = _mm_set1_ps(2.0f);
			uv  = _mm_mul_ps(uv, _mm_mul_ps(q_wwww, two));
			uuv = _mm_mul_ps(uuv, two);

			vec<4, float, Q> Result;
			Result.data = _mm_add_ps(v.data, _mm_add_ps(uv, uuv));
			return Result;
		}
	};
}//namespace detail
}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT
