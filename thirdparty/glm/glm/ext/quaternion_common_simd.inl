#if GLM_ARCH & GLM_ARCH_SSE2_BIT

namespace glm{
namespace detail
{
	template<qualifier Q>
	struct compute_dot<qua<float, Q>, float, true>
	{
		static GLM_FUNC_QUALIFIER float call(qua<float, Q> const& x, qua<float, Q> const& y)
		{
			return _mm_cvtss_f32(glm_vec1_dot(x.data, y.data));
		}
	};
}//namespace detail
}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

