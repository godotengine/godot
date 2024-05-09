namespace glm{
namespace detail
{
	template<length_t C, length_t R, typename T, qualifier Q, bool Aligned>
	struct compute_matrixCompMult_type<C, R, T, Q, false, Aligned> {
		GLM_FUNC_QUALIFIER static mat<C, R, T, Q> call(mat<C, R, T, Q> const& x, mat<C, R, T, Q> const& y)
		{
			return detail::compute_matrixCompMult<C, R, T, Q, detail::is_aligned<Q>::value>::call(x, y);
		}
	};

	template<length_t DA, length_t DB, typename T, qualifier Q>
	struct compute_outerProduct_type<DA, DB, T, Q, false> {
		GLM_FUNC_QUALIFIER static typename detail::outerProduct_trait<DA, DB, T, Q>::type call(vec<DA, T, Q> const& c, vec<DB, T, Q> const& r)
		{
			return detail::compute_outerProduct<DA, DB, T, Q>::call(c, r);
		}
	};

	template<length_t C, length_t R, typename T, qualifier Q, bool Aligned>
	struct compute_transpose_type<C, R, T, Q, false, Aligned>
	{
		GLM_FUNC_QUALIFIER static mat<R, C, T, Q> call(mat<C, R, T, Q> const& m)
		{
			return detail::compute_transpose<C, R, T, Q, detail::is_aligned<Q>::value>::call(m);
		}
	};

	template<length_t C, length_t R, typename T, qualifier Q, bool Aligned>
	struct compute_determinant_type<C, R, T, Q, false, Aligned>{
	
		GLM_FUNC_QUALIFIER static T call(mat<C, R, T, Q> const& m)
		{
			return detail::compute_determinant<C, R, T, Q, detail::is_aligned<Q>::value>::call(m);
		}
	};
}//namespace detail
}//namespace glm
