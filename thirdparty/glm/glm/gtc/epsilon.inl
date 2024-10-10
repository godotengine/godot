/// @ref gtc_epsilon

// Dependency:
#include "../vector_relational.hpp"
#include "../common.hpp"

namespace glm
{
	template<>
	GLM_FUNC_QUALIFIER bool epsilonEqual
	(
		float const& x,
		float const& y,
		float const& epsilon
	)
	{
		return abs(x - y) < epsilon;
	}

	template<>
	GLM_FUNC_QUALIFIER bool epsilonEqual
	(
		double const& x,
		double const& y,
		double const& epsilon
	)
	{
		return abs(x - y) < epsilon;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> epsilonEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, T const& epsilon)
	{
		return lessThan(abs(x - y), vec<L, T, Q>(epsilon));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> epsilonEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, vec<L, T, Q> const& epsilon)
	{
		return lessThan(abs(x - y), vec<L, T, Q>(epsilon));
	}

	template<>
	GLM_FUNC_QUALIFIER bool epsilonNotEqual(float const& x, float const& y, float const& epsilon)
	{
		return abs(x - y) >= epsilon;
	}

	template<>
	GLM_FUNC_QUALIFIER bool epsilonNotEqual(double const& x, double const& y, double const& epsilon)
	{
		return abs(x - y) >= epsilon;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> epsilonNotEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, T const& epsilon)
	{
		return greaterThanEqual(abs(x - y), vec<L, T, Q>(epsilon));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> epsilonNotEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, vec<L, T, Q> const& epsilon)
	{
		return greaterThanEqual(abs(x - y), vec<L, T, Q>(epsilon));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, bool, Q> epsilonEqual(qua<T, Q> const& x, qua<T, Q> const& y, T const& epsilon)
	{
		vec<4, T, Q> v(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
		return lessThan(abs(v), vec<4, T, Q>(epsilon));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, bool, Q> epsilonNotEqual(qua<T, Q> const& x, qua<T, Q> const& y, T const& epsilon)
	{
		vec<4, T, Q> v(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
		return greaterThanEqual(abs(v), vec<4, T, Q>(epsilon));
	}
}//namespace glm
