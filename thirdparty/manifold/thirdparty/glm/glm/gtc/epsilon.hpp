/// @ref gtc_epsilon
/// @file glm/gtc/epsilon.hpp
///
/// @see core (dependence)
/// @see gtc_quaternion (dependence)
///
/// @defgroup gtc_epsilon GLM_GTC_epsilon
/// @ingroup gtc
///
/// Include <glm/gtc/epsilon.hpp> to use the features of this extension.
///
/// Comparison functions for a user defined epsilon values.

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_epsilon extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_epsilon
	/// @{

	/// Returns the component-wise comparison of |x - y| < epsilon.
	/// True if this expression is satisfied.
	///
	/// @see gtc_epsilon
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, bool, Q> epsilonEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, T const& epsilon);

	/// Returns the component-wise comparison of |x - y| < epsilon.
	/// True if this expression is satisfied.
	///
	/// @see gtc_epsilon
	template<typename genType>
	GLM_FUNC_DECL bool epsilonEqual(genType const& x, genType const& y, genType const& epsilon);

	/// Returns the component-wise comparison of |x - y| < epsilon.
	/// True if this expression is not satisfied.
	///
	/// @see gtc_epsilon
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, bool, Q> epsilonNotEqual(vec<L, T, Q> const& x, vec<L, T, Q> const& y, T const& epsilon);

	/// Returns the component-wise comparison of |x - y| >= epsilon.
	/// True if this expression is not satisfied.
	///
	/// @see gtc_epsilon
	template<typename genType>
	GLM_FUNC_DECL bool epsilonNotEqual(genType const& x, genType const& y, genType const& epsilon);

	/// @}
}//namespace glm

#include "epsilon.inl"
