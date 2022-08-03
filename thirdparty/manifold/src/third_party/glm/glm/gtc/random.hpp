/// @ref gtc_random
/// @file glm/gtc/random.hpp
///
/// @see core (dependence)
/// @see gtx_random (extended)
///
/// @defgroup gtc_random GLM_GTC_random
/// @ingroup gtc
///
/// Include <glm/gtc/random.hpp> to use the features of this extension.
///
/// Generate random number from various distribution methods.

#pragma once

// Dependency:
#include "../ext/scalar_int_sized.hpp"
#include "../ext/scalar_uint_sized.hpp"
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_random extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_random
	/// @{

	/// Generate random numbers in the interval [Min, Max], according a linear distribution
	///
	/// @param Min Minimum value included in the sampling
	/// @param Max Maximum value included in the sampling
	/// @tparam genType Value type. Currently supported: float or double scalars.
	/// @see gtc_random
	template<typename genType>
	GLM_FUNC_DECL genType linearRand(genType Min, genType Max);

	/// Generate random numbers in the interval [Min, Max], according a linear distribution
	///
	/// @param Min Minimum value included in the sampling
	/// @param Max Maximum value included in the sampling
	/// @tparam T Value type. Currently supported: float or double.
	///
	/// @see gtc_random
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> linearRand(vec<L, T, Q> const& Min, vec<L, T, Q> const& Max);

	/// Generate random numbers in the interval [Min, Max], according a gaussian distribution
	///
	/// @see gtc_random
	template<typename genType>
	GLM_FUNC_DECL genType gaussRand(genType Mean, genType Deviation);

	/// Generate a random 2D vector which coordinates are regulary distributed on a circle of a given radius
	///
	/// @see gtc_random
	template<typename T>
	GLM_FUNC_DECL vec<2, T, defaultp> circularRand(T Radius);

	/// Generate a random 3D vector which coordinates are regulary distributed on a sphere of a given radius
	///
	/// @see gtc_random
	template<typename T>
	GLM_FUNC_DECL vec<3, T, defaultp> sphericalRand(T Radius);

	/// Generate a random 2D vector which coordinates are regulary distributed within the area of a disk of a given radius
	///
	/// @see gtc_random
	template<typename T>
	GLM_FUNC_DECL vec<2, T, defaultp> diskRand(T Radius);

	/// Generate a random 3D vector which coordinates are regulary distributed within the volume of a ball of a given radius
	///
	/// @see gtc_random
	template<typename T>
	GLM_FUNC_DECL vec<3, T, defaultp> ballRand(T Radius);

	/// @}
}//namespace glm

#include "random.inl"
