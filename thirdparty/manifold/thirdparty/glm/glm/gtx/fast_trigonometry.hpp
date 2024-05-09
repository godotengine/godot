/// @ref gtx_fast_trigonometry
/// @file glm/gtx/fast_trigonometry.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_fast_trigonometry GLM_GTX_fast_trigonometry
/// @ingroup gtx
///
/// Include <glm/gtx/fast_trigonometry.hpp> to use the features of this extension.
///
/// Fast but less accurate implementations of trigonometric functions.

#pragma once

// Dependency:
#include "../gtc/constants.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_fast_trigonometry is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_fast_trigonometry extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_fast_trigonometry
	/// @{

	/// Wrap an angle to [0 2pi[
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T wrapAngle(T angle);

	/// Faster than the common sin function but less accurate.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastSin(T angle);

	/// Faster than the common cos function but less accurate.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastCos(T angle);

	/// Faster than the common tan function but less accurate.
	/// Defined between -2pi and 2pi.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastTan(T angle);

	/// Faster than the common asin function but less accurate.
	/// Defined between -2pi and 2pi.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastAsin(T angle);

	/// Faster than the common acos function but less accurate.
	/// Defined between -2pi and 2pi.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastAcos(T angle);

	/// Faster than the common atan function but less accurate.
	/// Defined between -2pi and 2pi.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastAtan(T y, T x);

	/// Faster than the common atan function but less accurate.
	/// Defined between -2pi and 2pi.
	/// From GLM_GTX_fast_trigonometry extension.
	template<typename T>
	GLM_FUNC_DECL T fastAtan(T angle);

	/// @}
}//namespace glm

#include "fast_trigonometry.inl"
