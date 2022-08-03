/// @ref ext_quaternion_float
/// @file glm/ext/quaternion_float.hpp
///
/// @defgroup ext_quaternion_float GLM_EXT_quaternion_float
/// @ingroup ext
///
/// Exposes single-precision floating point quaternion type.
///
/// Include <glm/ext/quaternion_float.hpp> to use the features of this extension.
///
/// @see ext_quaternion_double
/// @see ext_quaternion_float_precision
/// @see ext_quaternion_common
/// @see ext_quaternion_exponential
/// @see ext_quaternion_geometric
/// @see ext_quaternion_relational
/// @see ext_quaternion_transform
/// @see ext_quaternion_trigonometric

#pragma once

// Dependency:
#include "../detail/type_quat.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_quaternion_float extension included")
#endif

namespace glm
{
	/// @addtogroup ext_quaternion_float
	/// @{

	/// Quaternion of single-precision floating-point numbers.
	typedef qua<float, defaultp>		quat;

	/// @}
} //namespace glm

