/// @ref ext_vector_double1
/// @file glm/ext/vector_double1.hpp
///
/// @defgroup ext_vector_double1 GLM_EXT_vector_double1
/// @ingroup ext
///
/// Exposes double-precision floating point vector type with one component.
///
/// Include <glm/ext/vector_double1.hpp> to use the features of this extension.
///
/// @see ext_vector_double1_precision extension.
/// @see ext_vector_float1 extension.

#pragma once

#include "../detail/type_vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_double1 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_double1
	/// @{

	/// 1 components vector of double-precision floating-point numbers.
	typedef vec<1, double, defaultp>		dvec1;

	/// @}
}//namespace glm
