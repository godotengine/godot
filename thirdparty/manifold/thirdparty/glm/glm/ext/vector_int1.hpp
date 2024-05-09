/// @ref ext_vector_int1
/// @file glm/ext/vector_int1.hpp
///
/// @defgroup ext_vector_int1 GLM_EXT_vector_int1
/// @ingroup ext
///
/// Exposes ivec1 vector type.
///
/// Include <glm/ext/vector_int1.hpp> to use the features of this extension.
///
/// @see ext_vector_uint1 extension.
/// @see ext_vector_int1_precision extension.

#pragma once

#include "../detail/type_vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_int1 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_int1
	/// @{

	/// 1 component vector of signed integer numbers.
	typedef vec<1, int, defaultp>			ivec1;

	/// @}
}//namespace glm

