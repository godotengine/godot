/// @ref ext_vector_bool1
/// @file glm/ext/vector_bool1.hpp
///
/// @defgroup ext_vector_bool1 GLM_EXT_vector_bool1
/// @ingroup ext
///
/// Exposes bvec1 vector type.
///
/// Include <glm/ext/vector_bool1.hpp> to use the features of this extension.
///
/// @see ext_vector_bool1_precision extension.

#pragma once

#include "../detail/type_vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_bool1 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_bool1
	/// @{

	/// 1 components vector of boolean.
	typedef vec<1, bool, defaultp>		bvec1;

	/// @}
}//namespace glm
