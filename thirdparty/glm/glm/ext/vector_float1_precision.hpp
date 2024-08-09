/// @ref ext_vector_float1_precision
/// @file glm/ext/vector_float1_precision.hpp
///
/// @defgroup ext_vector_float1_precision GLM_EXT_vector_float1_precision
/// @ingroup ext
///
/// Exposes highp_vec1, mediump_vec1 and lowp_vec1 types.
///
/// Include <glm/ext/vector_float1_precision.hpp> to use the features of this extension.
///
/// @see ext_vector_float1 extension.

#pragma once

#include "../detail/type_vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_float1_precision extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_float1_precision
	/// @{

	/// 1 component vector of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, float, highp>		highp_vec1;

	/// 1 component vector of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, float, mediump>		mediump_vec1;

	/// 1 component vector of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, float, lowp>			lowp_vec1;

	/// @}
}//namespace glm
