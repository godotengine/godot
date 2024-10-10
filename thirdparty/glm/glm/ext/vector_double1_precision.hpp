/// @ref ext_vector_double1_precision
/// @file glm/ext/vector_double1_precision.hpp
///
/// @defgroup ext_vector_double1_precision GLM_EXT_vector_double1_precision
/// @ingroup ext
///
/// Exposes highp_dvec1, mediump_dvec1 and lowp_dvec1 types.
///
/// Include <glm/ext/vector_double1_precision.hpp> to use the features of this extension.
///
/// @see ext_vector_double1

#pragma once

#include "../detail/type_vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_double1_precision extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_double1_precision
	/// @{

	/// 1 component vector of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, double, highp>		highp_dvec1;

	/// 1 component vector of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, double, mediump>		mediump_dvec1;

	/// 1 component vector of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, double, lowp>		lowp_dvec1;

	/// @}
}//namespace glm
