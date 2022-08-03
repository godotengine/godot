/// @ref gtx_matrix_cross_product
/// @file glm/gtx/matrix_cross_product.hpp
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_matrix_cross_product GLM_GTX_matrix_cross_product
/// @ingroup gtx
///
/// Include <glm/gtx/matrix_cross_product.hpp> to use the features of this extension.
///
/// Build cross product matrices

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_matrix_cross_product is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_matrix_cross_product extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_cross_product
	/// @{

	//! Build a cross product matrix.
	//! From GLM_GTX_matrix_cross_product extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<3, 3, T, Q> matrixCross3(
		vec<3, T, Q> const& x);

	//! Build a cross product matrix.
	//! From GLM_GTX_matrix_cross_product extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> matrixCross4(
		vec<3, T, Q> const& x);

	/// @}
}//namespace glm

#include "matrix_cross_product.inl"
