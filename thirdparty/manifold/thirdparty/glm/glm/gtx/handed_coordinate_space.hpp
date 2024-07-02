/// @ref gtx_handed_coordinate_space
/// @file glm/gtx/handed_coordinate_space.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_handed_coordinate_space GLM_GTX_handed_coordinate_space
/// @ingroup gtx
///
/// Include <glm/gtx/handed_coordinate_space.hpp> to use the features of this extension.
///
/// To know if a set of three basis vectors defines a right or left-handed coordinate system.

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_handed_coordinate_space is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_handed_coordinate_space extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_handed_coordinate_space
	/// @{

	//! Return if a trihedron right handed or not.
	//! From GLM_GTX_handed_coordinate_space extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool rightHanded(
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& binormal,
		vec<3, T, Q> const& normal);

	//! Return if a trihedron left handed or not.
	//! From GLM_GTX_handed_coordinate_space extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool leftHanded(
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& binormal,
		vec<3, T, Q> const& normal);

	/// @}
}// namespace glm

#include "handed_coordinate_space.inl"
