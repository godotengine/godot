/// @ref gtx_perpendicular
/// @file glm/gtx/perpendicular.hpp
///
/// @see core (dependence)
/// @see gtx_projection (dependence)
///
/// @defgroup gtx_perpendicular GLM_GTX_perpendicular
/// @ingroup gtx
///
/// Include <glm/gtx/perpendicular.hpp> to use the features of this extension.
///
/// Perpendicular of a vector from other one

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/projection.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_perpendicular is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_perpendicular extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_perpendicular
	/// @{

	//! Projects x a perpendicular axis of Normal.
	//! From GLM_GTX_perpendicular extension.
	template<typename genType>
	GLM_FUNC_DECL genType perp(genType const& x, genType const& Normal);

	/// @}
}//namespace glm

#include "perpendicular.inl"
