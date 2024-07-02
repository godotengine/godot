/// @ref gtx_extend
/// @file glm/gtx/extend.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_extend GLM_GTX_extend
/// @ingroup gtx
///
/// Include <glm/gtx/extend.hpp> to use the features of this extension.
///
/// Extend a position from a source to a position at a defined length.

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_extend is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_extend extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_extend
	/// @{

	/// Extends of Length the Origin position using the (Source - Origin) direction.
	/// @see gtx_extend
	template<typename genType>
	GLM_FUNC_DECL genType extend(
		genType const& Origin,
		genType const& Source,
		typename genType::value_type const Length);

	/// @}
}//namespace glm

#include "extend.inl"
