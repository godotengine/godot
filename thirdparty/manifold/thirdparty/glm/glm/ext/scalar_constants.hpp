/// @ref ext_scalar_constants
/// @file glm/ext/scalar_constants.hpp
///
/// @defgroup ext_scalar_constants GLM_EXT_scalar_constants
/// @ingroup ext
///
/// Provides a list of constants and precomputed useful values.
///
/// Include <glm/ext/scalar_constants.hpp> to use the features of this extension.

#pragma once

// Dependencies
#include "../detail/setup.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_scalar_constants extension included")
#endif

namespace glm
{
	/// @addtogroup ext_scalar_constants
	/// @{

	/// Return the epsilon constant for floating point types.
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR genType epsilon();

	/// Return the pi constant for floating point types.
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR genType pi();

	/// Return the value of cos(1 / 2) for floating point types.
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR genType cos_one_over_two();

	/// @}
} //namespace glm

#include "scalar_constants.inl"
