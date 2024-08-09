/// @ref gtx_raw_data
/// @file glm/gtx/raw_data.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_raw_data GLM_GTX_raw_data
/// @ingroup gtx
///
/// Include <glm/gtx/raw_data.hpp> to use the features of this extension.
///
/// Projection of a vector to other one

#pragma once

// Dependencies
#include "../ext/scalar_uint_sized.hpp"
#include "../detail/setup.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_raw_data is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_raw_data extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_raw_data
	/// @{

	//! Type for byte numbers.
	//! From GLM_GTX_raw_data extension.
	typedef detail::uint8		byte;

	//! Type for word numbers.
	//! From GLM_GTX_raw_data extension.
	typedef detail::uint16		word;

	//! Type for dword numbers.
	//! From GLM_GTX_raw_data extension.
	typedef detail::uint32		dword;

	//! Type for qword numbers.
	//! From GLM_GTX_raw_data extension.
	typedef detail::uint64		qword;

	/// @}
}// namespace glm

#include "raw_data.inl"
