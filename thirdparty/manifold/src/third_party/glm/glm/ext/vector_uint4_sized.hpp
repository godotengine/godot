/// @ref ext_vector_uint4_sized
/// @file glm/ext/vector_uint4_sized.hpp
///
/// @defgroup ext_vector_uint4_sized GLM_EXT_vector_uint4_sized
/// @ingroup ext
///
/// Exposes sized unsigned integer vector of 4 components type.
///
/// Include <glm/ext/vector_uint4_sized.hpp> to use the features of this extension.
///
/// @see ext_scalar_uint_sized
/// @see ext_vector_int4_sized

#pragma once

#include "../ext/vector_uint4.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint4_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_uint4_sized
	/// @{

	/// 8 bit unsigned integer vector of 4 components type.
	///
	/// @see ext_vector_uint4_sized
	typedef vec<4, uint8, defaultp>		u8vec4;

	/// 16 bit unsigned integer vector of 4 components type.
	///
	/// @see ext_vector_uint4_sized
	typedef vec<4, uint16, defaultp>	u16vec4;

	/// 32 bit unsigned integer vector of 4 components type.
	///
	/// @see ext_vector_uint4_sized
	typedef vec<4, uint32, defaultp>	u32vec4;

	/// 64 bit unsigned integer vector of 4 components type.
	///
	/// @see ext_vector_uint4_sized
	typedef vec<4, uint64, defaultp>	u64vec4;

	/// @}
}//namespace glm
