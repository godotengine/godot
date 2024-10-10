/// @ref ext_vector_uint1_sized
/// @file glm/ext/vector_uint1_sized.hpp
///
/// @defgroup ext_vector_uint1_sized GLM_EXT_vector_uint1_sized
/// @ingroup ext
///
/// Exposes sized unsigned integer vector types.
///
/// Include <glm/ext/vector_uint1_sized.hpp> to use the features of this extension.
///
/// @see ext_scalar_uint_sized
/// @see ext_vector_int1_sized

#pragma once

#include "../ext/vector_uint1.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint1_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_uint1_sized
	/// @{

	/// 8 bit unsigned integer vector of 1 component type.
	///
	/// @see ext_vector_uint1_sized
	typedef vec<1, uint8, defaultp>		u8vec1;

	/// 16 bit unsigned integer vector of 1 component type.
	///
	/// @see ext_vector_uint1_sized
	typedef vec<1, uint16, defaultp>	u16vec1;

	/// 32 bit unsigned integer vector of 1 component type.
	///
	/// @see ext_vector_uint1_sized
	typedef vec<1, uint32, defaultp>	u32vec1;

	/// 64 bit unsigned integer vector of 1 component type.
	///
	/// @see ext_vector_uint1_sized
	typedef vec<1, uint64, defaultp>	u64vec1;

	/// @}
}//namespace glm
