/// @ref gtx_range
/// @file glm/gtx/range.hpp
/// @author Joshua Moerman
///
/// @defgroup gtx_range GLM_GTX_range
/// @ingroup gtx
///
/// Include <glm/gtx/range.hpp> to use the features of this extension.
///
/// Defines begin and end for vectors and matrices. Useful for range-based for loop.
/// The range is defined over the elements, not over columns or rows (e.g. mat4 has 16 elements).

#pragma once

// Dependencies
#include "../detail/setup.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_range is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_range extension included")
#	endif
#endif

#include "../gtc/type_ptr.hpp"
#include "../gtc/vec1.hpp"

namespace glm
{
	/// @addtogroup gtx_range
	/// @{

#	if GLM_COMPILER & GLM_COMPILER_VC
#		pragma warning(push)
#		pragma warning(disable : 4100) // unreferenced formal parameter
#	endif

	template<typename T, qualifier Q>
	inline length_t components(vec<1, T, Q> const& v)
	{
		return v.length();
	}

	template<typename T, qualifier Q>
	inline length_t components(vec<2, T, Q> const& v)
	{
		return v.length();
	}

	template<typename T, qualifier Q>
	inline length_t components(vec<3, T, Q> const& v)
	{
		return v.length();
	}

	template<typename T, qualifier Q>
	inline length_t components(vec<4, T, Q> const& v)
	{
		return v.length();
	}

	template<typename genType>
	inline length_t components(genType const& m)
	{
		return m.length() * m[0].length();
	}

	template<typename genType>
	inline typename genType::value_type const * begin(genType const& v)
	{
		return value_ptr(v);
	}

	template<typename genType>
	inline typename genType::value_type const * end(genType const& v)
	{
		return begin(v) + components(v);
	}

	template<typename genType>
	inline typename genType::value_type * begin(genType& v)
	{
		return value_ptr(v);
	}

	template<typename genType>
	inline typename genType::value_type * end(genType& v)
	{
		return begin(v) + components(v);
	}

#	if GLM_COMPILER & GLM_COMPILER_VC
#		pragma warning(pop)
#	endif

	/// @}
}//namespace glm
