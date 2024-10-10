/// @ref gtx_fast_exponential
/// @file glm/gtx/fast_exponential.hpp
///
/// @see core (dependence)
/// @see gtx_half_float (dependence)
///
/// @defgroup gtx_fast_exponential GLM_GTX_fast_exponential
/// @ingroup gtx
///
/// Include <glm/gtx/fast_exponential.hpp> to use the features of this extension.
///
/// Fast but less accurate implementations of exponential based functions.

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_fast_exponential is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_fast_exponential extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_fast_exponential
	/// @{

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template<typename genType>
	GLM_FUNC_DECL genType fastPow(genType x, genType y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastPow(vec<L, T, Q> const& x, vec<L, T, Q> const& y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template<typename genTypeT, typename genTypeU>
	GLM_FUNC_DECL genTypeT fastPow(genTypeT x, genTypeU y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastPow(vec<L, T, Q> const& x);

	/// Faster than the common exp function but less accurate.
	/// @see gtx_fast_exponential
	template<typename T>
	GLM_FUNC_DECL T fastExp(T x);

	/// Faster than the common exp function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastExp(vec<L, T, Q> const& x);

	/// Faster than the common log function but less accurate.
	/// @see gtx_fast_exponential
	template<typename T>
	GLM_FUNC_DECL T fastLog(T x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastLog(vec<L, T, Q> const& x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template<typename T>
	GLM_FUNC_DECL T fastExp2(T x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastExp2(vec<L, T, Q> const& x);

	/// Faster than the common log2 function but less accurate.
	/// @see gtx_fast_exponential
	template<typename T>
	GLM_FUNC_DECL T fastLog2(T x);

	/// Faster than the common log2 function but less accurate.
	/// @see gtx_fast_exponential
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fastLog2(vec<L, T, Q> const& x);

	/// @}
}//namespace glm

#include "fast_exponential.inl"
