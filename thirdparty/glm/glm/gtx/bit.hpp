/// @ref gtx_bit
/// @file glm/gtx/bit.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_bit GLM_GTX_bit
/// @ingroup gtx
///
/// Include <glm/gtx/bit.hpp> to use the features of this extension.
///
/// Allow to perform bit operations on integer values

#pragma once

// Dependencies
#include "../gtc/bitfield.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_bit is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_bit extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_bit
	/// @{

	/// @see gtx_bit
	template<typename genIUType>
	GLM_FUNC_DECL genIUType highestBitValue(genIUType Value);

	/// @see gtx_bit
	template<typename genIUType>
	GLM_FUNC_DECL genIUType lowestBitValue(genIUType Value);

	/// Find the highest bit set to 1 in a integer variable and return its value.
	///
	/// @see gtx_bit
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> highestBitValue(vec<L, T, Q> const& value);

	/// Return the power of two number which value is just higher the input value.
	/// Deprecated, use ceilPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoAbove(genIUType Value);

	/// Return the power of two number which value is just higher the input value.
	/// Deprecated, use ceilPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<length_t L, typename T, qualifier Q>
	GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoAbove(vec<L, T, Q> const& value);

	/// Return the power of two number which value is just lower the input value.
	/// Deprecated, use floorPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoBelow(genIUType Value);

	/// Return the power of two number which value is just lower the input value.
	/// Deprecated, use floorPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<length_t L, typename T, qualifier Q>
	GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoBelow(vec<L, T, Q> const& value);

	/// Return the power of two number which value is the closet to the input value.
	/// Deprecated, use roundPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoNearest(genIUType Value);

	/// Return the power of two number which value is the closet to the input value.
	/// Deprecated, use roundPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template<length_t L, typename T, qualifier Q>
	GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoNearest(vec<L, T, Q> const& value);

	/// @}
} //namespace glm


#include "bit.inl"

