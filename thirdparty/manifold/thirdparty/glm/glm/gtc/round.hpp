/// @ref gtc_round
/// @file glm/gtc/round.hpp
///
/// @see core (dependence)
/// @see gtc_round (dependence)
///
/// @defgroup gtc_round GLM_GTC_round
/// @ingroup gtc
///
/// Include <glm/gtc/round.hpp> to use the features of this extension.
///
/// Rounding value to specific boundings

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../detail/qualifier.hpp"
#include "../detail/_vectorize.hpp"
#include "../vector_relational.hpp"
#include "../common.hpp"
#include <limits>

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_round extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_round
	/// @{

	/// Return the power of two number which value is just higher the input value,
	/// round up to a power of two.
	///
	/// @see gtc_round
	template<typename genIUType>
	GLM_FUNC_DECL genIUType ceilPowerOfTwo(genIUType v);

	/// Return the power of two number which value is just higher the input value,
	/// round up to a power of two.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> ceilPowerOfTwo(vec<L, T, Q> const& v);

	/// Return the power of two number which value is just lower the input value,
	/// round down to a power of two.
	///
	/// @see gtc_round
	template<typename genIUType>
	GLM_FUNC_DECL genIUType floorPowerOfTwo(genIUType v);

	/// Return the power of two number which value is just lower the input value,
	/// round down to a power of two.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> floorPowerOfTwo(vec<L, T, Q> const& v);

	/// Return the power of two number which value is the closet to the input value.
	///
	/// @see gtc_round
	template<typename genIUType>
	GLM_FUNC_DECL genIUType roundPowerOfTwo(genIUType v);

	/// Return the power of two number which value is the closet to the input value.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> roundPowerOfTwo(vec<L, T, Q> const& v);

	/// Higher multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	///
	/// @param v Source value to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<typename genType>
	GLM_FUNC_DECL genType ceilMultiple(genType v, genType Multiple);

	/// Higher multiple number of Source.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @param v Source values to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> ceilMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	///
	/// @param v Source value to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<typename genType>
	GLM_FUNC_DECL genType floorMultiple(genType v, genType Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @param v Source values to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> floorMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	///
	/// @param v Source value to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<typename genType>
	GLM_FUNC_DECL genType roundMultiple(genType v, genType Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @param v Source values to which is applied the function
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtc_round
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> roundMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

	/// @}
} //namespace glm

#include "round.inl"
