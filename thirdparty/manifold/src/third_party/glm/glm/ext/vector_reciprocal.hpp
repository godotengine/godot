/// @ref ext_vector_reciprocal
/// @file glm/ext/vector_reciprocal.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_reciprocal GLM_EXT_vector_reciprocal
/// @ingroup ext
///
/// Include <glm/ext/vector_reciprocal.hpp> to use the features of this extension.
///
/// Define secant, cosecant and cotangent functions.

#pragma once

// Dependencies
#include "../detail/setup.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_reciprocal extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_reciprocal
	/// @{

	/// Secant function.
	/// hypotenuse / adjacent or 1 / cos(x)
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType sec(genType angle);

	/// Cosecant function.
	/// hypotenuse / opposite or 1 / sin(x)
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType csc(genType angle);

	/// Cotangent function.
	/// adjacent / opposite or 1 / tan(x)
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType cot(genType angle);

	/// Inverse secant function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType asec(genType x);

	/// Inverse cosecant function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType acsc(genType x);

	/// Inverse cotangent function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType acot(genType x);

	/// Secant hyperbolic function.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType sech(genType angle);

	/// Cosecant hyperbolic function.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType csch(genType angle);

	/// Cotangent hyperbolic function.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType coth(genType angle);

	/// Inverse secant hyperbolic function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType asech(genType x);

	/// Inverse cosecant hyperbolic function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType acsch(genType x);

	/// Inverse cotangent hyperbolic function.
	///
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see ext_vector_reciprocal
	template<typename genType>
	GLM_FUNC_DECL genType acoth(genType x);

	/// @}
}//namespace glm

#include "vector_reciprocal.inl"
