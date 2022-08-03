/// @ref ext_scalar_relational
/// @file glm/ext/scalar_relational.hpp
///
/// @defgroup ext_scalar_relational GLM_EXT_scalar_relational
/// @ingroup ext
///
/// Exposes comparison functions for scalar types that take a user defined epsilon values.
///
/// Include <glm/ext/scalar_relational.hpp> to use the features of this extension.
///
/// @see core_vector_relational
/// @see ext_vector_relational
/// @see ext_matrix_relational

#pragma once

// Dependencies
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_scalar_relational extension included")
#endif

namespace glm
{
	/// Returns the component-wise comparison of |x - y| < epsilon.
	/// True if this expression is satisfied.
	///
	/// @tparam genType Floating-point or integer scalar types
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR bool equal(genType const& x, genType const& y, genType const& epsilon);

	/// Returns the component-wise comparison of |x - y| >= epsilon.
	/// True if this expression is not satisfied.
	///
	/// @tparam genType Floating-point or integer scalar types
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR bool notEqual(genType const& x, genType const& y, genType const& epsilon);

	/// Returns the component-wise comparison between two scalars in term of ULPs.
	/// True if this expression is satisfied.
	///
	/// @param x First operand.
	/// @param y Second operand.
	/// @param ULPs Maximum difference in ULPs between the two operators to consider them equal.
	///
	/// @tparam genType Floating-point or integer scalar types
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR bool equal(genType const& x, genType const& y, int ULPs);

	/// Returns the component-wise comparison between two scalars in term of ULPs.
	/// True if this expression is not satisfied.
	///
	/// @param x First operand.
	/// @param y Second operand.
	/// @param ULPs Maximum difference in ULPs between the two operators to consider them not equal.
	///
	/// @tparam genType Floating-point or integer scalar types
	template<typename genType>
	GLM_FUNC_DECL GLM_CONSTEXPR bool notEqual(genType const& x, genType const& y, int ULPs);

	/// @}
}//namespace glm

#include "scalar_relational.inl"
