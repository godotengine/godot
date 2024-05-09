/// @ref gtx_norm
/// @file glm/gtx/norm.hpp
///
/// @see core (dependence)
/// @see gtx_quaternion (dependence)
/// @see gtx_component_wise (dependence)
///
/// @defgroup gtx_norm GLM_GTX_norm
/// @ingroup gtx
///
/// Include <glm/gtx/norm.hpp> to use the features of this extension.
///
/// Various ways to compute vector norms.

#pragma once

// Dependency:
#include "../geometric.hpp"
#include "../gtx/quaternion.hpp"
#include "../gtx/component_wise.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_norm is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_norm extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_norm
	/// @{

	/// Returns the squared length of x.
	/// From GLM_GTX_norm extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL T length2(vec<L, T, Q> const& x);

	/// Returns the squared distance between p0 and p1, i.e., length2(p0 - p1).
	/// From GLM_GTX_norm extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL T distance2(vec<L, T, Q> const& p0, vec<L, T, Q> const& p1);

	//! Returns the L1 norm between x and y.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T l1Norm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

	//! Returns the L1 norm of v.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T l1Norm(vec<3, T, Q> const& v);

	//! Returns the L2 norm between x and y.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T l2Norm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

	//! Returns the L2 norm of v.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T l2Norm(vec<3, T, Q> const& x);

	//! Returns the L norm between x and y.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T lxNorm(vec<3, T, Q> const& x, vec<3, T, Q> const& y, unsigned int Depth);

	//! Returns the L norm of v.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T lxNorm(vec<3, T, Q> const& x, unsigned int Depth);

	//! Returns the LMax norm between x and y.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T lMaxNorm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

	//! Returns the LMax norm of v.
	//! From GLM_GTX_norm extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T lMaxNorm(vec<3, T, Q> const& x);

	/// @}
}//namespace glm

#include "norm.inl"
