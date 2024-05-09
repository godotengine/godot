/// @ref gtx_intersect
/// @file glm/gtx/intersect.hpp
///
/// @see core (dependence)
/// @see gtx_closest_point (dependence)
///
/// @defgroup gtx_intersect GLM_GTX_intersect
/// @ingroup gtx
///
/// Include <glm/gtx/intersect.hpp> to use the features of this extension.
///
/// Add intersection functions

#pragma once

// Dependency:
#include <cfloat>
#include <limits>
#include "../glm.hpp"
#include "../geometric.hpp"
#include "../gtx/closest_point.hpp"
#include "../gtx/vector_query.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_closest_point is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_closest_point extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_intersect
	/// @{

	//! Compute the intersection of a ray and a plane.
	//! Ray direction and plane normal must be unit length.
	//! From GLM_GTX_intersect extension.
	template<typename genType>
	GLM_FUNC_DECL bool intersectRayPlane(
		genType const& orig, genType const& dir,
		genType const& planeOrig, genType const& planeNormal,
		typename genType::value_type & intersectionDistance);

	//! Compute the intersection of a ray and a triangle.
	/// Based om Tomas Möller implementation http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
	//! From GLM_GTX_intersect extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool intersectRayTriangle(
		vec<3, T, Q> const& orig, vec<3, T, Q> const& dir,
		vec<3, T, Q> const& v0, vec<3, T, Q> const& v1, vec<3, T, Q> const& v2,
		vec<2, T, Q>& baryPosition, T& distance);

	//! Compute the intersection of a line and a triangle.
	//! From GLM_GTX_intersect extension.
	template<typename genType>
	GLM_FUNC_DECL bool intersectLineTriangle(
		genType const& orig, genType const& dir,
		genType const& vert0, genType const& vert1, genType const& vert2,
		genType & position);

	//! Compute the intersection distance of a ray and a sphere.
	//! The ray direction vector is unit length.
	//! From GLM_GTX_intersect extension.
	template<typename genType>
	GLM_FUNC_DECL bool intersectRaySphere(
		genType const& rayStarting, genType const& rayNormalizedDirection,
		genType const& sphereCenter, typename genType::value_type const sphereRadiusSquared,
		typename genType::value_type & intersectionDistance);

	//! Compute the intersection of a ray and a sphere.
	//! From GLM_GTX_intersect extension.
	template<typename genType>
	GLM_FUNC_DECL bool intersectRaySphere(
		genType const& rayStarting, genType const& rayNormalizedDirection,
		genType const& sphereCenter, const typename genType::value_type sphereRadius,
		genType & intersectionPosition, genType & intersectionNormal);

	//! Compute the intersection of a line and a sphere.
	//! From GLM_GTX_intersect extension
	template<typename genType>
	GLM_FUNC_DECL bool intersectLineSphere(
		genType const& point0, genType const& point1,
		genType const& sphereCenter, typename genType::value_type sphereRadius,
		genType & intersectionPosition1, genType & intersectionNormal1,
		genType & intersectionPosition2 = genType(), genType & intersectionNormal2 = genType());

	/// @}
}//namespace glm

#include "intersect.inl"
