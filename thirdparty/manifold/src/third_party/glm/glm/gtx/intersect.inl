/// @ref gtx_intersect

namespace glm
{
	template<typename genType>
	GLM_FUNC_QUALIFIER bool intersectRayPlane
	(
		genType const& orig, genType const& dir,
		genType const& planeOrig, genType const& planeNormal,
		typename genType::value_type & intersectionDistance
	)
	{
		typename genType::value_type d = glm::dot(dir, planeNormal);
		typename genType::value_type Epsilon = std::numeric_limits<typename genType::value_type>::epsilon();

		if(glm::abs(d) > Epsilon)  // if dir and planeNormal are not perpendicular
		{
			typename genType::value_type const tmp_intersectionDistance = 	glm::dot(planeOrig - orig, planeNormal) / d;
			if (tmp_intersectionDistance > static_cast<typename genType::value_type>(0)) { // allow only intersections
				intersectionDistance = tmp_intersectionDistance;
				return true;
			}
		}

		return false;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool intersectRayTriangle
	(
		vec<3, T, Q> const& orig, vec<3, T, Q> const& dir,
		vec<3, T, Q> const& vert0, vec<3, T, Q> const& vert1, vec<3, T, Q> const& vert2,
		vec<2, T, Q>& baryPosition, T& distance
	)
	{
		// find vectors for two edges sharing vert0
		vec<3, T, Q> const edge1 = vert1 - vert0;
		vec<3, T, Q> const edge2 = vert2 - vert0;

		// begin calculating determinant - also used to calculate U parameter
		vec<3, T, Q> const p = glm::cross(dir, edge2);

		// if determinant is near zero, ray lies in plane of triangle
		T const det = glm::dot(edge1, p);

		vec<3, T, Q> Perpendicular(0);

		if(det > std::numeric_limits<T>::epsilon())
		{
			// calculate distance from vert0 to ray origin
			vec<3, T, Q> const dist = orig - vert0;

			// calculate U parameter and test bounds
			baryPosition.x = glm::dot(dist, p);
			if(baryPosition.x < static_cast<T>(0) || baryPosition.x > det)
				return false;

			// prepare to test V parameter
			Perpendicular = glm::cross(dist, edge1);

			// calculate V parameter and test bounds
			baryPosition.y = glm::dot(dir, Perpendicular);
			if((baryPosition.y < static_cast<T>(0)) || ((baryPosition.x + baryPosition.y) > det))
				return false;
		}
		else if(det < -std::numeric_limits<T>::epsilon())
		{
			// calculate distance from vert0 to ray origin
			vec<3, T, Q> const dist = orig - vert0;

			// calculate U parameter and test bounds
			baryPosition.x = glm::dot(dist, p);
			if((baryPosition.x > static_cast<T>(0)) || (baryPosition.x < det))
				return false;

			// prepare to test V parameter
			Perpendicular = glm::cross(dist, edge1);

			// calculate V parameter and test bounds
			baryPosition.y = glm::dot(dir, Perpendicular);
			if((baryPosition.y > static_cast<T>(0)) || (baryPosition.x + baryPosition.y < det))
				return false;
		}
		else
			return false; // ray is parallel to the plane of the triangle

		T inv_det = static_cast<T>(1) / det;

		// calculate distance, ray intersects triangle
		distance = glm::dot(edge2, Perpendicular) * inv_det;
		baryPosition *= inv_det;

		return true;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER bool intersectLineTriangle
	(
		genType const& orig, genType const& dir,
		genType const& vert0, genType const& vert1, genType const& vert2,
		genType & position
	)
	{
		typename genType::value_type Epsilon = std::numeric_limits<typename genType::value_type>::epsilon();

		genType edge1 = vert1 - vert0;
		genType edge2 = vert2 - vert0;

		genType Perpendicular = cross(dir, edge2);

		typename genType::value_type det = dot(edge1, Perpendicular);

		if (det > -Epsilon && det < Epsilon)
			return false;
		typename genType::value_type inv_det = typename genType::value_type(1) / det;

		genType Tangent = orig - vert0;

		position.y = dot(Tangent, Perpendicular) * inv_det;
		if (position.y < typename genType::value_type(0) || position.y > typename genType::value_type(1))
			return false;

		genType Cotangent = cross(Tangent, edge1);

		position.z = dot(dir, Cotangent) * inv_det;
		if (position.z < typename genType::value_type(0) || position.y + position.z > typename genType::value_type(1))
			return false;

		position.x = dot(edge2, Cotangent) * inv_det;

		return true;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER bool intersectRaySphere
	(
		genType const& rayStarting, genType const& rayNormalizedDirection,
		genType const& sphereCenter, const typename genType::value_type sphereRadiusSquared,
		typename genType::value_type & intersectionDistance
	)
	{
		typename genType::value_type Epsilon = std::numeric_limits<typename genType::value_type>::epsilon();
		genType diff = sphereCenter - rayStarting;
		typename genType::value_type t0 = dot(diff, rayNormalizedDirection);
		typename genType::value_type dSquared = dot(diff, diff) - t0 * t0;
		if( dSquared > sphereRadiusSquared )
		{
			return false;
		}
		typename genType::value_type t1 = sqrt( sphereRadiusSquared - dSquared );
		intersectionDistance = t0 > t1 + Epsilon ? t0 - t1 : t0 + t1;
		return intersectionDistance > Epsilon;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER bool intersectRaySphere
	(
		genType const& rayStarting, genType const& rayNormalizedDirection,
		genType const& sphereCenter, const typename genType::value_type sphereRadius,
		genType & intersectionPosition, genType & intersectionNormal
	)
	{
		typename genType::value_type distance;
		if( intersectRaySphere( rayStarting, rayNormalizedDirection, sphereCenter, sphereRadius * sphereRadius, distance ) )
		{
			intersectionPosition = rayStarting + rayNormalizedDirection * distance;
			intersectionNormal = (intersectionPosition - sphereCenter) / sphereRadius;
			return true;
		}
		return false;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER bool intersectLineSphere
	(
		genType const& point0, genType const& point1,
		genType const& sphereCenter, typename genType::value_type sphereRadius,
		genType & intersectionPoint1, genType & intersectionNormal1,
		genType & intersectionPoint2, genType & intersectionNormal2
	)
	{
		typename genType::value_type Epsilon = std::numeric_limits<typename genType::value_type>::epsilon();
		genType dir = normalize(point1 - point0);
		genType diff = sphereCenter - point0;
		typename genType::value_type t0 = dot(diff, dir);
		typename genType::value_type dSquared = dot(diff, diff) - t0 * t0;
		if( dSquared > sphereRadius * sphereRadius )
		{
			return false;
		}
		typename genType::value_type t1 = sqrt( sphereRadius * sphereRadius - dSquared );
		if( t0 < t1 + Epsilon )
			t1 = -t1;
		intersectionPoint1 = point0 + dir * (t0 - t1);
		intersectionNormal1 = (intersectionPoint1 - sphereCenter) / sphereRadius;
		intersectionPoint2 = point0 + dir * (t0 + t1);
		intersectionNormal2 = (intersectionPoint2 - sphereCenter) / sphereRadius;
		return true;
	}
}//namespace glm
