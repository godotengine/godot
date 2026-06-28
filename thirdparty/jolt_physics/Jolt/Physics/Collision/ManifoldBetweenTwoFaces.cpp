// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Collision/ManifoldBetweenTwoFaces.h>
#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Geometry/ClipPoly.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

void PruneContactPoints(Vec3Arg inPenetrationAxis, ContactPoints &ioContactPointsOn1, ContactPoints &ioContactPointsOn2 JPH_IF_DEBUG_RENDERER(, RVec3Arg inCenterOfMass))
{
	// Makes no sense to call this with 4 or less points
	JPH_ASSERT(ioContactPointsOn1.size() > 4);

	// Both arrays should have the same size
	JPH_ASSERT(ioContactPointsOn1.size() == ioContactPointsOn2.size());

	// Penetration axis must be normalized
	JPH_ASSERT(inPenetrationAxis.IsNormalized());

	// We use a heuristic of (distance to center of mass) * (penetration depth) to find the contact point that we should keep
	// Neither of those two terms should ever become zero, so we clamp against this minimum value
	constexpr float cMinDistanceSq = 1.0e-6f; // 1 mm

	ContactPoints projected;
	StaticArray<float, 64> penetration_depth_sq;
	for (ContactPoints::size_type i = 0; i < ioContactPointsOn1.size(); ++i)
	{
		// Project contact points on the plane through inCenterOfMass with normal inPenetrationAxis and center around the center of mass of body 1
		// (note that since all points are relative to inCenterOfMass we can project onto the plane through the origin)
		Vec3 v1 = ioContactPointsOn1[i];
		projected.push_back(v1 - v1.Dot(inPenetrationAxis) * inPenetrationAxis);

		// Calculate penetration depth^2 of each point and clamp against the minimal distance
		Vec3 v2 = ioContactPointsOn2[i];
		penetration_depth_sq.push_back(max(cMinDistanceSq, (v2 - v1).LengthSq()));
	}

	// Find the point that is furthest away from the center of mass (its torque will have the biggest influence)
	// and the point that has the deepest penetration depth. Use the heuristic (distance to center of mass) * (penetration depth) for this.
	uint point1 = 0;
	float val = max(cMinDistanceSq, projected[0].LengthSq()) * penetration_depth_sq[0];
	for (uint i = 0; i < projected.size(); ++i)
	{
		float v = max(cMinDistanceSq, projected[i].LengthSq()) * penetration_depth_sq[i];
		if (v > val)
		{
			val = v;
			point1 = i;
		}
	}
	Vec3 point1v = projected[point1];

	// Find point furthest from the first point forming a line segment with point1. Again combine this with the heuristic
	// for deepest point as per above.
	uint point2 = uint(-1);
	val = -FLT_MAX;
	for (uint i = 0; i < projected.size(); ++i)
		if (i != point1)
		{
			float v = max(cMinDistanceSq, (projected[i] - point1v).LengthSq()) * penetration_depth_sq[i];
			if (v > val)
			{
				val = v;
				point2 = i;
			}
		}
	JPH_ASSERT(point2 != uint(-1));
	Vec3 point2v = projected[point2];

	// Find furthest points on both sides of the line segment in order to maximize the area
	uint point3 = uint(-1);
	uint point4 = uint(-1);
	float min_val = 0.0f;
	float max_val = 0.0f;
	Vec3 perp = (point2v - point1v).Cross(inPenetrationAxis);
	for (uint i = 0; i < projected.size(); ++i)
		if (i != point1 && i != point2)
		{
			float v = perp.Dot(projected[i] - point1v);
			if (v < min_val)
			{
				min_val = v;
				point3 = i;
			}
			else if (v > max_val)
			{
				max_val = v;
				point4 = i;
			}
		}

	// Add points to array (in order so they form a polygon)
	StaticArray<Vec3, 4> points_to_keep_on_1, points_to_keep_on_2;
	points_to_keep_on_1.push_back(ioContactPointsOn1[point1]);
	points_to_keep_on_2.push_back(ioContactPointsOn2[point1]);
	if (point3 != uint(-1))
	{
		points_to_keep_on_1.push_back(ioContactPointsOn1[point3]);
		points_to_keep_on_2.push_back(ioContactPointsOn2[point3]);
	}
	points_to_keep_on_1.push_back(ioContactPointsOn1[point2]);
	points_to_keep_on_2.push_back(ioContactPointsOn2[point2]);
	if (point4 != uint(-1))
	{
		JPH_ASSERT(point3 != point4);
		points_to_keep_on_1.push_back(ioContactPointsOn1[point4]);
		points_to_keep_on_2.push_back(ioContactPointsOn2[point4]);
	}

#ifdef JPH_DEBUG_RENDERER
	if (ContactConstraintManager::sDrawContactPointReduction)
	{
		// Draw input polygon
		DebugRenderer::sInstance->DrawWirePolygon(RMat44::sTranslation(inCenterOfMass), ioContactPointsOn1, Color::sOrange, 0.05f);

		// Draw primary axis
		DebugRenderer::sInstance->DrawArrow(inCenterOfMass + ioContactPointsOn1[point1], inCenterOfMass + ioContactPointsOn1[point2], Color::sRed, 0.05f);

		// Draw contact points we kept
		for (Vec3 p : points_to_keep_on_1)
			DebugRenderer::sInstance->DrawMarker(inCenterOfMass + p, Color::sGreen, 0.1f);
	}
#endif // JPH_DEBUG_RENDERER

	// Copy the points back to the input buffer
	ioContactPointsOn1 = points_to_keep_on_1;
	ioContactPointsOn2 = points_to_keep_on_2;
}

void ManifoldBetweenTwoFaces(Vec3Arg inContactPoint1, Vec3Arg inContactPoint2, Vec3Arg inPenetrationAxis, float inMaxContactDistance, const ConvexShape::SupportingFace &inShape1Face, const ConvexShape::SupportingFace &inShape2Face, ContactPoints &outContactPoints1, ContactPoints &outContactPoints2 JPH_IF_DEBUG_RENDERER(, RVec3Arg inCenterOfMass))
{
	JPH_ASSERT(inMaxContactDistance > 0.0f);

#ifdef JPH_DEBUG_RENDERER
	if (ContactConstraintManager::sDrawContactPoint)
	{
		RVec3 cp1 = inCenterOfMass + inContactPoint1;
		RVec3 cp2 = inCenterOfMass + inContactPoint2;

		// Draw contact points
		DebugRenderer::sInstance->DrawMarker(cp1, Color::sRed, 0.1f);
		DebugRenderer::sInstance->DrawMarker(cp2, Color::sGreen, 0.1f);

		// Draw contact normal
		DebugRenderer::sInstance->DrawArrow(cp1, cp1 + inPenetrationAxis.Normalized(), Color::sRed, 0.05f);
	}
#endif // JPH_DEBUG_RENDERER

	// Remember size before adding new points, to check at the end if we added some
	ContactPoints::size_type old_size = outContactPoints1.size();

	// Both faces need to have at least 2 points or else there can never be more than 1 contact point
	// At least one face needs to have at least 3 points (in the case that it has 2 points only if the edges match exactly you can have 2 contact points, but this situation is unstable anyhow)
	if (min(inShape1Face.size(), inShape2Face.size()) >= 2
		&& max(inShape1Face.size(), inShape2Face.size()) >= 3)
	{
		// Swap the shapes if the 2nd face doesn't have enough vertices
		const ConvexShape::SupportingFace *shape1_face, *shape2_face;
		ContactPoints *contact_points1, *contact_points2;
		Vec3 penetration_axis;
		if (inShape2Face.size() >= 3)
		{
			shape1_face = &inShape1Face;
			shape2_face = &inShape2Face;
			contact_points1 = &outContactPoints1;
			contact_points2 = &outContactPoints2;
			penetration_axis = inPenetrationAxis;
		}
		else
		{
			shape1_face = &inShape2Face;
			shape2_face = &inShape1Face;
			contact_points1 = &outContactPoints2;
			contact_points2 = &outContactPoints1;
			penetration_axis = -inPenetrationAxis;
		}

		// Determine plane origin and first edge direction
		Vec3 plane_origin = shape1_face->at(0);
		Vec3 first_edge = shape1_face->at(1) - plane_origin;

		Vec3 plane_normal;
		ConvexShape::SupportingFace clipped_face;
		if (shape1_face->size() >= 3)
		{
			// Clip the polygon of face 2 against that of 1
			ClipPolyVsPoly(*shape2_face, *shape1_face, penetration_axis, clipped_face);

			// Three vertices, can just calculate the normal
			plane_normal = first_edge.Cross(shape1_face->at(2) - plane_origin);
		}
		else
		{
			// Clip the polygon of face 2 against edge of 1
			ClipPolyVsEdge(*shape2_face, shape1_face->at(0), shape1_face->at(1), penetration_axis, clipped_face);

			// Two vertices, first find a perpendicular to the edge and penetration axis and then use the perpendicular together with the edge to form a normal
			plane_normal = first_edge.Cross(penetration_axis).Cross(first_edge);
		}

		// If penetration axis and plane normal are perpendicular, fall back to the contact points
		float penetration_axis_dot_plane_normal = penetration_axis.Dot(plane_normal);
		if (penetration_axis_dot_plane_normal != 0.0f)
		{
			float penetration_axis_len = penetration_axis.Length();

			for (Vec3 p2 : clipped_face)
			{
				// Project clipped face back onto the plane of face 1, we do this by solving:
				// p1 = p2 + distance * penetration_axis / |penetration_axis|
				// (p1 - plane_origin) . plane_normal = 0
				// This gives us:
				// distance = -|penetration_axis| * (p2 - plane_origin) . plane_normal / penetration_axis . plane_normal
				float distance = (p2 - plane_origin).Dot(plane_normal) / penetration_axis_dot_plane_normal; // note left out -|penetration_axis| term

				// If the point is less than inMaxContactDistance in front of the plane of face 2, add it as a contact point
				if (distance * penetration_axis_len < inMaxContactDistance)
				{
					Vec3 p1 = p2 - distance * penetration_axis;
					contact_points1->push_back(p1);
					contact_points2->push_back(p2);
				}
			}
		}

	#ifdef JPH_DEBUG_RENDERER
		if (ContactConstraintManager::sDrawSupportingFaces)
		{
			RMat44 com = RMat44::sTranslation(inCenterOfMass);

			// Draw clipped poly
			DebugRenderer::sInstance->DrawWirePolygon(com, clipped_face, Color::sOrange);

			// Draw supporting faces
			DebugRenderer::sInstance->DrawWirePolygon(com, inShape1Face, Color::sRed, 0.05f);
			DebugRenderer::sInstance->DrawWirePolygon(com, inShape2Face, Color::sGreen, 0.05f);

			// Draw normal
			float plane_normal_len = plane_normal.Length();
			if (plane_normal_len > 0.0f)
			{
				RVec3 plane_origin_ws = inCenterOfMass + plane_origin;
				DebugRenderer::sInstance->DrawArrow(plane_origin_ws, plane_origin_ws + plane_normal / plane_normal_len, Color::sYellow, 0.05f);
			}

			// Draw contact points that remain after distance check
			for (ContactPoints::size_type p = old_size; p < outContactPoints1.size(); ++p)
			{
				DebugRenderer::sInstance->DrawMarker(inCenterOfMass + outContactPoints1[p], Color::sYellow, 0.1f);
				DebugRenderer::sInstance->DrawMarker(inCenterOfMass + outContactPoints2[p], Color::sOrange, 0.1f);
			}
		}
	#endif // JPH_DEBUG_RENDERER
	}

	// If the clipping result is empty, use the contact point itself
	if (outContactPoints1.size() == old_size)
	{
		outContactPoints1.push_back(inContactPoint1);
		outContactPoints2.push_back(inContactPoint2);
	}
}

JPH_NAMESPACE_END
