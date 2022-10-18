/*************************************************************************/
/*  overlap_check.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "overlap_check.h"

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include <BulletCollision/CollisionShapes/btBoxShape.h>
#include <BulletCollision/CollisionShapes/btCapsuleShape.h>
#include <BulletCollision/CollisionShapes/btConeShape.h>
#include <BulletCollision/CollisionShapes/btConvexPointCloudShape.h>
#include <BulletCollision/CollisionShapes/btConvexPolyhedron.h>
#include <BulletCollision/CollisionShapes/btCylinderShape.h>
#include <BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h>
#include <BulletCollision/CollisionShapes/btSphereShape.h>
#include <BulletCollision/CollisionShapes/btTriangleShape.h>
#include <BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h>
#include <LinearMath/btMatrix3x3.h>

namespace SAT {
struct Range {
	real_t min;
	real_t max;
};

Range project(const btConvexShape *p_shape, const btVector3 &p_axis, const btTransform &p_transform) {
	btVector3 wmin;
	btVector3 wmax;
	Range r;
	p_shape->project(p_transform, p_axis, r.min, r.max, wmin, wmax);
	return r;
}

bool is_separated(const btConvexShape *p_shape_1, const btTransform &p_shape_1_transform, const btConvexShape *p_shape_2, const btTransform &p_shape_2_transform, const btVector3 &p_axis) {
	if (unlikely(p_axis.fuzzyZero())) {
		// Don't allow 0 axis test.
		return false;
	}

	const Range s_1_range = project(p_shape_1, p_axis, p_shape_1_transform);
	const Range s_2_range = project(p_shape_2, p_axis, p_shape_2_transform);

	return s_1_range.max < s_2_range.min || s_2_range.max < s_1_range.min;
}
}; // namespace SAT

OverlappingFunc OverlapCheck::overlapping_funcs[MAX_BROADPHASE_COLLISION_TYPES][MAX_BROADPHASE_COLLISION_TYPES];

const static btVector3 direction_axis[3] = { btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1) };

// Function ported from bullet/BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.cpp
static SIMD_FORCE_INLINE void segmentsClosestPoints(
		btVector3 &ptsVector,
		btVector3 &offsetA,
		btVector3 &offsetB,
		real_t &tA, real_t &tB,
		const btVector3 &translation,
		const btVector3 &dirA, real_t hlenA,
		const btVector3 &dirB, real_t hlenB) {
	// compute the parameters of the closest points on each line segment
	real_t dirA_dot_dirB = btDot(dirA, dirB);
	real_t dirA_dot_trans = btDot(dirA, translation);
	real_t dirB_dot_trans = btDot(dirB, translation);

	real_t denom = 1.0f - dirA_dot_dirB * dirA_dot_dirB;

	if (denom == 0.0f) {
		tA = 0.0f;
	} else {
		tA = (dirA_dot_trans - dirB_dot_trans * dirA_dot_dirB) / denom;
		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	}

	tB = tA * dirA_dot_dirB - dirB_dot_trans;

	if (tB < -hlenB) {
		tB = -hlenB;
		tA = tB * dirA_dot_dirB + dirA_dot_trans;

		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	} else if (tB > hlenB) {
		tB = hlenB;
		tA = tB * dirA_dot_dirB + dirA_dot_trans;

		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	}

	// compute the closest points relative to segment centers.

	offsetA = dirA * tA;
	offsetB = dirB * tB;

	ptsVector = translation - offsetA + offsetB;
}

/// Returns a vector that points toward the target but perpendicular to the up
/// vector
btVector3 toward(const btVector3 &p_up, const btVector3 &p_target_axis) {
	btVector3 side = p_up.cross(p_target_axis);
	if (side.fuzzyZero()) {
		return p_target_axis;
	} else {
		return side.cross(p_up).safeNormalize();
	}
}

// Test the axis of the shape_1 but rotated toward the object and not along
// the basis. So even if the objects are orientated in other directions is
// possible to find the separation axis.
// One way so the check is only done with the axis of the shape_1
bool is_separated_oriented_axis_one_way(
		const btConvexShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		const btConvexShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btVector3 dir = (p_shape_2_transform.getOrigin() - p_shape_1_transform.getOrigin());
	if (dir.length2() > CMP_EPSILON) {
		dir.normalize();

		for (uint32_t i = 0; i < 3; i += 1) {
			const btVector3 axis = toward(p_shape_1_transform.getBasis().getColumn(i).normalized(), dir);

			if (SAT::is_separated(p_shape_1, p_shape_1_transform, p_shape_2, p_shape_2_transform, axis)) {
				return true;
			}
		}
	}

	return false;
}

// Test the axis of the shape_1 but rotated toward the object and not along
// the basis. So even if the objects are orientated in other directions is
// possible to find the separation axis.
// Bi way so the check is done on both shapes.
bool is_separated_oriented_axis_bi_way(
		const btConvexShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		const btConvexShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btVector3 dir = (p_shape_2_transform.getOrigin() - p_shape_1_transform.getOrigin());
	if (dir.length2() > CMP_EPSILON) {
		dir.normalize();

		for (uint32_t i = 0; i < 3; i += 1) {
			const btVector3 axis = toward(p_shape_1_transform.getBasis().getColumn(i).normalized(), dir);

			if (SAT::is_separated(p_shape_1, p_shape_1_transform, p_shape_2, p_shape_2_transform, axis)) {
				return true;
			}
		}

		// Test the other way around now.
		dir *= -1.0;
		for (uint32_t i = 0; i < 3; i += 1) {
			const btVector3 axis = toward(p_shape_2_transform.getBasis().getColumn(i), dir);

			if (SAT::is_separated(p_shape_1, p_shape_1_transform, p_shape_2, p_shape_2_transform, axis)) {
				return true;
			}
		}
	}

	return false;
}

bool is_separated_polyhedron_faces_check(
		btPolyhedralConvexShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		const btConvexShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	if (p_shape_1->getConvexPolyhedron() == nullptr) {
		p_shape_1->initializePolyhedralFeatures();
	}

	// Check all faces of the polygon.
	for (int i = 0; i < p_shape_1->getConvexPolyhedron()->m_faces.size(); i += 1) {
		const btVector3 local_axis(
				p_shape_1->getConvexPolyhedron()->m_faces[i].m_plane[0],
				p_shape_1->getConvexPolyhedron()->m_faces[i].m_plane[1],
				p_shape_1->getConvexPolyhedron()->m_faces[i].m_plane[2]);

		const btVector3 axis(p_shape_1_transform.getBasis() * local_axis);
		if (SAT::is_separated(p_shape_1, p_shape_1_transform, p_shape_2, p_shape_2_transform, axis)) {
			return true;
		}
	}

	return false;
}

bool overlap_check_sphere_sphere(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const real_t combined_radius =
			static_cast<btSphereShape *>(p_shape_1)->getRadius() +
			static_cast<btSphereShape *>(p_shape_2)->getRadius();
	return (p_shape_1_transform.inverse() * p_shape_2_transform).getOrigin().length2() <= combined_radius * combined_radius;
}

bool overlap_check_box_sphere(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btBoxShape *box = static_cast<btBoxShape *>(p_shape_1);
	const btSphereShape *sphere = static_cast<btSphereShape *>(p_shape_2);

	btVector3 const &box_half_extent = box->getHalfExtentsWithoutMargin();

	// Convert the sphere position to the box's local space
	const btVector3 sphere_rel_pos = (p_shape_1_transform.inverse() * p_shape_2_transform).getOrigin();

	// Determine the closest point to the sphere center in the box
	btVector3 closest_point = sphere_rel_pos;
	closest_point.setX(btMin(box_half_extent.getX(), closest_point.getX()));
	closest_point.setX(btMax(-box_half_extent.getX(), closest_point.getX()));
	closest_point.setY(btMin(box_half_extent.getY(), closest_point.getY()));
	closest_point.setY(btMax(-box_half_extent.getY(), closest_point.getY()));
	closest_point.setZ(btMin(box_half_extent.getZ(), closest_point.getZ()));
	closest_point.setZ(btMax(-box_half_extent.getZ(), closest_point.getZ()));

	const real_t intersection_dist = sphere->getRadius() + box->getMargin();
	const real_t contact_dist = intersection_dist;
	const btVector3 normal = sphere_rel_pos - closest_point;

	const real_t dist2 = normal.length2();

	// Check if the sphere is inside the box.
	return dist2 <= contact_dist * contact_dist;
}

bool overlap_check_sphere_box(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_box_sphere(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_capsule_sphere(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCapsuleShape *capsule = static_cast<btCapsuleShape *>(p_shape_1);
	const btSphereShape *sphere = static_cast<btSphereShape *>(p_shape_2);

	const btVector3 sphere_relative_position = (p_shape_1_transform.inverse() * p_shape_2_transform).getOrigin();
	const btVector3 capsule_axis = direction_axis[capsule->getUpAxis()];

	const btVector3 capsule_near_point = capsule_axis * CLAMP(capsule_axis.dot(sphere_relative_position), -capsule->getHalfHeight(), capsule->getHalfHeight());

	const real_t combined_radius = sphere->getRadius() + capsule->getRadius();

	return (capsule_near_point - sphere_relative_position).length2() <= combined_radius * combined_radius;
}

bool overlap_check_sphere_capsule(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_capsule_sphere(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_capsule_capsule(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCapsuleShape *capsule_1 = static_cast<btCapsuleShape *>(p_shape_1);
	const btCapsuleShape *capsule_2 = static_cast<btCapsuleShape *>(p_shape_2);

	btVector3 directionA = p_shape_1_transform.getBasis().getColumn(capsule_1->getUpAxis());
	btVector3 translationA = p_shape_1_transform.getOrigin();
	btVector3 directionB = p_shape_2_transform.getBasis().getColumn(capsule_2->getUpAxis());
	btVector3 translationB = p_shape_2_transform.getOrigin();

	// Translation between centers
	btVector3 translation = translationB - translationA;

	// Compute the closest points of the capsule line segments.

	btVector3 ptsVector; // The vector between the closest points.
	btVector3 offsetA, offsetB; // Offsets from segment centers to their closest points.
	real_t tA, tB; // Parameters on line segment.

	segmentsClosestPoints(
			ptsVector,
			offsetA,
			offsetB,
			tA,
			tB,
			translation,
			directionA,
			capsule_1->getHalfHeight(),
			directionB,
			capsule_2->getHalfHeight());

	const real_t combined_radius = capsule_1->getRadius() + capsule_2->getRadius();
	return ptsVector.length2() <= combined_radius * combined_radius;
}

bool overlap_check_convex_sphere(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btPolyhedralConvexShape *convex = static_cast<btPolyhedralConvexShape *>(p_shape_1);
	const btSphereShape *sphere = static_cast<btSphereShape *>(p_shape_2);

	if (is_separated_oriented_axis_one_way(
				convex,
				p_shape_1_transform,
				sphere,
				p_shape_2_transform)) {
		return false;
	}

	if (is_separated_polyhedron_faces_check(
				convex,
				p_shape_1_transform,
				sphere,
				p_shape_2_transform)) {
		return false;
	}

	return true;
}

bool overlap_check_sphere_convex(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_convex_sphere(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_convex_cylinder(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btPolyhedralConvexShape *convex = static_cast<btPolyhedralConvexShape *>(p_shape_1);
	const btCylinderShape *cylinder = static_cast<btCylinderShape *>(p_shape_2);

	if (is_separated_oriented_axis_bi_way(
				convex,
				p_shape_1_transform,
				cylinder,
				p_shape_2_transform)) {
		return false;
	}

	// Test faces of the cylinder
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_2_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(convex, p_shape_1_transform, cylinder, p_shape_2_transform, axis)) {
			return false;
		}
	}

	if (is_separated_polyhedron_faces_check(
				convex,
				p_shape_1_transform,
				cylinder,
				p_shape_2_transform)) {
		return false;
	}

	return true;
}

bool overlap_check_cylinder_convex(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_convex_cylinder(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_cylinder_sphere(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCylinderShape *cylinder = static_cast<btCylinderShape *>(p_shape_1);
	const btSphereShape *sphere = static_cast<btSphereShape *>(p_shape_2);

	// Test faces of cylinder
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_1_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(cylinder, p_shape_1_transform, sphere, p_shape_2_transform, axis)) {
			return false;
		}
	}

	if (is_separated_oriented_axis_one_way(
				cylinder,
				p_shape_1_transform,
				sphere,
				p_shape_2_transform)) {
		return false;
	}

	return true;
}

bool overlap_check_sphere_cylinder(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_cylinder_sphere(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_box_box(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	// TODO use an acelerated algorithm instead of SAT

	const btBoxShape *box_1 = static_cast<btBoxShape *>(p_shape_1);
	const btBoxShape *box_2 = static_cast<btBoxShape *>(p_shape_2);

	// Test faces of
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_1_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(box_1, p_shape_1_transform, box_2, p_shape_2_transform, axis)) {
			return false;
		}
	}

	// Test faces of B.
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_2_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(box_1, p_shape_1_transform, box_2, p_shape_2_transform, axis)) {
			return false;
		}
	}

	// Test faces but oriented toward target.
	if (is_separated_oriented_axis_bi_way(
				box_1,
				p_shape_1_transform,
				box_2,
				p_shape_2_transform)) {
		return false;
	}

	// Overlapping.
	return true;
}

bool overlap_check_box_cylinder(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btBoxShape *box = static_cast<btBoxShape *>(p_shape_1);
	const btCylinderShape *cylinder = static_cast<btCylinderShape *>(p_shape_2);

	// Test faces of box.
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_1_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(box, p_shape_1_transform, cylinder, p_shape_2_transform, axis)) {
			return false;
		}
	}

	// Test axis of the Cylinder
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_2_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(box, p_shape_1_transform, cylinder, p_shape_2_transform, axis)) {
			return false;
		}
	}

	// Test faces but oriented toward target.
	if (is_separated_oriented_axis_bi_way(
				box,
				p_shape_1_transform,
				cylinder,
				p_shape_2_transform)) {
		return false;
	}

	// Overlapping.
	return true;
}

bool overlap_check_cylinder_box(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_box_cylinder(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_box_capsule(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCapsuleShape *capsule = static_cast<btCapsuleShape *>(p_shape_2);

	{
		// Test shape 1
		btSphereShape sphere(capsule->getRadius());

		if (overlap_check_box_sphere(
					p_shape_1,
					p_shape_1_transform,
					&sphere,
					p_shape_2_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * capsule->getHalfHeight()))) {
			return true;
		}

		if (overlap_check_box_sphere(
					p_shape_1,
					p_shape_1_transform,
					&sphere,
					p_shape_2_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * (-capsule->getHalfHeight())))) {
			return true;
		}
	}

	{
		btVector3 cylinder_half_extent;
		cylinder_half_extent[capsule->getUpAxis()] = capsule->getHalfHeight();
		cylinder_half_extent[(capsule->getUpAxis() + 1) % 3] = capsule->getRadius();
		cylinder_half_extent[(capsule->getUpAxis() + 2) % 3] = capsule->getRadius();
		btCylinderShapeZ cylinder(cylinder_half_extent); // <-- TODO
		if (overlap_check_box_cylinder(p_shape_1, p_shape_1_transform, &cylinder, p_shape_2_transform)) {
			return true;
		}
	}

	// This check is special compared to the others. If nothing overlaps at this
	// point, the capsule is not colliding.
	return false;
}

bool overlap_check_capsule_box(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_box_capsule(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_cylinder_cylinder(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCylinderShape *cylinder_1 = static_cast<btCylinderShape *>(p_shape_1);
	const btCylinderShape *cylinder_2 = static_cast<btCylinderShape *>(p_shape_2);

	// Test faces of cylinder 1.
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_1_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(cylinder_1, p_shape_1_transform, cylinder_2, p_shape_2_transform, axis)) {
			return false;
		}
	}

	// Test faces of cylinder 2.
	for (uint32_t i = 0; i < 3; i += 1) {
		const btVector3 axis = p_shape_2_transform.getBasis().getColumn(i).normalized();

		if (SAT::is_separated(cylinder_1, p_shape_1_transform, cylinder_2, p_shape_2_transform, axis)) {
			return false;
		}
	}

	if (is_separated_oriented_axis_bi_way(
				cylinder_1,
				p_shape_1_transform,
				cylinder_2,
				p_shape_2_transform)) {
		return false;
	}

	return true;
}

bool overlap_check_cylinder_capsule(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCapsuleShape *capsule = static_cast<btCapsuleShape *>(p_shape_2);

	{
		// Test shape 1
		btSphereShape sphere(capsule->getRadius());

		if (overlap_check_cylinder_sphere(
					p_shape_1,
					p_shape_1_transform,
					&sphere,
					p_shape_2_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * capsule->getHalfHeight()))) {
			return true;
		}

		if (overlap_check_cylinder_sphere(
					p_shape_1,
					p_shape_1_transform,
					&sphere,
					p_shape_2_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * (-capsule->getHalfHeight())))) {
			return true;
		}
	}

	{
		btVector3 cylinder_half_extent;
		cylinder_half_extent[capsule->getUpAxis()] = capsule->getHalfHeight();
		cylinder_half_extent[(capsule->getUpAxis() + 1) % 3] = capsule->getRadius();
		cylinder_half_extent[(capsule->getUpAxis() + 2) % 3] = capsule->getRadius();
		btCylinderShapeZ cylinder(cylinder_half_extent); // <-- TODO
		if (overlap_check_cylinder_cylinder(p_shape_1, p_shape_1_transform, &cylinder, p_shape_2_transform)) {
			return true;
		}
	}

	// This check is special compared to the others. If nothing overlaps at this
	// point, the capsule is not colliding.
	return false;
}

bool overlap_check_capsule_cylinder(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_cylinder_capsule(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_capsule_convex(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	const btCapsuleShape *capsule = static_cast<btCapsuleShape *>(p_shape_1);

	{
		// Test shape 1
		btSphereShape sphere(capsule->getRadius());

		if (overlap_check_sphere_convex(
					&sphere,
					p_shape_1_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * capsule->getHalfHeight()),
					p_shape_2,
					p_shape_2_transform)) {
			return true;
		}

		if (overlap_check_sphere_convex(
					&sphere,
					p_shape_1_transform * btTransform(btMatrix3x3::getIdentity(), direction_axis[capsule->getUpAxis()] * (-capsule->getHalfHeight())),
					p_shape_2,
					p_shape_2_transform)) {
			return true;
		}
	}

	{
		btVector3 cylinder_half_extent;
		cylinder_half_extent[capsule->getUpAxis()] = capsule->getHalfHeight();
		cylinder_half_extent[(capsule->getUpAxis() + 1) % 3] = capsule->getRadius();
		cylinder_half_extent[(capsule->getUpAxis() + 2) % 3] = capsule->getRadius();
		btCylinderShapeZ cylinder(cylinder_half_extent); // <-- TODO
		if (overlap_check_cylinder_convex(&cylinder, p_shape_1_transform, p_shape_2, p_shape_2_transform)) {
			return true;
		}
	}

	// This check is special compared to the others. If nothing overlaps at this
	// point, the capsule is not colliding.
	return false;
}

bool overlap_check_convex_capsule(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_capsule_convex(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_polyhedron_polyhedron(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btPolyhedralConvexShape *poly_1 = static_cast<btPolyhedralConvexShape *>(p_shape_1);
	btPolyhedralConvexShape *poly_2 = static_cast<btPolyhedralConvexShape *>(p_shape_2);

	btVector3 sep;
	struct NoRes : public btDiscreteCollisionDetectorInterface::Result {
		virtual void setShapeIdentifiersA(int partId0, int index0) {}
		virtual void setShapeIdentifiersB(int partId1, int index1) {}
		virtual void addContactPoint(const btVector3 &normalOnBInWorld, const btVector3 &pointInWorld, real_t depth) {}
	} no_res;

	if (poly_1->getConvexPolyhedron() == nullptr) {
		poly_1->initializePolyhedralFeatures(0);
	}

	if (poly_2->getConvexPolyhedron() == nullptr) {
		poly_2->initializePolyhedralFeatures(0);
	}

	return btPolyhedralContactClipping::findSeparatingAxis(
			*poly_1->getConvexPolyhedron(),
			*poly_2->getConvexPolyhedron(),
			p_shape_1_transform,
			p_shape_2_transform,
			sep,
			no_res);
}

bool overlap_check_any_convex_concave(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	btConvexShape *convex = static_cast<btConvexShape *>(p_shape_1);
	btScaledBvhTriangleMeshShape *concave = static_cast<btScaledBvhTriangleMeshShape *>(p_shape_2);

	// Per each triangle check if it overlaps with the convex object.
	struct ConvexConcave : public btTriangleCallback {
		const btConvexShape *convex;
		const btTransform &shape_1_transform;
		const btCollisionShape *shape_2;
		const btTransform &shape_2_transform;
		bool overlap = false;
		uint32_t triangle_checked = 0;

		ConvexConcave(
				btConvexShape *p_shape_1,
				const btTransform &p_shape_1_transform,
				btCollisionShape *p_shape_2,
				const btTransform &p_shape_2_transform) :
				convex(p_shape_1),
				shape_1_transform(p_shape_1_transform),
				shape_2(p_shape_2),
				shape_2_transform(p_shape_2_transform) {
		}

		virtual void processTriangle(btVector3 *vertices, int partId, int triangleIndex) override {
			if (overlap) {
				// Nothing to do, separation found.
				return;
			}

			triangle_checked += 1;

			// Check if this triangle is overlapped with the body.
			const btTriangleShape triangle(vertices[0], vertices[1], vertices[2]);

			// Test faces of convex.
			for (uint32_t i = 0; i < 3; i += 1) {
				const btVector3 axis = shape_1_transform.getBasis().getColumn(i).normalized();

				if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, axis)) {
					return;
				}
			}

			// ~~ Check along the edges direction.

			// Triangle Axis 1
			const btVector3 triangle_edge_1 = (vertices[1] - vertices[0]).safeNormalize();
			const btVector3 global_triangle_edge_1 = shape_2_transform.getBasis() * triangle_edge_1;
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, global_triangle_edge_1)) {
				return;
			}

			// Triangle Axis 2
			const btVector3 triangle_edge_2 = (vertices[2] - vertices[1]).safeNormalize();
			const btVector3 global_triangle_edge_2 = shape_2_transform.getBasis() * triangle_edge_2;
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, global_triangle_edge_2)) {
				return;
			}

			// Triangle Axis 3
			const btVector3 triangle_edge_3 = (vertices[0] - vertices[2]).safeNormalize();
			const btVector3 global_triangle_edge_3 = shape_2_transform.getBasis() * triangle_edge_3;
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, global_triangle_edge_3)) {
				return;
			}

			// Check along the triangle normal.
			const btVector3 triangle_normal = triangle_edge_1.cross(triangle_edge_2).safeNormalize();
			const btVector3 global_triangle_normal = shape_2_transform.getBasis() * triangle_normal;
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, global_triangle_normal)) {
				return;
			}

			// Check along the direction to the objects but looking from the
			// prospective of edge + face.
			const btVector3 triangle_origin = (vertices[0] + vertices[1] + vertices[2]) / 3.0;
			const btVector3 triangle_origin_global = shape_2_transform * triangle_origin;
			const btVector3 dir_to_obj = (shape_1_transform.getOrigin() - triangle_origin_global).safeNormalize();

			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, toward(global_triangle_edge_1, dir_to_obj))) {
				return;
			}
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, toward(global_triangle_edge_2, dir_to_obj))) {
				return;
			}
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, toward(global_triangle_edge_3, dir_to_obj))) {
				return;
			}
			if (SAT::is_separated(convex, shape_1_transform, &triangle, shape_2_transform, toward(global_triangle_normal, dir_to_obj))) {
				return;
			}

			//if (is_separated_oriented_axis_one_way(
			//			convex,
			//			shape_1_transform,
			//			&triangle,
			//			shape_2_transform)) {
			//	return;
			//}

			overlap = true;
		}
	};

	ConvexConcave callback(
			convex,
			p_shape_1_transform,
			p_shape_2,
			p_shape_2_transform);

	btVector3 aabbMin;
	btVector3 aabbMax;

	p_shape_1->getAabb(p_shape_2_transform.inverse() * p_shape_1_transform, aabbMin, aabbMax);

	// Start the check.
	concave->processAllTriangles(
			&callback,
			aabbMin,
			aabbMax);

	return callback.triangle_checked > 0 && callback.overlap;
}

bool overlap_check_concave_any_convex(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_any_convex_concave(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_plane_any_convex(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	// Just check along the Y of the plane is enough.

	const btConvexShape *convex = static_cast<btConvexShape *>(p_shape_2);
	btVector3 axis = p_shape_1_transform.getBasis().getColumn(1);

	if (unlikely(axis.fuzzyZero())) {
		// Don't allow 0 axis test.
		return false;
	}

	axis.normalize();

	const SAT::Range s_1_range{ -FLT_MAX, axis.dot(p_shape_1_transform.getOrigin()) };
	const SAT::Range s_2_range = SAT::project(convex, axis, p_shape_2_transform);

	const bool is_separated = s_1_range.max < s_2_range.min || s_2_range.max < s_1_range.min;
	return is_separated == false;
}

bool overlap_check_any_convex_plane(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return overlap_check_plane_any_convex(
			p_shape_2,
			p_shape_2_transform,
			p_shape_1,
			p_shape_1_transform);
}

bool overlap_check_none(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform) {
	return false;
}

void OverlapCheck::init() {
	for (int i = 0; i < MAX_BROADPHASE_COLLISION_TYPES; i += 1) {
		for (int y = 0; y < MAX_BROADPHASE_COLLISION_TYPES; y += 1) {
			OverlapCheck::overlapping_funcs[i][y] = nullptr;
		}
	}

	// Sphere
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_sphere_sphere;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_sphere_box;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_sphere_capsule;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_sphere_cylinder;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_sphere_convex;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[SPHERE_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Box
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_box_box;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_box_sphere;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_box_capsule;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_box_cylinder;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_polyhedron_polyhedron;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[BOX_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Capsule
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_capsule_capsule;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_capsule_sphere;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_capsule_box;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_capsule_cylinder;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_capsule_convex;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[CAPSULE_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Cylinder
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_cylinder_box;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_cylinder_sphere;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_cylinder_capsule;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_cylinder_cylinder;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_cylinder_convex;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[CYLINDER_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Cone
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[CONE_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Convex
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_polyhedron_polyhedron;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_convex_sphere;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_convex_capsule;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_convex_cylinder;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = nullptr; // TODO
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_polyhedron_polyhedron;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_any_convex_concave;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_any_convex_plane;
	overlapping_funcs[CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Concave
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_concave_any_convex;
	overlapping_funcs[SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	// Plane
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_plane_any_convex;
	overlapping_funcs[STATIC_PLANE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;

	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][SPHERE_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][BOX_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][CAPSULE_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][CYLINDER_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][CONE_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][STATIC_PLANE_PROXYTYPE] = overlap_check_none;
	overlapping_funcs[EMPTY_SHAPE_PROXYTYPE][EMPTY_SHAPE_PROXYTYPE] = overlap_check_none;
}

OverlappingFunc OverlapCheck::find_algorithm(int body_1, int body_2) {
	if (body_1 < 0 || body_1 >= MAX_BROADPHASE_COLLISION_TYPES) {
		return nullptr;
	}

	if (body_2 < 0 || body_2 >= MAX_BROADPHASE_COLLISION_TYPES) {
		return nullptr;
	}

	return overlapping_funcs[body_1][body_2];
}
