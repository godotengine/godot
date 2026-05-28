/**************************************************************************/
/*  spring_bone_collision_capsule_3d.cpp                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "spring_bone_collision_capsule_3d.h"

#include "core/object/class_db.h"
#include "scene/3d/spring_bone_collision_sphere_3d.h"

void SpringBoneCollisionCapsule3D::set_radius(float p_radius) {
	radius = p_radius;
	if (radius > height * 0.5) {
		height = radius * 2.0;
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float SpringBoneCollisionCapsule3D::get_radius() const {
	return radius;
}

void SpringBoneCollisionCapsule3D::set_height(float p_height) {
	height = p_height;
	if (radius > height * 0.5) {
		radius = height * 0.5;
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float SpringBoneCollisionCapsule3D::get_height() const {
	return height;
}

void SpringBoneCollisionCapsule3D::set_mid_height(real_t p_mid_height) {
	ERR_FAIL_COND_MSG(p_mid_height < 0.0f, "SpringBoneCollisionCapsule3D mid-height cannot be negative.");
	height = p_mid_height + radius * 2.0f;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

real_t SpringBoneCollisionCapsule3D::get_mid_height() const {
	return height - radius * 2.0f;
}

SpringBoneCollision3D::CollideMode SpringBoneCollisionCapsule3D::get_collide_mode() const {
	return collide_mode;
}

void SpringBoneCollisionCapsule3D::set_collide_mode(CollideMode p_collide_mode) {
	collide_mode = p_collide_mode;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Pair<Vector3, Vector3> SpringBoneCollisionCapsule3D::get_head_and_tail(const Transform3D &p_center) const {
	Transform3D tr = get_transform_from_skeleton(p_center);
	return Pair<Vector3, Vector3>(tr.origin + tr.basis.xform(Vector3::UP * (height * 0.5 - radius)), tr.origin + tr.basis.xform(Vector3::DOWN * (height * 0.5 - radius)));
}

void SpringBoneCollisionCapsule3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &SpringBoneCollisionCapsule3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &SpringBoneCollisionCapsule3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &SpringBoneCollisionCapsule3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &SpringBoneCollisionCapsule3D::get_height);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &SpringBoneCollisionCapsule3D::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &SpringBoneCollisionCapsule3D::get_mid_height);
	ClassDB::bind_method(D_METHOD("set_collide_mode", "collide_mode"), &SpringBoneCollisionCapsule3D::set_collide_mode);
	ClassDB::bind_method(D_METHOD("get_collide_mode"), &SpringBoneCollisionCapsule3D::get_collide_mode);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m", PROPERTY_USAGE_NONE), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collide_mode", PROPERTY_HINT_ENUM, "Joint,Inside,Chain"), "set_collide_mode", "get_collide_mode");
}

Vector3 closest_capsule_sphere(const Vector3 &head, const Vector3 &tail, const Vector3 &bone_sphere_center) {
	Vector3 p = tail - head;
	Vector3 q = bone_sphere_center - head;
	float dot = p.dot(q);
	if (dot <= 0) {
		return head;
	}
	float pls = p.length_squared();
	if ((pls <= dot) || Math::is_zero_approx(pls)) {
		return tail;
	}
	return head + p * (dot / pls);
}

bool closest_capsule_sphere_to_taper(Vector3 &capsule_sphere_center, const Vector3 &head, const Vector3 &tail, float radius, const Vector3 &p_bone_little_end, float p_bone_little_end_radius, const Vector3 &p_bone_big_end, float p_bone_big_end_radius, float p_bone_length) {
	// The collision capsule is (head, tail), radius.
	// The tapered bone capsule is from (p_bone_little_end, p_bone_little_end_radius) to (p_bone_big_end, p_bone_big_end_radius).
	DEV_ASSERT(p_bone_little_end_radius <= p_bone_big_end_radius);

	Vector3 bone_axis = p_bone_big_end - p_bone_little_end;  // should be length p_bone_length due to calls to limit_length()
	Vector3 p = tail - head;
	DEV_ASSERT(Math::is_equal_approx(bone_axis.length(), p_bone_length));

	// The bone_axis and p (the capsule axis) are skew lines, 
	// so the cross-product vector is the shortest distance between them. 
	Vector3 perp = bone_axis.cross(p);
	real_t perp_len = perp.length();
	if (Math::is_zero_approx(perp_len)) {
		capsule_sphere_center = head;
		return true;
	}
	real_t perp_bone = perp.dot(p_bone_little_end);
	real_t perp_capsule = perp.dot(head);
	real_t perp_dist = (perp_capsule - perp_bone) / perp_len;
	if (Math::abs(perp_dist) > radius + p_bone_big_end_radius) {
		return false;
	}

	// Calculate the points of closest approach between these two skew lines
	// by solving: p_bone_little_end + bone_axis * lam + perp = head + p * mu

	Vector3 hh = p_bone_little_end - head;
	// dot bone_axis: hh.dot(bone_axis) + bone_axis.dot(bone_axis) * lam = p.dot(bone_axis) * mu
	// dot p: hh.dot(p) + bone_axis.dot(p) * lam = p.dot(p) * mu
	real_t badp = bone_axis.dot(p);
	real_t badba = bone_axis.dot(bone_axis);
	real_t pdp = p.dot(p);
	real_t hhdba = hh.dot(bone_axis);
	real_t hhdp = hh.dot(p);
	// hhdba = -badba * lam + badp * mu
	// hhdp = -badp * lam + pdp * mu
	// ( -badba  badp )   ( lam )   ( hhdba )
	// ( -badp   pdp  ) * (  mu ) = (  hhdp )
	real_t det = -badba * pdp + badp * badp;
	// ( pdp    -badp )   ( hhdba )   ( lam )
	// ( badp  -badba ) * (  hhdp ) = (  mu ) * det
	real_t lam = (pdp * hhdba - badp * hhdp) / det;
	real_t mu = (badp * hhdba - badba * hhdp) / det;

	// Check calculation
	real_t Dhhdba = -badba * lam + badp * mu;
	real_t Dhhdp = -badp * lam + pdp * mu;
	Vector3 Dperpvec = perp * (perp_dist / perp_len);
	printf(" %d %.03f %0.3f small ", SpringBoneCollision3D::Dsegmentindexbeingcalculated, Dhhdba - hhdba, Dhhdp - hhdp);
	Vector3 Dlammuvec = (p_bone_little_end + bone_axis * lam + Dperpvec) - (head + p * mu);
	printf(" Dlamuvec %.04f small ", Dlammuvec.length());

	// handle cylinder case
	if (p_bone_little_end_radius == p_bone_big_end_radius) {
		printf("\n");
		if ((mu < 0.0) || (mu > 1.0)) {
			return false;
		}
		capsule_sphere_center = head + p * mu;
		return true;
	}
	real_t cone_slope = p_bone_length / (p_bone_big_end_radius - p_bone_little_end_radius);

	// Now consider the plane perpendicular to perp containing the head-tail vector of the capsule
	// Set its origin to be at (head + p * mu) with y-vector along the normalized_bone_axis
	// and x-vector perpendicular to bone_axis and perp.
	// The intersection of this plane with the bone cone will be a hyperbola (conic section).
	// When the capsule intrudes by a distance of intrad into the bone cone, then the cone 
	// from (p_bone_little_end, p_bone_little_end_radius + radius - intrad) to (p_bone_big_end, p_bone_big_end_radius + radius - intrad), 
	// and therefore the hyperbola will be tangential to the capsule axis.

	real_t cone_rad_at_plane_origin = p_bone_little_end_radius + (p_bone_little_end_radius - p_bone_big_end_radius) * lam;

	Vector3 cap_plane_origin = head + p * mu;
	real_t bone_axis_length = sqrt(badba);
	real_t pcomponent_y = badp / bone_axis_length;
	real_t pcomponent_x = sqrt(pdp - pcomponent_y * pcomponent_y);
	Vector3 Dcomponenty = bone_axis.normalized();
	Vector3 Dcomponentx = (p - Dcomponenty * p.dot(Dcomponenty)).normalized();
	printf(" compvs %.02f %.02f ", Dcomponenty.dot(p) - pcomponent_y, Dcomponentx.dot(p) - pcomponent_x);
	real_t capslope = pcomponent_y / pcomponent_x;

	// The collision capsule is assumed to intrude into the tapered bone capsule, 
	// and we can add the radius of the collision capsule to the bone capsule so we can consider it as a line.
	// So we are looking for value of intrad where 
	// The (p_bone_little_end, p_bone_little_end_radius + radius - intrad) to (p_bone_big_end, p_bone_big_end_radius + radius - intrad)

	// The apex of cone in plane is (0, -cone_rad_at_plane_origin/cone_slope)
	// Relative to this origin in the plane, a point (x,y) in the plane is on the cone (where it intersects the plane) 
	// when (perp_dist^2 + x^2) = (y*cone_slope)^2, 
	// so: y = sqrt(perp_dist^2 + x^2)/cone_slope which asymtotes to x/cone_slope as x tends to infinity.
	// The gradient of y at x is:
	//  dy/dx = (perp_dist^2 + x^2) ^ (-1/2) * x / cone_slope 
	//  which needs to equal capslope.
	real_t ccsq = capslope * cone_slope * capslope * cone_slope;
	// capslope*cone_slope = (perp_dist^2 + x^2) ^ (-1/2) * x 
	real_t perp_dist_sq = perp_dist * perp_dist;
	real_t xsq = ccsq * perp_dist_sq / (1.0 - ccsq);
	if (xsq < 0.0) {
		printf(" %.2f xsq<0\n", xsq);
		return false;
	}
	real_t tangent_axis_distance = Math::sqrt(perp_dist_sq + xsq);

	real_t x = Math::sqrt(xsq);
	real_t y_line = Math::abs(capslope)*x;
	real_t mu2 = mu + x/pcomponent_x * SIGN(pcomponent_y);
	// discard if mu2 is outside of unit range
	real_t lam2 = lam + y_line / bone_axis_length;
	// discard if lam2 is outside of unit range

	real_t cone_rad_at_tangent = p_bone_little_end_radius + (p_bone_little_end_radius - p_bone_big_end_radius) * lam2;
	real_t intrad = tangent_axis_distance - cone_rad_at_tangent;
	// discard if intrad < 0

	// calculations here to prove that this is the minimum (tangential) approach

	printf("\n");
	if ((mu2 >= 0.0) and (mu2 < 1.0)) {
		capsule_sphere_center = head + p * mu2;
		return true;
	}
	return false;
}

Vector3 SpringBoneCollisionCapsule3D::_collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3 &p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	// The tapered bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius).
	Pair<Vector3, Vector3> head_tail = get_head_and_tail(p_center);
	Vector3 head = head_tail.first;
	Vector3 tail = head_tail.second;
	if (collide_mode == COLLIDE_MODE_CHAIN) {
		// Pick sphere in collider capsule that best collides with the tapered bone.
		Vector3 capsule_sphere_center;
		bool collision_detected;
		if (p_bone_origin_radius <= p_bone_radius) {
			collision_detected = closest_capsule_sphere_to_taper(capsule_sphere_center, head, tail, radius, p_current_origin, p_bone_origin_radius, p_current, p_bone_radius, p_bone_length);
		} else {
			collision_detected = closest_capsule_sphere_to_taper(capsule_sphere_center, head, tail, radius, p_current, p_bone_radius, p_current_origin, p_bone_origin_radius, p_bone_length);
		}
		return _collide_sphere_taper(capsule_sphere_center, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	}
	// Pick sphere in collider capsule that best collides with the bone end point (the joint).
	Vector3 capsule_sphere_center = closest_capsule_sphere(head, tail, p_current);
	return _collide_sphere(capsule_sphere_center, radius, (collide_mode == COLLIDE_MODE_INSIDE), p_bone_radius, p_current);
}
