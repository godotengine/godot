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

Vector3 closest_capsule_sphere_to_taper(head, tail, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current) {
	// The bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)

	Vector3 bone_axis = p_current - p_current_origin;  // should be length p_bone_radius due to calls to limit_length()
	Vector3 p = tail - head;
	Vector3 perp = bone_axis.cross(p);
	float perp_len = perp.length();
	if (is_zero_approx(perp_len)) {
		return head;
	}
	float perp_bone = perp.dot(p_current);
	float perp_capsule = perp.dot(head);
	float perp_dist = (perp_capsule - perp_bone) / perp_len;
	if (Math::abs(perp_dist) > radius + MAX(p_bone_origin_radius, p_bone_radius)) {
		return head;  // Too far away.
	}
	// Solve p_current_origin + bone_axis * lam + perp * perp_dist = head + p * mu
	Vector3 hh = p_current_origin - head;
	// dot bone_axis: hh.dot(bone_axis) + bone_axis.dot(bone_axis) * lam = p.dot(bone_axis) * mu
	// dot p: hh.dot(p) + bone_axis.dot(p) * lam = p.dot(p) * mu
	float badp = bone_axis.dot(p);
	float badba = bone_axis.dot(bone_axis)
	float pdp = p.dot(p);
	float hhdba = hh.dot(bone_axis);
	float hhdp = hh.dot(p);
	// hhdba = -badba * lam + badp * mu
	// hhdp = -badp * lam + pdp * mu
	// ( -badba  badp )   ( lam )   ( hhdba )
	// ( -badp   pdp  ) * (  mu ) = (  hhdp )
	float det = -badba * pdp + badp * badp;
	// ( pdp    -badp )   ( hhdba )   ( lam )
	// ( badp  -badba ) * (  hhdp ) = (  mu ) * det
	float lam = (pdp * hhdba - badp * hhdp) / det;
	float mu = (badp * hhdba - badba * hhdp) / det;

	// Now we account for the taper at this closest point of approach
	float taper_fore = (p_bone_origin_radius - p_bone_radius) / p_bone_length;
	float taper_side = Math::sqrt(1.0 - taper_fore * taper_fore);

	// Drive mud in (mu + mud) along the capsule axis.
	float component_along_bone = badp / Math::sqrt(badba);
	float component_perp_bone = Math::sqrt(pdp * pdp - component_along_bone * component_along_bone);
	
	float taperlamfac = taper_fore / taper_side / bone_axis_length
	// Then radial_distance = hypot(perp_dist, component_perp_bone * mud)
	//  and lammd = lam + component_along_bone * mud
	//  so lamcone = lammd - radial_distance * taperlamfac
	//  where coneradius = p_bone_origin_radius + (p_bone_radius - p_bone_origin_radius) * lamcone;

// We must minimize the distance between the conepoint and the capsule point minus the coneradius

	return head;
}

Vector3 SpringBoneCollisionCapsule3D::_collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3 &p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	// The bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)
	// Pick sphere in collider capsule that best collides with the bone.
	Pair<Vector3, Vector3> head_tail = get_head_and_tail(p_center);
	Vector3 head = head_tail.first;
	Vector3 tail = head_tail.second;
	if (collide_mode == COLLIDE_MODE_CHAIN) {
		Vector3 bone_sphere_center = closest_capsule_sphere_to_taper(head, tail, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
		return _collide_sphere_taper(origin, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	}
	Vector3 bone_sphere_center = closest_capsule_sphere(head, tail, p_current);
	return _collide_sphere(bone_sphere_center, radius, (collide_mode == COLLIDE_MODE_INSIDE), p_bone_radius, p_current);
}
