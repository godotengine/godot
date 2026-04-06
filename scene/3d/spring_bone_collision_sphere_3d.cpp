/**************************************************************************/
/*  spring_bone_collision_sphere_3d.cpp                                   */
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

#include "spring_bone_collision_sphere_3d.h"

#include "core/object/class_db.h"

void SpringBoneCollisionSphere3D::set_radius(float p_radius) {
	radius = p_radius;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float SpringBoneCollisionSphere3D::get_radius() const {
	return radius;
}

void SpringBoneCollisionSphere3D::set_inside(bool p_enabled) {
	inside = p_enabled;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

bool SpringBoneCollisionSphere3D::is_inside() const {
	return inside;
}

void SpringBoneCollisionSphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &SpringBoneCollisionSphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &SpringBoneCollisionSphere3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_inside", "enabled"), &SpringBoneCollisionSphere3D::set_inside);
	ClassDB::bind_method(D_METHOD("is_inside"), &SpringBoneCollisionSphere3D::is_inside);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "inside"), "set_inside", "is_inside");
}


Vector3 SpringBoneCollisionSphere3D::_collide_sphere(const Vector3 &p_origin, float p_radius, bool p_inside, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) {
	Vector3 diff = p_current - p_origin;
	float diff_length = diff.length();
	float r = p_inside ? p_radius - p_bone_radius : p_bone_radius + p_radius;
	float distance = p_inside ? r - diff_length : diff_length - r;
	if (distance > 0) {
		return p_current;
	}
	return p_origin + diff.normalized() * r;
}

Vector3 SpringBoneCollisionSphere3D::_collide_sphere_taper(const Vector3 &p_origin, float p_radius, bool p_inside, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) {
	// (p_origin, p_radius) defines the external collider, 
	// The bone caspsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)
	// where p_current is to be displaced

	float taper_fore = (p_bone_origin_radius - p_bone_radius) / p_bone_length;
	
	// send the troublesome inside mode and short taper case into old implementation
	if (p_inside || (fabs(taper_fore) >= 1.0)) { 
		return _collide_sphere(p_origin, p_radius, p_inside, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	}

	Vector3 diff = p_current - p_origin;
	Vector3 bone_axis = p_current - p_current_origin;  // should be length p_bone_radius due to calls to limit_length()
	float taper_side = sqrt(1.0 - taper_fore * taper_fore);
	float bone_axis_sq = bone_axis.dot(bone_axis);
	float lam = 1.0 - bone_axis.dot(diff) / bone_axis_sq;  // calculated from the tail end
	Vector3 vecside = p_origin - (p_current_origin + bone_axis * lam);
	// printf(" zz=%f ", vecside.dot(bone_axis)); // should be zero
	float radial_distance = vecside.length();
	if (radial_distance > std::max(p_bone_origin_radius, p_bone_radius) + p_radius) {
		return p_current;
	}
	float bone_axis_length = sqrt(bone_axis_sq);

	// limit contact with the cone close to the root where it gets twitchy
	float gapdistance = p_bone_radius * 0.5 + p_bone_origin_radius * 0.5 + p_radius * sqrt(0.5);
	float lamconemin = gapdistance / bone_axis_length * 0.5;

	// case of collide sphere being very large.
	if (lamconemin > 1.0) {  // apply this case before the beyond origin end to avoid twitchiness
		return _collide_sphere(p_origin, p_radius, p_inside, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	}

	float lamd = radial_distance * taper_fore / taper_side / bone_axis_length;
	float lamcone = lam - lamd;
	if (lamcone <= 0.0) {  // beyond origin end
		return p_current;
	}
	if (lamcone >= 1.0) {  // beyond tail end
		return _collide_sphere(p_origin, p_radius, p_inside, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	}

	// prove numerically this is the closest approach to the cone
	/*float lam1 = lamcone;
	float m1 = (p_current_origin + bone_axis * lam1 - p_origin).length() - (p_bone_origin_radius + (p_bone_radius - p_bone_origin_radius) * lam1);
	float lam0 = lamcone - 0.01;
	float m0 = (p_current_origin + bone_axis * lam0 - p_origin).length() - (p_bone_origin_radius + (p_bone_radius - p_bone_origin_radius) * lam0);
	float lam2 = lamcone + 0.01;
	float m2 = (p_current_origin + bone_axis * lam2 - p_origin).length() - (p_bone_origin_radius + (p_bone_radius - p_bone_origin_radius) * lam2);
	printf(" check %f>0 ", std::min(m0, m2) - m1);
	*/

	if (lamcone < lamconemin) {
		lamcone = lamconemin;
	}
	
	// Check collision with this cone
	Vector3 coneaxispoint = p_current_origin + bone_axis * lamcone;
	Vector3 conepointdiff = coneaxispoint - p_origin;
	float coneaxisradius = p_bone_origin_radius + (p_bone_radius - p_bone_origin_radius) * lamcone;

	float r = coneaxisradius + p_radius;
	float conepointdifflength = conepointdiff.length();
	float distance = conepointdifflength - r;
	if (distance > 0.0) {
		return p_current;
	}
	//printf(" hh=%f; ", distance); 

	// We could model a rotation of the bone_axis about p_current_origin to move the (coneaxispoint, coneaxisradius) sphere 
	// away from its intersection with (p_origin, p_radius) [not quite accurate since as it rotates the lamcone position 	
	// of the virtual sphere inside the cone changes].
	// But instead we will just project it as a simple lever and rely on limit_length() and the iteration to settle it into the correct place.

	// position virtual sphere of contact in the cone is pushed to
	Vector3 p_coneaxispointnew = p_origin + conepointdiff.normalized() * r;

	// projection out to the end point of the cone as though it were a lever
	// (this isn't necessary since the change it makes is masked by the limit_length() function)
	// also limit the size of the multiplier to avoid extreme movement when near the joint
	Vector3 p_current_new = p_current_origin + (p_coneaxispointnew - p_current_origin) / std::max<float>(0.1, lamcone);
	
	return p_current_new;
}

Vector3 SpringBoneCollisionSphere3D::_collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	return _collide_sphere_taper(get_transform_from_skeleton(p_center).origin, radius, inside, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
}
