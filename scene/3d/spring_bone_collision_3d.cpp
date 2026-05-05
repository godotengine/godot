/**************************************************************************/
/*  spring_bone_collision_3d.cpp                                          */
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

#include "spring_bone_collision_3d.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "scene/3d/spring_bone_simulator_3d.h"

PackedStringArray SpringBoneCollision3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	SpringBoneSimulator3D *parent = Object::cast_to<SpringBoneSimulator3D>(get_parent());
	if (!parent) {
		warnings.push_back(RTR("Parent node should be a SpringBoneSimulator3D node."));
	}

	return warnings;
}

void SpringBoneCollision3D::_validate_property(PropertyInfo &p_property) const {
	if (Engine::get_singleton()->is_editor_hint() && p_property.name == "bone_name") {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			p_property.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			p_property.hint_string = sk->get_concatenated_bone_names();
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	} else if (bone < 0 && (p_property.name == "position_offset" || p_property.name == "rotation_offset")) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void SpringBoneCollision3D::_validate_bone_name() {
	// Prior bone name.
	if (!bone_name.is_empty()) {
		set_bone_name(bone_name);
	} else if (bone != -1) {
		set_bone(bone);
	}
}

Skeleton3D *SpringBoneCollision3D::get_skeleton() const {
	SpringBoneSimulator3D *parent = Object::cast_to<SpringBoneSimulator3D>(get_parent());
	if (!parent) {
		return nullptr;
	}
	return parent->get_skeleton();
}

void SpringBoneCollision3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_bone(sk->find_bone(bone_name));
	}
}

String SpringBoneCollision3D::get_bone_name() const {
	return bone_name;
}

void SpringBoneCollision3D::set_bone(int p_bone) {
	bone = p_bone;

	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (bone < -1 || bone >= sk->get_bone_count()) {
			WARN_PRINT("Bone index '" + itos(p_bone) + "' is out of range! Cannot connect BoneAttachment to node!");
			bone = -1;
		} else {
			bone_name = sk->get_bone_name(bone);
		}
	}

	notify_property_list_changed();
}

int SpringBoneCollision3D::get_bone() const {
	return bone;
}

void SpringBoneCollision3D::set_position_offset(const Vector3 &p_offset) {
	if (position_offset == p_offset) {
		return;
	}
	position_offset = p_offset;
	sync_pose();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Vector3 SpringBoneCollision3D::get_position_offset() const {
	return position_offset;
}

void SpringBoneCollision3D::set_rotation_offset(const Quaternion &p_offset) {
	if (rotation_offset == p_offset) {
		return;
	}
	rotation_offset = p_offset;
	sync_pose();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Quaternion SpringBoneCollision3D::get_rotation_offset() const {
	return rotation_offset;
}

void SpringBoneCollision3D::sync_pose() {
	if (bone >= 0) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			Transform3D tr = sk->get_global_transform() * sk->get_bone_global_pose(bone);
			tr.origin += tr.basis.get_rotation_quaternion().xform(position_offset);
			tr.basis *= Basis(rotation_offset);
			set_global_transform(tr);
		}
	}
}

Transform3D SpringBoneCollision3D::get_transform_from_skeleton(const Transform3D &p_center) const {
	Transform3D gtr = get_global_transform();
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		Transform3D tr = sk->get_global_transform();
		gtr = tr.affine_inverse() * p_center * gtr;
	}
	return gtr;
}

void SpringBoneCollision3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skeleton"), &SpringBoneCollision3D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &SpringBoneCollision3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &SpringBoneCollision3D::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone", "bone"), &SpringBoneCollision3D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &SpringBoneCollision3D::get_bone);

	ClassDB::bind_method(D_METHOD("set_position_offset", "offset"), &SpringBoneCollision3D::set_position_offset);
	ClassDB::bind_method(D_METHOD("get_position_offset"), &SpringBoneCollision3D::get_position_offset);

	ClassDB::bind_method(D_METHOD("set_rotation_offset", "offset"), &SpringBoneCollision3D::set_rotation_offset);
	ClassDB::bind_method(D_METHOD("get_rotation_offset"), &SpringBoneCollision3D::get_rotation_offset);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_bone", "get_bone");

	ADD_GROUP("Offset", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position_offset"), "set_position_offset", "get_position_offset");
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "rotation_offset"), "set_rotation_offset", "get_rotation_offset");

	BIND_ENUM_CONSTANT(COLLIDE_MODE_JOINT);
	BIND_ENUM_CONSTANT(COLLIDE_MODE_INSIDE);
	BIND_ENUM_CONSTANT(COLLIDE_MODE_CHAIN);
}

void SpringBoneCollision3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			_validate_bone_name();
		} break;
	}
}

// static
Vector3 SpringBoneCollision3D::_collide_sphere(const Vector3 &p_origin, float p_radius, bool p_inside, float p_bone_radius, const Vector3 &p_current) {
	Vector3 diff = p_current - p_origin;
	float diff_length = diff.length();
	float r = p_inside ? p_radius - p_bone_radius : p_bone_radius + p_radius;
	float distance = p_inside ? r - diff_length : diff_length - r;
	if (distance > 0) {
		return p_current;
	}
	return p_origin + diff.normalized() * r;
}

// static
Vector3 SpringBoneCollision3D::_collide_sphere_taper(const Vector3 &p_origin, float p_radius, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) {
	// (p_origin, p_radius) defines the external collider
	// The bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)
	// where p_current is to be displaced

	float taper_fore = (p_bone_origin_radius - p_bone_radius) / p_bone_length;

	// send the short taper case into old implementation
	if (Math::abs(taper_fore) >= 1.0) {
		return _collide_sphere(p_origin, p_radius, false, p_bone_radius, p_current);
	}

	Vector3 diff = p_current - p_origin;
	Vector3 bone_axis = p_current - p_current_origin;  // should be length p_bone_radius due to calls to limit_length()
	float taper_side = Math::sqrt(1.0 - taper_fore * taper_fore);
	float bone_axis_sq = bone_axis.dot(bone_axis);
	float lam = 1.0 - bone_axis.dot(diff) / bone_axis_sq;  // calculated from the tail end
	Vector3 vecside = p_origin - (p_current_origin + bone_axis * lam);
	// printf(" zz=%f ", vecside.dot(bone_axis)); // should be zero
	float radial_distance = vecside.length();
	if (radial_distance > MAX(p_bone_origin_radius, p_bone_radius) + p_radius) {
		return p_current;
	}
	float bone_axis_length = Math::sqrt(bone_axis_sq);

	// limit contact with the cone close to the root where it gets twitchy
	float gapdistance = p_bone_radius * 0.5 + p_bone_origin_radius * 0.5 + p_radius * Math::sqrt(0.5);
	float lamconemin = gapdistance / bone_axis_length * 0.5;

	// case of collide sphere being very large.
	if (lamconemin > 1.0) {  // apply this case before the beyond origin end to avoid twitchiness
		return _collide_sphere(p_origin, p_radius, false, p_bone_radius, p_current);
	}

	float lamd = radial_distance * taper_fore / taper_side / bone_axis_length;
	float lamcone = lam - lamd;
	if (lamcone <= 0.0) {  // beyond origin end
		return p_current;
	}
	if (lamcone >= 1.0) {  // beyond tail end
		return _collide_sphere(p_origin, p_radius, false, p_bone_radius, p_current);
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
	Vector3 p_current_new = p_current_origin + (p_coneaxispointnew - p_current_origin) / MAX(0.1f, lamcone);
	
	return p_current_new;
}

Vector3 SpringBoneCollision3D::collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	return _collide(p_center, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
}

Vector3 SpringBoneCollision3D::_collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3& p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	return Vector3(0, 0, 0);
}
