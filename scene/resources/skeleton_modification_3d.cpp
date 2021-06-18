/*************************************************************************/
/*  skeleton_modification_3d.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "skeleton_modification_3d.h"
#include "scene/3d/skeleton_3d.h"

void SkeletonModification3D::execute(float delta) {
	if (get_script_instance()) {
		if (get_script_instance()->has_method("execute")) {
			get_script_instance()->call("execute", delta);
		}
	}

	if (!enabled)
		return;
}

void SkeletonModification3D::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack) {
		is_setup = true;
	} else {
		WARN_PRINT("Could not setup modification with name " + this->get_name());
	}

	if (get_script_instance()) {
		if (get_script_instance()->has_method("setup_modification")) {
			get_script_instance()->call("setup_modification", p_stack);
		}
	}
}

void SkeletonModification3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModification3D::get_enabled() {
	return enabled;
}

// Helper function. Copied from the 2D IK PR. Needed for CCDIK.
float SkeletonModification3D::clamp_angle(float angle, float min_bound, float max_bound, bool invert) {
	// Map to the 0 to 360 range (in radians though) instead of the -180 to 180 range.
	if (angle < 0) {
		angle = (Math_PI * 2) + angle;
	}

	// Make min and max in the range of 0 to 360 (in radians), and make sure they are in the right order
	if (min_bound < 0) {
		min_bound = (Math_PI * 2) + min_bound;
	}
	if (max_bound < 0) {
		max_bound = (Math_PI * 2) + max_bound;
	}
	if (min_bound > max_bound) {
		float tmp = min_bound;
		min_bound = max_bound;
		max_bound = tmp;
	}

	// Note: May not be the most optimal way to clamp, but it always constraints to the nearest angle.
	if (invert == false) {
		if (angle < min_bound || angle > max_bound) {
			Vector2 min_bound_vec = Vector2(Math::cos(min_bound), Math::sin(min_bound));
			Vector2 max_bound_vec = Vector2(Math::cos(max_bound), Math::sin(max_bound));
			Vector2 angle_vec = Vector2(Math::cos(angle), Math::sin(angle));

			if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
				angle = min_bound;
			} else {
				angle = max_bound;
			}
		}
	} else {
		if (angle > min_bound && angle < max_bound) {
			Vector2 min_bound_vec = Vector2(Math::cos(min_bound), Math::sin(min_bound));
			Vector2 max_bound_vec = Vector2(Math::cos(max_bound), Math::sin(max_bound));
			Vector2 angle_vec = Vector2(Math::cos(angle), Math::sin(angle));

			if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
				angle = min_bound;
			} else {
				angle = max_bound;
			}
		}
	}
	return angle;
}

bool SkeletonModification3D::_print_execution_error(bool p_condition, String p_message) {
	// If the modification is not setup, don't bother printing the error
	if (!is_setup) {
		return p_condition;
	}

	if (p_condition && !execution_error_found) {
		ERR_PRINT(p_message);
		execution_error_found = true;
	}
	return p_condition;
}

Ref<SkeletonModificationStack3D> SkeletonModification3D::get_modification_stack() {
	return stack;
}

void SkeletonModification3D::set_is_setup(bool p_is_setup) {
	is_setup = p_is_setup;
}

bool SkeletonModification3D::get_is_setup() const {
	return is_setup;
}

void SkeletonModification3D::set_execution_mode(int p_mode) {
	execution_mode = p_mode;
}

int SkeletonModification3D::get_execution_mode() const {
	return execution_mode;
}

void SkeletonModification3D::_bind_methods() {
	BIND_VMETHOD(MethodInfo("execute", PropertyInfo(Variant::FLOAT, "delta")));
	BIND_VMETHOD(MethodInfo("setup_modification", PropertyInfo(Variant::OBJECT, "modification_stack", PROPERTY_HINT_RESOURCE_TYPE, "SkeletonModificationStack3D")));

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification3D::get_enabled);
	ClassDB::bind_method(D_METHOD("get_modification_stack"), &SkeletonModification3D::get_modification_stack);
	ClassDB::bind_method(D_METHOD("set_is_setup", "is_setup"), &SkeletonModification3D::set_is_setup);
	ClassDB::bind_method(D_METHOD("get_is_setup"), &SkeletonModification3D::get_is_setup);
	ClassDB::bind_method(D_METHOD("set_execution_mode", "execution_mode"), &SkeletonModification3D::set_execution_mode);
	ClassDB::bind_method(D_METHOD("get_execution_mode"), &SkeletonModification3D::get_execution_mode);
	ClassDB::bind_method(D_METHOD("clamp_angle", "angle", "min", "max", "invert"), &SkeletonModification3D::clamp_angle);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "execution_mode", PROPERTY_HINT_ENUM, "process, physics_process"), "set_execution_mode", "get_execution_mode");
}

SkeletonModification3D::SkeletonModification3D() {
	stack = nullptr;
	is_setup = false;
}
