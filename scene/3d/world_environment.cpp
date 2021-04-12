/*************************************************************************/
/*  world_environment.cpp                                                */
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

#include "world_environment.h"

#include "scene/main/window.h"

void WorldEnvironment::_notification(int p_what) {
	if (p_what == Node3D::NOTIFICATION_ENTER_WORLD || p_what == Node3D::NOTIFICATION_ENTER_TREE) {
		if (environment.is_valid()) {
			add_to_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
			_update_current_environment();
		}

		if (camera_effects.is_valid()) {
			add_to_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
			_update_current_camera_effects();
		}

	} else if (p_what == Node3D::NOTIFICATION_EXIT_WORLD || p_what == Node3D::NOTIFICATION_EXIT_TREE) {
		if (environment.is_valid()) {
			remove_from_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
			_update_current_environment();
		}

		if (camera_effects.is_valid()) {
			remove_from_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
			_update_current_camera_effects();
		}
	}
}

void WorldEnvironment::_update_current_environment() {
	WorldEnvironment *first = Object::cast_to<WorldEnvironment>(get_tree()->get_first_node_in_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id())));

	if (first) {
		get_viewport()->find_world_3d()->set_environment(first->environment);
	} else {
		get_viewport()->find_world_3d()->set_environment(Ref<Environment>());
	}
	get_tree()->call_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()), "update_configuration_warnings");
}

void WorldEnvironment::_update_current_camera_effects() {
	WorldEnvironment *first = Object::cast_to<WorldEnvironment>(get_tree()->get_first_node_in_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id())));
	if (first) {
		get_viewport()->find_world_3d()->set_camera_effects(first->camera_effects);
	} else {
		get_viewport()->find_world_3d()->set_camera_effects(Ref<CameraEffects>());
	}

	get_tree()->call_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()), "update_configuration_warnings");
}

void WorldEnvironment::set_environment(const Ref<Environment> &p_environment) {
	if (environment == p_environment) {
		return;
	}
	if (is_inside_tree() && environment.is_valid()) {
		remove_from_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	environment = p_environment;

	if (is_inside_tree() && environment.is_valid()) {
		add_to_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	if (is_inside_tree()) {
		_update_current_environment();
	} else {
		update_configuration_warnings();
	}
}

Ref<Environment> WorldEnvironment::get_environment() const {
	return environment;
}

void WorldEnvironment::set_camera_effects(const Ref<CameraEffects> &p_camera_effects) {
	if (camera_effects == p_camera_effects) {
		return;
	}

	if (is_inside_tree() && camera_effects.is_valid() && get_viewport()->find_world_3d()->get_camera_effects() == camera_effects) {
		remove_from_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	camera_effects = p_camera_effects;
	if (is_inside_tree() && camera_effects.is_valid()) {
		add_to_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	if (is_inside_tree()) {
		_update_current_camera_effects();
	} else {
		update_configuration_warnings();
	}
}

Ref<CameraEffects> WorldEnvironment::get_camera_effects() const {
	return camera_effects;
}

TypedArray<String> WorldEnvironment::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!environment.is_valid()) {
		warnings.push_back(TTR("WorldEnvironment requires its \"Environment\" property to contain an Environment to have a visible effect."));
	}

	if (!is_inside_tree()) {
		return warnings;
	}

	if (environment.is_valid() && get_viewport()->find_world_3d()->get_environment() != environment) {
		warnings.push_back(("Only the first Environment has an effect in a scene (or set of instantiated scenes)."));
	}

	if (camera_effects.is_valid() && get_viewport()->find_world_3d()->get_camera_effects() != camera_effects) {
		warnings.push_back(TTR("Only one WorldEnvironment is allowed per scene (or set of instanced scenes)."));
	}

	return warnings;
}

void WorldEnvironment::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_environment", "env"), &WorldEnvironment::set_environment);
	ClassDB::bind_method(D_METHOD("get_environment"), &WorldEnvironment::get_environment);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_environment", "get_environment");

	ClassDB::bind_method(D_METHOD("set_camera_effects", "env"), &WorldEnvironment::set_camera_effects);
	ClassDB::bind_method(D_METHOD("get_camera_effects"), &WorldEnvironment::get_camera_effects);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "camera_effects", PROPERTY_HINT_RESOURCE_TYPE, "CameraEffects"), "set_camera_effects", "get_camera_effects");
}

WorldEnvironment::WorldEnvironment() {
}
