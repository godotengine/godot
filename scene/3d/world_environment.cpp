/*************************************************************************/
/*  world_environment.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
			if (get_viewport()->find_world_3d()->get_environment().is_valid()) {
				WARN_PRINT("World already has an environment (Another WorldEnvironment?), overriding.");
			}
			get_viewport()->find_world_3d()->set_environment(environment);
			add_to_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		}

		if (camera_effects.is_valid()) {
			if (get_viewport()->find_world_3d()->get_camera_effects().is_valid()) {
				WARN_PRINT("World already has a camera effects (Another WorldEnvironment?), overriding.");
			}
			get_viewport()->find_world_3d()->set_camera_effects(camera_effects);
			add_to_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		}

	} else if (p_what == Node3D::NOTIFICATION_EXIT_WORLD || p_what == Node3D::NOTIFICATION_EXIT_TREE) {
		if (environment.is_valid() && get_viewport()->find_world_3d()->get_environment() == environment) {
			get_viewport()->find_world_3d()->set_environment(Ref<Environment>());
			remove_from_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		}

		if (camera_effects.is_valid() && get_viewport()->find_world_3d()->get_camera_effects() == camera_effects) {
			get_viewport()->find_world_3d()->set_camera_effects(Ref<CameraEffects>());
			remove_from_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		}
	}
}

void WorldEnvironment::set_environment(const Ref<Environment> &p_environment) {
	if (is_inside_tree() && environment.is_valid() && get_viewport()->find_world_3d()->get_environment() == environment) {
		get_viewport()->find_world_3d()->set_environment(Ref<Environment>());
		remove_from_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		//clean up
	}

	environment = p_environment;
	if (is_inside_tree() && environment.is_valid()) {
		if (get_viewport()->find_world_3d()->get_environment().is_valid()) {
			WARN_PRINT("World already has an environment (Another WorldEnvironment?), overriding.");
		}
		get_viewport()->find_world_3d()->set_environment(environment);
		add_to_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	update_configuration_warning();
}

Ref<Environment> WorldEnvironment::get_environment() const {
	return environment;
}

void WorldEnvironment::set_camera_effects(const Ref<CameraEffects> &p_camera_effects) {
	if (is_inside_tree() && camera_effects.is_valid() && get_viewport()->find_world_3d()->get_camera_effects() == camera_effects) {
		get_viewport()->find_world_3d()->set_camera_effects(Ref<CameraEffects>());
		remove_from_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
		//clean up
	}

	camera_effects = p_camera_effects;
	if (is_inside_tree() && camera_effects.is_valid()) {
		if (get_viewport()->find_world_3d()->get_camera_effects().is_valid()) {
			WARN_PRINT("World already has an camera_effects (Another WorldEnvironment?), overriding.");
		}
		get_viewport()->find_world_3d()->set_camera_effects(camera_effects);
		add_to_group("_world_camera_effects_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()));
	}

	update_configuration_warning();
}

Ref<CameraEffects> WorldEnvironment::get_camera_effects() const {
	return camera_effects;
}

String WorldEnvironment::get_configuration_warning() const {
	String warning = Node::get_configuration_warning();

	if (!environment.is_valid()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("WorldEnvironment requires its \"Environment\" property to contain an Environment to have a visible effect.");
	}

	if (!is_inside_tree()) {
		return warning;
	}

	List<Node *> nodes;
	get_tree()->get_nodes_in_group("_world_environment_" + itos(get_viewport()->find_world_3d()->get_scenario().get_id()), &nodes);

	if (nodes.size() > 1) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("Only one WorldEnvironment is allowed per scene (or set of instanced scenes).");
	}

	return warning;
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
