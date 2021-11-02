/*************************************************************************/
/*  world_3d.cpp                                                         */
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

#include "world_3d.h"

#include "core/math/camera_matrix.h"
#include "core/math/octree.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/visible_on_screen_notifier_3d.h"
#include "scene/scene_string_names.h"
#include "servers/navigation_server_3d.h"

void World3D::_register_camera(Camera3D *p_camera) {
#ifndef _3D_DISABLED
	cameras.insert(p_camera);
#endif
}

void World3D::_remove_camera(Camera3D *p_camera) {
#ifndef _3D_DISABLED
	cameras.erase(p_camera);
#endif
}

RID World3D::get_space() const {
	return space;
}

RID World3D::get_navigation_map() const {
	return navigation_map;
}

RID World3D::get_scenario() const {
	return scenario;
}

void World3D::set_environment(const Ref<Environment> &p_environment) {
	if (environment == p_environment) {
		return;
	}

	environment = p_environment;
	if (environment.is_valid()) {
		RS::get_singleton()->scenario_set_environment(scenario, environment->get_rid());
	} else {
		RS::get_singleton()->scenario_set_environment(scenario, RID());
	}

	emit_changed();
}

Ref<Environment> World3D::get_environment() const {
	return environment;
}

void World3D::set_fallback_environment(const Ref<Environment> &p_environment) {
	if (fallback_environment == p_environment) {
		return;
	}

	fallback_environment = p_environment;
	if (fallback_environment.is_valid()) {
		RS::get_singleton()->scenario_set_fallback_environment(scenario, p_environment->get_rid());
	} else {
		RS::get_singleton()->scenario_set_fallback_environment(scenario, RID());
	}

	emit_changed();
}

Ref<Environment> World3D::get_fallback_environment() const {
	return fallback_environment;
}

void World3D::set_camera_effects(const Ref<CameraEffects> &p_camera_effects) {
	camera_effects = p_camera_effects;
	if (camera_effects.is_valid()) {
		RS::get_singleton()->scenario_set_camera_effects(scenario, camera_effects->get_rid());
	} else {
		RS::get_singleton()->scenario_set_camera_effects(scenario, RID());
	}
}

Ref<CameraEffects> World3D::get_camera_effects() const {
	return camera_effects;
}

PhysicsDirectSpaceState3D *World3D::get_direct_space_state() {
	return PhysicsServer3D::get_singleton()->space_get_direct_state(space);
}

void World3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_space"), &World3D::get_space);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &World3D::get_navigation_map);
	ClassDB::bind_method(D_METHOD("get_scenario"), &World3D::get_scenario);
	ClassDB::bind_method(D_METHOD("set_environment", "env"), &World3D::set_environment);
	ClassDB::bind_method(D_METHOD("get_environment"), &World3D::get_environment);
	ClassDB::bind_method(D_METHOD("set_fallback_environment", "env"), &World3D::set_fallback_environment);
	ClassDB::bind_method(D_METHOD("get_fallback_environment"), &World3D::get_fallback_environment);
	ClassDB::bind_method(D_METHOD("set_camera_effects", "effects"), &World3D::set_camera_effects);
	ClassDB::bind_method(D_METHOD("get_camera_effects"), &World3D::get_camera_effects);
	ClassDB::bind_method(D_METHOD("get_direct_space_state"), &World3D::get_direct_space_state);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_environment", "get_environment");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fallback_environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_fallback_environment", "get_fallback_environment");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "camera_effects", PROPERTY_HINT_RESOURCE_TYPE, "CameraEffects"), "set_camera_effects", "get_camera_effects");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_space");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "navigation_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_navigation_map");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "scenario", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_scenario");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "direct_space_state", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsDirectSpaceState3D", PROPERTY_USAGE_NONE), "", "get_direct_space_state");
}

World3D::World3D() {
	space = PhysicsServer3D::get_singleton()->space_create();
	scenario = RenderingServer::get_singleton()->scenario_create();

	PhysicsServer3D::get_singleton()->space_set_active(space, true);
	PhysicsServer3D::get_singleton()->area_set_param(space, PhysicsServer3D::AREA_PARAM_GRAVITY, GLOBAL_DEF("physics/3d/default_gravity", 9.8));
	PhysicsServer3D::get_singleton()->area_set_param(space, PhysicsServer3D::AREA_PARAM_GRAVITY_VECTOR, GLOBAL_DEF("physics/3d/default_gravity_vector", Vector3(0, -1, 0)));
	PhysicsServer3D::get_singleton()->area_set_param(space, PhysicsServer3D::AREA_PARAM_LINEAR_DAMP, GLOBAL_DEF("physics/3d/default_linear_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/3d/default_linear_damp", PropertyInfo(Variant::FLOAT, "physics/3d/default_linear_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"));
	PhysicsServer3D::get_singleton()->area_set_param(space, PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP, GLOBAL_DEF("physics/3d/default_angular_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/3d/default_angular_damp", PropertyInfo(Variant::FLOAT, "physics/3d/default_angular_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"));

	navigation_map = NavigationServer3D::get_singleton()->map_create();
	NavigationServer3D::get_singleton()->map_set_active(navigation_map, true);
	NavigationServer3D::get_singleton()->map_set_cell_size(navigation_map, GLOBAL_DEF("navigation/3d/default_cell_size", 0.3));
	NavigationServer3D::get_singleton()->map_set_edge_connection_margin(navigation_map, GLOBAL_DEF("navigation/3d/default_edge_connection_margin", 0.3));
}

World3D::~World3D() {
	PhysicsServer3D::get_singleton()->free(space);
	RenderingServer::get_singleton()->free(scenario);
	NavigationServer3D::get_singleton()->free(navigation_map);
}
