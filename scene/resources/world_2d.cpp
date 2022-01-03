/*************************************************************************/
/*  world_2d.cpp                                                         */
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

#include "world_2d.h"

#include "core/config/project_settings.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/visible_on_screen_notifier_2d.h"
#include "scene/main/window.h"
#include "servers/navigation_server_2d.h"
#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

RID World2D::get_canvas() const {
	return canvas;
}

RID World2D::get_space() const {
	return space;
}

RID World2D::get_navigation_map() const {
	return navigation_map;
}

void World2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_canvas"), &World2D::get_canvas);
	ClassDB::bind_method(D_METHOD("get_space"), &World2D::get_space);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &World2D::get_navigation_map);

	ClassDB::bind_method(D_METHOD("get_direct_space_state"), &World2D::get_direct_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "canvas", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_canvas");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_space");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "navigation_map", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_navigation_map");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "direct_space_state", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsDirectSpaceState2D", PROPERTY_USAGE_NONE), "", "get_direct_space_state");
}

PhysicsDirectSpaceState2D *World2D::get_direct_space_state() {
	return PhysicsServer2D::get_singleton()->space_get_direct_state(space);
}

World2D::World2D() {
	canvas = RenderingServer::get_singleton()->canvas_create();

	// Create and configure space2D to be more friendly with pixels than meters
	space = PhysicsServer2D::get_singleton()->space_create();
	PhysicsServer2D::get_singleton()->space_set_active(space, true);
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_GRAVITY, GLOBAL_DEF("physics/2d/default_gravity", 980.0));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_GRAVITY_VECTOR, GLOBAL_DEF("physics/2d/default_gravity_vector", Vector2(0, 1)));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_LINEAR_DAMP, GLOBAL_DEF("physics/2d/default_linear_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/default_linear_damp", PropertyInfo(Variant::FLOAT, "physics/2d/default_linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_ANGULAR_DAMP, GLOBAL_DEF("physics/2d/default_angular_damp", 1.0));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/default_angular_damp", PropertyInfo(Variant::FLOAT, "physics/2d/default_angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));

	// Create and configure the navigation_map to be more friendly with pixels than meters.
	navigation_map = NavigationServer2D::get_singleton()->map_create();
	NavigationServer2D::get_singleton()->map_set_active(navigation_map, true);
	NavigationServer2D::get_singleton()->map_set_cell_size(navigation_map, GLOBAL_DEF("navigation/2d/default_cell_size", 10));
	NavigationServer2D::get_singleton()->map_set_edge_connection_margin(navigation_map, GLOBAL_DEF("navigation/2d/default_edge_connection_margin", 5));
}

World2D::~World2D() {
	RenderingServer::get_singleton()->free(canvas);
	PhysicsServer2D::get_singleton()->free(space);
	NavigationServer2D::get_singleton()->free(navigation_map);
}
