/*************************************************************************/
/*  navigation_server_3d.cpp                                             */
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

/**
	@author AndreaCatania
*/

#include "navigation_server_3d.h"

NavigationServer3D *NavigationServer3D::singleton = nullptr;

void NavigationServer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("map_create"), &NavigationServer3D::map_create);
	ClassDB::bind_method(D_METHOD("map_set_active", "map", "active"), &NavigationServer3D::map_set_active);
	ClassDB::bind_method(D_METHOD("map_is_active", "nap"), &NavigationServer3D::map_is_active);
	ClassDB::bind_method(D_METHOD("map_set_up", "map", "up"), &NavigationServer3D::map_set_up);
	ClassDB::bind_method(D_METHOD("map_get_up", "map"), &NavigationServer3D::map_get_up);
	ClassDB::bind_method(D_METHOD("map_set_cell_size", "map", "cell_size"), &NavigationServer3D::map_set_cell_size);
	ClassDB::bind_method(D_METHOD("map_get_cell_size", "map"), &NavigationServer3D::map_get_cell_size);
	ClassDB::bind_method(D_METHOD("map_set_edge_connection_margin", "map", "margin"), &NavigationServer3D::map_set_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("map_get_edge_connection_margin", "map"), &NavigationServer3D::map_get_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("map_get_path", "map", "origin", "destination", "optimize", "layers"), &NavigationServer3D::map_get_path, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("map_get_closest_point_to_segment", "map", "start", "end", "use_collision"), &NavigationServer3D::map_get_closest_point_to_segment, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("map_get_closest_point", "map", "to_point"), &NavigationServer3D::map_get_closest_point);
	ClassDB::bind_method(D_METHOD("map_get_closest_point_normal", "map", "to_point"), &NavigationServer3D::map_get_closest_point_normal);
	ClassDB::bind_method(D_METHOD("map_get_closest_point_owner", "map", "to_point"), &NavigationServer3D::map_get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("region_create"), &NavigationServer3D::region_create);
	ClassDB::bind_method(D_METHOD("region_set_map", "region", "map"), &NavigationServer3D::region_set_map);
	ClassDB::bind_method(D_METHOD("region_set_layers", "region", "layers"), &NavigationServer3D::region_set_layers);
	ClassDB::bind_method(D_METHOD("region_get_layers", "region"), &NavigationServer3D::region_get_layers);
	ClassDB::bind_method(D_METHOD("region_set_transform", "region", "transform"), &NavigationServer3D::region_set_transform);
	ClassDB::bind_method(D_METHOD("region_set_navmesh", "region", "nav_mesh"), &NavigationServer3D::region_set_navmesh);
	ClassDB::bind_method(D_METHOD("region_bake_navmesh", "mesh", "node"), &NavigationServer3D::region_bake_navmesh);
	ClassDB::bind_method(D_METHOD("region_get_connections_count", "region"), &NavigationServer3D::region_get_connections_count);
	ClassDB::bind_method(D_METHOD("region_get_connection_pathway_start", "region", "connection"), &NavigationServer3D::region_get_connection_pathway_start);
	ClassDB::bind_method(D_METHOD("region_get_connection_pathway_end", "region", "connection"), &NavigationServer3D::region_get_connection_pathway_end);

	ClassDB::bind_method(D_METHOD("agent_create"), &NavigationServer3D::agent_create);
	ClassDB::bind_method(D_METHOD("agent_set_map", "agent", "map"), &NavigationServer3D::agent_set_map);
	ClassDB::bind_method(D_METHOD("agent_set_neighbor_dist", "agent", "dist"), &NavigationServer3D::agent_set_neighbor_dist);
	ClassDB::bind_method(D_METHOD("agent_set_max_neighbors", "agent", "count"), &NavigationServer3D::agent_set_max_neighbors);
	ClassDB::bind_method(D_METHOD("agent_set_time_horizon", "agent", "time"), &NavigationServer3D::agent_set_time_horizon);
	ClassDB::bind_method(D_METHOD("agent_set_radius", "agent", "radius"), &NavigationServer3D::agent_set_radius);
	ClassDB::bind_method(D_METHOD("agent_set_max_speed", "agent", "max_speed"), &NavigationServer3D::agent_set_max_speed);
	ClassDB::bind_method(D_METHOD("agent_set_velocity", "agent", "velocity"), &NavigationServer3D::agent_set_velocity);
	ClassDB::bind_method(D_METHOD("agent_set_target_velocity", "agent", "target_velocity"), &NavigationServer3D::agent_set_target_velocity);
	ClassDB::bind_method(D_METHOD("agent_set_position", "agent", "position"), &NavigationServer3D::agent_set_position);
	ClassDB::bind_method(D_METHOD("agent_is_map_changed", "agent"), &NavigationServer3D::agent_is_map_changed);
	ClassDB::bind_method(D_METHOD("agent_set_callback", "agent", "receiver", "method", "userdata"), &NavigationServer3D::agent_set_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("free", "object"), &NavigationServer3D::free);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &NavigationServer3D::set_active);
	ClassDB::bind_method(D_METHOD("process", "delta_time"), &NavigationServer3D::process);

	ADD_SIGNAL(MethodInfo("map_changed", PropertyInfo(Variant::RID, "map")));
}

const NavigationServer3D *NavigationServer3D::get_singleton() {
	return singleton;
}

NavigationServer3D *NavigationServer3D::get_singleton_mut() {
	return singleton;
}

NavigationServer3D::NavigationServer3D() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

NavigationServer3D::~NavigationServer3D() {
	singleton = nullptr;
}

NavigationServer3DCallback NavigationServer3DManager::create_callback = nullptr;

void NavigationServer3DManager::set_default_server(NavigationServer3DCallback p_callback) {
	create_callback = p_callback;
}

NavigationServer3D *NavigationServer3DManager::new_default_server() {
	ERR_FAIL_COND_V(create_callback == nullptr, nullptr);
	return create_callback();
}
