/**************************************************************************/
/*  navigation_server_dummy.h                                             */
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

#ifndef NAVIGATION_SERVER_DUMMY_H
#define NAVIGATION_SERVER_DUMMY_H

#include "servers/navigation_server.h"

class NavigationServerDummy : public NavigationServer {
	GDCLASS(NavigationServerDummy, NavigationServer);

public:
	Array get_maps() const { return Array(); }
	RID map_create() const { return RID(); }
	void map_set_active(RID p_map, bool p_active) const {}
	bool map_is_active(RID p_map) const { return false; }
	void map_set_up(RID p_map, Vector3 p_up) const {}
	Vector3 map_get_up(RID p_map) const { return Vector3(); }
	void map_set_cell_size(RID p_map, real_t p_cell_size) const {}
	real_t map_get_cell_size(RID p_map) const { return 0; }
	void map_set_cell_height(RID p_map, real_t p_cell_height) const {}
	real_t map_get_cell_height(RID p_map) const { return 0; }
	void map_set_edge_connection_margin(RID p_map, real_t p_connection_margin) const {}
	real_t map_get_edge_connection_margin(RID p_map) const { return 0; }
	Vector<Vector3> map_get_path(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers) const { return Vector<Vector3>(); }
	Vector3 map_get_closest_point_to_segment(RID p_map, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const { return Vector3(); }
	Vector3 map_get_closest_point(RID p_map, const Vector3 &p_point) const { return Vector3(); }
	Vector3 map_get_closest_point_normal(RID p_map, const Vector3 &p_point) const { return Vector3(); }
	RID map_get_closest_point_owner(RID p_map, const Vector3 &p_point) const { return RID(); }
	Array map_get_regions(RID p_map) const { return Array(); }
	Array map_get_agents(RID p_map) const { return Array(); }
	void map_force_update(RID p_map) {}

	RID region_create() const { return RID(); }
	void region_set_enter_cost(RID p_region, real_t p_enter_cost) const {}
	real_t region_get_enter_cost(RID p_region) const { return 0; }
	void region_set_travel_cost(RID p_region, real_t p_travel_cost) const {}
	real_t region_get_travel_cost(RID p_region) const { return 0; }
	bool region_owns_point(RID p_region, const Vector3 &p_point) const { return false; }
	void region_set_map(RID p_region, RID p_map) const {}
	RID region_get_map(RID p_region) const { return RID(); }
	void region_set_navigation_layers(RID p_region, uint32_t p_navigation_layers) const {}
	uint32_t region_get_navigation_layers(RID p_region) const { return 0; }
	void region_set_transform(RID p_region, Transform p_transform) const {}
	void region_set_navmesh(RID p_region, Ref<NavigationMesh> p_nav_mesh) const {}
	void region_bake_navmesh(Ref<NavigationMesh> r_mesh, Node *p_node) const {}
	int region_get_connections_count(RID p_region) const { return 0; }
	Vector3 region_get_connection_pathway_start(RID p_region, int p_connection_id) const { return Vector3(); }
	Vector3 region_get_connection_pathway_end(RID p_region, int p_connection_id) const { return Vector3(); }

	RID agent_create() const { return RID(); }
	void agent_set_map(RID p_agent, RID p_map) const {}
	RID agent_get_map(RID p_agent) const { return RID(); }
	void agent_set_neighbor_dist(RID p_agent, real_t p_dist) const {}
	void agent_set_max_neighbors(RID p_agent, int p_count) const {}
	void agent_set_time_horizon(RID p_agent, real_t p_time) const {}
	void agent_set_radius(RID p_agent, real_t p_radius) const {}
	void agent_set_max_speed(RID p_agent, real_t p_max_speed) const {}
	void agent_set_velocity(RID p_agent, Vector3 p_velocity) const {}
	void agent_set_target_velocity(RID p_agent, Vector3 p_velocity) const {}
	void agent_set_position(RID p_agent, Vector3 p_position) const {}
	void agent_set_ignore_y(RID p_agent, bool p_ignore) const {}
	bool agent_is_map_changed(RID p_agent) const { return false; }
	void agent_set_callback(RID p_agent, ObjectID p_object_id, StringName p_method, Variant p_udata = Variant()) const {}
	void free(RID p_object) const {}
	void set_active(bool p_active) const {}
	void process(real_t delta_time) {}
};

#endif // NAVIGATION_SERVER_DUMMY_H
