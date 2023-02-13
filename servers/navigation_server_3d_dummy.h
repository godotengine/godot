/**************************************************************************/
/*  navigation_server_3d_dummy.h                                          */
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

#ifndef NAVIGATION_SERVER_3D_DUMMY_H
#define NAVIGATION_SERVER_3D_DUMMY_H

#include "servers/navigation_server_3d.h"

class NavigationServer3DDummy : public NavigationServer3D {
	GDCLASS(NavigationServer3DDummy, NavigationServer3D);

public:
	TypedArray<RID> get_maps() const override { return TypedArray<RID>(); }
	RID map_create() override { return RID(); }
	void map_set_active(RID p_map, bool p_active) override {}
	bool map_is_active(RID p_map) const override { return false; }
	void map_set_up(RID p_map, Vector3 p_up) override {}
	Vector3 map_get_up(RID p_map) const override { return Vector3(); }
	void map_set_cell_size(RID p_map, real_t p_cell_size) override {}
	real_t map_get_cell_size(RID p_map) const override { return 0; }
	void map_set_edge_connection_margin(RID p_map, real_t p_connection_margin) override {}
	real_t map_get_edge_connection_margin(RID p_map) const override { return 0; }
	void map_set_link_connection_radius(RID p_map, real_t p_connection_radius) override {}
	real_t map_get_link_connection_radius(RID p_map) const override { return 0; }
	Vector<Vector3> map_get_path(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers) const override { return Vector<Vector3>(); }
	Vector3 map_get_closest_point_to_segment(RID p_map, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const override { return Vector3(); }
	Vector3 map_get_closest_point(RID p_map, const Vector3 &p_point) const override { return Vector3(); }
	Vector3 map_get_closest_point_normal(RID p_map, const Vector3 &p_point) const override { return Vector3(); }
	RID map_get_closest_point_owner(RID p_map, const Vector3 &p_point) const override { return RID(); }
	TypedArray<RID> map_get_links(RID p_map) const override { return TypedArray<RID>(); }
	TypedArray<RID> map_get_regions(RID p_map) const override { return TypedArray<RID>(); }
	TypedArray<RID> map_get_agents(RID p_map) const override { return TypedArray<RID>(); }
	void map_force_update(RID p_map) override {}
	RID region_create() override { return RID(); }
	void region_set_enter_cost(RID p_region, real_t p_enter_cost) override {}
	real_t region_get_enter_cost(RID p_region) const override { return 0; }
	void region_set_travel_cost(RID p_region, real_t p_travel_cost) override {}
	real_t region_get_travel_cost(RID p_region) const override { return 0; }
	void region_set_owner_id(RID p_region, ObjectID p_owner_id) override {}
	ObjectID region_get_owner_id(RID p_region) const override { return ObjectID(); }
	bool region_owns_point(RID p_region, const Vector3 &p_point) const override { return false; }
	void region_set_map(RID p_region, RID p_map) override {}
	RID region_get_map(RID p_region) const override { return RID(); }
	void region_set_navigation_layers(RID p_region, uint32_t p_navigation_layers) override {}
	uint32_t region_get_navigation_layers(RID p_region) const override { return 0; }
	void region_set_transform(RID p_region, Transform3D p_transform) override {}
	void region_set_navigation_mesh(RID p_region, Ref<NavigationMesh> p_navigation_mesh) override {}
	void region_bake_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node) override {}
	int region_get_connections_count(RID p_region) const override { return 0; }
	Vector3 region_get_connection_pathway_start(RID p_region, int p_connection_id) const override { return Vector3(); }
	Vector3 region_get_connection_pathway_end(RID p_region, int p_connection_id) const override { return Vector3(); }
	RID link_create() override { return RID(); }
	void link_set_map(RID p_link, RID p_map) override {}
	RID link_get_map(RID p_link) const override { return RID(); }
	void link_set_bidirectional(RID p_link, bool p_bidirectional) override {}
	bool link_is_bidirectional(RID p_link) const override { return false; }
	void link_set_navigation_layers(RID p_link, uint32_t p_navigation_layers) override {}
	uint32_t link_get_navigation_layers(RID p_link) const override { return 0; }
	void link_set_start_position(RID p_link, Vector3 p_position) override {}
	Vector3 link_get_start_position(RID p_link) const override { return Vector3(); }
	void link_set_end_position(RID p_link, Vector3 p_position) override {}
	Vector3 link_get_end_position(RID p_link) const override { return Vector3(); }
	void link_set_enter_cost(RID p_link, real_t p_enter_cost) override {}
	real_t link_get_enter_cost(RID p_link) const override { return 0; }
	void link_set_travel_cost(RID p_link, real_t p_travel_cost) override {}
	real_t link_get_travel_cost(RID p_link) const override { return 0; }
	void link_set_owner_id(RID p_link, ObjectID p_owner_id) override {}
	ObjectID link_get_owner_id(RID p_link) const override { return ObjectID(); }
	RID agent_create() override { return RID(); }
	void agent_set_map(RID p_agent, RID p_map) override {}
	RID agent_get_map(RID p_agent) const override { return RID(); }
	void agent_set_neighbor_distance(RID p_agent, real_t p_distance) override {}
	void agent_set_max_neighbors(RID p_agent, int p_count) override {}
	void agent_set_time_horizon(RID p_agent, real_t p_time) override {}
	void agent_set_radius(RID p_agent, real_t p_radius) override {}
	void agent_set_max_speed(RID p_agent, real_t p_max_speed) override {}
	void agent_set_velocity(RID p_agent, Vector3 p_velocity) override {}
	void agent_set_target_velocity(RID p_agent, Vector3 p_velocity) override {}
	void agent_set_position(RID p_agent, Vector3 p_position) override {}
	void agent_set_ignore_y(RID p_agent, bool p_ignore) override {}
	bool agent_is_map_changed(RID p_agent) const override { return false; }
	void agent_set_callback(RID p_agent, Callable p_callback) override {}
	void free(RID p_object) override {}
	void set_active(bool p_active) override {}
	void process(real_t delta_time) override {}
	NavigationUtilities::PathQueryResult _query_path(const NavigationUtilities::PathQueryParameters &p_parameters) const override { return NavigationUtilities::PathQueryResult(); }
	int get_process_info(ProcessInfo p_info) const override { return 0; }
	void set_debug_enabled(bool p_enabled) {}
	bool get_debug_enabled() const { return false; }
};

#endif // NAVIGATION_SERVER_3D_DUMMY_H
