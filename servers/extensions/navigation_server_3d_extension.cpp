/**************************************************************************/
/*  navigation_server_3d_extension.cpp                                    */
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

#include "navigation_server_3d_extension.h"

void NavigationServer3DExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_maps)
	GDVIRTUAL_BIND(_map_create)
	GDVIRTUAL_BIND(_map_set_active, "map", "active")
	GDVIRTUAL_BIND(_map_is_active, "map")
	GDVIRTUAL_BIND(_map_set_up, "map", "up")
	GDVIRTUAL_BIND(_map_get_up, "map")
	GDVIRTUAL_BIND(_map_set_cell_size, "map", "cell_size")
	GDVIRTUAL_BIND(_map_get_cell_size, "map")
	GDVIRTUAL_BIND(_map_set_edge_connection_margin, "map", "margin")
	GDVIRTUAL_BIND(_map_get_edge_connection_margin, "map")
	GDVIRTUAL_BIND(_map_set_link_connection_radius, "map", "radius")
	GDVIRTUAL_BIND(_map_get_link_connection_radius, "map")
	GDVIRTUAL_BIND(_map_get_path, "map", "origin", "destination", "optimize", "navigation_layers")
	GDVIRTUAL_BIND(_map_get_closest_point_to_segment, "map", "start", "end", "use_collision")
	GDVIRTUAL_BIND(_map_get_closest_point, "map", "to_point")
	GDVIRTUAL_BIND(_map_get_closest_point_normal, "map", "to_point")
	GDVIRTUAL_BIND(_map_get_closest_point_owner, "map", "to_point")
	GDVIRTUAL_BIND(_map_get_links, "map")
	GDVIRTUAL_BIND(_map_get_regions, "map")
	GDVIRTUAL_BIND(_map_get_agents, "map")
	GDVIRTUAL_BIND(_map_force_update, "map")
	GDVIRTUAL_BIND(_query_path_extension, "parameters", "result")

	GDVIRTUAL_BIND(_region_create)
	GDVIRTUAL_BIND(_region_set_enter_cost, "region", "enter_cost")
	GDVIRTUAL_BIND(_region_get_enter_cost, "region")
	GDVIRTUAL_BIND(_region_set_travel_cost, "region", "travel_cost")
	GDVIRTUAL_BIND(_region_get_travel_cost, "region")
	GDVIRTUAL_BIND(_region_set_owner_id, "region", "owner_id")
	GDVIRTUAL_BIND(_region_get_owner_id, "region")
	GDVIRTUAL_BIND(_region_owns_point, "region", "point")
	GDVIRTUAL_BIND(_region_set_map, "region", "map")
	GDVIRTUAL_BIND(_region_get_map, "region")
	GDVIRTUAL_BIND(_region_set_navigation_layers, "region", "navigation_layers")
	GDVIRTUAL_BIND(_region_get_navigation_layers, "region")
	GDVIRTUAL_BIND(_region_set_transform, "region", "transform")
	GDVIRTUAL_BIND(_region_set_navigation_mesh, "region", "navigation_mesh")
	GDVIRTUAL_BIND(_region_bake_navigation_mesh, "navigation_mesh", "root_node")
	GDVIRTUAL_BIND(_region_get_connections_count, "region")
	GDVIRTUAL_BIND(_region_get_connection_pathway_start, "region", "connection")
	GDVIRTUAL_BIND(_region_get_connection_pathway_end, "region", "connection")

	GDVIRTUAL_BIND(_link_create)
	GDVIRTUAL_BIND(_link_set_map, "link", "map")
	GDVIRTUAL_BIND(_link_get_map, "link")
	GDVIRTUAL_BIND(_link_set_bidirectional, "link", "bidirectional")
	GDVIRTUAL_BIND(_link_is_bidirectional, "link")
	GDVIRTUAL_BIND(_link_set_navigation_layers, "link", "navigation_layers")
	GDVIRTUAL_BIND(_link_get_navigation_layers, "link")
	GDVIRTUAL_BIND(_link_set_start_position, "link", "position")
	GDVIRTUAL_BIND(_link_get_start_position, "link")
	GDVIRTUAL_BIND(_link_set_end_position, "link", "position")
	GDVIRTUAL_BIND(_link_get_end_position, "link")
	GDVIRTUAL_BIND(_link_set_enter_cost, "link", "enter_cost")
	GDVIRTUAL_BIND(_link_get_enter_cost, "link")
	GDVIRTUAL_BIND(_link_set_travel_cost, "link", "travel_cost")
	GDVIRTUAL_BIND(_link_get_travel_cost, "link")
	GDVIRTUAL_BIND(_link_set_owner_id, "link", "owner_id")
	GDVIRTUAL_BIND(_link_get_owner_id, "link")

	GDVIRTUAL_BIND(_agent_create)
	GDVIRTUAL_BIND(_agent_set_map, "agent", "map")
	GDVIRTUAL_BIND(_agent_get_map, "agent")
	GDVIRTUAL_BIND(_agent_set_neighbor_distance, "agent", "distance")
	GDVIRTUAL_BIND(_agent_set_max_neighbors, "agent", "count")
	GDVIRTUAL_BIND(_agent_set_time_horizon, "agent", "time")
	GDVIRTUAL_BIND(_agent_set_radius, "agent", "radius")
	GDVIRTUAL_BIND(_agent_set_max_speed, "agent", "max_speed")
	GDVIRTUAL_BIND(_agent_set_velocity, "agent", "velocity")
	GDVIRTUAL_BIND(_agent_set_target_velocity, "agent", "target_velocity")
	GDVIRTUAL_BIND(_agent_set_position, "agent", "position")
	GDVIRTUAL_BIND(_agent_set_ignore_y, "agent", "ignore_y")
	GDVIRTUAL_BIND(_agent_is_map_changed, "agent")
	GDVIRTUAL_BIND(_agent_set_callback, "agent", "callback")

	GDVIRTUAL_BIND(_free, "rid")
	GDVIRTUAL_BIND(_set_active, "active")
	GDVIRTUAL_BIND(_process, "delta_time")
	GDVIRTUAL_BIND(_get_process_info, "info")
}
