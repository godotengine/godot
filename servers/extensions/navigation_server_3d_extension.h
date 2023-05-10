/**************************************************************************/
/*  navigation_server_3d_extension.h                                      */
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

#ifndef NAVIGATION_SERVER_3D_EXTENSION_H
#define NAVIGATION_SERVER_3D_EXTENSION_H

#include "core/extension/ext_wrappers.gen.inc"
#include "core/object/script_language.h"
#include "core/variant/native_ptr.h"
#include "servers/navigation_server_3d.h"

typedef NavigationUtilities::PathQueryParameters NavigationServer3DExtensionPathQueryParameters;
typedef NavigationUtilities::PathQueryResult NavigationServer3DExtensionPathQueryResult;

GDVIRTUAL_NATIVE_PTR(NavigationServer3DExtensionPathQueryParameters)
GDVIRTUAL_NATIVE_PTR(NavigationServer3DExtensionPathQueryResult)

class NavigationServer3DExtension : public NavigationServer3D {
	GDCLASS(NavigationServer3DExtension, NavigationServer3D)

protected:
	static void _bind_methods();

public:
	EXBIND0RC(TypedArray<RID>, get_maps)
	EXBIND0R(RID, map_create)
	EXBIND2(map_set_active, RID, bool)
	EXBIND1RC(bool, map_is_active, RID)
	EXBIND2(map_set_up, RID, Vector3)
	EXBIND1RC(Vector3, map_get_up, RID)
	EXBIND2(map_set_cell_size, RID, real_t)
	EXBIND1RC(real_t, map_get_cell_size, RID)
	EXBIND2(map_set_edge_connection_margin, RID, real_t)
	EXBIND1RC(real_t, map_get_edge_connection_margin, RID)
	EXBIND2(map_set_link_connection_radius, RID, real_t)
	EXBIND1RC(real_t, map_get_link_connection_radius, RID)
	EXBIND5RC(Vector<Vector3>, map_get_path, RID, Vector3, Vector3, bool, uint32_t)
	EXBIND4RC(Vector3, map_get_closest_point_to_segment, RID, const Vector3 &, const Vector3 &, bool)
	EXBIND2RC(Vector3, map_get_closest_point, RID, const Vector3 &)
	EXBIND2RC(Vector3, map_get_closest_point_normal, RID, const Vector3 &)
	EXBIND2RC(RID, map_get_closest_point_owner, RID, const Vector3 &)
	EXBIND1RC(TypedArray<RID>, map_get_links, RID)
	EXBIND1RC(TypedArray<RID>, map_get_regions, RID)
	EXBIND1RC(TypedArray<RID>, map_get_agents, RID)
	EXBIND1(map_force_update, RID)
	EXBIND0R(RID, region_create)
	EXBIND2(region_set_enter_cost, RID, real_t)
	EXBIND1RC(real_t, region_get_enter_cost, RID)
	EXBIND2(region_set_travel_cost, RID, real_t)
	EXBIND1RC(real_t, region_get_travel_cost, RID)
	EXBIND2(region_set_owner_id, RID, ObjectID)
	EXBIND1RC(ObjectID, region_get_owner_id, RID)
	EXBIND2RC(bool, region_owns_point, RID, const Vector3 &)
	EXBIND2(region_set_map, RID, RID)
	EXBIND1RC(RID, region_get_map, RID)
	EXBIND2(region_set_navigation_layers, RID, uint32_t)
	EXBIND1RC(uint32_t, region_get_navigation_layers, RID)
	EXBIND2(region_set_transform, RID, Transform3D)
	EXBIND2(region_set_navigation_mesh, RID, Ref<NavigationMesh>)
	EXBIND2(region_bake_navigation_mesh, Ref<NavigationMesh>, Node *)
	EXBIND1RC(int, region_get_connections_count, RID)
	EXBIND2RC(Vector3, region_get_connection_pathway_start, RID, int)
	EXBIND2RC(Vector3, region_get_connection_pathway_end, RID, int)
	EXBIND0R(RID, link_create)
	EXBIND2(link_set_map, RID, RID)
	EXBIND1RC(RID, link_get_map, RID)
	EXBIND2(link_set_bidirectional, RID, bool)
	EXBIND1RC(bool, link_is_bidirectional, RID)
	EXBIND2(link_set_navigation_layers, RID, uint32_t)
	EXBIND1RC(uint32_t, link_get_navigation_layers, RID)
	EXBIND2(link_set_start_position, RID, Vector3)
	EXBIND1RC(Vector3, link_get_start_position, RID)
	EXBIND2(link_set_end_position, RID, Vector3)
	EXBIND1RC(Vector3, link_get_end_position, RID)
	EXBIND2(link_set_enter_cost, RID, real_t)
	EXBIND1RC(real_t, link_get_enter_cost, RID)
	EXBIND2(link_set_travel_cost, RID, real_t)
	EXBIND1RC(real_t, link_get_travel_cost, RID)
	EXBIND2(link_set_owner_id, RID, ObjectID)
	EXBIND1RC(ObjectID, link_get_owner_id, RID)
	EXBIND0R(RID, agent_create)
	EXBIND2(agent_set_map, RID, RID)
	EXBIND1RC(RID, agent_get_map, RID)
	EXBIND2(agent_set_neighbor_distance, RID, real_t)
	EXBIND2(agent_set_max_neighbors, RID, int)
	EXBIND2(agent_set_time_horizon, RID, real_t)
	EXBIND2(agent_set_radius, RID, real_t)
	EXBIND2(agent_set_max_speed, RID, real_t)
	EXBIND2(agent_set_velocity, RID, Vector3)
	EXBIND2(agent_set_target_velocity, RID, Vector3)
	EXBIND2(agent_set_position, RID, Vector3)
	EXBIND2(agent_set_ignore_y, RID, bool)
	EXBIND1RC(bool, agent_is_map_changed, RID)
	EXBIND2(agent_set_callback, RID, Callable)
	EXBIND1(free, RID)
	EXBIND1(set_active, bool)
	EXBIND1(process, real_t)
	EXBIND1RC(int, get_process_info, ProcessInfo)

	GDVIRTUAL2C(_query_path_extension, GDExtensionConstPtr<const NavigationServer3DExtensionPathQueryParameters>, GDExtensionPtr<NavigationServer3DExtensionPathQueryResult>)

	NavigationUtilities::PathQueryResult _query_path(const NavigationUtilities::PathQueryParameters &p_parameters) const override {
		NavigationUtilities::PathQueryResult res;

		GDVIRTUAL_REQUIRED_CALL(_query_path_extension, &p_parameters, &res);

		return res;
	}
};

#endif // NAVIGATION_SERVER_3D_EXTENSION_H
