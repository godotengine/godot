/*************************************************************************/
/*  gd_navigation_server.h                                               */
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

#ifndef GD_NAVIGATION_SERVER_H
#define GD_NAVIGATION_SERVER_H

#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"
#include "servers/navigation_server_3d.h"

#include "nav_map.h"
#include "nav_region.h"
#include "rvo_agent.h"

/**
	@author AndreaCatania
*/

/// The commands are functions executed during the `sync` phase.

#define MERGE_INTERNAL(A, B) A##B
#define MERGE(A, B) MERGE_INTERNAL(A, B)

#define COMMAND_1(F_NAME, T_0, D_0)     \
	virtual void F_NAME(T_0 D_0) const; \
	void MERGE(_cmd_, F_NAME)(T_0 D_0)

#define COMMAND_2(F_NAME, T_0, D_0, T_1, D_1)    \
	virtual void F_NAME(T_0 D_0, T_1 D_1) const; \
	void MERGE(_cmd_, F_NAME)(T_0 D_0, T_1 D_1)

#define COMMAND_4_DEF(F_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, D_3_DEF) \
	virtual void F_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3 = D_3_DEF) const;   \
	void MERGE(_cmd_, F_NAME)(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3)

class GdNavigationServer;

struct SetCommand {
	virtual ~SetCommand() {}
	virtual void exec(GdNavigationServer *server) = 0;
};

class GdNavigationServer : public NavigationServer3D {
	Mutex commands_mutex;
	/// Mutex used to make any operation threadsafe.
	Mutex operations_mutex;

	std::vector<SetCommand *> commands;

	mutable RID_PtrOwner<NavMap> map_owner;
	mutable RID_PtrOwner<NavRegion> region_owner;
	mutable RID_PtrOwner<RvoAgent> agent_owner;

	bool active = true;
	LocalVector<NavMap *> active_maps;
	LocalVector<uint32_t> active_maps_update_id;

public:
	GdNavigationServer();
	virtual ~GdNavigationServer();

	void add_command(SetCommand *command) const;

	virtual RID map_create() const;
	COMMAND_2(map_set_active, RID, p_map, bool, p_active);
	virtual bool map_is_active(RID p_map) const;

	COMMAND_2(map_set_up, RID, p_map, Vector3, p_up);
	virtual Vector3 map_get_up(RID p_map) const;

	COMMAND_2(map_set_cell_size, RID, p_map, real_t, p_cell_size);
	virtual real_t map_get_cell_size(RID p_map) const;

	COMMAND_2(map_set_edge_connection_margin, RID, p_map, real_t, p_connection_margin);
	virtual real_t map_get_edge_connection_margin(RID p_map) const;

	virtual Vector<Vector3> map_get_path(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_layers = 1) const;

	virtual Vector3 map_get_closest_point_to_segment(RID p_map, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision = false) const;
	virtual Vector3 map_get_closest_point(RID p_map, const Vector3 &p_point) const;
	virtual Vector3 map_get_closest_point_normal(RID p_map, const Vector3 &p_point) const;
	virtual RID map_get_closest_point_owner(RID p_map, const Vector3 &p_point) const;

	virtual RID region_create() const;
	COMMAND_2(region_set_map, RID, p_region, RID, p_map);
	COMMAND_2(region_set_layers, RID, p_region, uint32_t, p_layers);
	virtual uint32_t region_get_layers(RID p_region) const;
	COMMAND_2(region_set_transform, RID, p_region, Transform, p_transform);
	COMMAND_2(region_set_navmesh, RID, p_region, Ref<NavigationMesh>, p_nav_mesh);
	virtual void region_bake_navmesh(Ref<NavigationMesh> r_mesh, Node *p_node) const;
	virtual int region_get_connections_count(RID p_region) const;
	virtual Vector3 region_get_connection_pathway_start(RID p_region, int p_connection_id) const;
	virtual Vector3 region_get_connection_pathway_end(RID p_region, int p_connection_id) const;

	virtual RID agent_create() const;
	COMMAND_2(agent_set_map, RID, p_agent, RID, p_map);
	COMMAND_2(agent_set_neighbor_dist, RID, p_agent, real_t, p_dist);
	COMMAND_2(agent_set_max_neighbors, RID, p_agent, int, p_count);
	COMMAND_2(agent_set_time_horizon, RID, p_agent, real_t, p_time);
	COMMAND_2(agent_set_radius, RID, p_agent, real_t, p_radius);
	COMMAND_2(agent_set_max_speed, RID, p_agent, real_t, p_max_speed);
	COMMAND_2(agent_set_velocity, RID, p_agent, Vector3, p_velocity);
	COMMAND_2(agent_set_target_velocity, RID, p_agent, Vector3, p_velocity);
	COMMAND_2(agent_set_position, RID, p_agent, Vector3, p_position);
	COMMAND_2(agent_set_ignore_y, RID, p_agent, bool, p_ignore);
	virtual bool agent_is_map_changed(RID p_agent) const;
	COMMAND_4_DEF(agent_set_callback, RID, p_agent, Object *, p_receiver, StringName, p_method, Variant, p_udata, Variant());

	COMMAND_1(free, RID, p_object);

	virtual void set_active(bool p_active) const;

	void flush_queries();
	virtual void process(real_t p_delta_time);
};

#undef COMMAND_1
#undef COMMAND_2
#undef COMMAND_4_DEF

#endif // GD_NAVIGATION_SERVER_H
