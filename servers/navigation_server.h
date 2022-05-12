/*************************************************************************/
/*  navigation_server.h                                                  */
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

#ifndef NAVIGATION_SERVER_H
#define NAVIGATION_SERVER_H

#include "core/object.h"
#include "core/rid.h"
#include "scene/3d/navigation_mesh_instance.h"

/// This server uses the concept of internal mutability.
/// All the constant functions can be called in multithread because internally
/// the server takes care to schedule the functions access.
///
/// Note: All the `set` functions are commands executed during the `sync` phase,
/// don't expect that a change is immediately propagated.
class NavigationServer : public Object {
	GDCLASS(NavigationServer, Object);

	static NavigationServer *singleton;

protected:
	static void _bind_methods();

public:
	/// Thread safe, can be used across many threads.
	static const NavigationServer *get_singleton();

	/// MUST be used in single thread!
	static NavigationServer *get_singleton_mut();

	/// Create a new map.
	virtual RID map_create() const = 0;

	/// Set map active.
	virtual void map_set_active(RID p_map, bool p_active) const = 0;

	/// Returns true if the map is active.
	virtual bool map_is_active(RID p_map) const = 0;

	/// Set the map UP direction.
	virtual void map_set_up(RID p_map, Vector3 p_up) const = 0;

	/// Returns the map UP direction.
	virtual Vector3 map_get_up(RID p_map) const = 0;

	/// Set the map cell size used to weld the navigation mesh polygons.
	virtual void map_set_cell_size(RID p_map, real_t p_cell_size) const = 0;

	/// Returns the map cell size.
	virtual real_t map_get_cell_size(RID p_map) const = 0;

	/// Set the map cell height used to weld the navigation mesh polygons.
	virtual void map_set_cell_height(RID p_map, real_t p_cell_height) const = 0;

	/// Returns the map cell height.
	virtual real_t map_get_cell_height(RID p_map) const = 0;

	/// Set the map edge connection margin used to weld the compatible region edges.
	virtual void map_set_edge_connection_margin(RID p_map, real_t p_connection_margin) const = 0;

	/// Returns the edge connection margin of this map.
	virtual real_t map_get_edge_connection_margin(RID p_map) const = 0;

	/// Returns the navigation path to reach the destination from the origin.
	virtual Vector<Vector3> map_get_path(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize) const = 0;

	virtual Vector3 map_get_closest_point_to_segment(RID p_map, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision = false) const = 0;
	virtual Vector3 map_get_closest_point(RID p_map, const Vector3 &p_point) const = 0;
	virtual Vector3 map_get_closest_point_normal(RID p_map, const Vector3 &p_point) const = 0;
	virtual RID map_get_closest_point_owner(RID p_map, const Vector3 &p_point) const = 0;

	virtual Array map_get_regions(RID p_map) const = 0;
	virtual Array map_get_agents(RID p_map) const = 0;

	/// Creates a new region.
	virtual RID region_create() const = 0;

	/// Set the map of this region.
	virtual void region_set_map(RID p_region, RID p_map) const = 0;
	virtual RID region_get_map(RID p_region) const = 0;

	/// Set the global transformation of this region.
	virtual void region_set_transform(RID p_region, Transform p_transform) const = 0;

	/// Set the navigation mesh of this region.
	virtual void region_set_navmesh(RID p_region, Ref<NavigationMesh> p_nav_mesh) const = 0;

	/// Bake the navigation mesh
	virtual void region_bake_navmesh(Ref<NavigationMesh> r_mesh, Node *p_node) const = 0;

	/// Creates the agent.
	virtual RID agent_create() const = 0;

	/// Put the agent in the map.
	virtual void agent_set_map(RID p_agent, RID p_map) const = 0;
	virtual RID agent_get_map(RID p_agent) const = 0;

	/// The maximum distance (center point to
	/// center point) to other agents this agent
	/// takes into account in the navigation. The
	/// larger this number, the longer the running
	/// time of the simulation. If the number is too
	/// low, the simulation will not be safe.
	/// Must be non-negative.
	virtual void agent_set_neighbor_dist(RID p_agent, real_t p_dist) const = 0;

	/// The maximum number of other agents this
	/// agent takes into account in the navigation.
	/// The larger this number, the longer the
	/// running time of the simulation. If the
	/// number is too low, the simulation will not
	/// be safe.
	virtual void agent_set_max_neighbors(RID p_agent, int p_count) const = 0;

	/// The minimal amount of time for which this
	/// agent's velocities that are computed by the
	/// simulation are safe with respect to other
	/// agents. The larger this number, the sooner
	/// this agent will respond to the presence of
	/// other agents, but the less freedom this
	/// agent has in choosing its velocities.
	/// Must be positive.
	virtual void agent_set_time_horizon(RID p_agent, real_t p_time) const = 0;

	/// The radius of this agent.
	/// Must be non-negative.
	virtual void agent_set_radius(RID p_agent, real_t p_radius) const = 0;

	/// The maximum speed of this agent.
	/// Must be non-negative.
	virtual void agent_set_max_speed(RID p_agent, real_t p_max_speed) const = 0;

	/// Current velocity of the agent
	virtual void agent_set_velocity(RID p_agent, Vector3 p_velocity) const = 0;

	/// The new target velocity.
	virtual void agent_set_target_velocity(RID p_agent, Vector3 p_velocity) const = 0;

	/// Position of the agent in world space.
	virtual void agent_set_position(RID p_agent, Vector3 p_position) const = 0;

	/// Agent ignore the Y axis and avoid collisions by moving only on the horizontal plane
	virtual void agent_set_ignore_y(RID p_agent, bool p_ignore) const = 0;

	/// Returns true if the map got changed the previous frame.
	virtual bool agent_is_map_changed(RID p_agent) const = 0;

	/// Callback called at the end of the RVO process
	virtual void agent_set_callback(RID p_agent, Object *p_receiver, StringName p_method, Variant p_udata = Variant()) const = 0;

	/// Destroy the `RID`
	virtual void free(RID p_object) const = 0;

	/// Control activation of this server.
	virtual void set_active(bool p_active) const = 0;

	/// Process the collision avoidance agents.
	/// The result of this process is needed by the physics server,
	/// so this must be called in the main thread.
	/// Note: This function is not thread safe.
	virtual void process(real_t delta_time) = 0;

	NavigationServer();
	virtual ~NavigationServer();
};

typedef NavigationServer *(*NavigationServerCallback)();

/// Manager used for the server singleton registration
class NavigationServerManager {
	static NavigationServerCallback create_callback;

public:
	static void set_default_server(NavigationServerCallback p_callback);
	static NavigationServer *new_default_server();
};

#endif
