/**************************************************************************/
/*  navigation_server_3d.h                                                */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/os/rw_lock.h"
#include "core/templates/rid_owner.h"
#include "core/variant/type_info.h"

#ifdef DEBUG_ENABLED
class NavigationServer3DDebug;
#endif // DEBUG_ENABLED

class Node;
class NavigationMesh;
class NavigationMeshSourceGeometryData3D;
class NavigationPathQueryParameters3D;
class NavigationPathQueryResult3D;

struct NavMeshGeometryParser3D {
	RID self;
	Callable callback;
};

class NavigationServer3D : public Object {
	GDCLASS(NavigationServer3D, Object);

	static NavigationServer3D *singleton;

protected:
	static void _bind_methods();

public:
	static NavigationServer3D *get_singleton();

	virtual TypedArray<RID> get_maps() const = 0;

	/* MAP API */

	virtual RID map_create() = 0;

	virtual void map_set_active(RID p_map, bool p_active) = 0;
	virtual bool map_is_active(RID p_map) const = 0;

	virtual void map_set_up(RID p_map, Vector3 p_up) = 0;
	virtual Vector3 map_get_up(RID p_map) const = 0;

	virtual void map_set_cell_size(RID p_map, real_t p_cell_size) = 0;
	virtual real_t map_get_cell_size(RID p_map) const = 0;

	virtual void map_set_cell_height(RID p_map, real_t p_height) = 0;
	virtual real_t map_get_cell_height(RID p_map) const = 0;

	virtual void map_set_merge_rasterizer_cell_scale(RID p_map, float p_value) = 0;
	virtual float map_get_merge_rasterizer_cell_scale(RID p_map) const = 0;

	virtual void map_set_use_edge_connections(RID p_map, bool p_enabled) = 0;
	virtual bool map_get_use_edge_connections(RID p_map) const = 0;

	virtual void map_set_edge_connection_margin(RID p_map, real_t p_connection_margin) = 0;
	virtual real_t map_get_edge_connection_margin(RID p_map) const = 0;

	virtual void map_set_link_connection_radius(RID p_map, real_t p_connection_radius) = 0;
	virtual real_t map_get_link_connection_radius(RID p_map) const = 0;

	virtual Vector<Vector3> map_get_path(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers = 1) = 0;

	virtual Vector3 map_get_closest_point_to_segment(RID p_map, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision = false) const = 0;
	virtual Vector3 map_get_closest_point(RID p_map, const Vector3 &p_point) const = 0;
	virtual Vector3 map_get_closest_point_normal(RID p_map, const Vector3 &p_point) const = 0;
	virtual RID map_get_closest_point_owner(RID p_map, const Vector3 &p_point) const = 0;

	virtual TypedArray<RID> map_get_links(RID p_map) const = 0;
	virtual TypedArray<RID> map_get_regions(RID p_map) const = 0;
	virtual TypedArray<RID> map_get_agents(RID p_map) const = 0;
	virtual TypedArray<RID> map_get_obstacles(RID p_map) const = 0;

	virtual void map_force_update(RID p_map) = 0;
	virtual uint32_t map_get_iteration_id(RID p_map) const = 0;

	virtual void map_set_use_async_iterations(RID p_map, bool p_enabled) = 0;
	virtual bool map_get_use_async_iterations(RID p_map) const = 0;

	virtual Vector3 map_get_random_point(RID p_map, uint32_t p_navigation_layers, bool p_uniformly) const = 0;

	/* REGION API */

	virtual RID region_create() = 0;
	virtual uint32_t region_get_iteration_id(RID p_region) const = 0;

	virtual void region_set_use_async_iterations(RID p_region, bool p_enabled) = 0;
	virtual bool region_get_use_async_iterations(RID p_region) const = 0;

	virtual void region_set_enabled(RID p_region, bool p_enabled) = 0;
	virtual bool region_get_enabled(RID p_region) const = 0;

	virtual void region_set_use_edge_connections(RID p_region, bool p_enabled) = 0;
	virtual bool region_get_use_edge_connections(RID p_region) const = 0;

	virtual void region_set_enter_cost(RID p_region, real_t p_enter_cost) = 0;
	virtual real_t region_get_enter_cost(RID p_region) const = 0;

	virtual void region_set_travel_cost(RID p_region, real_t p_travel_cost) = 0;
	virtual real_t region_get_travel_cost(RID p_region) const = 0;

	virtual void region_set_owner_id(RID p_region, ObjectID p_owner_id) = 0;
	virtual ObjectID region_get_owner_id(RID p_region) const = 0;

	virtual bool region_owns_point(RID p_region, const Vector3 &p_point) const = 0;

	virtual void region_set_map(RID p_region, RID p_map) = 0;
	virtual RID region_get_map(RID p_region) const = 0;

	virtual void region_set_navigation_layers(RID p_region, uint32_t p_navigation_layers) = 0;
	virtual uint32_t region_get_navigation_layers(RID p_region) const = 0;

	virtual void region_set_transform(RID p_region, Transform3D p_transform) = 0;
	virtual Transform3D region_get_transform(RID p_region) const = 0;

	virtual void region_set_navigation_mesh(RID p_region, Ref<NavigationMesh> p_navigation_mesh) = 0;

#ifndef DISABLE_DEPRECATED
	virtual void region_bake_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node) = 0;
#endif // DISABLE_DEPRECATED

	virtual int region_get_connections_count(RID p_region) const = 0;
	virtual Vector3 region_get_connection_pathway_start(RID p_region, int p_connection_id) const = 0;
	virtual Vector3 region_get_connection_pathway_end(RID p_region, int p_connection_id) const = 0;

	virtual Vector3 region_get_closest_point_to_segment(RID p_region, const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision = false) const = 0;
	virtual Vector3 region_get_closest_point(RID p_region, const Vector3 &p_point) const = 0;
	virtual Vector3 region_get_closest_point_normal(RID p_region, const Vector3 &p_point) const = 0;
	virtual Vector3 region_get_random_point(RID p_region, uint32_t p_navigation_layers, bool p_uniformly) const = 0;

	virtual AABB region_get_bounds(RID p_region) const = 0;

	/* LINK API */

	virtual RID link_create() = 0;
	virtual uint32_t link_get_iteration_id(RID p_link) const = 0;

	virtual void link_set_map(RID p_link, RID p_map) = 0;
	virtual RID link_get_map(RID p_link) const = 0;

	virtual void link_set_enabled(RID p_link, bool p_enabled) = 0;
	virtual bool link_get_enabled(RID p_link) const = 0;

	virtual void link_set_bidirectional(RID p_link, bool p_bidirectional) = 0;
	virtual bool link_is_bidirectional(RID p_link) const = 0;

	virtual void link_set_navigation_layers(RID p_link, uint32_t p_navigation_layers) = 0;
	virtual uint32_t link_get_navigation_layers(RID p_link) const = 0;

	virtual void link_set_start_position(RID p_link, Vector3 p_position) = 0;
	virtual Vector3 link_get_start_position(RID p_link) const = 0;

	virtual void link_set_end_position(RID p_link, Vector3 p_position) = 0;
	virtual Vector3 link_get_end_position(RID p_link) const = 0;

	virtual void link_set_enter_cost(RID p_link, real_t p_enter_cost) = 0;
	virtual real_t link_get_enter_cost(RID p_link) const = 0;

	virtual void link_set_travel_cost(RID p_link, real_t p_travel_cost) = 0;
	virtual real_t link_get_travel_cost(RID p_link) const = 0;

	virtual void link_set_owner_id(RID p_link, ObjectID p_owner_id) = 0;
	virtual ObjectID link_get_owner_id(RID p_link) const = 0;

	/* AGENT API */

	virtual RID agent_create() = 0;

	virtual void agent_set_map(RID p_agent, RID p_map) = 0;
	virtual RID agent_get_map(RID p_agent) const = 0;

	virtual void agent_set_paused(RID p_agent, bool p_paused) = 0;
	virtual bool agent_get_paused(RID p_agent) const = 0;

	virtual void agent_set_avoidance_enabled(RID p_agent, bool p_enabled) = 0;
	virtual bool agent_get_avoidance_enabled(RID p_agent) const = 0;

	virtual void agent_set_use_3d_avoidance(RID p_agent, bool p_enabled) = 0;
	virtual bool agent_get_use_3d_avoidance(RID p_agent) const = 0;

	virtual void agent_set_neighbor_distance(RID p_agent, real_t p_distance) = 0;
	virtual real_t agent_get_neighbor_distance(RID p_agent) const = 0;

	virtual void agent_set_max_neighbors(RID p_agent, int p_count) = 0;
	virtual int agent_get_max_neighbors(RID p_agent) const = 0;

	virtual void agent_set_time_horizon_agents(RID p_agent, real_t p_time_horizon) = 0;
	virtual real_t agent_get_time_horizon_agents(RID p_agent) const = 0;

	virtual void agent_set_time_horizon_obstacles(RID p_agent, real_t p_time_horizon) = 0;
	virtual real_t agent_get_time_horizon_obstacles(RID p_agent) const = 0;

	virtual void agent_set_radius(RID p_agent, real_t p_radius) = 0;
	virtual real_t agent_get_radius(RID p_agent) const = 0;

	virtual void agent_set_height(RID p_agent, real_t p_height) = 0;
	virtual real_t agent_get_height(RID p_agent) const = 0;

	virtual void agent_set_max_speed(RID p_agent, real_t p_max_speed) = 0;
	virtual real_t agent_get_max_speed(RID p_agent) const = 0;

	virtual void agent_set_velocity_forced(RID p_agent, Vector3 p_velocity) = 0;

	virtual void agent_set_velocity(RID p_agent, Vector3 p_velocity) = 0;
	virtual Vector3 agent_get_velocity(RID p_agent) const = 0;

	virtual void agent_set_position(RID p_agent, Vector3 p_position) = 0;
	virtual Vector3 agent_get_position(RID p_agent) const = 0;

	virtual bool agent_is_map_changed(RID p_agent) const = 0;

	virtual void agent_set_avoidance_callback(RID p_agent, Callable p_callback) = 0;
	virtual bool agent_has_avoidance_callback(RID p_agent) const = 0;

	virtual void agent_set_avoidance_layers(RID p_agent, uint32_t p_layers) = 0;
	virtual uint32_t agent_get_avoidance_layers(RID p_agent) const = 0;

	virtual void agent_set_avoidance_mask(RID p_agent, uint32_t p_mask) = 0;
	virtual uint32_t agent_get_avoidance_mask(RID p_agent) const = 0;

	virtual void agent_set_avoidance_priority(RID p_agent, real_t p_priority) = 0;
	virtual real_t agent_get_avoidance_priority(RID p_agent) const = 0;

	/* OBSTACLE API */

	virtual RID obstacle_create() = 0;

	virtual void obstacle_set_map(RID p_obstacle, RID p_map) = 0;
	virtual RID obstacle_get_map(RID p_obstacle) const = 0;

	virtual void obstacle_set_paused(RID p_obstacle, bool p_paused) = 0;
	virtual bool obstacle_get_paused(RID p_obstacle) const = 0;

	virtual void obstacle_set_avoidance_enabled(RID p_obstacle, bool p_enabled) = 0;
	virtual bool obstacle_get_avoidance_enabled(RID p_obstacle) const = 0;

	virtual void obstacle_set_use_3d_avoidance(RID p_obstacle, bool p_enabled) = 0;
	virtual bool obstacle_get_use_3d_avoidance(RID p_obstacle) const = 0;

	virtual void obstacle_set_radius(RID p_obstacle, real_t p_radius) = 0;
	virtual real_t obstacle_get_radius(RID p_obstacle) const = 0;
	virtual void obstacle_set_height(RID p_obstacle, real_t p_height) = 0;
	virtual real_t obstacle_get_height(RID p_obstacle) const = 0;
	virtual void obstacle_set_velocity(RID p_obstacle, Vector3 p_velocity) = 0;
	virtual Vector3 obstacle_get_velocity(RID p_obstacle) const = 0;
	virtual void obstacle_set_position(RID p_obstacle, Vector3 p_position) = 0;
	virtual Vector3 obstacle_get_position(RID p_obstacle) const = 0;
	virtual void obstacle_set_vertices(RID p_obstacle, const Vector<Vector3> &p_vertices) = 0;
	virtual Vector<Vector3> obstacle_get_vertices(RID p_obstacle) const = 0;
	virtual void obstacle_set_avoidance_layers(RID p_obstacle, uint32_t p_layers) = 0;
	virtual uint32_t obstacle_get_avoidance_layers(RID p_obstacle) const = 0;

	/* QUERY API */

	virtual void query_path(const Ref<NavigationPathQueryParameters3D> &p_query_parameters, Ref<NavigationPathQueryResult3D> p_query_result, const Callable &p_callback = Callable()) = 0;

	/* NAVMESH BAKE API */

#ifndef _3D_DISABLED
	virtual void parse_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, Node *p_root_node, const Callable &p_callback = Callable()) = 0;
	virtual void bake_from_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, const Callable &p_callback = Callable()) = 0;
	virtual void bake_from_source_geometry_data_async(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, const Callable &p_callback = Callable()) = 0;
	virtual bool is_baking_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh) const = 0;
	virtual String get_baking_navigation_mesh_state_msg(Ref<NavigationMesh> p_navigation_mesh) const = 0;
#endif // _3D_DISABLED

protected:
	static RWLock geometry_parser_rwlock;
	static RID_Owner<NavMeshGeometryParser3D> geometry_parser_owner;
	static LocalVector<NavMeshGeometryParser3D *> generator_parsers;

public:
	virtual RID source_geometry_parser_create() = 0;
	virtual void source_geometry_parser_set_callback(RID p_parser, const Callable &p_callback) = 0;

	virtual Vector<Vector3> simplify_path(const Vector<Vector3> &p_path, real_t p_epsilon) = 0;

	/* SERVER API */

	virtual void set_active(bool p_active) = 0;
	virtual void process(double p_delta_time) = 0;
	virtual void physics_process(double p_delta_time) = 0;
	virtual void init() = 0;
	virtual void sync() = 0;
	virtual void finish() = 0;
	virtual void free_rid(RID p_rid) = 0;
#ifndef DISABLE_DEPRECATED
	[[deprecated("Use `free_rid()` instead.")]] void free(RID p_rid) {
		free_rid(p_rid);
	}
#endif // DISABLE_DEPRECATED

	NavigationServer3D();
	~NavigationServer3D() override;

	/* DEBUG API */

	enum ProcessInfo {
		INFO_ACTIVE_MAPS,
		INFO_REGION_COUNT,
		INFO_AGENT_COUNT,
		INFO_LINK_COUNT,
		INFO_POLYGON_COUNT,
		INFO_EDGE_COUNT,
		INFO_EDGE_MERGE_COUNT,
		INFO_EDGE_CONNECTION_COUNT,
		INFO_EDGE_FREE_COUNT,
		INFO_OBSTACLE_COUNT,
	};

	virtual int get_process_info(ProcessInfo p_info) const = 0;

	void set_debug_enabled(bool p_enabled);
	bool get_debug_enabled() const;

protected:
#ifndef DISABLE_DEPRECATED
	Vector<Vector3> _map_get_path_bind_compat_100129(RID p_map, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers = 1) const;
	void _query_path_bind_compat_100129(const Ref<NavigationPathQueryParameters3D> &p_query_parameters, Ref<NavigationPathQueryResult3D> p_query_result) const;
	static void _bind_compatibility_methods();
#endif

#ifdef DEBUG_ENABLED
public:
	void emit_navigation_debug_changed();
	void emit_avoidance_debug_changed();

private:
	NavigationServer3DDebug *debug = nullptr;

	bool navigation_debug_dirty = false;
	void _emit_navigation_debug_changed_signal();

	bool avoidance_debug_dirty = false;
	void _emit_avoidance_debug_changed_signal();
#endif // DEBUG_ENABLED

public:
	void register_settings();
	void update_from_settings();
};

VARIANT_ENUM_CAST(NavigationServer3D::ProcessInfo);
