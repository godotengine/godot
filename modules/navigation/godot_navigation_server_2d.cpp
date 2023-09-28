/**************************************************************************/
/*  godot_navigation_server_2d.cpp                                        */
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

#include "godot_navigation_server_2d.h"

#ifdef CLIPPER2_ENABLED
#include "nav_mesh_generator_2d.h"
#endif // CLIPPER2_ENABLED

#include "servers/navigation_server_3d.h"

#define FORWARD_0(FUNC_NAME)                                     \
	GodotNavigationServer2D::FUNC_NAME() {                       \
		return NavigationServer3D::get_singleton()->FUNC_NAME(); \
	}

#define FORWARD_0_C(FUNC_NAME)                                   \
	GodotNavigationServer2D::FUNC_NAME()                         \
			const {                                              \
		return NavigationServer3D::get_singleton()->FUNC_NAME(); \
	}

#define FORWARD_1(FUNC_NAME, T_0, D_0, CONV_0)                              \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0) {                           \
		return NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0)); \
	}

#define FORWARD_1_C(FUNC_NAME, T_0, D_0, CONV_0)                            \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0)                             \
			const {                                                         \
		return NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0)); \
	}

#define FORWARD_1_R_C(CONV_R, FUNC_NAME, T_0, D_0, CONV_0)                          \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0)                                     \
			const {                                                                 \
		return CONV_R(NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0))); \
	}

#define FORWARD_2(FUNC_NAME, T_0, D_0, T_1, D_1, CONV_0, CONV_1)                         \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0, T_1 D_1) {                               \
		return NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1)); \
	}

#define FORWARD_2_C(FUNC_NAME, T_0, D_0, T_1, D_1, CONV_0, CONV_1)                       \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0, T_1 D_1)                                 \
			const {                                                                      \
		return NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1)); \
	}

#define FORWARD_2_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, CONV_0, CONV_1)                     \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0, T_1 D_1)                                         \
			const {                                                                              \
		return CONV_R(NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1))); \
	}

#define FORWARD_5_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, T_4, D_4, CONV_0, CONV_1, CONV_2, CONV_3, CONV_4)      \
	GodotNavigationServer2D::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3, T_4 D_4)                                                     \
			const {                                                                                                                     \
		return CONV_R(NavigationServer3D::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2), CONV_3(D_3), CONV_4(D_4))); \
	}

static RID rid_to_rid(const RID d) {
	return d;
}

static bool bool_to_bool(const bool d) {
	return d;
}

static int int_to_int(const int d) {
	return d;
}

static uint32_t uint32_to_uint32(const uint32_t d) {
	return d;
}

static real_t real_to_real(const real_t d) {
	return d;
}

static Vector3 v2_to_v3(const Vector2 d) {
	return Vector3(d.x, 0.0, d.y);
}

static Vector2 v3_to_v2(const Vector3 &d) {
	return Vector2(d.x, d.z);
}

static Vector<Vector3> vector_v2_to_v3(const Vector<Vector2> &d) {
	Vector<Vector3> nd;
	nd.resize(d.size());
	for (int i(0); i < nd.size(); i++) {
		nd.write[i] = v2_to_v3(d[i]);
	}
	return nd;
}

static Vector<Vector2> vector_v3_to_v2(const Vector<Vector3> &d) {
	Vector<Vector2> nd;
	nd.resize(d.size());
	for (int i(0); i < nd.size(); i++) {
		nd.write[i] = v3_to_v2(d[i]);
	}
	return nd;
}

static Transform3D trf2_to_trf3(const Transform2D &d) {
	Vector3 o(v2_to_v3(d.get_origin()));
	Basis b;
	b.rotate(Vector3(0, -1, 0), d.get_rotation());
	b.scale(v2_to_v3(d.get_scale()));
	return Transform3D(b, o);
}

static ObjectID id_to_id(const ObjectID &id) {
	return id;
}

static Callable callable_to_callable(const Callable &c) {
	return c;
}

static Ref<NavigationMesh> poly_to_mesh(Ref<NavigationPolygon> d) {
	if (d.is_valid()) {
		return d->get_navigation_mesh();
	} else {
		return Ref<NavigationMesh>();
	}
}

void GodotNavigationServer2D::init() {
#ifdef CLIPPER2_ENABLED
	navmesh_generator_2d = memnew(NavMeshGenerator2D);
	ERR_FAIL_NULL_MSG(navmesh_generator_2d, "Failed to init NavMeshGenerator2D.");
#endif // CLIPPER2_ENABLED
}

void GodotNavigationServer2D::sync() {
#ifdef CLIPPER2_ENABLED
	if (navmesh_generator_2d) {
		navmesh_generator_2d->sync();
	}
#endif // CLIPPER2_ENABLED
}

void GodotNavigationServer2D::finish() {
#ifdef CLIPPER2_ENABLED
	if (navmesh_generator_2d) {
		navmesh_generator_2d->finish();
		memdelete(navmesh_generator_2d);
		navmesh_generator_2d = nullptr;
	}
#endif // CLIPPER2_ENABLED
}

void GodotNavigationServer2D::parse_source_geometry_data(const Ref<NavigationPolygon> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, Node *p_root_node, const Callable &p_callback) {
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "The SceneTree can only be parsed on the main thread. Call this function from the main thread or use call_deferred().");
	ERR_FAIL_COND_MSG(!p_navigation_mesh.is_valid(), "Invalid navigation polygon.");
	ERR_FAIL_NULL_MSG(p_root_node, "No parsing root node specified.");
	ERR_FAIL_COND_MSG(!p_root_node->is_inside_tree(), "The root node needs to be inside the SceneTree.");

#ifdef CLIPPER2_ENABLED
	ERR_FAIL_NULL(NavMeshGenerator2D::get_singleton());
	NavMeshGenerator2D::get_singleton()->parse_source_geometry_data(p_navigation_mesh, p_source_geometry_data, p_root_node, p_callback);
#endif // CLIPPER2_ENABLED
}

void GodotNavigationServer2D::bake_from_source_geometry_data(const Ref<NavigationPolygon> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, const Callable &p_callback) {
	ERR_FAIL_COND_MSG(!p_navigation_mesh.is_valid(), "Invalid navigation polygon.");
	ERR_FAIL_COND_MSG(!p_source_geometry_data.is_valid(), "Invalid NavigationMeshSourceGeometryData2D.");

#ifdef CLIPPER2_ENABLED
	ERR_FAIL_NULL(NavMeshGenerator2D::get_singleton());
	NavMeshGenerator2D::get_singleton()->bake_from_source_geometry_data(p_navigation_mesh, p_source_geometry_data, p_callback);
#endif // CLIPPER2_ENABLED
}

void GodotNavigationServer2D::bake_from_source_geometry_data_async(const Ref<NavigationPolygon> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, const Callable &p_callback) {
	ERR_FAIL_COND_MSG(!p_navigation_mesh.is_valid(), "Invalid navigation mesh.");
	ERR_FAIL_COND_MSG(!p_source_geometry_data.is_valid(), "Invalid NavigationMeshSourceGeometryData2D.");

#ifdef CLIPPER2_ENABLED
	ERR_FAIL_NULL(NavMeshGenerator2D::get_singleton());
	NavMeshGenerator2D::get_singleton()->bake_from_source_geometry_data_async(p_navigation_mesh, p_source_geometry_data, p_callback);
#endif // CLIPPER2_ENABLED
}

GodotNavigationServer2D::GodotNavigationServer2D() {}

GodotNavigationServer2D::~GodotNavigationServer2D() {}

TypedArray<RID> FORWARD_0_C(get_maps);

TypedArray<RID> FORWARD_1_C(map_get_links, RID, p_map, rid_to_rid);

TypedArray<RID> FORWARD_1_C(map_get_regions, RID, p_map, rid_to_rid);

TypedArray<RID> FORWARD_1_C(map_get_agents, RID, p_map, rid_to_rid);

TypedArray<RID> FORWARD_1_C(map_get_obstacles, RID, p_map, rid_to_rid);

RID FORWARD_1_C(region_get_map, RID, p_region, rid_to_rid);

RID FORWARD_1_C(agent_get_map, RID, p_agent, rid_to_rid);

RID FORWARD_0(map_create);

void FORWARD_2(map_set_active, RID, p_map, bool, p_active, rid_to_rid, bool_to_bool);

bool FORWARD_1_C(map_is_active, RID, p_map, rid_to_rid);

void GodotNavigationServer2D::map_force_update(RID p_map) {
	NavigationServer3D::get_singleton()->map_force_update(p_map);
}

void FORWARD_2(map_set_cell_size, RID, p_map, real_t, p_cell_size, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_cell_size, RID, p_map, rid_to_rid);

void FORWARD_2(map_set_use_edge_connections, RID, p_map, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(map_get_use_edge_connections, RID, p_map, rid_to_rid);

void FORWARD_2(map_set_edge_connection_margin, RID, p_map, real_t, p_connection_margin, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_edge_connection_margin, RID, p_map, rid_to_rid);

void FORWARD_2(map_set_link_connection_radius, RID, p_map, real_t, p_connection_radius, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_link_connection_radius, RID, p_map, rid_to_rid);

Vector<Vector2> FORWARD_5_R_C(vector_v3_to_v2, map_get_path, RID, p_map, Vector2, p_origin, Vector2, p_destination, bool, p_optimize, uint32_t, p_layers, rid_to_rid, v2_to_v3, v2_to_v3, bool_to_bool, uint32_to_uint32);

Vector2 FORWARD_2_R_C(v3_to_v2, map_get_closest_point, RID, p_map, const Vector2 &, p_point, rid_to_rid, v2_to_v3);
RID FORWARD_2_C(map_get_closest_point_owner, RID, p_map, const Vector2 &, p_point, rid_to_rid, v2_to_v3);

RID FORWARD_0(region_create);

void FORWARD_2(region_set_enabled, RID, p_region, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(region_get_enabled, RID, p_region, rid_to_rid);
void FORWARD_2(region_set_use_edge_connections, RID, p_region, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(region_get_use_edge_connections, RID, p_region, rid_to_rid);

void FORWARD_2(region_set_enter_cost, RID, p_region, real_t, p_enter_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(region_get_enter_cost, RID, p_region, rid_to_rid);
void FORWARD_2(region_set_travel_cost, RID, p_region, real_t, p_travel_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(region_get_travel_cost, RID, p_region, rid_to_rid);
void FORWARD_2(region_set_owner_id, RID, p_region, ObjectID, p_owner_id, rid_to_rid, id_to_id);
ObjectID FORWARD_1_C(region_get_owner_id, RID, p_region, rid_to_rid);
bool FORWARD_2_C(region_owns_point, RID, p_region, const Vector2 &, p_point, rid_to_rid, v2_to_v3);

void FORWARD_2(region_set_map, RID, p_region, RID, p_map, rid_to_rid, rid_to_rid);
void FORWARD_2(region_set_navigation_layers, RID, p_region, uint32_t, p_navigation_layers, rid_to_rid, uint32_to_uint32);
uint32_t FORWARD_1_C(region_get_navigation_layers, RID, p_region, rid_to_rid);
void FORWARD_2(region_set_transform, RID, p_region, Transform2D, p_transform, rid_to_rid, trf2_to_trf3);

void GodotNavigationServer2D::region_set_navigation_polygon(RID p_region, Ref<NavigationPolygon> p_navigation_polygon) {
	NavigationServer3D::get_singleton()->region_set_navigation_mesh(p_region, poly_to_mesh(p_navigation_polygon));
}

int FORWARD_1_C(region_get_connections_count, RID, p_region, rid_to_rid);
Vector2 FORWARD_2_R_C(v3_to_v2, region_get_connection_pathway_start, RID, p_region, int, p_connection_id, rid_to_rid, int_to_int);
Vector2 FORWARD_2_R_C(v3_to_v2, region_get_connection_pathway_end, RID, p_region, int, p_connection_id, rid_to_rid, int_to_int);

RID FORWARD_0(link_create);

void FORWARD_2(link_set_map, RID, p_link, RID, p_map, rid_to_rid, rid_to_rid);
RID FORWARD_1_C(link_get_map, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_enabled, RID, p_link, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(link_get_enabled, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_bidirectional, RID, p_link, bool, p_bidirectional, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(link_is_bidirectional, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_navigation_layers, RID, p_link, uint32_t, p_navigation_layers, rid_to_rid, uint32_to_uint32);
uint32_t FORWARD_1_C(link_get_navigation_layers, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_start_position, RID, p_link, Vector2, p_position, rid_to_rid, v2_to_v3);
Vector2 FORWARD_1_R_C(v3_to_v2, link_get_start_position, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_end_position, RID, p_link, Vector2, p_position, rid_to_rid, v2_to_v3);
Vector2 FORWARD_1_R_C(v3_to_v2, link_get_end_position, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_enter_cost, RID, p_link, real_t, p_enter_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(link_get_enter_cost, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_travel_cost, RID, p_link, real_t, p_travel_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(link_get_travel_cost, RID, p_link, rid_to_rid);
void FORWARD_2(link_set_owner_id, RID, p_link, ObjectID, p_owner_id, rid_to_rid, id_to_id);
ObjectID FORWARD_1_C(link_get_owner_id, RID, p_link, rid_to_rid);

RID GodotNavigationServer2D::agent_create() {
	RID agent = NavigationServer3D::get_singleton()->agent_create();
	return agent;
}

void FORWARD_2(agent_set_avoidance_enabled, RID, p_agent, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(agent_get_avoidance_enabled, RID, p_agent, rid_to_rid);
void FORWARD_2(agent_set_map, RID, p_agent, RID, p_map, rid_to_rid, rid_to_rid);
void FORWARD_2(agent_set_neighbor_distance, RID, p_agent, real_t, p_dist, rid_to_rid, real_to_real);
void FORWARD_2(agent_set_max_neighbors, RID, p_agent, int, p_count, rid_to_rid, int_to_int);
void FORWARD_2(agent_set_time_horizon_agents, RID, p_agent, real_t, p_time_horizon, rid_to_rid, real_to_real);
void FORWARD_2(agent_set_time_horizon_obstacles, RID, p_agent, real_t, p_time_horizon, rid_to_rid, real_to_real);
void FORWARD_2(agent_set_radius, RID, p_agent, real_t, p_radius, rid_to_rid, real_to_real);
void FORWARD_2(agent_set_max_speed, RID, p_agent, real_t, p_max_speed, rid_to_rid, real_to_real);
void FORWARD_2(agent_set_velocity_forced, RID, p_agent, Vector2, p_velocity, rid_to_rid, v2_to_v3);
void FORWARD_2(agent_set_velocity, RID, p_agent, Vector2, p_velocity, rid_to_rid, v2_to_v3);
void FORWARD_2(agent_set_position, RID, p_agent, Vector2, p_position, rid_to_rid, v2_to_v3);

bool FORWARD_1_C(agent_is_map_changed, RID, p_agent, rid_to_rid);
void FORWARD_2(agent_set_paused, RID, p_agent, bool, p_paused, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(agent_get_paused, RID, p_agent, rid_to_rid);

void FORWARD_1(free, RID, p_object, rid_to_rid);
void FORWARD_2(agent_set_avoidance_callback, RID, p_agent, Callable, p_callback, rid_to_rid, callable_to_callable);

void FORWARD_2(agent_set_avoidance_layers, RID, p_agent, uint32_t, p_layers, rid_to_rid, uint32_to_uint32);
void FORWARD_2(agent_set_avoidance_mask, RID, p_agent, uint32_t, p_mask, rid_to_rid, uint32_to_uint32);
void FORWARD_2(agent_set_avoidance_priority, RID, p_agent, real_t, p_priority, rid_to_rid, real_to_real);

RID GodotNavigationServer2D::obstacle_create() {
	RID obstacle = NavigationServer3D::get_singleton()->obstacle_create();
	return obstacle;
}
void FORWARD_2(obstacle_set_avoidance_enabled, RID, p_obstacle, bool, p_enabled, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(obstacle_get_avoidance_enabled, RID, p_obstacle, rid_to_rid);
void FORWARD_2(obstacle_set_map, RID, p_obstacle, RID, p_map, rid_to_rid, rid_to_rid);
RID FORWARD_1_C(obstacle_get_map, RID, p_obstacle, rid_to_rid);
void FORWARD_2(obstacle_set_paused, RID, p_obstacle, bool, p_paused, rid_to_rid, bool_to_bool);
bool FORWARD_1_C(obstacle_get_paused, RID, p_obstacle, rid_to_rid);
void FORWARD_2(obstacle_set_radius, RID, p_obstacle, real_t, p_radius, rid_to_rid, real_to_real);
void FORWARD_2(obstacle_set_velocity, RID, p_obstacle, Vector2, p_velocity, rid_to_rid, v2_to_v3);
void FORWARD_2(obstacle_set_position, RID, p_obstacle, Vector2, p_position, rid_to_rid, v2_to_v3);
void FORWARD_2(obstacle_set_avoidance_layers, RID, p_obstacle, uint32_t, p_layers, rid_to_rid, uint32_to_uint32);

void GodotNavigationServer2D::obstacle_set_vertices(RID p_obstacle, const Vector<Vector2> &p_vertices) {
	NavigationServer3D::get_singleton()->obstacle_set_vertices(p_obstacle, vector_v2_to_v3(p_vertices));
}

void GodotNavigationServer2D::query_path(const Ref<NavigationPathQueryParameters2D> &p_query_parameters, Ref<NavigationPathQueryResult2D> p_query_result) const {
	ERR_FAIL_COND(!p_query_parameters.is_valid());
	ERR_FAIL_COND(!p_query_result.is_valid());

	const NavigationUtilities::PathQueryResult _query_result = NavigationServer3D::get_singleton()->_query_path(p_query_parameters->get_parameters());

	p_query_result->set_path(vector_v3_to_v2(_query_result.path));
	p_query_result->set_path_types(_query_result.path_types);
	p_query_result->set_path_rids(_query_result.path_rids);
	p_query_result->set_path_owner_ids(_query_result.path_owner_ids);
}
