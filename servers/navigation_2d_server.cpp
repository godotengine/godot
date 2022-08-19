/*************************************************************************/
/*  navigation_2d_server.cpp                                             */
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

#include "servers/navigation_2d_server.h"

#include "core/math/transform.h"
#include "core/math/transform_2d.h"
#include "servers/navigation_server.h"

Navigation2DServer *Navigation2DServer::singleton = nullptr;

#define FORWARD_0_C(FUNC_NAME)                                 \
	Navigation2DServer::FUNC_NAME()                            \
			const {                                            \
		return NavigationServer::get_singleton()->FUNC_NAME(); \
	}

#define FORWARD_1(FUNC_NAME, T_0, D_0, CONV_0)                                \
	Navigation2DServer::FUNC_NAME(T_0 D_0) {                                  \
		return NavigationServer::get_singleton_mut()->FUNC_NAME(CONV_0(D_0)); \
	}

#define FORWARD_1_C(FUNC_NAME, T_0, D_0, CONV_0)                          \
	Navigation2DServer::FUNC_NAME(T_0 D_0)                                \
			const {                                                       \
		return NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0)); \
	}

#define FORWARD_2_C(FUNC_NAME, T_0, D_0, T_1, D_1, CONV_0, CONV_1)                     \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1)                                    \
			const {                                                                    \
		return NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1)); \
	}

#define FORWARD_2_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, CONV_0, CONV_1)                   \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1)                                            \
			const {                                                                            \
		return CONV_R(NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1))); \
	}

#define FORWARD_3_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, CONV_0, CONV_1, CONV_2)              \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2)                                                \
			const {                                                                                         \
		return CONV_R(NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2))); \
	}

#define FORWARD_3_C(FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, CONV_0, CONV_1, CONV_2)                \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2)                                        \
			const {                                                                                 \
		return NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2)); \
	}

#define FORWARD_4_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, CONV_0, CONV_1, CONV_2, CONV_3)         \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3)                                                    \
			const {                                                                                                      \
		return CONV_R(NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2), CONV_3(D_3))); \
	}

#define FORWARD_4_C(FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, CONV_0, CONV_1, CONV_2, CONV_3)           \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3)                                            \
			const {                                                                                              \
		return NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2), CONV_3(D_3)); \
	}

#define FORWARD_5_R_C(CONV_R, FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, T_4, D_4, CONV_0, CONV_1, CONV_2, CONV_3, CONV_4)    \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3, T_4 D_4)                                                        \
			const {                                                                                                                   \
		return CONV_R(NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2), CONV_3(D_3), CONV_4(D_4))); \
	}

#define FORWARD_5_C(FUNC_NAME, T_0, D_0, T_1, D_1, T_2, D_2, T_3, D_3, T_4, D_4, CONV_0, CONV_1, CONV_2, CONV_3, CONV_4)      \
	Navigation2DServer::FUNC_NAME(T_0 D_0, T_1 D_1, T_2 D_2, T_3 D_3, T_4 D_4)                                                \
			const {                                                                                                           \
		return NavigationServer::get_singleton()->FUNC_NAME(CONV_0(D_0), CONV_1(D_1), CONV_2(D_2), CONV_3(D_3), CONV_4(D_4)); \
	}

RID rid_to_rid(const RID d) {
	return d;
}

bool bool_to_bool(const bool d) {
	return d;
}

int int_to_int(const int d) {
	return d;
}

uint32_t uint32_to_uint32(const uint32_t d) {
	return d;
}

real_t real_to_real(const real_t d) {
	return d;
}

Vector3 v2_to_v3(const Vector2 d) {
	return Vector3(d.x, 0.0, d.y);
}

Vector2 v3_to_v2(const Vector3 &d) {
	return Vector2(d.x, d.z);
}

Vector<Vector2> vector_v3_to_v2(const Vector<Vector3> &d) {
	Vector<Vector2> nd;
	nd.resize(d.size());
	for (int i(0); i < nd.size(); i++) {
		nd.write[i] = v3_to_v2(d[i]);
	}
	return nd;
}

Transform trf2_to_trf3(const Transform2D &d) {
	Vector3 o(v2_to_v3(d.get_origin()));
	Basis b;
	b.rotate(Vector3(0, -1, 0), d.get_rotation());
	b.scale(v2_to_v3(d.get_scale()));
	return Transform(b, o);
}

Object *obj_to_obj(Object *d) {
	return d;
}

StringName sn_to_sn(StringName &d) {
	return d;
}

Variant var_to_var(Variant &d) {
	return d;
}

Ref<NavigationMesh> poly_to_mesh(Ref<NavigationPolygon> d) {
	if (d.is_valid()) {
		return d->get_mesh();
	} else {
		return Ref<NavigationMesh>();
	}
}

void Navigation2DServer::_emit_map_changed(RID p_map) {
	emit_signal("map_changed", p_map);
}

void Navigation2DServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_maps"), &Navigation2DServer::get_maps);

	ClassDB::bind_method(D_METHOD("map_create"), &Navigation2DServer::map_create);
	ClassDB::bind_method(D_METHOD("map_set_active", "map", "active"), &Navigation2DServer::map_set_active);
	ClassDB::bind_method(D_METHOD("map_is_active", "map"), &Navigation2DServer::map_is_active);
	ClassDB::bind_method(D_METHOD("map_set_cell_size", "map", "cell_size"), &Navigation2DServer::map_set_cell_size);
	ClassDB::bind_method(D_METHOD("map_get_cell_size", "map"), &Navigation2DServer::map_get_cell_size);
	ClassDB::bind_method(D_METHOD("map_set_cell_height", "map", "cell_height"), &Navigation2DServer::map_set_cell_size);
	ClassDB::bind_method(D_METHOD("map_get_cell_height", "map"), &Navigation2DServer::map_get_cell_size);
	ClassDB::bind_method(D_METHOD("map_set_edge_connection_margin", "map", "margin"), &Navigation2DServer::map_set_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("map_get_edge_connection_margin", "map"), &Navigation2DServer::map_get_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("map_get_path", "map", "origin", "destination", "optimize", "navigation_layers"), &Navigation2DServer::map_get_path, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("map_get_closest_point", "map", "to_point"), &Navigation2DServer::map_get_closest_point);
	ClassDB::bind_method(D_METHOD("map_get_closest_point_owner", "map", "to_point"), &Navigation2DServer::map_get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("map_get_regions", "map"), &Navigation2DServer::map_get_regions);
	ClassDB::bind_method(D_METHOD("map_get_agents", "map"), &Navigation2DServer::map_get_agents);
	ClassDB::bind_method(D_METHOD("map_force_update", "map"), &Navigation2DServer::map_force_update);

	ClassDB::bind_method(D_METHOD("region_create"), &Navigation2DServer::region_create);
	ClassDB::bind_method(D_METHOD("region_set_enter_cost", "region", "enter_cost"), &Navigation2DServer::region_set_enter_cost);
	ClassDB::bind_method(D_METHOD("region_get_enter_cost", "region"), &Navigation2DServer::region_get_enter_cost);
	ClassDB::bind_method(D_METHOD("region_set_travel_cost", "region", "travel_cost"), &Navigation2DServer::region_set_travel_cost);
	ClassDB::bind_method(D_METHOD("region_get_travel_cost", "region"), &Navigation2DServer::region_get_travel_cost);
	ClassDB::bind_method(D_METHOD("region_owns_point", "region", "point"), &Navigation2DServer::region_owns_point);
	ClassDB::bind_method(D_METHOD("region_set_map", "region", "map"), &Navigation2DServer::region_set_map);
	ClassDB::bind_method(D_METHOD("region_get_map", "region"), &Navigation2DServer::region_get_map);
	ClassDB::bind_method(D_METHOD("region_set_navigation_layers", "region", "navigation_layers"), &Navigation2DServer::region_set_navigation_layers);
	ClassDB::bind_method(D_METHOD("region_get_navigation_layers", "region"), &Navigation2DServer::region_get_navigation_layers);
	ClassDB::bind_method(D_METHOD("region_set_transform", "region", "transform"), &Navigation2DServer::region_set_transform);
	ClassDB::bind_method(D_METHOD("region_set_navpoly", "region", "nav_poly"), &Navigation2DServer::region_set_navpoly);
	ClassDB::bind_method(D_METHOD("region_get_connections_count", "region"), &Navigation2DServer::region_get_connections_count);
	ClassDB::bind_method(D_METHOD("region_get_connection_pathway_start", "region", "connection"), &Navigation2DServer::region_get_connection_pathway_start);
	ClassDB::bind_method(D_METHOD("region_get_connection_pathway_end", "region", "connection"), &Navigation2DServer::region_get_connection_pathway_end);

	ClassDB::bind_method(D_METHOD("agent_create"), &Navigation2DServer::agent_create);
	ClassDB::bind_method(D_METHOD("agent_set_map", "agent", "map"), &Navigation2DServer::agent_set_map);
	ClassDB::bind_method(D_METHOD("agent_get_map", "agent"), &Navigation2DServer::agent_get_map);
	ClassDB::bind_method(D_METHOD("agent_set_neighbor_dist", "agent", "dist"), &Navigation2DServer::agent_set_neighbor_dist);
	ClassDB::bind_method(D_METHOD("agent_set_max_neighbors", "agent", "count"), &Navigation2DServer::agent_set_max_neighbors);
	ClassDB::bind_method(D_METHOD("agent_set_time_horizon", "agent", "time"), &Navigation2DServer::agent_set_time_horizon);
	ClassDB::bind_method(D_METHOD("agent_set_radius", "agent", "radius"), &Navigation2DServer::agent_set_radius);
	ClassDB::bind_method(D_METHOD("agent_set_max_speed", "agent", "max_speed"), &Navigation2DServer::agent_set_max_speed);
	ClassDB::bind_method(D_METHOD("agent_set_velocity", "agent", "velocity"), &Navigation2DServer::agent_set_velocity);
	ClassDB::bind_method(D_METHOD("agent_set_target_velocity", "agent", "target_velocity"), &Navigation2DServer::agent_set_target_velocity);
	ClassDB::bind_method(D_METHOD("agent_set_position", "agent", "position"), &Navigation2DServer::agent_set_position);
	ClassDB::bind_method(D_METHOD("agent_is_map_changed", "agent"), &Navigation2DServer::agent_is_map_changed);
	ClassDB::bind_method(D_METHOD("agent_set_callback", "agent", "receiver", "method", "userdata"), &Navigation2DServer::agent_set_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &Navigation2DServer::free);

	ClassDB::bind_method(D_METHOD("_emit_map_changed"), &Navigation2DServer::_emit_map_changed);
	ADD_SIGNAL(MethodInfo("map_changed", PropertyInfo(Variant::_RID, "map")));
}

Navigation2DServer::Navigation2DServer() {
	singleton = this;
	ERR_FAIL_COND_MSG(!NavigationServer::get_singleton(), "The NavigationServer singleton should be initialized before the Navigation2DServer one.");
	NavigationServer::get_singleton_mut()->connect("map_changed", this, "_emit_map_changed");
}

Navigation2DServer::~Navigation2DServer() {
	singleton = nullptr;
}

Array FORWARD_0_C(get_maps);

Array FORWARD_1_C(map_get_regions, RID, p_map, rid_to_rid);

Array FORWARD_1_C(map_get_agents, RID, p_map, rid_to_rid);

RID FORWARD_1_C(region_get_map, RID, p_region, rid_to_rid);

RID FORWARD_1_C(agent_get_map, RID, p_agent, rid_to_rid);

RID FORWARD_0_C(map_create);

void FORWARD_2_C(map_set_active, RID, p_map, bool, p_active, rid_to_rid, bool_to_bool);

bool FORWARD_1_C(map_is_active, RID, p_map, rid_to_rid);

void Navigation2DServer::map_force_update(RID p_map) {
	NavigationServer::get_singleton_mut()->map_force_update(p_map);
}

void FORWARD_2_C(map_set_cell_size, RID, p_map, real_t, p_cell_size, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_cell_size, RID, p_map, rid_to_rid);

void FORWARD_2_C(map_set_cell_height, RID, p_map, real_t, p_cell_height, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_cell_height, RID, p_map, rid_to_rid);

void FORWARD_2_C(map_set_edge_connection_margin, RID, p_map, real_t, p_connection_margin, rid_to_rid, real_to_real);
real_t FORWARD_1_C(map_get_edge_connection_margin, RID, p_map, rid_to_rid);

Vector<Vector2> FORWARD_5_R_C(vector_v3_to_v2, map_get_path, RID, p_map, Vector2, p_origin, Vector2, p_destination, bool, p_optimize, uint32_t, p_navigation_layers, rid_to_rid, v2_to_v3, v2_to_v3, bool_to_bool, uint32_to_uint32);

Vector2 FORWARD_2_R_C(v3_to_v2, map_get_closest_point, RID, p_map, const Vector2 &, p_point, rid_to_rid, v2_to_v3);
RID FORWARD_2_C(map_get_closest_point_owner, RID, p_map, const Vector2 &, p_point, rid_to_rid, v2_to_v3);

RID FORWARD_0_C(region_create);

void FORWARD_2_C(region_set_enter_cost, RID, p_region, real_t, p_enter_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(region_get_enter_cost, RID, p_region, rid_to_rid);
void FORWARD_2_C(region_set_travel_cost, RID, p_region, real_t, p_travel_cost, rid_to_rid, real_to_real);
real_t FORWARD_1_C(region_get_travel_cost, RID, p_region, rid_to_rid);
bool FORWARD_2_C(region_owns_point, RID, p_region, const Vector2 &, p_point, rid_to_rid, v2_to_v3);

void FORWARD_2_C(region_set_map, RID, p_region, RID, p_map, rid_to_rid, rid_to_rid);
void FORWARD_2_C(region_set_navigation_layers, RID, p_region, uint32_t, p_navigation_layers, rid_to_rid, uint32_to_uint32);
uint32_t FORWARD_1_C(region_get_navigation_layers, RID, p_region, rid_to_rid);
void FORWARD_2_C(region_set_transform, RID, p_region, Transform2D, p_transform, rid_to_rid, trf2_to_trf3);

void Navigation2DServer::region_set_navpoly(RID p_region, Ref<NavigationPolygon> p_nav_mesh) const {
	NavigationServer::get_singleton()->region_set_navmesh(p_region, poly_to_mesh(p_nav_mesh));
}

int FORWARD_1_C(region_get_connections_count, RID, p_region, rid_to_rid);
Vector2 FORWARD_2_R_C(v3_to_v2, region_get_connection_pathway_start, RID, p_region, int, p_connection_id, rid_to_rid, int_to_int);
Vector2 FORWARD_2_R_C(v3_to_v2, region_get_connection_pathway_end, RID, p_region, int, p_connection_id, rid_to_rid, int_to_int);

RID Navigation2DServer::agent_create() const {
	RID agent = NavigationServer::get_singleton()->agent_create();
	NavigationServer::get_singleton()->agent_set_ignore_y(agent, true);
	return agent;
}

void FORWARD_2_C(agent_set_map, RID, p_agent, RID, p_map, rid_to_rid, rid_to_rid);

void FORWARD_2_C(agent_set_neighbor_dist, RID, p_agent, real_t, p_dist, rid_to_rid, real_to_real);

void FORWARD_2_C(agent_set_max_neighbors, RID, p_agent, int, p_count, rid_to_rid, int_to_int);

void FORWARD_2_C(agent_set_time_horizon, RID, p_agent, real_t, p_time, rid_to_rid, real_to_real);

void FORWARD_2_C(agent_set_radius, RID, p_agent, real_t, p_radius, rid_to_rid, real_to_real);

void FORWARD_2_C(agent_set_max_speed, RID, p_agent, real_t, p_max_speed, rid_to_rid, real_to_real);

void FORWARD_2_C(agent_set_velocity, RID, p_agent, Vector2, p_velocity, rid_to_rid, v2_to_v3);

void FORWARD_2_C(agent_set_target_velocity, RID, p_agent, Vector2, p_velocity, rid_to_rid, v2_to_v3);

void FORWARD_2_C(agent_set_position, RID, p_agent, Vector2, p_position, rid_to_rid, v2_to_v3);

void FORWARD_2_C(agent_set_ignore_y, RID, p_agent, bool, p_ignore, rid_to_rid, bool_to_bool);

bool FORWARD_1_C(agent_is_map_changed, RID, p_agent, rid_to_rid);

void FORWARD_4_C(agent_set_callback, RID, p_agent, Object *, p_receiver, StringName, p_method, Variant, p_udata, rid_to_rid, obj_to_obj, sn_to_sn, var_to_var);

void FORWARD_1_C(free, RID, p_object, rid_to_rid);
