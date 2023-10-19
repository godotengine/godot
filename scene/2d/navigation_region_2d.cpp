/**************************************************************************/
/*  navigation_region_2d.cpp                                              */
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

#include "navigation_region_2d.h"

#include "core/math/geometry_2d.h"
#include "scene/2d/navigation_obstacle_2d.h"
#include "scene/resources/world_2d.h"
#include "servers/navigation_server_2d.h"

void NavigationRegion2D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}

	enabled = p_enabled;

	NavigationServer2D::get_singleton()->region_set_enabled(region, enabled);

#ifdef DEBUG_ENABLED
	if (Engine::get_singleton()->is_editor_hint() || NavigationServer2D::get_singleton()->get_debug_navigation_enabled()) {
		queue_redraw();
	}
#endif // DEBUG_ENABLED
}

bool NavigationRegion2D::is_enabled() const {
	return enabled;
}

void NavigationRegion2D::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections == p_enabled) {
		return;
	}

	use_edge_connections = p_enabled;

	NavigationServer2D::get_singleton()->region_set_use_edge_connections(region, use_edge_connections);
}

bool NavigationRegion2D::get_use_edge_connections() const {
	return use_edge_connections;
}

void NavigationRegion2D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}

	navigation_layers = p_navigation_layers;

	NavigationServer2D::get_singleton()->region_set_navigation_layers(region, navigation_layers);
}

uint32_t NavigationRegion2D::get_navigation_layers() const {
	return navigation_layers;
}

void NavigationRegion2D::set_navigation_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Navigation layer number must be between 1 and 32 inclusive.");

	uint32_t _navigation_layers = get_navigation_layers();

	if (p_value) {
		_navigation_layers |= 1 << (p_layer_number - 1);
	} else {
		_navigation_layers &= ~(1 << (p_layer_number - 1));
	}

	set_navigation_layers(_navigation_layers);
}

bool NavigationRegion2D::get_navigation_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Navigation layer number must be between 1 and 32 inclusive.");

	return get_navigation_layers() & (1 << (p_layer_number - 1));
}

void NavigationRegion2D::set_enter_cost(real_t p_enter_cost) {
	ERR_FAIL_COND_MSG(p_enter_cost < 0.0, "The enter_cost must be positive.");
	if (Math::is_equal_approx(enter_cost, p_enter_cost)) {
		return;
	}

	enter_cost = p_enter_cost;

	NavigationServer2D::get_singleton()->region_set_enter_cost(region, enter_cost);
}

real_t NavigationRegion2D::get_enter_cost() const {
	return enter_cost;
}

void NavigationRegion2D::set_travel_cost(real_t p_travel_cost) {
	ERR_FAIL_COND_MSG(p_travel_cost < 0.0, "The travel_cost must be positive.");
	if (Math::is_equal_approx(travel_cost, p_travel_cost)) {
		return;
	}

	travel_cost = p_travel_cost;

	NavigationServer2D::get_singleton()->region_set_travel_cost(region, travel_cost);
}

real_t NavigationRegion2D::get_travel_cost() const {
	return travel_cost;
}

RID NavigationRegion2D::get_region_rid() const {
	return region;
}

#ifdef TOOLS_ENABLED
Rect2 NavigationRegion2D::_edit_get_rect() const {
	return navigation_polygon.is_valid() ? navigation_polygon->_edit_get_rect() : Rect2();
}

bool NavigationRegion2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return navigation_polygon.is_valid() ? navigation_polygon->_edit_is_selected_on_click(p_point, p_tolerance) : false;
}
#endif

void NavigationRegion2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_region_enter_navigation_map();
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			set_physics_process_internal(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_region_exit_navigation_map();
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			set_physics_process_internal(false);
		} break;

		case NOTIFICATION_DRAW: {
#ifdef DEBUG_ENABLED
			if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || NavigationServer2D::get_singleton()->get_debug_enabled()) && navigation_polygon.is_valid()) {
				_update_debug_mesh();
				_update_debug_edge_connections_mesh();
			}
#endif // DEBUG_ENABLED
		} break;
	}
}

void NavigationRegion2D::set_navigation_polygon(const Ref<NavigationPolygon> &p_navigation_polygon) {
	if (navigation_polygon.is_valid()) {
		navigation_polygon->disconnect_changed(callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}

	navigation_polygon = p_navigation_polygon;
	NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, p_navigation_polygon);

	if (navigation_polygon.is_valid()) {
		navigation_polygon->connect_changed(callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}
	_navigation_polygon_changed();

	update_configuration_warnings();
}

Ref<NavigationPolygon> NavigationRegion2D::get_navigation_polygon() const {
	return navigation_polygon;
}

void NavigationRegion2D::set_navigation_map(RID p_navigation_map) {
	if (map_override == p_navigation_map) {
		return;
	}

	map_override = p_navigation_map;

	NavigationServer2D::get_singleton()->region_set_map(region, map_override);
	for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
		if (constrain_avoidance_obstacles[i].is_valid()) {
			NavigationServer2D::get_singleton()->obstacle_set_map(constrain_avoidance_obstacles[i], map_override);
		}
	}
}

RID NavigationRegion2D::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (is_inside_tree()) {
		return get_world_2d()->get_navigation_map();
	}
	return RID();
}

void NavigationRegion2D::bake_navigation_polygon(bool p_on_thread) {
	ERR_FAIL_COND_MSG(!Thread::is_main_thread(), "The SceneTree can only be parsed on the main thread. Call this function from the main thread or use call_deferred().");
	ERR_FAIL_COND_MSG(!navigation_polygon.is_valid(), "Baking the navigation polygon requires a valid `NavigationPolygon` resource.");

	Ref<NavigationMeshSourceGeometryData2D> source_geometry_data;
	source_geometry_data.instantiate();

	NavigationServer2D::get_singleton()->parse_source_geometry_data(navigation_polygon, source_geometry_data, this);

	if (p_on_thread) {
		NavigationServer2D::get_singleton()->bake_from_source_geometry_data_async(navigation_polygon, source_geometry_data, callable_mp(this, &NavigationRegion2D::_bake_finished).bind(navigation_polygon));
	} else {
		NavigationServer2D::get_singleton()->bake_from_source_geometry_data(navigation_polygon, source_geometry_data, callable_mp(this, &NavigationRegion2D::_bake_finished).bind(navigation_polygon));
	}
}

void NavigationRegion2D::_bake_finished(Ref<NavigationPolygon> p_navigation_polygon) {
	if (!Thread::is_main_thread()) {
		call_deferred(SNAME("_bake_finished"), p_navigation_polygon);
		return;
	}

	set_navigation_polygon(p_navigation_polygon);
	emit_signal(SNAME("bake_finished"));
}

void NavigationRegion2D::_navigation_polygon_changed() {
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_navigation_hint())) {
		queue_redraw();
	}
	if (navigation_polygon.is_valid()) {
		NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, navigation_polygon);
	}
	_update_avoidance_constrain();
}

#ifdef DEBUG_ENABLED
void NavigationRegion2D::_navigation_map_changed(RID p_map) {
	if (is_inside_tree() && get_world_2d()->get_navigation_map() == p_map) {
		queue_redraw();
	}
}
#endif // DEBUG_ENABLED

PackedStringArray NavigationRegion2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!navigation_polygon.is_valid()) {
			warnings.push_back(RTR("A NavigationMesh resource must be set or created for this node to work. Please set a property or draw a polygon."));
		}
	}

	return warnings;
}

void NavigationRegion2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_polygon", "navigation_polygon"), &NavigationRegion2D::set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("get_navigation_polygon"), &NavigationRegion2D::get_navigation_polygon);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationRegion2D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationRegion2D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationRegion2D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationRegion2D::get_navigation_map);

	ClassDB::bind_method(D_METHOD("set_use_edge_connections", "enabled"), &NavigationRegion2D::set_use_edge_connections);
	ClassDB::bind_method(D_METHOD("get_use_edge_connections"), &NavigationRegion2D::get_use_edge_connections);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationRegion2D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationRegion2D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationRegion2D::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationRegion2D::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("set_constrain_avoidance", "enabled"), &NavigationRegion2D::set_constrain_avoidance);
	ClassDB::bind_method(D_METHOD("get_constrain_avoidance"), &NavigationRegion2D::get_constrain_avoidance);
	ClassDB::bind_method(D_METHOD("set_avoidance_layers", "layers"), &NavigationRegion2D::set_avoidance_layers);
	ClassDB::bind_method(D_METHOD("get_avoidance_layers"), &NavigationRegion2D::get_avoidance_layers);
	ClassDB::bind_method(D_METHOD("set_avoidance_layer_value", "layer_number", "value"), &NavigationRegion2D::set_avoidance_layer_value);
	ClassDB::bind_method(D_METHOD("get_avoidance_layer_value", "layer_number"), &NavigationRegion2D::get_avoidance_layer_value);

	ClassDB::bind_method(D_METHOD("get_region_rid"), &NavigationRegion2D::get_region_rid);

	ClassDB::bind_method(D_METHOD("set_enter_cost", "enter_cost"), &NavigationRegion2D::set_enter_cost);
	ClassDB::bind_method(D_METHOD("get_enter_cost"), &NavigationRegion2D::get_enter_cost);

	ClassDB::bind_method(D_METHOD("set_travel_cost", "travel_cost"), &NavigationRegion2D::set_travel_cost);
	ClassDB::bind_method(D_METHOD("get_travel_cost"), &NavigationRegion2D::get_travel_cost);

	ClassDB::bind_method(D_METHOD("bake_navigation_polygon", "on_thread"), &NavigationRegion2D::bake_navigation_polygon, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("_navigation_polygon_changed"), &NavigationRegion2D::_navigation_polygon_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_polygon", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"), "set_navigation_polygon", "get_navigation_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_edge_connections"), "set_use_edge_connections", "get_use_edge_connections");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_2D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "enter_cost"), "set_enter_cost", "get_enter_cost");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "travel_cost"), "set_travel_cost", "get_travel_cost");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constrain_avoidance"), "set_constrain_avoidance", "get_constrain_avoidance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "avoidance_layers", PROPERTY_HINT_LAYERS_AVOIDANCE), "set_avoidance_layers", "get_avoidance_layers");

	ADD_SIGNAL(MethodInfo("navigation_polygon_changed"));
	ADD_SIGNAL(MethodInfo("bake_finished"));
}

#ifndef DISABLE_DEPRECATED
// Compatibility with earlier 4.0 betas.
bool NavigationRegion2D::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "navpoly") {
		set_navigation_polygon(p_value);
		return true;
	}
	return false;
}

bool NavigationRegion2D::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "navpoly") {
		r_ret = get_navigation_polygon();
		return true;
	}
	return false;
}
#endif // DISABLE_DEPRECATED

NavigationRegion2D::NavigationRegion2D() {
	set_notify_transform(true);
	set_hide_clip_children(true);

	region = NavigationServer2D::get_singleton()->region_create();
	NavigationServer2D::get_singleton()->region_set_owner_id(region, get_instance_id());
	NavigationServer2D::get_singleton()->region_set_enter_cost(region, get_enter_cost());
	NavigationServer2D::get_singleton()->region_set_travel_cost(region, get_travel_cost());

#ifdef DEBUG_ENABLED
	NavigationServer2D::get_singleton()->connect(SNAME("map_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
	NavigationServer2D::get_singleton()->connect(SNAME("navigation_debug_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
#endif // DEBUG_ENABLED
}

NavigationRegion2D::~NavigationRegion2D() {
	ERR_FAIL_NULL(NavigationServer2D::get_singleton());
	NavigationServer2D::get_singleton()->free(region);

	for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
		if (constrain_avoidance_obstacles[i].is_valid()) {
			NavigationServer2D::get_singleton()->free(constrain_avoidance_obstacles[i]);
		}
	}
	constrain_avoidance_obstacles.clear();

#ifdef DEBUG_ENABLED
	NavigationServer2D::get_singleton()->disconnect(SNAME("map_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
	NavigationServer2D::get_singleton()->disconnect(SNAME("navigation_debug_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
#endif // DEBUG_ENABLED
}

void NavigationRegion2D::_update_avoidance_constrain() {
	for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
		if (constrain_avoidance_obstacles[i].is_valid()) {
			NavigationServer2D::get_singleton()->free(constrain_avoidance_obstacles[i]);
			constrain_avoidance_obstacles[i] = RID();
		}
	}
	constrain_avoidance_obstacles.clear();

	if (!constrain_avoidance) {
		return;
	}

	if (get_navigation_polygon() == nullptr) {
		return;
	}

	Ref<NavigationPolygon> _navpoly = get_navigation_polygon();
	int _outline_count = _navpoly->get_outline_count();
	if (_outline_count == 0) {
		return;
	}

	for (int outline_index(0); outline_index < _outline_count; outline_index++) {
		const Vector<Vector2> &_outline = _navpoly->get_outline(outline_index);

		const int outline_size = _outline.size();
		if (outline_size < 3) {
			ERR_FAIL_COND_MSG(_outline.size() < 3, "NavigationPolygon outline needs to have at least 3 vertex to create avoidance obstacles to constrain avoidance agent's");
			continue;
		}

		RID obstacle_rid = NavigationServer2D::get_singleton()->obstacle_create();
		constrain_avoidance_obstacles.push_back(obstacle_rid);

		Vector<Vector2> new_obstacle_outline;

		if (outline_index == 0) {
			for (int i(0); i < outline_size; i++) {
				new_obstacle_outline.push_back(_outline[outline_size - i - 1]);
			}
			ERR_FAIL_COND_MSG(Geometry2D::is_polygon_clockwise(_outline), "Outer most outline needs to be clockwise to push avoidance agent inside");
		} else {
			for (int i(0); i < outline_size; i++) {
				new_obstacle_outline.push_back(_outline[i]);
			}
		}
		new_obstacle_outline.resize(outline_size);

		NavigationServer2D::get_singleton()->obstacle_set_vertices(obstacle_rid, new_obstacle_outline);
		NavigationServer2D::get_singleton()->obstacle_set_avoidance_layers(obstacle_rid, avoidance_layers);
		if (is_inside_tree()) {
			if (map_override.is_valid()) {
				NavigationServer2D::get_singleton()->obstacle_set_map(obstacle_rid, map_override);
			} else {
				NavigationServer2D::get_singleton()->obstacle_set_map(obstacle_rid, get_world_2d()->get_navigation_map());
			}
			NavigationServer2D::get_singleton()->obstacle_set_position(obstacle_rid, get_global_position());
		}
	}
	constrain_avoidance_obstacles.resize(_outline_count);
}

void NavigationRegion2D::set_constrain_avoidance(bool p_enabled) {
	constrain_avoidance = p_enabled;
	_update_avoidance_constrain();
	notify_property_list_changed();
}

bool NavigationRegion2D::get_constrain_avoidance() const {
	return constrain_avoidance;
}

void NavigationRegion2D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "avoidance_layers") {
		if (!constrain_avoidance) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}
}

void NavigationRegion2D::set_avoidance_layers(uint32_t p_layers) {
	avoidance_layers = p_layers;
	if (constrain_avoidance_obstacles.size() > 0) {
		for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
			NavigationServer2D::get_singleton()->obstacle_set_avoidance_layers(constrain_avoidance_obstacles[i], avoidance_layers);
		}
	}
}

uint32_t NavigationRegion2D::get_avoidance_layers() const {
	return avoidance_layers;
}

void NavigationRegion2D::set_avoidance_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Avoidance layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Avoidance layer number must be between 1 and 32 inclusive.");
	uint32_t avoidance_layers_new = get_avoidance_layers();
	if (p_value) {
		avoidance_layers_new |= 1 << (p_layer_number - 1);
	} else {
		avoidance_layers_new &= ~(1 << (p_layer_number - 1));
	}
	set_avoidance_layers(avoidance_layers_new);
}

bool NavigationRegion2D::get_avoidance_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Avoidance layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Avoidance layer number must be between 1 and 32 inclusive.");
	return get_avoidance_layers() & (1 << (p_layer_number - 1));
}

void NavigationRegion2D::_region_enter_navigation_map() {
	if (!is_inside_tree()) {
		return;
	}

	if (map_override.is_valid()) {
		NavigationServer2D::get_singleton()->region_set_map(region, map_override);
		for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
			if (constrain_avoidance_obstacles[i].is_valid()) {
				NavigationServer2D::get_singleton()->obstacle_set_map(constrain_avoidance_obstacles[i], map_override);
			}
		}
	} else {
		NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
		for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
			if (constrain_avoidance_obstacles[i].is_valid()) {
				NavigationServer2D::get_singleton()->obstacle_set_map(constrain_avoidance_obstacles[i], get_world_2d()->get_navigation_map());
			}
		}
	}

	current_global_transform = get_global_transform();
	NavigationServer2D::get_singleton()->region_set_transform(region, current_global_transform);
	for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
		if (constrain_avoidance_obstacles[i].is_valid()) {
			NavigationServer2D::get_singleton()->obstacle_set_position(constrain_avoidance_obstacles[i], get_global_position());
		}
	}

	NavigationServer2D::get_singleton()->region_set_enabled(region, enabled);

	queue_redraw();
}

void NavigationRegion2D::_region_exit_navigation_map() {
	NavigationServer2D::get_singleton()->region_set_map(region, RID());
	for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
		if (constrain_avoidance_obstacles[i].is_valid()) {
			NavigationServer2D::get_singleton()->obstacle_set_map(constrain_avoidance_obstacles[i], RID());
		}
	}
}

void NavigationRegion2D::_region_update_transform() {
	if (!is_inside_tree()) {
		return;
	}

	Transform2D new_global_transform = get_global_transform();
	if (current_global_transform != new_global_transform) {
		current_global_transform = new_global_transform;
		NavigationServer2D::get_singleton()->region_set_transform(region, current_global_transform);
		for (uint32_t i = 0; i < constrain_avoidance_obstacles.size(); i++) {
			if (constrain_avoidance_obstacles[i].is_valid()) {
				NavigationServer2D::get_singleton()->obstacle_set_position(constrain_avoidance_obstacles[i], get_global_position());
			}
		}
	}

	queue_redraw();
}

#ifdef DEBUG_ENABLED
void NavigationRegion2D::_update_debug_mesh() {
	Vector<Vector2> navigation_polygon_vertices = navigation_polygon->get_vertices();
	if (navigation_polygon_vertices.size() < 3) {
		return;
	}

	const NavigationServer2D *ns2d = NavigationServer2D::get_singleton();

	bool enabled_geometry_face_random_color = ns2d->get_debug_navigation_enable_geometry_face_random_color();
	bool enabled_edge_lines = ns2d->get_debug_navigation_enable_edge_lines();

	Color debug_face_color = ns2d->get_debug_navigation_geometry_face_color();
	Color debug_edge_color = ns2d->get_debug_navigation_geometry_edge_color();

	if (!enabled) {
		debug_face_color = ns2d->get_debug_navigation_geometry_face_disabled_color();
		debug_edge_color = ns2d->get_debug_navigation_geometry_edge_disabled_color();
	}

	RandomPCG rand;

	for (int i = 0; i < navigation_polygon->get_polygon_count(); i++) {
		// An array of vertices for this polygon.
		Vector<int> polygon = navigation_polygon->get_polygon(i);
		Vector<Vector2> debug_polygon_vertices;
		debug_polygon_vertices.resize(polygon.size());
		for (int j = 0; j < polygon.size(); j++) {
			ERR_FAIL_INDEX(polygon[j], navigation_polygon_vertices.size());
			debug_polygon_vertices.write[j] = navigation_polygon_vertices[polygon[j]];
		}

		// Generate the polygon color, slightly randomly modified from the settings one.
		Color random_variation_color = debug_face_color;
		if (enabled_geometry_face_random_color) {
			random_variation_color.set_hsv(
					debug_face_color.get_h() + rand.random(-1.0, 1.0) * 0.1,
					debug_face_color.get_s(),
					debug_face_color.get_v() + rand.random(-1.0, 1.0) * 0.2);
		}
		random_variation_color.a = debug_face_color.a;

		Vector<Color> debug_face_colors;
		debug_face_colors.push_back(random_variation_color);
		RS::get_singleton()->canvas_item_add_polygon(get_canvas_item(), debug_polygon_vertices, debug_face_colors);

		if (enabled_edge_lines) {
			Vector<Color> debug_edge_colors;
			debug_edge_colors.push_back(debug_edge_color);
			debug_polygon_vertices.push_back(debug_polygon_vertices[0]); // Add first again for closing polyline.
			RS::get_singleton()->canvas_item_add_polyline(get_canvas_item(), debug_polygon_vertices, debug_edge_colors);
		}
	}
}
#endif // DEBUG_ENABLED

#ifdef DEBUG_ENABLED
void NavigationRegion2D::_update_debug_edge_connections_mesh() {
	const NavigationServer2D *ns2d = NavigationServer2D::get_singleton();
	bool enable_edge_connections = use_edge_connections && ns2d->get_debug_navigation_enable_edge_connections() && ns2d->map_get_use_edge_connections(get_world_2d()->get_navigation_map());

	if (enable_edge_connections) {
		Color debug_edge_connection_color = ns2d->get_debug_navigation_edge_connection_color();
		// Draw the region edge connections.
		Transform2D xform = get_global_transform();
		real_t radius = ns2d->map_get_edge_connection_margin(get_world_2d()->get_navigation_map()) / 2.0;
		for (int i = 0; i < ns2d->region_get_connections_count(region); i++) {
			// Two main points
			Vector2 a = ns2d->region_get_connection_pathway_start(region, i);
			a = xform.affine_inverse().xform(a);
			Vector2 b = ns2d->region_get_connection_pathway_end(region, i);
			b = xform.affine_inverse().xform(b);
			draw_line(a, b, debug_edge_connection_color);

			// Draw a circle to illustrate the margins.
			real_t angle = a.angle_to_point(b);
			draw_arc(a, radius, angle + Math_PI / 2.0, angle - Math_PI / 2.0 + Math_TAU, 10, debug_edge_connection_color);
			draw_arc(b, radius, angle - Math_PI / 2.0, angle + Math_PI / 2.0, 10, debug_edge_connection_color);
		}
	}
}
#endif // DEBUG_ENABLED
