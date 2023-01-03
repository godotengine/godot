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

#include "core/core_string_names.h"
#include "scene/resources/world_2d.h"
#include "servers/navigation/navigation_mesh_generator.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"

void NavigationRegion2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_region_rid"), &NavigationRegion2D::get_region_rid);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationRegion2D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationRegion2D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_navigation_polygon", "navigation_polygon"), &NavigationRegion2D::set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("get_navigation_polygon"), &NavigationRegion2D::get_navigation_polygon);

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationRegion2D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationRegion2D::get_navigation_map);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationRegion2D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationRegion2D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationRegion2D::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationRegion2D::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("set_enter_cost", "enter_cost"), &NavigationRegion2D::set_enter_cost);
	ClassDB::bind_method(D_METHOD("get_enter_cost"), &NavigationRegion2D::get_enter_cost);

	ClassDB::bind_method(D_METHOD("set_travel_cost", "travel_cost"), &NavigationRegion2D::set_travel_cost);
	ClassDB::bind_method(D_METHOD("get_travel_cost"), &NavigationRegion2D::get_travel_cost);

	ClassDB::bind_method(D_METHOD("bake_navigation_polygon", "on_thread"), &NavigationRegion2D::bake_navigation_polygon, DEFVAL(true));

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_polygon", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"), "set_navigation_polygon", "get_navigation_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_2D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "enter_cost"), "set_enter_cost", "get_enter_cost");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "travel_cost"), "set_travel_cost", "get_travel_cost");

	ADD_SIGNAL(MethodInfo("navigation_polygon_changed"));
	ADD_SIGNAL(MethodInfo("bake_finished"));
}

void NavigationRegion2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (enabled) {
				if (map_override.is_valid()) {
					NavigationServer2D::get_singleton()->region_set_map(region, map_override);
				} else {
					NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
				}
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			NavigationServer2D::get_singleton()->region_set_transform(region, get_global_transform());
		} break;

		case NOTIFICATION_EXIT_TREE: {
			NavigationServer2D::get_singleton()->region_set_map(region, RID());
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

RID NavigationRegion2D::get_region_rid() const {
	return region;
}

void NavigationRegion2D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}

	enabled = p_enabled;

	if (!is_inside_tree()) {
		return;
	}

	if (!enabled) {
		NavigationServer2D::get_singleton()->region_set_map(region, RID());
	} else {
		if (map_override.is_valid()) {
			NavigationServer2D::get_singleton()->region_set_map(region, map_override);
		} else {
			NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
		}
	}

#ifdef DEBUG_ENABLED
	if (Engine::get_singleton()->is_editor_hint() || NavigationServer2D::get_singleton()->get_debug_enabled()) {
		queue_redraw();
	}
#endif // DEBUG_ENABLED
}

bool NavigationRegion2D::is_enabled() const {
	return enabled;
}

void NavigationRegion2D::set_navigation_polygon(const Ref<NavigationPolygon> &p_navigation_polygon) {
	if (navigation_polygon.is_valid()) {
		navigation_polygon->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}

	navigation_polygon = p_navigation_polygon;

	if (navigation_polygon.is_valid()) {
		navigation_polygon->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}

	_navigation_polygon_changed();
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
}

RID NavigationRegion2D::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (is_inside_tree()) {
		return get_world_2d()->get_navigation_map();
	}
	return RID();
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

void NavigationRegion2D::bake_navigation_polygon(bool p_on_thread) {
	ERR_FAIL_COND_MSG(!get_navigation_polygon().is_valid(), "Can't bake the navigation polygon if the `NavigationPolygon` resource doesn't exist");
	if (baking_started) {
		WARN_PRINT("NavigationPolygon baking already started. Wait for it to finish.");
		return;
	}
	baking_started = true;

	navigation_polygon->clear();

	if (p_on_thread && OS::get_singleton()->get_render_thread_mode() == OS::RenderThreadMode::RENDER_SEPARATE_THREAD) {
		NavigationMeshGenerator::get_singleton()->parse_and_bake_2d(navigation_polygon, this, Callable(this, "_bake_finished"));
	} else {
		Ref<NavigationMeshSourceGeometryData2D> source_geometry_data = NavigationMeshGenerator::get_singleton()->parse_2d_source_geometry_data(navigation_polygon, this);
		NavigationMeshGenerator::get_singleton()->bake_2d_from_source_geometry_data(navigation_polygon, source_geometry_data);
		_bake_finished();
	}
}

void NavigationRegion2D::_bake_finished() {
	baking_started = false;
	set_navigation_polygon(navigation_polygon);
	emit_signal(SNAME("bake_finished"));
}

PackedStringArray NavigationRegion2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!navigation_polygon.is_valid()) {
			warnings.push_back(RTR("A NavigationMesh resource must be set or created for this node to work. Please set a property or draw a polygon."));
		}
	}

	return warnings;
}

void NavigationRegion2D::_navigation_polygon_changed() {
	NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, navigation_polygon);

	update_configuration_warnings();

	emit_signal(SNAME("navigation_polygon_changed"));

#ifdef DEBUG_ENABLED
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_navigation_hint())) {
		queue_redraw();
	}
#endif // DEBUG_ENABLED
}

#ifdef DEBUG_ENABLED
void NavigationRegion2D::_navigation_map_changed(RID p_map) {
	if (is_inside_tree() && get_world_2d()->get_navigation_map() == p_map) {
		queue_redraw();
	}
}
#endif // DEBUG_ENABLED

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

#ifdef DEBUG_ENABLED
	NavigationServer2D::get_singleton()->disconnect(SNAME("map_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
	NavigationServer2D::get_singleton()->disconnect(SNAME("navigation_debug_changed"), callable_mp(this, &NavigationRegion2D::_navigation_map_changed));
#endif // DEBUG_ENABLED
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
	bool enable_edge_connections = ns2d->get_debug_navigation_enable_edge_connections();
	Color debug_edge_connection_color = ns2d->get_debug_navigation_edge_connection_color();

	if (enable_edge_connections) {
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

#ifdef TOOLS_ENABLED
Rect2 NavigationRegion2D::_edit_get_rect() const {
	return navigation_polygon.is_valid() ? navigation_polygon->_edit_get_rect() : Rect2();
}

bool NavigationRegion2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return navigation_polygon.is_valid() ? navigation_polygon->_edit_is_selected_on_click(p_point, p_tolerance) : false;
}
#endif
