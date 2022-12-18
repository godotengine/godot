/*************************************************************************/
/*  navigation_region_2d.cpp                                             */
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

#include "navigation_region_2d.h"

#include "core/core_string_names.h"
#include "scene/resources/world_2d.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"

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
		NavigationServer2D::get_singleton_mut()->disconnect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
	} else {
		NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
		NavigationServer2D::get_singleton_mut()->connect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
	}

#ifdef DEBUG_ENABLED
	if (Engine::get_singleton()->is_editor_hint() || NavigationServer3D::get_singleton()->get_debug_enabled()) {
		queue_redraw();
	}
#endif // DEBUG_ENABLED
}

bool NavigationRegion2D::is_enabled() const {
	return enabled;
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
			if (enabled) {
				NavigationServer2D::get_singleton()->region_set_map(region, get_world_2d()->get_navigation_map());
				NavigationServer2D::get_singleton_mut()->connect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			NavigationServer2D::get_singleton()->region_set_transform(region, get_global_transform());
		} break;

		case NOTIFICATION_EXIT_TREE: {
			NavigationServer2D::get_singleton()->region_set_map(region, RID());
			if (enabled) {
				NavigationServer2D::get_singleton_mut()->disconnect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
			}
		} break;

		case NOTIFICATION_DRAW: {
#ifdef DEBUG_ENABLED
			if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || NavigationServer3D::get_singleton()->get_debug_enabled()) && navigation_polygon.is_valid()) {
				Vector<Vector2> verts = navigation_polygon->get_vertices();
				if (verts.size() < 3) {
					return;
				}

				Color color;
				if (enabled) {
					color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_color();
				} else {
					color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_disabled_color();
				}
				Color doors_color = NavigationServer3D::get_singleton()->get_debug_navigation_edge_connection_color();

				RandomPCG rand;

				for (int i = 0; i < navigation_polygon->get_polygon_count(); i++) {
					// An array of vertices for this polygon.
					Vector<int> polygon = navigation_polygon->get_polygon(i);
					Vector<Vector2> vertices;
					vertices.resize(polygon.size());
					for (int j = 0; j < polygon.size(); j++) {
						ERR_FAIL_INDEX(polygon[j], verts.size());
						vertices.write[j] = verts[polygon[j]];
					}

					// Generate the polygon color, slightly randomly modified from the settings one.
					Color random_variation_color;
					random_variation_color.set_hsv(color.get_h() + rand.random(-1.0, 1.0) * 0.1, color.get_s(), color.get_v() + rand.random(-1.0, 1.0) * 0.2);
					random_variation_color.a = color.a;
					Vector<Color> colors;
					colors.push_back(random_variation_color);

					RS::get_singleton()->canvas_item_add_polygon(get_canvas_item(), vertices, colors);
				}

				// Draw the region
				Transform2D xform = get_global_transform();
				const NavigationServer2D *ns = NavigationServer2D::get_singleton();
				real_t radius = ns->map_get_edge_connection_margin(get_world_2d()->get_navigation_map()) / 2.0;
				for (int i = 0; i < ns->region_get_connections_count(region); i++) {
					// Two main points
					Vector2 a = ns->region_get_connection_pathway_start(region, i);
					a = xform.affine_inverse().xform(a);
					Vector2 b = ns->region_get_connection_pathway_end(region, i);
					b = xform.affine_inverse().xform(b);
					draw_line(a, b, doors_color);

					// Draw a circle to illustrate the margins.
					real_t angle = a.angle_to_point(b);
					draw_arc(a, radius, angle + Math_PI / 2.0, angle - Math_PI / 2.0 + Math_TAU, 10, doors_color);
					draw_arc(b, radius, angle - Math_PI / 2.0, angle + Math_PI / 2.0, 10, doors_color);
				}
			}
#endif // DEBUG_ENABLED
		} break;
	}
}

void NavigationRegion2D::set_navigation_polygon(const Ref<NavigationPolygon> &p_navigation_polygon) {
	if (p_navigation_polygon == navigation_polygon) {
		return;
	}

	if (navigation_polygon.is_valid()) {
		navigation_polygon->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}

	navigation_polygon = p_navigation_polygon;
	NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, p_navigation_polygon);

	if (navigation_polygon.is_valid()) {
		navigation_polygon->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NavigationRegion2D::_navigation_polygon_changed));
	}
	_navigation_polygon_changed();

	update_configuration_warnings();
}

Ref<NavigationPolygon> NavigationRegion2D::get_navigation_polygon() const {
	return navigation_polygon;
}

void NavigationRegion2D::_navigation_polygon_changed() {
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_navigation_hint())) {
		queue_redraw();
	}
	if (navigation_polygon.is_valid()) {
		NavigationServer2D::get_singleton()->region_set_navigation_polygon(region, navigation_polygon);
	}
}

void NavigationRegion2D::_map_changed(RID p_map) {
#ifdef DEBUG_ENABLED
	if (is_inside_tree() && get_world_2d()->get_navigation_map() == p_map) {
		queue_redraw();
	}
#endif // DEBUG_ENABLED
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

void NavigationRegion2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_polygon", "navigation_polygon"), &NavigationRegion2D::set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("get_navigation_polygon"), &NavigationRegion2D::get_navigation_polygon);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationRegion2D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationRegion2D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationRegion2D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationRegion2D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationRegion2D::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationRegion2D::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("get_region_rid"), &NavigationRegion2D::get_region_rid);

	ClassDB::bind_method(D_METHOD("set_enter_cost", "enter_cost"), &NavigationRegion2D::set_enter_cost);
	ClassDB::bind_method(D_METHOD("get_enter_cost"), &NavigationRegion2D::get_enter_cost);

	ClassDB::bind_method(D_METHOD("set_travel_cost", "travel_cost"), &NavigationRegion2D::set_travel_cost);
	ClassDB::bind_method(D_METHOD("get_travel_cost"), &NavigationRegion2D::get_travel_cost);

	ClassDB::bind_method(D_METHOD("_navigation_polygon_changed"), &NavigationRegion2D::_navigation_polygon_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_polygon", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"), "set_navigation_polygon", "get_navigation_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_2D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "enter_cost"), "set_enter_cost", "get_enter_cost");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "travel_cost"), "set_travel_cost", "get_travel_cost");
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

	region = NavigationServer2D::get_singleton()->region_create();
	NavigationServer2D::get_singleton()->region_set_owner_id(region, get_instance_id());
	NavigationServer2D::get_singleton()->region_set_enter_cost(region, get_enter_cost());
	NavigationServer2D::get_singleton()->region_set_travel_cost(region, get_travel_cost());

#ifdef DEBUG_ENABLED
	NavigationServer3D::get_singleton_mut()->connect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
	NavigationServer3D::get_singleton_mut()->connect("navigation_debug_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
#endif // DEBUG_ENABLED
}

NavigationRegion2D::~NavigationRegion2D() {
	NavigationServer2D::get_singleton()->free(region);

#ifdef DEBUG_ENABLED
	NavigationServer3D::get_singleton_mut()->disconnect("map_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
	NavigationServer3D::get_singleton_mut()->disconnect("navigation_debug_changed", callable_mp(this, &NavigationRegion2D::_map_changed));
#endif // DEBUG_ENABLED
}
