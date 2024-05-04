/**************************************************************************/
/*  navigation_obstacle_2d.cpp                                            */
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

#include "navigation_obstacle_2d.h"

#include "core/math/geometry_2d.h"
#include "scene/resources/world_2d.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"

void NavigationObstacle2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationObstacle2D::get_rid);

	ClassDB::bind_method(D_METHOD("set_avoidance_enabled", "enabled"), &NavigationObstacle2D::set_avoidance_enabled);
	ClassDB::bind_method(D_METHOD("get_avoidance_enabled"), &NavigationObstacle2D::get_avoidance_enabled);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationObstacle2D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationObstacle2D::get_navigation_map);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationObstacle2D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationObstacle2D::get_radius);

	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &NavigationObstacle2D::set_velocity);
	ClassDB::bind_method(D_METHOD("get_velocity"), &NavigationObstacle2D::get_velocity);

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationObstacle2D::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationObstacle2D::get_vertices);

	ClassDB::bind_method(D_METHOD("set_avoidance_layers", "layers"), &NavigationObstacle2D::set_avoidance_layers);
	ClassDB::bind_method(D_METHOD("get_avoidance_layers"), &NavigationObstacle2D::get_avoidance_layers);

	ClassDB::bind_method(D_METHOD("set_avoidance_layer_value", "layer_number", "value"), &NavigationObstacle2D::set_avoidance_layer_value);
	ClassDB::bind_method(D_METHOD("get_avoidance_layer_value", "layer_number"), &NavigationObstacle2D::get_avoidance_layer_value);

	ClassDB::bind_method(D_METHOD("set_affect_navigation_mesh", "enabled"), &NavigationObstacle2D::set_affect_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_affect_navigation_mesh"), &NavigationObstacle2D::get_affect_navigation_mesh);

	ClassDB::bind_method(D_METHOD("set_carve_navigation_mesh", "enabled"), &NavigationObstacle2D::set_carve_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_carve_navigation_mesh"), &NavigationObstacle2D::get_carve_navigation_mesh);

	ClassDB::bind_method(D_METHOD("set_avoidance_space", "avoidance_space"), &NavigationObstacle2D::set_avoidance_space);
	ClassDB::bind_method(D_METHOD("get_avoidance_space"), &NavigationObstacle2D::get_avoidance_space);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.0,500,0.01,suffix:px"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "vertices"), "set_vertices", "get_vertices");
	ADD_GROUP("NavigationMesh", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "affect_navigation_mesh"), "set_affect_navigation_mesh", "get_affect_navigation_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "carve_navigation_mesh"), "set_carve_navigation_mesh", "get_carve_navigation_mesh");
	ADD_GROUP("Avoidance", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "avoidance_enabled"), "set_avoidance_enabled", "get_avoidance_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "velocity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_velocity", "get_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "avoidance_layers", PROPERTY_HINT_LAYERS_AVOIDANCE), "set_avoidance_layers", "get_avoidance_layers");
}

void NavigationObstacle2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			_obstacle_enter_tree();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_obstacle_exit_tree();
		} break;

		case NOTIFICATION_PAUSED:
		case NOTIFICATION_UNPAUSED: {
			NavigationServer2D::get_singleton()->obstacle_set_paused(obstacle, !can_process());
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
#ifdef DEBUG_ENABLED
			if (debug_canvas_item.is_valid()) {
				RS::get_singleton()->canvas_item_set_visible(debug_canvas_item, is_visible_in_tree());
			}
#endif // DEBUG_ENABLED
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			_obstacle_physics_process();
		} break;

		case NOTIFICATION_DRAW: {
#ifdef DEBUG_ENABLED
			_obstacle_debug_update();
#endif // DEBUG_ENABLED
		} break;
	}
}

NavigationObstacle2D::NavigationObstacle2D() {
	NavigationServer2D *ns2d = NavigationServer2D::get_singleton();

	obstacle = ns2d->obstacle_create();

	ns2d->obstacle_set_radius(obstacle, radius);
	ns2d->obstacle_set_vertices(obstacle, vertices);
	ns2d->obstacle_set_avoidance_layers(obstacle, avoidance_layers);
	ns2d->obstacle_set_avoidance_enabled(obstacle, avoidance_enabled);

#ifdef DEBUG_ENABLED
	ns2d->connect("avoidance_debug_changed", callable_mp((CanvasItem *)this, &NavigationObstacle2D::queue_redraw));
#endif // DEBUG_ENABLED
}

NavigationObstacle2D::~NavigationObstacle2D() {
	NavigationServer2D *ns2d = NavigationServer2D::get_singleton();
	ERR_FAIL_NULL(ns2d);

	ns2d->free(obstacle);
	obstacle = RID();

#ifdef DEBUG_ENABLED
	ns2d->disconnect("avoidance_debug_changed", callable_mp((CanvasItem *)this, &NavigationObstacle2D::queue_redraw));
	_obstacle_debug_free();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::set_vertices(const Vector<Vector2> &p_vertices) {
	vertices = p_vertices;
	NavigationServer2D::get_singleton()->obstacle_set_vertices(obstacle, vertices);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::set_avoidance_space(RID p_avoidance_space) {
	if (avoidance_space_override == p_avoidance_space) {
		return;
	}

	avoidance_space_override = p_avoidance_space;

	NavigationServer2D::get_singleton()->obstacle_set_avoidance_space(obstacle, get_avoidance_space());
}

RID NavigationObstacle2D::get_avoidance_space() const {
	if (!avoidance_enabled) {
		return RID();
	}
	if (avoidance_space_override.is_valid()) {
		return avoidance_space_override;
	} else if (is_inside_tree()) {
		return get_world_2d()->get_avoidance_space();
	}
	return RID();
}

void NavigationObstacle2D::set_radius(real_t p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0.0, "Radius must be positive.");
	if (Math::is_equal_approx(radius, p_radius)) {
		return;
	}

	radius = p_radius;

	NavigationServer2D::get_singleton()->obstacle_set_radius(obstacle, radius);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::set_avoidance_layers(uint32_t p_layers) {
	if (avoidance_layers == p_layers) {
		return;
	}
	avoidance_layers = p_layers;
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_layers(obstacle, avoidance_layers);
}

uint32_t NavigationObstacle2D::get_avoidance_layers() const {
	return avoidance_layers;
}

void NavigationObstacle2D::set_avoidance_layer_value(int p_layer_number, bool p_value) {
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

bool NavigationObstacle2D::get_avoidance_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Avoidance layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Avoidance layer number must be between 1 and 32 inclusive.");
	return get_avoidance_layers() & (1 << (p_layer_number - 1));
}

void NavigationObstacle2D::set_avoidance_enabled(bool p_enabled) {
	if (avoidance_enabled == p_enabled) {
		return;
	}

	avoidance_enabled = p_enabled;
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_enabled(obstacle, avoidance_enabled);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

bool NavigationObstacle2D::get_avoidance_enabled() const {
	return avoidance_enabled;
}

void NavigationObstacle2D::set_velocity(const Vector2 p_velocity) {
	velocity = p_velocity;
	velocity_submitted = true;
}

void NavigationObstacle2D::set_affect_navigation_mesh(bool p_enabled) {
	affect_navigation_mesh = p_enabled;
}

bool NavigationObstacle2D::get_affect_navigation_mesh() const {
	return affect_navigation_mesh;
}

void NavigationObstacle2D::set_carve_navigation_mesh(bool p_enabled) {
	carve_navigation_mesh = p_enabled;
}

bool NavigationObstacle2D::get_carve_navigation_mesh() const {
	return carve_navigation_mesh;
}

void NavigationObstacle2D::_update_position(const Vector2 p_position) {
	NavigationServer2D::get_singleton()->obstacle_set_position(obstacle, p_position);
#ifdef DEBUG_ENABLED
	if (debug_canvas_item.is_valid()) {
		Transform2D debug_transform = Transform2D(0.0, get_global_position());
		RenderingServer::get_singleton()->canvas_item_set_transform(debug_canvas_item, debug_transform);
	}
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::_update_avoidance_space(RID p_avoidance_space) {
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_space(obstacle, get_avoidance_space());
}

void NavigationObstacle2D::_obstacle_enter_tree() {
	if (!is_inside_tree()) {
		return;
	}

	previous_transform = get_global_transform();

	// Need to trigger map controlled agent assignment somehow for the fake_agent since obstacles use no callback like regular agents.
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_enabled(obstacle, avoidance_enabled);
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_space(obstacle, get_avoidance_space());

	_update_position(get_global_position());
	set_physics_process_internal(true);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::_obstacle_exit_tree() {
	set_physics_process_internal(false);
	_update_avoidance_space(RID());
#ifdef DEBUG_ENABLED
	_obstacle_debug_free();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::_obstacle_physics_process() {
	if (!is_inside_tree()) {
		return;
	}

	_update_position(get_global_position());

	if (velocity_submitted) {
		velocity_submitted = false;
		// Only update if there is a noticeable change, else the rvo agent preferred velocity stays the same.
		if (!previous_velocity.is_equal_approx(velocity)) {
			NavigationServer2D::get_singleton()->obstacle_set_velocity(obstacle, velocity);
		}
		previous_velocity = velocity;
	}

#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

#ifdef DEBUG_ENABLED
void NavigationObstacle2D::_obstacle_debug_update() {
	NavigationServer2D *ns2d = NavigationServer2D::get_singleton();
	bool is_debug_enabled = false;

	if (Engine::get_singleton()->is_editor_hint()) {
		is_debug_enabled = true;
	} else if (ns2d->get_debug_enabled() && ns2d->get_debug_avoidance_enabled()) {
		is_debug_enabled = true;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);

	if (debug_canvas_item.is_null()) {
		debug_canvas_item = rs->canvas_item_create();
	}

	rs->canvas_item_clear(debug_canvas_item);

	if (!is_debug_enabled) {
		if (debug_canvas_item.is_valid()) {
			rs->free(debug_canvas_item);
			debug_canvas_item = RID();
		}
		return;
	}

	if (!is_inside_tree()) {
		return;
	}

	rs->canvas_item_set_parent(debug_canvas_item, get_canvas());
	rs->canvas_item_set_z_index(debug_canvas_item, RS::CANVAS_ITEM_Z_MAX - 1);
	rs->canvas_item_set_visible(debug_canvas_item, is_visible_in_tree());

	Transform2D debug_transform = Transform2D(0.0, get_global_position());
	rs->canvas_item_set_transform(debug_canvas_item, debug_transform);

	if (radius > 0.0 && ns2d->get_debug_navigation_avoidance_enable_obstacles_radius()) {
		Color debug_radius_color = ns2d->get_debug_navigation_avoidance_obstacles_radius_color();

		RS::get_singleton()->canvas_item_add_circle(debug_canvas_item, Vector2(), radius, debug_radius_color);
	}

	if (get_vertices().size() > 2 && ns2d->get_debug_navigation_avoidance_enable_obstacles_static()) {
		bool obstacle_pushes_inward = Geometry2D::is_polygon_clockwise(get_vertices());

		Color debug_static_obstacle_face_color;

		if (obstacle_pushes_inward) {
			debug_static_obstacle_face_color = ns2d->get_debug_navigation_avoidance_static_obstacle_pushin_face_color();
		} else {
			debug_static_obstacle_face_color = ns2d->get_debug_navigation_avoidance_static_obstacle_pushout_face_color();
		}

		Vector<Vector2> debug_obstacle_polygon_vertices = get_vertices();

		Vector<Color> debug_obstacle_polygon_colors;
		debug_obstacle_polygon_colors.resize(debug_obstacle_polygon_vertices.size());
		debug_obstacle_polygon_colors.fill(debug_static_obstacle_face_color);

		RS::get_singleton()->canvas_item_add_polygon(debug_canvas_item, debug_obstacle_polygon_vertices, debug_obstacle_polygon_colors);

		Color debug_static_obstacle_edge_color;

		if (obstacle_pushes_inward) {
			debug_static_obstacle_edge_color = ns2d->get_debug_navigation_avoidance_static_obstacle_pushin_edge_color();
		} else {
			debug_static_obstacle_edge_color = ns2d->get_debug_navigation_avoidance_static_obstacle_pushout_edge_color();
		}

		Vector<Vector2> debug_obstacle_line_vertices = get_vertices();
		debug_obstacle_line_vertices.push_back(debug_obstacle_line_vertices[0]);
		debug_obstacle_line_vertices.resize(debug_obstacle_line_vertices.size());

		Vector<Color> debug_obstacle_line_colors;
		debug_obstacle_line_colors.resize(debug_obstacle_line_vertices.size());
		debug_obstacle_line_colors.fill(debug_static_obstacle_edge_color);

		RS::get_singleton()->canvas_item_add_polyline(debug_canvas_item, debug_obstacle_line_vertices, debug_obstacle_line_colors, 4.0);
	}
}
#endif // DEBUG_ENABLED

#ifdef DEBUG_ENABLED
void NavigationObstacle2D::_obstacle_debug_free() {
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);

	if (debug_canvas_item.is_valid()) {
		rs->free(debug_canvas_item);
		debug_canvas_item = RID();
	}
}
#endif // DEBUG_ENABLED

#ifndef DISABLE_DEPRECATED
void NavigationObstacle2D::set_navigation_map(RID p_navigation_map) {
	WARN_PRINT_ONCE("An Obstacle is no longer an assigned part of a navigation map. See 'set_avoidance_space()' to set the obstacle's avoidance space.");
};
RID NavigationObstacle2D::get_navigation_map() const {
	WARN_PRINT_ONCE("An Obstacle is no longer an assigned part of a navigation map. See 'get_avoidance_space()' to get the obstacle's avoidance space.");
	return RID();
};
#endif // DISABLE_DEPRECATED