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

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationObstacle2D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationObstacle2D::get_navigation_map);

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

	ADD_GROUP("Avoidance", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "avoidance_enabled"), "set_avoidance_enabled", "get_avoidance_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "velocity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_velocity", "get_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.0,500,0.01,suffix:px"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "vertices"), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "avoidance_layers", PROPERTY_HINT_LAYERS_AVOIDANCE), "set_avoidance_layers", "get_avoidance_layers");
}

void NavigationObstacle2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			if (map_override.is_valid()) {
				_update_map(map_override);
			} else if (is_inside_tree()) {
				_update_map(get_world_2d()->get_navigation_map());
			} else {
				_update_map(RID());
			}
			previous_transform = get_global_transform();
			// need to trigger map controlled agent assignment somehow for the fake_agent since obstacles use no callback like regular agents
			NavigationServer2D::get_singleton()->obstacle_set_avoidance_enabled(obstacle, avoidance_enabled);
			_update_position(get_global_position());
			set_physics_process_internal(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_physics_process_internal(false);
			_update_map(RID());
		} break;

		case NOTIFICATION_PAUSED: {
			if (!can_process()) {
				map_before_pause = map_current;
				_update_map(RID());
			} else if (can_process() && !(map_before_pause == RID())) {
				_update_map(map_before_pause);
				map_before_pause = RID();
			}
			NavigationServer2D::get_singleton()->obstacle_set_paused(obstacle, !can_process());
		} break;

		case NOTIFICATION_UNPAUSED: {
			if (!can_process()) {
				map_before_pause = map_current;
				_update_map(RID());
			} else if (can_process() && !(map_before_pause == RID())) {
				_update_map(map_before_pause);
				map_before_pause = RID();
			}
			NavigationServer2D::get_singleton()->obstacle_set_paused(obstacle, !can_process());
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (is_inside_tree()) {
				_update_position(get_global_position());

				if (velocity_submitted) {
					velocity_submitted = false;
					// only update if there is a noticeable change, else the rvo agent preferred velocity stays the same
					if (!previous_velocity.is_equal_approx(velocity)) {
						NavigationServer2D::get_singleton()->obstacle_set_velocity(obstacle, velocity);
					}
					previous_velocity = velocity;
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
#ifdef DEBUG_ENABLED
			if (is_inside_tree()) {
				bool is_debug_enabled = false;
				if (Engine::get_singleton()->is_editor_hint()) {
					is_debug_enabled = true;
				} else if (NavigationServer2D::get_singleton()->get_debug_enabled() && NavigationServer2D::get_singleton()->get_debug_avoidance_enabled()) {
					is_debug_enabled = true;
				}

				if (is_debug_enabled) {
					_update_fake_agent_radius_debug();
					_update_static_obstacle_debug();
				}
			}
#endif // DEBUG_ENABLED
		} break;
	}
}

NavigationObstacle2D::NavigationObstacle2D() {
	obstacle = NavigationServer2D::get_singleton()->obstacle_create();

	NavigationServer2D::get_singleton()->obstacle_set_radius(obstacle, radius);
	NavigationServer2D::get_singleton()->obstacle_set_vertices(obstacle, vertices);
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_layers(obstacle, avoidance_layers);
	NavigationServer2D::get_singleton()->obstacle_set_avoidance_enabled(obstacle, avoidance_enabled);
}

NavigationObstacle2D::~NavigationObstacle2D() {
	ERR_FAIL_NULL(NavigationServer2D::get_singleton());

	NavigationServer2D::get_singleton()->free(obstacle);
	obstacle = RID();
}

void NavigationObstacle2D::set_vertices(const Vector<Vector2> &p_vertices) {
	vertices = p_vertices;
	NavigationServer2D::get_singleton()->obstacle_set_vertices(obstacle, vertices);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

void NavigationObstacle2D::set_navigation_map(RID p_navigation_map) {
	if (map_override == p_navigation_map) {
		return;
	}
	map_override = p_navigation_map;
	_update_map(map_override);
}

RID NavigationObstacle2D::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (is_inside_tree()) {
		return get_world_2d()->get_navigation_map();
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

void NavigationObstacle2D::_update_map(RID p_map) {
	map_current = p_map;
	NavigationServer2D::get_singleton()->obstacle_set_map(obstacle, p_map);
}

void NavigationObstacle2D::_update_position(const Vector2 p_position) {
	NavigationServer2D::get_singleton()->obstacle_set_position(obstacle, p_position);
#ifdef DEBUG_ENABLED
	queue_redraw();
#endif // DEBUG_ENABLED
}

#ifdef DEBUG_ENABLED
void NavigationObstacle2D::_update_fake_agent_radius_debug() {
	if (radius > 0.0 && NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_enable_obstacles_radius()) {
		Color debug_radius_color = NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_obstacles_radius_color();
		RS::get_singleton()->canvas_item_add_circle(get_canvas_item(), Vector2(), radius, debug_radius_color);
	}
}
#endif // DEBUG_ENABLED

#ifdef DEBUG_ENABLED
void NavigationObstacle2D::_update_static_obstacle_debug() {
	if (get_vertices().size() > 2 && NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_enable_obstacles_static()) {
		bool obstacle_pushes_inward = Geometry2D::is_polygon_clockwise(get_vertices());

		Color debug_static_obstacle_face_color;

		if (obstacle_pushes_inward) {
			debug_static_obstacle_face_color = NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_static_obstacle_pushin_face_color();
		} else {
			debug_static_obstacle_face_color = NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_static_obstacle_pushout_face_color();
		}

		Vector<Vector2> debug_obstacle_polygon_vertices = get_vertices();

		Vector<Color> debug_obstacle_polygon_colors;
		debug_obstacle_polygon_colors.resize(debug_obstacle_polygon_vertices.size());
		debug_obstacle_polygon_colors.fill(debug_static_obstacle_face_color);

		RS::get_singleton()->canvas_item_add_polygon(get_canvas_item(), debug_obstacle_polygon_vertices, debug_obstacle_polygon_colors);

		Color debug_static_obstacle_edge_color;

		if (obstacle_pushes_inward) {
			debug_static_obstacle_edge_color = NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_static_obstacle_pushin_edge_color();
		} else {
			debug_static_obstacle_edge_color = NavigationServer2D::get_singleton()->get_debug_navigation_avoidance_static_obstacle_pushout_edge_color();
		}

		Vector<Vector2> debug_obstacle_line_vertices = get_vertices();
		debug_obstacle_line_vertices.push_back(debug_obstacle_line_vertices[0]);
		debug_obstacle_line_vertices.resize(debug_obstacle_line_vertices.size());

		Vector<Color> debug_obstacle_line_colors;
		debug_obstacle_line_colors.resize(debug_obstacle_line_vertices.size());
		debug_obstacle_line_colors.fill(debug_static_obstacle_edge_color);

		RS::get_singleton()->canvas_item_add_polyline(get_canvas_item(), debug_obstacle_line_vertices, debug_obstacle_line_colors, 4.0);
	}
}
#endif // DEBUG_ENABLED
