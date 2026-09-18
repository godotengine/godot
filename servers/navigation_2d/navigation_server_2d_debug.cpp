/**************************************************************************/
/*  navigation_server_2d_debug.cpp                                        */
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

#ifdef DEBUG_ENABLED

#include "navigation_server_2d_debug.h"

#include "core/config/project_settings.h"
#include "servers/navigation_2d/navigation_server_2d.h"

NavigationServer2DDebug *NavigationServer2DDebug::singleton = nullptr;

NavigationServer2DDebug *NavigationServer2DDebug::get_singleton() {
	return singleton;
}

NavigationServer2DDebug::NavigationServer2DDebug() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

NavigationServer2DDebug::~NavigationServer2DDebug() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void NavigationServer2DDebug::register_settings() {
	debug_navigation_edge_connection_color = GLOBAL_DEF("debug/shapes/navigation/2d/edge_connection_color", Color(1.0, 0.0, 1.0, 1.0));
	debug_navigation_geometry_edge_color = GLOBAL_DEF("debug/shapes/navigation/2d/geometry_edge_color", Color(0.5, 1.0, 1.0, 1.0));
	debug_navigation_geometry_face_color = GLOBAL_DEF("debug/shapes/navigation/2d/geometry_face_color", Color(0.5, 1.0, 1.0, 0.4));
	debug_navigation_geometry_edge_disabled_color = GLOBAL_DEF("debug/shapes/navigation/2d/geometry_edge_disabled_color", Color(0.5, 0.5, 0.5, 1.0));
	debug_navigation_geometry_face_disabled_color = GLOBAL_DEF("debug/shapes/navigation/2d/geometry_face_disabled_color", Color(0.5, 0.5, 0.5, 0.4));
	debug_navigation_link_connection_color = GLOBAL_DEF("debug/shapes/navigation/2d/link_connection_color", Color(1.0, 0.5, 1.0, 1.0));
	debug_navigation_link_connection_disabled_color = GLOBAL_DEF("debug/shapes/navigation/2d/link_connection_disabled_color", Color(0.5, 0.5, 0.5, 1.0));
	debug_navigation_agent_path_color = GLOBAL_DEF("debug/shapes/navigation/2d/agent_path_color", Color(1.0, 0.0, 0.0, 1.0));

	debug_navigation_enable_edge_connections = GLOBAL_DEF("debug/shapes/navigation/2d/enable_edge_connections", true);
	debug_navigation_enable_edge_lines = GLOBAL_DEF("debug/shapes/navigation/2d/enable_edge_lines", true);
	debug_navigation_enable_geometry_face_random_color = GLOBAL_DEF("debug/shapes/navigation/2d/enable_geometry_face_random_color", true);
	debug_navigation_enable_link_connections = GLOBAL_DEF("debug/shapes/navigation/2d/enable_link_connections", true);

	debug_navigation_enable_agent_paths = GLOBAL_DEF("debug/shapes/navigation/2d/enable_agent_paths", true);
	debug_navigation_agent_path_point_size = GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "debug/shapes/navigation/2d/agent_path_point_size", PROPERTY_HINT_RANGE, "0.01,10,0.001,or_greater"), 4.0);

	debug_navigation_avoidance_agents_radius_color = GLOBAL_DEF("debug/shapes/avoidance/2d/agents_radius_color", Color(1.0, 1.0, 0.0, 0.25));
	debug_navigation_avoidance_obstacles_radius_color = GLOBAL_DEF("debug/shapes/avoidance/2d/obstacles_radius_color", Color(1.0, 0.5, 0.0, 0.25));
	debug_navigation_avoidance_static_obstacle_pushin_face_color = GLOBAL_DEF("debug/shapes/avoidance/2d/obstacles_static_face_pushin_color", Color(1.0, 0.0, 0.0, 0.0));
	debug_navigation_avoidance_static_obstacle_pushin_edge_color = GLOBAL_DEF("debug/shapes/avoidance/2d/obstacles_static_edge_pushin_color", Color(1.0, 0.0, 0.0, 1.0));
	debug_navigation_avoidance_static_obstacle_pushout_face_color = GLOBAL_DEF("debug/shapes/avoidance/2d/obstacles_static_face_pushout_color", Color(1.0, 1.0, 0.0, 0.5));
	debug_navigation_avoidance_static_obstacle_pushout_edge_color = GLOBAL_DEF("debug/shapes/avoidance/2d/obstacles_static_edge_pushout_color", Color(1.0, 1.0, 0.0, 1.0));
	debug_navigation_avoidance_enable_agents_radius = GLOBAL_DEF("debug/shapes/avoidance/2d/enable_agents_radius", true);
	debug_navigation_avoidance_enable_obstacles_radius = GLOBAL_DEF("debug/shapes/avoidance/2d/enable_obstacles_radius", true);
	debug_navigation_avoidance_enable_obstacles_static = GLOBAL_DEF("debug/shapes/avoidance/2d/enable_obstacles_static", true);
}

void NavigationServer2DDebug::update_from_settings() {
	if (!ProjectSettings::get_singleton()->check_changed_settings_in_group("debug/shapes/navigation/2d")) {
		return;
	}
	set_debug_navigation_edge_connection_color(GLOBAL_GET("debug/shapes/navigation/2d/edge_connection_color"));
	set_debug_navigation_geometry_edge_color(GLOBAL_GET("debug/shapes/navigation/2d/geometry_edge_color"));
	set_debug_navigation_geometry_face_color(GLOBAL_GET("debug/shapes/navigation/2d/geometry_face_color"));
	set_debug_navigation_geometry_edge_disabled_color(GLOBAL_GET("debug/shapes/navigation/2d/geometry_edge_disabled_color"));
	set_debug_navigation_geometry_face_disabled_color(GLOBAL_GET("debug/shapes/navigation/2d/geometry_face_disabled_color"));
	set_debug_navigation_enable_edge_connections(GLOBAL_GET("debug/shapes/navigation/2d/enable_edge_connections"));
	set_debug_navigation_enable_edge_lines(GLOBAL_GET("debug/shapes/navigation/2d/enable_edge_lines"));
	set_debug_navigation_enable_geometry_face_random_color(GLOBAL_GET("debug/shapes/navigation/2d/enable_geometry_face_random_color"));
}

void NavigationServer2DDebug::set_debug_enabled(bool p_enabled) {
	debug_enabled = p_enabled;
	NavigationServer2D::get_singleton()->emit_navigation_debug_changed();
	NavigationServer2D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer2DDebug::get_debug_enabled() const {
	return debug_enabled;
}

void NavigationServer2DDebug::set_debug_navigation_enabled(bool p_enabled) {
	debug_navigation_enabled = p_enabled;
	NavigationServer2D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer2DDebug::get_debug_navigation_enabled() const {
	return debug_navigation_enabled;
}

void NavigationServer2DDebug::set_debug_avoidance_enabled(bool p_enabled) {
	debug_avoidance_enabled = p_enabled;
	NavigationServer2D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer2DDebug::get_debug_avoidance_enabled() const {
	return debug_avoidance_enabled;
}

void NavigationServer2DDebug::set_debug_navigation_edge_connection_color(const Color &p_color) {
	debug_navigation_edge_connection_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_edge_connection_color() const {
	return debug_navigation_edge_connection_color;
}

void NavigationServer2DDebug::set_debug_navigation_geometry_face_color(const Color &p_color) {
	debug_navigation_geometry_face_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_geometry_face_color() const {
	return debug_navigation_geometry_face_color;
}

void NavigationServer2DDebug::set_debug_navigation_geometry_face_disabled_color(const Color &p_color) {
	debug_navigation_geometry_face_disabled_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_geometry_face_disabled_color() const {
	return debug_navigation_geometry_face_disabled_color;
}

void NavigationServer2DDebug::set_debug_navigation_link_connection_color(const Color &p_color) {
	debug_navigation_link_connection_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_link_connection_color() const {
	return debug_navigation_link_connection_color;
}

void NavigationServer2DDebug::set_debug_navigation_link_connection_disabled_color(const Color &p_color) {
	debug_navigation_link_connection_disabled_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_link_connection_disabled_color() const {
	return debug_navigation_link_connection_disabled_color;
}

void NavigationServer2DDebug::set_debug_navigation_geometry_edge_color(const Color &p_color) {
	debug_navigation_geometry_edge_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_geometry_edge_color() const {
	return debug_navigation_geometry_edge_color;
}

void NavigationServer2DDebug::set_debug_navigation_geometry_edge_disabled_color(const Color &p_color) {
	debug_navigation_geometry_edge_disabled_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_geometry_edge_disabled_color() const {
	return debug_navigation_geometry_edge_disabled_color;
}

void NavigationServer2DDebug::set_debug_navigation_enable_edge_connections(const bool p_value) {
	debug_navigation_enable_edge_connections = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_enable_edge_connections() const {
	return debug_navigation_enable_edge_connections;
}

void NavigationServer2DDebug::set_debug_navigation_enable_geometry_face_random_color(const bool p_value) {
	debug_navigation_enable_geometry_face_random_color = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_enable_geometry_face_random_color() const {
	return debug_navigation_enable_geometry_face_random_color;
}

void NavigationServer2DDebug::set_debug_navigation_enable_edge_lines(const bool p_value) {
	debug_navigation_enable_edge_lines = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_enable_edge_lines() const {
	return debug_navigation_enable_edge_lines;
}

void NavigationServer2DDebug::set_debug_navigation_agent_path_color(const Color &p_color) {
	debug_navigation_agent_path_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_agent_path_color() const {
	return debug_navigation_agent_path_color;
}

void NavigationServer2DDebug::set_debug_navigation_enable_agent_paths(const bool p_value) {
	debug_navigation_enable_agent_paths = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_enable_agent_paths() const {
	return debug_navigation_enable_agent_paths;
}

void NavigationServer2DDebug::set_debug_navigation_agent_path_point_size(float p_point_size) {
	debug_navigation_agent_path_point_size = p_point_size;
}

float NavigationServer2DDebug::get_debug_navigation_agent_path_point_size() const {
	return debug_navigation_agent_path_point_size;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_enable_agents_radius(const bool p_value) {
	debug_navigation_avoidance_enable_agents_radius = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_avoidance_enable_agents_radius() const {
	return debug_navigation_avoidance_enable_agents_radius;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_enable_obstacles_radius(const bool p_value) {
	debug_navigation_avoidance_enable_obstacles_radius = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_avoidance_enable_obstacles_radius() const {
	return debug_navigation_avoidance_enable_obstacles_radius;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_agents_radius_color(const Color &p_color) {
	debug_navigation_avoidance_agents_radius_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_agents_radius_color() const {
	return debug_navigation_avoidance_agents_radius_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_obstacles_radius_color(const Color &p_color) {
	debug_navigation_avoidance_obstacles_radius_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_obstacles_radius_color() const {
	return debug_navigation_avoidance_obstacles_radius_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_static_obstacle_pushin_face_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushin_face_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_face_color() const {
	return debug_navigation_avoidance_static_obstacle_pushin_face_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_static_obstacle_pushout_face_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushout_face_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_face_color() const {
	return debug_navigation_avoidance_static_obstacle_pushout_face_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_static_obstacle_pushin_edge_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushin_edge_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_edge_color() const {
	return debug_navigation_avoidance_static_obstacle_pushin_edge_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_static_obstacle_pushout_edge_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushout_edge_color = p_color;
}

Color NavigationServer2DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_edge_color() const {
	return debug_navigation_avoidance_static_obstacle_pushout_edge_color;
}

void NavigationServer2DDebug::set_debug_navigation_avoidance_enable_obstacles_static(const bool p_value) {
	debug_navigation_avoidance_enable_obstacles_static = p_value;
}

bool NavigationServer2DDebug::get_debug_navigation_avoidance_enable_obstacles_static() const {
	return debug_navigation_avoidance_enable_obstacles_static;
}

#endif // DEBUG_ENABLED
