/**************************************************************************/
/*  navigation_server_3d_debug.cpp                                        */
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

#include "navigation_server_3d_debug.h"

#include "core/config/project_settings.h"
#include "scene/resources/material.h"
#include "servers/navigation_3d/navigation_server_3d.h"

NavigationServer3DDebug *NavigationServer3DDebug::singleton = nullptr;

NavigationServer3DDebug *NavigationServer3DDebug::get_singleton() {
	return singleton;
}

NavigationServer3DDebug::NavigationServer3DDebug() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

NavigationServer3DDebug::~NavigationServer3DDebug() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void NavigationServer3DDebug::register_settings() {
#ifndef DISABLE_DEPRECATED
#define MOVE_PROJECT_SETTING_1(m_old_setting, m_new_setting) \
	if (!ProjectSettings::get_singleton()->has_setting(m_new_setting) && ProjectSettings::get_singleton()->has_setting(m_old_setting)) { \
		Variant value = GLOBAL_GET(m_old_setting); \
		ProjectSettings::get_singleton()->set_setting(m_new_setting, value); \
		ProjectSettings::get_singleton()->clear(m_old_setting); \
	}
#define MOVE_PROJECT_SETTING_2(m_old_setting, m_new_setting_1, m_new_setting_2) \
	if ((!ProjectSettings::get_singleton()->has_setting(m_new_setting_1) || !ProjectSettings::get_singleton()->has_setting(m_new_setting_2)) && \
			ProjectSettings::get_singleton()->has_setting(m_old_setting)) { \
		Variant value = GLOBAL_GET(m_old_setting); \
		if (!ProjectSettings::get_singleton()->has_setting(m_new_setting_1)) { \
			ProjectSettings::get_singleton()->set_setting(m_new_setting_1, value); \
		} \
		if (!ProjectSettings::get_singleton()->has_setting(m_new_setting_2)) { \
			ProjectSettings::get_singleton()->set_setting(m_new_setting_2, value); \
		} \
		ProjectSettings::get_singleton()->clear(m_old_setting); \
	}
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/edge_connection_color", "debug/shapes/navigation/2d/edge_connection_color", "debug/shapes/navigation/3d/edge_connection_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/geometry_edge_color", "debug/shapes/navigation/2d/geometry_edge_color", "debug/shapes/navigation/3d/geometry_edge_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/geometry_face_color", "debug/shapes/navigation/2d/geometry_face_color", "debug/shapes/navigation/3d/geometry_face_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/geometry_edge_disabled_color", "debug/shapes/navigation/2d/geometry_edge_disabled_color", "debug/shapes/navigation/3d/geometry_edge_disabled_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/geometry_face_disabled_color", "debug/shapes/navigation/2d/geometry_face_disabled_color", "debug/shapes/navigation/3d/geometry_face_disabled_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/link_connection_color", "debug/shapes/navigation/2d/link_connection_color", "debug/shapes/navigation/3d/link_connection_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/link_connection_disabled_color", "debug/shapes/navigation/2d/link_connection_disabled_color", "debug/shapes/navigation/3d/link_connection_disabled_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/agent_path_color", "debug/shapes/navigation/2d/agent_path_color", "debug/shapes/navigation/3d/agent_path_color");

	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/enable_edge_connections", "debug/shapes/navigation/2d/enable_edge_connections", "debug/shapes/navigation/3d/enable_edge_connections");
	MOVE_PROJECT_SETTING_1("debug/shapes/navigation/enable_edge_connections_xray", "debug/shapes/navigation/3d/enable_edge_connections_xray");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/enable_edge_lines", "debug/shapes/navigation/2d/enable_edge_lines", "debug/shapes/navigation/3d/enable_edge_lines");
	MOVE_PROJECT_SETTING_1("debug/shapes/navigation/enable_edge_lines_xray", "debug/shapes/navigation/3d/enable_edge_lines_xray");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/enable_geometry_face_random_color", "debug/shapes/navigation/2d/enable_geometry_face_random_color", "debug/shapes/navigation/3d/enable_geometry_face_random_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/enable_link_connections", "debug/shapes/navigation/2d/enable_link_connections", "debug/shapes/navigation/3d/enable_link_connections");
	MOVE_PROJECT_SETTING_1("debug/shapes/navigation/enable_link_connections_xray", "debug/shapes/navigation/3d/enable_link_connections_xray");

	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/enable_agent_paths", "debug/shapes/navigation/2d/enable_agent_paths", "debug/shapes/navigation/3d/enable_agent_paths");
	MOVE_PROJECT_SETTING_1("debug/shapes/navigation/enable_agent_paths_xray", "debug/shapes/navigation/3d/enable_agent_paths_xray");
	MOVE_PROJECT_SETTING_2("debug/shapes/navigation/agent_path_point_size", "debug/shapes/navigation/2d/agent_path_point_size", "debug/shapes/navigation/3d/agent_path_point_size");

	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/agents_radius_color", "debug/shapes/avoidance/2d/agents_radius_color", "debug/shapes/avoidance/3d/agents_radius_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/obstacles_radius_color", "debug/shapes/avoidance/2d/obstacles_radius_color", "debug/shapes/avoidance/3d/obstacles_radius_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/obstacles_static_face_pushin_color", "debug/shapes/avoidance/2d/obstacles_static_face_pushin_color", "debug/shapes/avoidance/3d/obstacles_static_face_pushin_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/obstacles_static_edge_pushin_color", "debug/shapes/avoidance/2d/obstacles_static_edge_pushin_color", "debug/shapes/avoidance/3d/obstacles_static_edge_pushin_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/obstacles_static_face_pushout_color", "debug/shapes/avoidance/2d/obstacles_static_face_pushout_color", "debug/shapes/avoidance/3d/obstacles_static_face_pushout_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/obstacles_static_edge_pushout_color", "debug/shapes/avoidance/2d/obstacles_static_edge_pushout_color", "debug/shapes/avoidance/3d/obstacles_static_edge_pushout_color");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/enable_agents_radius", "debug/shapes/avoidance/2d/enable_agents_radius", "debug/shapes/avoidance/3d/enable_agents_radius");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/enable_obstacles_radius", "debug/shapes/avoidance/2d/enable_obstacles_radius", "debug/shapes/avoidance/2d/enable_obstacles_static");
	MOVE_PROJECT_SETTING_2("debug/shapes/avoidance/enable_obstacles_radius", "debug/shapes/avoidance/3d/enable_obstacles_radius", "debug/shapes/avoidance/3d/enable_obstacles_static");
#undef MOVE_PROJECT_SETTING_1
#undef MOVE_PROJECT_SETTING_2
#endif // DISABLE_DEPRECATED

	debug_navigation_edge_connection_color = GLOBAL_DEF("debug/shapes/navigation/3d/edge_connection_color", Color(1.0, 0.0, 1.0, 1.0));
	debug_navigation_geometry_edge_color = GLOBAL_DEF("debug/shapes/navigation/3d/geometry_edge_color", Color(0.5, 1.0, 1.0, 1.0));
	debug_navigation_geometry_face_color = GLOBAL_DEF("debug/shapes/navigation/3d/geometry_face_color", Color(0.5, 1.0, 1.0, 0.4));
	debug_navigation_geometry_edge_disabled_color = GLOBAL_DEF("debug/shapes/navigation/3d/geometry_edge_disabled_color", Color(0.5, 0.5, 0.5, 1.0));
	debug_navigation_geometry_face_disabled_color = GLOBAL_DEF("debug/shapes/navigation/3d/geometry_face_disabled_color", Color(0.5, 0.5, 0.5, 0.4));
	debug_navigation_link_connection_color = GLOBAL_DEF("debug/shapes/navigation/3d/link_connection_color", Color(1.0, 0.5, 1.0, 1.0));
	debug_navigation_link_connection_disabled_color = GLOBAL_DEF("debug/shapes/navigation/3d/link_connection_disabled_color", Color(0.5, 0.5, 0.5, 1.0));
	debug_navigation_agent_path_color = GLOBAL_DEF("debug/shapes/navigation/3d/agent_path_color", Color(1.0, 0.0, 0.0, 1.0));

	debug_navigation_enable_edge_connections = GLOBAL_DEF("debug/shapes/navigation/3d/enable_edge_connections", true);
	debug_navigation_enable_edge_connections_xray = GLOBAL_DEF("debug/shapes/navigation/3d/enable_edge_connections_xray", true);
	debug_navigation_enable_edge_lines = GLOBAL_DEF("debug/shapes/navigation/3d/enable_edge_lines", true);
	debug_navigation_enable_edge_lines_xray = GLOBAL_DEF("debug/shapes/navigation/3d/enable_edge_lines_xray", true);
	debug_navigation_enable_geometry_face_random_color = GLOBAL_DEF("debug/shapes/navigation/3d/enable_geometry_face_random_color", true);
	debug_navigation_enable_link_connections = GLOBAL_DEF("debug/shapes/navigation/3d/enable_link_connections", true);
	debug_navigation_enable_link_connections_xray = GLOBAL_DEF("debug/shapes/navigation/3d/enable_link_connections_xray", true);

	debug_navigation_enable_agent_paths = GLOBAL_DEF("debug/shapes/navigation/3d/enable_agent_paths", true);
	debug_navigation_enable_agent_paths_xray = GLOBAL_DEF("debug/shapes/navigation/3d/enable_agent_paths_xray", true);
	debug_navigation_agent_path_point_size = GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "debug/shapes/navigation/3d/agent_path_point_size", PROPERTY_HINT_RANGE, "0.01,10,0.001,or_greater"), 4.0);

	debug_navigation_avoidance_agents_radius_color = GLOBAL_DEF("debug/shapes/avoidance/3d/agents_radius_color", Color(1.0, 1.0, 0.0, 0.25));
	debug_navigation_avoidance_obstacles_radius_color = GLOBAL_DEF("debug/shapes/avoidance/3d/obstacles_radius_color", Color(1.0, 0.5, 0.0, 0.25));
	debug_navigation_avoidance_static_obstacle_pushin_face_color = GLOBAL_DEF("debug/shapes/avoidance/3d/obstacles_static_face_pushin_color", Color(1.0, 0.0, 0.0, 0.0));
	debug_navigation_avoidance_static_obstacle_pushin_edge_color = GLOBAL_DEF("debug/shapes/avoidance/3d/obstacles_static_edge_pushin_color", Color(1.0, 0.0, 0.0, 1.0));
	debug_navigation_avoidance_static_obstacle_pushout_face_color = GLOBAL_DEF("debug/shapes/avoidance/3d/obstacles_static_face_pushout_color", Color(1.0, 1.0, 0.0, 0.5));
	debug_navigation_avoidance_static_obstacle_pushout_edge_color = GLOBAL_DEF("debug/shapes/avoidance/3d/obstacles_static_edge_pushout_color", Color(1.0, 1.0, 0.0, 1.0));
	debug_navigation_avoidance_enable_agents_radius = GLOBAL_DEF("debug/shapes/avoidance/3d/enable_agents_radius", true);
	debug_navigation_avoidance_enable_obstacles_radius = GLOBAL_DEF("debug/shapes/avoidance/3d/enable_obstacles_radius", true);
	debug_navigation_avoidance_enable_obstacles_static = GLOBAL_DEF("debug/shapes/avoidance/3d/enable_obstacles_static", true);
}

void NavigationServer3DDebug::update_from_settings() {
	if (!ProjectSettings::get_singleton()->check_changed_settings_in_group("debug/shapes/navigation/3d")) {
		return;
	}
	set_debug_navigation_edge_connection_color(GLOBAL_GET("debug/shapes/navigation/3d/edge_connection_color"));
	set_debug_navigation_geometry_edge_color(GLOBAL_GET("debug/shapes/navigation/3d/geometry_edge_color"));
	set_debug_navigation_geometry_face_color(GLOBAL_GET("debug/shapes/navigation/3d/geometry_face_color"));
	set_debug_navigation_geometry_edge_disabled_color(GLOBAL_GET("debug/shapes/navigation/3d/geometry_edge_disabled_color"));
	set_debug_navigation_geometry_face_disabled_color(GLOBAL_GET("debug/shapes/navigation/3d/geometry_face_disabled_color"));
	set_debug_navigation_enable_edge_connections(GLOBAL_GET("debug/shapes/navigation/3d/enable_edge_connections"));
	set_debug_navigation_enable_edge_connections_xray(GLOBAL_GET("debug/shapes/navigation/3d/enable_edge_connections_xray"));
	set_debug_navigation_enable_edge_lines(GLOBAL_GET("debug/shapes/navigation/3d/enable_edge_lines"));
	set_debug_navigation_enable_edge_lines_xray(GLOBAL_GET("debug/shapes/navigation/3d/enable_edge_lines_xray"));
	set_debug_navigation_enable_geometry_face_random_color(GLOBAL_GET("debug/shapes/navigation/3d/enable_geometry_face_random_color"));
}

void NavigationServer3DDebug::set_debug_enabled(bool p_enabled) {
	debug_enabled = p_enabled;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
	NavigationServer3D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer3DDebug::get_debug_enabled() const {
	return debug_enabled;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_geometry_face_material() {
	if (debug_navigation_geometry_face_material.is_valid()) {
		return debug_navigation_geometry_face_material;
	}

	bool enabled_geometry_face_random_color = get_debug_navigation_enable_geometry_face_random_color();

	Ref<StandardMaterial3D> face_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	face_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	face_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	face_material->set_albedo(get_debug_navigation_geometry_face_color());
	face_material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
	face_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (enabled_geometry_face_random_color) {
		face_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
		face_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	}

	debug_navigation_geometry_face_material = face_material;

	return debug_navigation_geometry_face_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_geometry_edge_material() {
	if (debug_navigation_geometry_edge_material.is_valid()) {
		return debug_navigation_geometry_edge_material;
	}

	bool enabled_edge_lines_xray = get_debug_navigation_enable_edge_lines_xray();

	Ref<StandardMaterial3D> line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_albedo(get_debug_navigation_geometry_edge_color());
	line_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (enabled_edge_lines_xray) {
		line_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}

	debug_navigation_geometry_edge_material = line_material;

	return debug_navigation_geometry_edge_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_geometry_face_disabled_material() {
	if (debug_navigation_geometry_face_disabled_material.is_valid()) {
		return debug_navigation_geometry_face_disabled_material;
	}

	Ref<StandardMaterial3D> face_disabled_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	face_disabled_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	face_disabled_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	face_disabled_material->set_albedo(get_debug_navigation_geometry_face_disabled_color());
	face_disabled_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);

	debug_navigation_geometry_face_disabled_material = face_disabled_material;

	return debug_navigation_geometry_face_disabled_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_geometry_edge_disabled_material() {
	if (debug_navigation_geometry_edge_disabled_material.is_valid()) {
		return debug_navigation_geometry_edge_disabled_material;
	}

	bool enabled_edge_lines_xray = get_debug_navigation_enable_edge_lines_xray();

	Ref<StandardMaterial3D> line_disabled_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_disabled_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_disabled_material->set_albedo(get_debug_navigation_geometry_edge_disabled_color());
	line_disabled_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (enabled_edge_lines_xray) {
		line_disabled_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}

	debug_navigation_geometry_edge_disabled_material = line_disabled_material;

	return debug_navigation_geometry_edge_disabled_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_edge_connections_material() {
	if (debug_navigation_edge_connections_material.is_valid()) {
		return debug_navigation_edge_connections_material;
	}

	bool enabled_edge_connections_xray = get_debug_navigation_enable_edge_connections_xray();

	Ref<StandardMaterial3D> edge_connections_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	edge_connections_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	edge_connections_material->set_albedo(get_debug_navigation_edge_connection_color());
	edge_connections_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (enabled_edge_connections_xray) {
		edge_connections_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	edge_connections_material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MAX - 2);

	debug_navigation_edge_connections_material = edge_connections_material;

	return debug_navigation_edge_connections_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_link_connections_material() {
	if (debug_navigation_link_connections_material.is_valid()) {
		return debug_navigation_link_connections_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_albedo(debug_navigation_link_connection_color);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (debug_navigation_enable_link_connections_xray) {
		material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MAX - 2);

	debug_navigation_link_connections_material = material;
	return debug_navigation_link_connections_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_link_connections_disabled_material() {
	if (debug_navigation_link_connections_disabled_material.is_valid()) {
		return debug_navigation_link_connections_disabled_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_albedo(debug_navigation_link_connection_disabled_color);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (debug_navigation_enable_link_connections_xray) {
		material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MAX - 2);

	debug_navigation_link_connections_disabled_material = material;
	return debug_navigation_link_connections_disabled_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_agent_path_line_material() {
	if (debug_navigation_agent_path_line_material.is_valid()) {
		return debug_navigation_agent_path_line_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);

	material->set_albedo(debug_navigation_agent_path_color);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (debug_navigation_enable_agent_paths_xray) {
		material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MAX - 2);

	debug_navigation_agent_path_line_material = material;
	return debug_navigation_agent_path_line_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_agent_path_point_material() {
	if (debug_navigation_agent_path_point_material.is_valid()) {
		return debug_navigation_agent_path_point_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_albedo(debug_navigation_agent_path_color);
	material->set_flag(StandardMaterial3D::FLAG_USE_POINT_SIZE, true);
	material->set_point_size(debug_navigation_agent_path_point_size);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	if (debug_navigation_enable_agent_paths_xray) {
		material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MAX - 2);

	debug_navigation_agent_path_point_material = material;
	return debug_navigation_agent_path_point_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_agents_radius_material() {
	if (debug_navigation_avoidance_agents_radius_material.is_valid()) {
		return debug_navigation_avoidance_agents_radius_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
	material->set_albedo(debug_navigation_avoidance_agents_radius_color);
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MIN + 2);

	debug_navigation_avoidance_agents_radius_material = material;
	return debug_navigation_avoidance_agents_radius_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_obstacles_radius_material() {
	if (debug_navigation_avoidance_obstacles_radius_material.is_valid()) {
		return debug_navigation_avoidance_obstacles_radius_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
	material->set_albedo(debug_navigation_avoidance_obstacles_radius_color);
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MIN + 2);

	debug_navigation_avoidance_obstacles_radius_material = material;
	return debug_navigation_avoidance_obstacles_radius_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_face_material() {
	if (debug_navigation_avoidance_static_obstacle_pushin_face_material.is_valid()) {
		return debug_navigation_avoidance_static_obstacle_pushin_face_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
	material->set_albedo(debug_navigation_avoidance_static_obstacle_pushin_face_color);
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MIN + 2);

	debug_navigation_avoidance_static_obstacle_pushin_face_material = material;
	return debug_navigation_avoidance_static_obstacle_pushin_face_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_face_material() {
	if (debug_navigation_avoidance_static_obstacle_pushout_face_material.is_valid()) {
		return debug_navigation_avoidance_static_obstacle_pushout_face_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
	material->set_albedo(debug_navigation_avoidance_static_obstacle_pushout_face_color);
	material->set_render_priority(StandardMaterial3D::RENDER_PRIORITY_MIN + 2);

	debug_navigation_avoidance_static_obstacle_pushout_face_material = material;
	return debug_navigation_avoidance_static_obstacle_pushout_face_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_edge_material() {
	if (debug_navigation_avoidance_static_obstacle_pushin_edge_material.is_valid()) {
		return debug_navigation_avoidance_static_obstacle_pushin_edge_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	material->set_albedo(debug_navigation_avoidance_static_obstacle_pushin_edge_color);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);

	debug_navigation_avoidance_static_obstacle_pushin_edge_material = material;
	return debug_navigation_avoidance_static_obstacle_pushin_edge_material;
}

Ref<StandardMaterial3D> NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_edge_material() {
	if (debug_navigation_avoidance_static_obstacle_pushout_edge_material.is_valid()) {
		return debug_navigation_avoidance_static_obstacle_pushout_edge_material;
	}

	Ref<StandardMaterial3D> material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	material->set_albedo(debug_navigation_avoidance_static_obstacle_pushout_edge_color);
	material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);

	debug_navigation_avoidance_static_obstacle_pushout_edge_material = material;
	return debug_navigation_avoidance_static_obstacle_pushout_edge_material;
}

void NavigationServer3DDebug::set_debug_navigation_edge_connection_color(const Color &p_color) {
	debug_navigation_edge_connection_color = p_color;
	if (debug_navigation_edge_connections_material.is_valid()) {
		debug_navigation_edge_connections_material->set_albedo(debug_navigation_edge_connection_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_edge_connection_color() const {
	return debug_navigation_edge_connection_color;
}

void NavigationServer3DDebug::set_debug_navigation_geometry_edge_color(const Color &p_color) {
	debug_navigation_geometry_edge_color = p_color;
	if (debug_navigation_geometry_edge_material.is_valid()) {
		debug_navigation_geometry_edge_material->set_albedo(debug_navigation_geometry_edge_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_geometry_edge_color() const {
	return debug_navigation_geometry_edge_color;
}

void NavigationServer3DDebug::set_debug_navigation_geometry_face_color(const Color &p_color) {
	debug_navigation_geometry_face_color = p_color;
	if (debug_navigation_geometry_face_material.is_valid()) {
		debug_navigation_geometry_face_material->set_albedo(debug_navigation_geometry_face_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_geometry_face_color() const {
	return debug_navigation_geometry_face_color;
}

void NavigationServer3DDebug::set_debug_navigation_geometry_edge_disabled_color(const Color &p_color) {
	debug_navigation_geometry_edge_disabled_color = p_color;
	if (debug_navigation_geometry_edge_disabled_material.is_valid()) {
		debug_navigation_geometry_edge_disabled_material->set_albedo(debug_navigation_geometry_edge_disabled_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_geometry_edge_disabled_color() const {
	return debug_navigation_geometry_edge_disabled_color;
}

void NavigationServer3DDebug::set_debug_navigation_geometry_face_disabled_color(const Color &p_color) {
	debug_navigation_geometry_face_disabled_color = p_color;
	if (debug_navigation_geometry_face_disabled_material.is_valid()) {
		debug_navigation_geometry_face_disabled_material->set_albedo(debug_navigation_geometry_face_disabled_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_geometry_face_disabled_color() const {
	return debug_navigation_geometry_face_disabled_color;
}

void NavigationServer3DDebug::set_debug_navigation_link_connection_color(const Color &p_color) {
	debug_navigation_link_connection_color = p_color;
	if (debug_navigation_link_connections_material.is_valid()) {
		debug_navigation_link_connections_material->set_albedo(debug_navigation_link_connection_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_link_connection_color() const {
	return debug_navigation_link_connection_color;
}

void NavigationServer3DDebug::set_debug_navigation_link_connection_disabled_color(const Color &p_color) {
	debug_navigation_link_connection_disabled_color = p_color;
	if (debug_navigation_link_connections_disabled_material.is_valid()) {
		debug_navigation_link_connections_disabled_material->set_albedo(debug_navigation_link_connection_disabled_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_link_connection_disabled_color() const {
	return debug_navigation_link_connection_disabled_color;
}

void NavigationServer3DDebug::set_debug_navigation_agent_path_point_size(float p_point_size) {
	debug_navigation_agent_path_point_size = MAX(0.1, p_point_size);
	if (debug_navigation_agent_path_point_material.is_valid()) {
		debug_navigation_agent_path_point_material->set_point_size(debug_navigation_agent_path_point_size);
	}
}

float NavigationServer3DDebug::get_debug_navigation_agent_path_point_size() const {
	return debug_navigation_agent_path_point_size;
}

void NavigationServer3DDebug::set_debug_navigation_agent_path_color(const Color &p_color) {
	debug_navigation_agent_path_color = p_color;
	if (debug_navigation_agent_path_line_material.is_valid()) {
		debug_navigation_agent_path_line_material->set_albedo(debug_navigation_agent_path_color);
	}
	if (debug_navigation_agent_path_point_material.is_valid()) {
		debug_navigation_agent_path_point_material->set_albedo(debug_navigation_agent_path_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_agent_path_color() const {
	return debug_navigation_agent_path_color;
}

void NavigationServer3DDebug::set_debug_navigation_enable_edge_connections(const bool p_value) {
	debug_navigation_enable_edge_connections = p_value;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enable_edge_connections() const {
	return debug_navigation_enable_edge_connections;
}

void NavigationServer3DDebug::set_debug_navigation_enable_edge_connections_xray(const bool p_value) {
	debug_navigation_enable_edge_connections_xray = p_value;
	if (debug_navigation_edge_connections_material.is_valid()) {
		debug_navigation_edge_connections_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, debug_navigation_enable_edge_connections_xray);
	}
}

bool NavigationServer3DDebug::get_debug_navigation_enable_edge_connections_xray() const {
	return debug_navigation_enable_edge_connections_xray;
}

void NavigationServer3DDebug::set_debug_navigation_enable_edge_lines(const bool p_value) {
	debug_navigation_enable_edge_lines = p_value;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enable_edge_lines() const {
	return debug_navigation_enable_edge_lines;
}

void NavigationServer3DDebug::set_debug_navigation_enable_edge_lines_xray(const bool p_value) {
	debug_navigation_enable_edge_lines_xray = p_value;
	if (debug_navigation_geometry_edge_material.is_valid()) {
		debug_navigation_geometry_edge_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, debug_navigation_enable_edge_lines_xray);
	}
}

bool NavigationServer3DDebug::get_debug_navigation_enable_edge_lines_xray() const {
	return debug_navigation_enable_edge_lines_xray;
}

void NavigationServer3DDebug::set_debug_navigation_enable_geometry_face_random_color(const bool p_value) {
	debug_navigation_enable_geometry_face_random_color = p_value;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enable_geometry_face_random_color() const {
	return debug_navigation_enable_geometry_face_random_color;
}

void NavigationServer3DDebug::set_debug_navigation_enable_link_connections(const bool p_value) {
	debug_navigation_enable_link_connections = p_value;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enable_link_connections() const {
	return debug_navigation_enable_link_connections;
}

void NavigationServer3DDebug::set_debug_navigation_enable_link_connections_xray(const bool p_value) {
	debug_navigation_enable_link_connections_xray = p_value;
	if (debug_navigation_link_connections_material.is_valid()) {
		debug_navigation_link_connections_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, debug_navigation_enable_link_connections_xray);
	}
}

bool NavigationServer3DDebug::get_debug_navigation_enable_link_connections_xray() const {
	return debug_navigation_enable_link_connections_xray;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_enable_agents_radius(const bool p_value) {
	debug_navigation_avoidance_enable_agents_radius = p_value;
	NavigationServer3D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_avoidance_enable_agents_radius() const {
	return debug_navigation_avoidance_enable_agents_radius;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_enable_obstacles_radius(const bool p_value) {
	debug_navigation_avoidance_enable_obstacles_radius = p_value;
	NavigationServer3D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_avoidance_enable_obstacles_radius() const {
	return debug_navigation_avoidance_enable_obstacles_radius;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_enable_obstacles_static(const bool p_value) {
	debug_navigation_avoidance_enable_obstacles_static = p_value;
	NavigationServer3D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_avoidance_enable_obstacles_static() const {
	return debug_navigation_avoidance_enable_obstacles_static;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_agents_radius_color(const Color &p_color) {
	debug_navigation_avoidance_agents_radius_color = p_color;
	if (debug_navigation_avoidance_agents_radius_material.is_valid()) {
		debug_navigation_avoidance_agents_radius_material->set_albedo(debug_navigation_avoidance_agents_radius_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_agents_radius_color() const {
	return debug_navigation_avoidance_agents_radius_color;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_obstacles_radius_color(const Color &p_color) {
	debug_navigation_avoidance_obstacles_radius_color = p_color;
	if (debug_navigation_avoidance_obstacles_radius_material.is_valid()) {
		debug_navigation_avoidance_obstacles_radius_material->set_albedo(debug_navigation_avoidance_obstacles_radius_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_obstacles_radius_color() const {
	return debug_navigation_avoidance_obstacles_radius_color;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_static_obstacle_pushin_face_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushin_face_color = p_color;
	if (debug_navigation_avoidance_static_obstacle_pushin_face_material.is_valid()) {
		debug_navigation_avoidance_static_obstacle_pushin_face_material->set_albedo(debug_navigation_avoidance_static_obstacle_pushin_face_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_face_color() const {
	return debug_navigation_avoidance_static_obstacle_pushin_face_color;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_static_obstacle_pushout_face_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushout_face_color = p_color;
	if (debug_navigation_avoidance_static_obstacle_pushout_face_material.is_valid()) {
		debug_navigation_avoidance_static_obstacle_pushout_face_material->set_albedo(debug_navigation_avoidance_static_obstacle_pushout_face_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_face_color() const {
	return debug_navigation_avoidance_static_obstacle_pushout_face_color;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_static_obstacle_pushin_edge_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushin_edge_color = p_color;
	if (debug_navigation_avoidance_static_obstacle_pushin_edge_material.is_valid()) {
		debug_navigation_avoidance_static_obstacle_pushin_edge_material->set_albedo(debug_navigation_avoidance_static_obstacle_pushin_edge_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushin_edge_color() const {
	return debug_navigation_avoidance_static_obstacle_pushin_edge_color;
}

void NavigationServer3DDebug::set_debug_navigation_avoidance_static_obstacle_pushout_edge_color(const Color &p_color) {
	debug_navigation_avoidance_static_obstacle_pushout_edge_color = p_color;
	if (debug_navigation_avoidance_static_obstacle_pushout_edge_material.is_valid()) {
		debug_navigation_avoidance_static_obstacle_pushout_edge_material->set_albedo(debug_navigation_avoidance_static_obstacle_pushout_edge_color);
	}
}

Color NavigationServer3DDebug::get_debug_navigation_avoidance_static_obstacle_pushout_edge_color() const {
	return debug_navigation_avoidance_static_obstacle_pushout_edge_color;
}

void NavigationServer3DDebug::set_debug_navigation_enable_agent_paths(const bool p_value) {
	debug_navigation_enable_agent_paths = p_value;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enable_agent_paths() const {
	return debug_navigation_enable_agent_paths;
}

void NavigationServer3DDebug::set_debug_navigation_enable_agent_paths_xray(const bool p_value) {
	debug_navigation_enable_agent_paths_xray = p_value;
	if (debug_navigation_agent_path_line_material.is_valid()) {
		debug_navigation_agent_path_line_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, debug_navigation_enable_agent_paths_xray);
	}
	if (debug_navigation_agent_path_point_material.is_valid()) {
		debug_navigation_agent_path_point_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, debug_navigation_enable_agent_paths_xray);
	}
}

bool NavigationServer3DDebug::get_debug_navigation_enable_agent_paths_xray() const {
	return debug_navigation_enable_agent_paths_xray;
}

void NavigationServer3DDebug::set_debug_navigation_enabled(bool p_enabled) {
	debug_navigation_enabled = p_enabled;
	NavigationServer3D::get_singleton()->emit_navigation_debug_changed();
}

bool NavigationServer3DDebug::get_debug_navigation_enabled() const {
	return debug_navigation_enabled;
}

void NavigationServer3DDebug::set_debug_avoidance_enabled(bool p_enabled) {
	debug_avoidance_enabled = p_enabled;
	NavigationServer3D::get_singleton()->emit_avoidance_debug_changed();
}

bool NavigationServer3DDebug::get_debug_avoidance_enabled() const {
	return debug_avoidance_enabled;
}

#endif // DEBUG_ENABLED
