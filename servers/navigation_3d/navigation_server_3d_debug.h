/**************************************************************************/
/*  navigation_server_3d_debug.h                                          */
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

#ifdef DEBUG_ENABLED

#include "core/math/color.h"
#include "core/object/ref_counted.h"

class StandardMaterial3D;

class NavigationServer3DDebug {
	static NavigationServer3DDebug *singleton;

	bool debug_enabled = false;
	bool debug_navigation_enabled = false;
	bool debug_avoidance_enabled = false;

	Color debug_navigation_edge_connection_color;
	Color debug_navigation_geometry_edge_color;
	Color debug_navigation_geometry_face_color;
	Color debug_navigation_geometry_edge_disabled_color;
	Color debug_navigation_geometry_face_disabled_color;
	Color debug_navigation_link_connection_color;
	Color debug_navigation_link_connection_disabled_color;
	Color debug_navigation_agent_path_color;

	float debug_navigation_agent_path_point_size = 4.0;

	Color debug_navigation_avoidance_agents_radius_color;
	Color debug_navigation_avoidance_obstacles_radius_color;

	Color debug_navigation_avoidance_static_obstacle_pushin_face_color;
	Color debug_navigation_avoidance_static_obstacle_pushout_face_color;
	Color debug_navigation_avoidance_static_obstacle_pushin_edge_color;
	Color debug_navigation_avoidance_static_obstacle_pushout_edge_color;

	bool debug_navigation_enable_edge_connections = true;
	bool debug_navigation_enable_edge_connections_xray = true;
	bool debug_navigation_enable_edge_lines = true;
	bool debug_navigation_enable_edge_lines_xray = true;
	bool debug_navigation_enable_geometry_face_random_color = true;
	bool debug_navigation_enable_link_connections = true;
	bool debug_navigation_enable_link_connections_xray = true;
	bool debug_navigation_enable_agent_paths = true;
	bool debug_navigation_enable_agent_paths_xray = true;

	bool debug_navigation_avoidance_enable_agents_radius = true;
	bool debug_navigation_avoidance_enable_obstacles_radius = true;
	bool debug_navigation_avoidance_enable_obstacles_static = true;

	Ref<StandardMaterial3D> debug_navigation_geometry_edge_material;
	Ref<StandardMaterial3D> debug_navigation_geometry_face_material;
	Ref<StandardMaterial3D> debug_navigation_geometry_edge_disabled_material;
	Ref<StandardMaterial3D> debug_navigation_geometry_face_disabled_material;
	Ref<StandardMaterial3D> debug_navigation_edge_connections_material;
	Ref<StandardMaterial3D> debug_navigation_link_connections_material;
	Ref<StandardMaterial3D> debug_navigation_link_connections_disabled_material;
	Ref<StandardMaterial3D> debug_navigation_avoidance_agents_radius_material;
	Ref<StandardMaterial3D> debug_navigation_avoidance_obstacles_radius_material;

	Ref<StandardMaterial3D> debug_navigation_avoidance_static_obstacle_pushin_face_material;
	Ref<StandardMaterial3D> debug_navigation_avoidance_static_obstacle_pushout_face_material;
	Ref<StandardMaterial3D> debug_navigation_avoidance_static_obstacle_pushin_edge_material;
	Ref<StandardMaterial3D> debug_navigation_avoidance_static_obstacle_pushout_edge_material;

	Ref<StandardMaterial3D> debug_navigation_agent_path_line_material;
	Ref<StandardMaterial3D> debug_navigation_agent_path_point_material;

public:
	static NavigationServer3DDebug *get_singleton();

	void set_debug_enabled(bool p_enabled);
	bool get_debug_enabled() const;

	void register_settings();
	void update_from_settings();

	void set_debug_navigation_enabled(bool p_enabled);
	bool get_debug_navigation_enabled() const;

	void set_debug_avoidance_enabled(bool p_enabled);
	bool get_debug_avoidance_enabled() const;

	void set_debug_navigation_edge_connection_color(const Color &p_color);
	Color get_debug_navigation_edge_connection_color() const;

	void set_debug_navigation_geometry_edge_color(const Color &p_color);
	Color get_debug_navigation_geometry_edge_color() const;

	void set_debug_navigation_geometry_face_color(const Color &p_color);
	Color get_debug_navigation_geometry_face_color() const;

	void set_debug_navigation_geometry_edge_disabled_color(const Color &p_color);
	Color get_debug_navigation_geometry_edge_disabled_color() const;

	void set_debug_navigation_geometry_face_disabled_color(const Color &p_color);
	Color get_debug_navigation_geometry_face_disabled_color() const;

	void set_debug_navigation_link_connection_color(const Color &p_color);
	Color get_debug_navigation_link_connection_color() const;

	void set_debug_navigation_link_connection_disabled_color(const Color &p_color);
	Color get_debug_navigation_link_connection_disabled_color() const;

	void set_debug_navigation_agent_path_color(const Color &p_color);
	Color get_debug_navigation_agent_path_color() const;

	void set_debug_navigation_avoidance_agents_radius_color(const Color &p_color);
	Color get_debug_navigation_avoidance_agents_radius_color() const;

	void set_debug_navigation_avoidance_obstacles_radius_color(const Color &p_color);
	Color get_debug_navigation_avoidance_obstacles_radius_color() const;

	void set_debug_navigation_avoidance_static_obstacle_pushin_face_color(const Color &p_color);
	Color get_debug_navigation_avoidance_static_obstacle_pushin_face_color() const;

	void set_debug_navigation_avoidance_static_obstacle_pushout_face_color(const Color &p_color);
	Color get_debug_navigation_avoidance_static_obstacle_pushout_face_color() const;

	void set_debug_navigation_avoidance_static_obstacle_pushin_edge_color(const Color &p_color);
	Color get_debug_navigation_avoidance_static_obstacle_pushin_edge_color() const;

	void set_debug_navigation_avoidance_static_obstacle_pushout_edge_color(const Color &p_color);
	Color get_debug_navigation_avoidance_static_obstacle_pushout_edge_color() const;

	void set_debug_navigation_enable_edge_connections(const bool p_value);
	bool get_debug_navigation_enable_edge_connections() const;

	void set_debug_navigation_enable_edge_connections_xray(const bool p_value);
	bool get_debug_navigation_enable_edge_connections_xray() const;

	void set_debug_navigation_enable_edge_lines(const bool p_value);
	bool get_debug_navigation_enable_edge_lines() const;

	void set_debug_navigation_enable_edge_lines_xray(const bool p_value);
	bool get_debug_navigation_enable_edge_lines_xray() const;

	void set_debug_navigation_enable_geometry_face_random_color(const bool p_value);
	bool get_debug_navigation_enable_geometry_face_random_color() const;

	void set_debug_navigation_enable_link_connections(const bool p_value);
	bool get_debug_navigation_enable_link_connections() const;

	void set_debug_navigation_enable_link_connections_xray(const bool p_value);
	bool get_debug_navigation_enable_link_connections_xray() const;

	void set_debug_navigation_enable_agent_paths(const bool p_value);
	bool get_debug_navigation_enable_agent_paths() const;

	void set_debug_navigation_enable_agent_paths_xray(const bool p_value);
	bool get_debug_navigation_enable_agent_paths_xray() const;

	void set_debug_navigation_agent_path_point_size(float p_point_size);
	float get_debug_navigation_agent_path_point_size() const;

	void set_debug_navigation_avoidance_enable_agents_radius(const bool p_value);
	bool get_debug_navigation_avoidance_enable_agents_radius() const;

	void set_debug_navigation_avoidance_enable_obstacles_radius(const bool p_value);
	bool get_debug_navigation_avoidance_enable_obstacles_radius() const;

	void set_debug_navigation_avoidance_enable_obstacles_static(const bool p_value);
	bool get_debug_navigation_avoidance_enable_obstacles_static() const;

	Ref<StandardMaterial3D> get_debug_navigation_geometry_face_material();
	Ref<StandardMaterial3D> get_debug_navigation_geometry_edge_material();
	Ref<StandardMaterial3D> get_debug_navigation_geometry_face_disabled_material();
	Ref<StandardMaterial3D> get_debug_navigation_geometry_edge_disabled_material();
	Ref<StandardMaterial3D> get_debug_navigation_edge_connections_material();
	Ref<StandardMaterial3D> get_debug_navigation_link_connections_material();
	Ref<StandardMaterial3D> get_debug_navigation_link_connections_disabled_material();

	Ref<StandardMaterial3D> get_debug_navigation_agent_path_line_material();
	Ref<StandardMaterial3D> get_debug_navigation_agent_path_point_material();

	Ref<StandardMaterial3D> get_debug_navigation_avoidance_agents_radius_material();
	Ref<StandardMaterial3D> get_debug_navigation_avoidance_obstacles_radius_material();

	Ref<StandardMaterial3D> get_debug_navigation_avoidance_static_obstacle_pushin_face_material();
	Ref<StandardMaterial3D> get_debug_navigation_avoidance_static_obstacle_pushout_face_material();
	Ref<StandardMaterial3D> get_debug_navigation_avoidance_static_obstacle_pushin_edge_material();
	Ref<StandardMaterial3D> get_debug_navigation_avoidance_static_obstacle_pushout_edge_material();

	NavigationServer3DDebug();
	~NavigationServer3DDebug();
};

#endif // DEBUG_ENABLED
