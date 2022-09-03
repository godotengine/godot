/*************************************************************************/
/*  navigation_server_2d.h                                               */
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

#ifndef NAVIGATION_SERVER_2D_H
#define NAVIGATION_SERVER_2D_H

#include "core/object/class_db.h"
#include "core/templates/rid.h"
#include "scene/2d/navigation_region_2d.h"

// This server exposes the `NavigationServer3D` features in the 2D world.
class NavigationServer2D : public Object {
	GDCLASS(NavigationServer2D, Object);

	static NavigationServer2D *singleton;

	void _emit_map_changed(RID p_map);

protected:
	static void _bind_methods();

public:
	/// Thread safe, can be used across many threads.
	static const NavigationServer2D *get_singleton() { return singleton; }

	/// MUST be used in single thread!
	static NavigationServer2D *get_singleton_mut() { return singleton; }

	virtual TypedArray<RID> get_maps() const;

	/// Create a new map.
	virtual RID map_create() const;

	/// Set map active.
	virtual void map_set_active(RID p_map, bool p_active) const;

	/// Returns true if the map is active.
	virtual bool map_is_active(RID p_map) const;

	/// Set the map cell size used to weld the navigation mesh polygons.
	virtual void map_set_cell_size(RID p_map, real_t p_cell_size) const;

	/// Returns the map cell size.
	virtual real_t map_get_cell_size(RID p_map) const;

	/// Set the map edge connection margin used to weld the compatible region edges.
	virtual void map_set_edge_connection_margin(RID p_map, real_t p_connection_margin) const;

	/// Returns the edge connection margin of this map.
	virtual real_t map_get_edge_connection_margin(RID p_map) const;

	/// Set the map link connection radius used to attach links to the nav mesh.
	virtual void map_set_link_connection_radius(RID p_map, real_t p_connection_radius) const;

	/// Returns the link connection radius of this map.
	virtual real_t map_get_link_connection_radius(RID p_map) const;

	/// Returns the navigation path to reach the destination from the origin.
	virtual Vector<Vector2> map_get_path(RID p_map, Vector2 p_origin, Vector2 p_destination, bool p_optimize, uint32_t p_navigation_layers = 1) const;

	virtual Vector2 map_get_closest_point(RID p_map, const Vector2 &p_point) const;
	virtual RID map_get_closest_point_owner(RID p_map, const Vector2 &p_point) const;

	virtual TypedArray<RID> map_get_links(RID p_map) const;
	virtual TypedArray<RID> map_get_regions(RID p_map) const;
	virtual TypedArray<RID> map_get_agents(RID p_map) const;

	virtual void map_force_update(RID p_map);

	/// Creates a new region.
	virtual RID region_create() const;

	/// Set the enter_cost of a region
	virtual void region_set_enter_cost(RID p_region, real_t p_enter_cost) const;
	virtual real_t region_get_enter_cost(RID p_region) const;

	/// Set the travel_cost of a region
	virtual void region_set_travel_cost(RID p_region, real_t p_travel_cost) const;
	virtual real_t region_get_travel_cost(RID p_region) const;

	virtual bool region_owns_point(RID p_region, const Vector2 &p_point) const;

	/// Set the map of this region.
	virtual void region_set_map(RID p_region, RID p_map) const;
	virtual RID region_get_map(RID p_region) const;

	/// Set the region's layers
	virtual void region_set_navigation_layers(RID p_region, uint32_t p_navigation_layers) const;
	virtual uint32_t region_get_navigation_layers(RID p_region) const;

	/// Set the global transformation of this region.
	virtual void region_set_transform(RID p_region, Transform2D p_transform) const;

	/// Set the navigation poly of this region.
	virtual void region_set_navpoly(RID p_region, Ref<NavigationPolygon> p_nav_mesh) const;

	/// Get a list of a region's connection to other regions.
	virtual int region_get_connections_count(RID p_region) const;
	virtual Vector2 region_get_connection_pathway_start(RID p_region, int p_connection_id) const;
	virtual Vector2 region_get_connection_pathway_end(RID p_region, int p_connection_id) const;

	/// Creates a new link between locations in the nav map.
	virtual RID link_create() const;

	/// Set the map of this link.
	virtual void link_set_map(RID p_link, RID p_map) const;
	virtual RID link_get_map(RID p_link) const;

	/// Set whether this link travels in both directions.
	virtual void link_set_bidirectional(RID p_link, bool p_bidirectional) const;
	virtual bool link_is_bidirectional(RID p_link) const;

	/// Set the link's layers.
	virtual void link_set_navigation_layers(RID p_link, uint32_t p_navigation_layers) const;
	virtual uint32_t link_get_navigation_layers(RID p_link) const;

	/// Set the start location of the link.
	virtual void link_set_start_location(RID p_link, Vector2 p_location) const;
	virtual Vector2 link_get_start_location(RID p_link) const;

	/// Set the end location of the link.
	virtual void link_set_end_location(RID p_link, Vector2 p_location) const;
	virtual Vector2 link_get_end_location(RID p_link) const;

	/// Set the enter cost of the link.
	virtual void link_set_enter_cost(RID p_link, real_t p_enter_cost) const;
	virtual real_t link_get_enter_cost(RID p_link) const;

	/// Set the travel cost of the link.
	virtual void link_set_travel_cost(RID p_link, real_t p_travel_cost) const;
	virtual real_t link_get_travel_cost(RID p_link) const;

	/// Creates the agent.
	virtual RID agent_create() const;

	/// Put the agent in the map.
	virtual void agent_set_map(RID p_agent, RID p_map) const;
	virtual RID agent_get_map(RID p_agent) const;

	/// The maximum distance (center point to
	/// center point) to other agents this agent
	/// takes into account in the navigation. The
	/// larger this number, the longer the running
	/// time of the simulation. If the number is too
	/// low, the simulation will not be safe.
	/// Must be non-negative.
	virtual void agent_set_neighbor_distance(RID p_agent, real_t p_distance) const;

	/// The maximum number of other agents this
	/// agent takes into account in the navigation.
	/// The larger this number, the longer the
	/// running time of the simulation. If the
	/// number is too low, the simulation will not
	/// be safe.
	virtual void agent_set_max_neighbors(RID p_agent, int p_count) const;

	/// The minimal amount of time for which this
	/// agent's velocities that are computed by the
	/// simulation are safe with respect to other
	/// agents. The larger this number, the sooner
	/// this agent will respond to the presence of
	/// other agents, but the less freedom this
	/// agent has in choosing its velocities.
	/// Must be positive.
	virtual void agent_set_time_horizon(RID p_agent, real_t p_time) const;

	/// The radius of this agent.
	/// Must be non-negative.
	virtual void agent_set_radius(RID p_agent, real_t p_radius) const;

	/// The maximum speed of this agent.
	/// Must be non-negative.
	virtual void agent_set_max_speed(RID p_agent, real_t p_max_speed) const;

	/// Current velocity of the agent
	virtual void agent_set_velocity(RID p_agent, Vector2 p_velocity) const;

	/// The new target velocity.
	virtual void agent_set_target_velocity(RID p_agent, Vector2 p_velocity) const;

	/// Position of the agent in world space.
	virtual void agent_set_position(RID p_agent, Vector2 p_position) const;

	/// Agent ignore the Y axis and avoid collisions by moving only on the horizontal plane
	virtual void agent_set_ignore_y(RID p_agent, bool p_ignore) const;

	/// Returns true if the map got changed the previous frame.
	virtual bool agent_is_map_changed(RID p_agent) const;

	/// Callback called at the end of the RVO process
	virtual void agent_set_callback(RID p_agent, Object *p_receiver, StringName p_method, Variant p_udata = Variant()) const;

	/// Destroy the `RID`
	virtual void free(RID p_object) const;

	NavigationServer2D();
	virtual ~NavigationServer2D();

#ifdef DEBUG_ENABLED
	void set_debug_enabled(bool p_enabled);
	bool get_debug_enabled() const;

	void set_debug_navigation_edge_connection_color(const Color &p_color);
	Color get_debug_navigation_edge_connection_color() const;

	void set_debug_navigation_geometry_face_color(const Color &p_color);
	Color get_debug_navigation_geometry_face_color() const;

	void set_debug_navigation_geometry_face_disabled_color(const Color &p_color);
	Color get_debug_navigation_geometry_face_disabled_color() const;

	void set_debug_navigation_link_connection_color(const Color &p_color);
	Color get_debug_navigation_link_connection_color() const;

	void set_debug_navigation_link_connection_disabled_color(const Color &p_color);
	Color get_debug_navigation_link_connection_disabled_color() const;

	void set_debug_navigation_enable_edge_connections(const bool p_value);
	bool get_debug_navigation_enable_edge_connections() const;
#endif // DEBUG_ENABLED
};

#endif // NAVIGATION_SERVER_2D_H
