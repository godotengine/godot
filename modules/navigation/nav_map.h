/*************************************************************************/
/*  nav_map.h                                                            */
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

#ifndef NAV_MAP_H
#define NAV_MAP_H

#include "nav_rid.h"

#include "core/map.h"
#include "core/math/math_defs.h"
#include "core/os/thread_work_pool.h"
#include "nav_utils.h"

#include <KdTree.h>

class NavRegion;
class RvoAgent;
class NavRegion;

class NavMap : public NavRid {
	/// Map Up
	Vector3 up = Vector3(0, 1, 0);

	/// To find the polygons edges the vertices are displaced in a grid where
	/// each cell has the following cell_size and cell_height.
	real_t cell_size = 0.25;
	real_t cell_height = 0.25;

	/// This value is used to detect the near edges to connect.
	real_t edge_connection_margin = 0.25;

	bool regenerate_polygons = true;
	bool regenerate_links = true;

	LocalVector<NavRegion *> regions;

	/// Map polygons
	LocalVector<gd::Polygon> polygons;

	/// Rvo world
	RVO::KdTree rvo;

	/// Is agent array modified?
	bool agents_dirty = false;

	/// All the Agents (even the controlled one)
	LocalVector<RvoAgent *> agents;

	/// Controlled agents
	LocalVector<RvoAgent *> controlled_agents;

	/// Physics delta time
	real_t deltatime = 0.0;

	/// Change the id each time the map is updated.
	uint32_t map_update_id = 0;

#ifndef NO_THREADS
	/// Pooled threads for computing steps
	ThreadWorkPool step_work_pool;
#endif // NO_THREADS

public:
	NavMap();
	~NavMap();

	void set_up(Vector3 p_up);
	Vector3 get_up() const {
		return up;
	}

	void set_cell_size(float p_cell_size);
	float get_cell_size() const {
		return cell_size;
	}

	void set_cell_height(float p_cell_height);
	float get_cell_height() const {
		return cell_height;
	}

	void set_edge_connection_margin(float p_edge_connection_margin);
	float get_edge_connection_margin() const {
		return edge_connection_margin;
	}

	gd::PointKey get_point_key(const Vector3 &p_pos) const;

	Vector<Vector3> get_path(Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers = 1) const;
	Vector3 get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const;
	Vector3 get_closest_point(const Vector3 &p_point) const;
	Vector3 get_closest_point_normal(const Vector3 &p_point) const;
	gd::ClosestPointQueryResult get_closest_point_info(const Vector3 &p_point) const;
	RID get_closest_point_owner(const Vector3 &p_point) const;

	void add_region(NavRegion *p_region);
	void remove_region(NavRegion *p_region);
	const LocalVector<NavRegion *> &get_regions() const {
		return regions;
	}

	bool has_agent(RvoAgent *agent) const;
	void add_agent(RvoAgent *agent);
	void remove_agent(RvoAgent *agent);
	const LocalVector<RvoAgent *> &get_agents() const {
		return agents;
	}

	void set_agent_as_controlled(RvoAgent *agent);
	void remove_agent_as_controlled(RvoAgent *agent);

	uint32_t get_map_update_id() const {
		return map_update_id;
	}

	void sync();
	void step(real_t p_deltatime);
	void dispatch_callbacks();

private:
	void compute_single_step(uint32_t index, RvoAgent **agent);
	void clip_path(const LocalVector<gd::NavigationPoly> &p_navigation_polys, Vector<Vector3> &path, const gd::NavigationPoly *from_poly, const Vector3 &p_to_point, const gd::NavigationPoly *p_to_poly) const;
};

#endif // NAV_MAP_H
