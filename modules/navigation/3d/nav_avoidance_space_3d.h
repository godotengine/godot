/**************************************************************************/
/*  nav_avoidance_space_3d.h                                              */
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

#ifndef NAV_AVOIDANCE_SPACE_3D_H
#define NAV_AVOIDANCE_SPACE_3D_H

#include "../nav_agent.h"
#include "../nav_obstacle.h"
#include "../nav_rid.h"
#include "../nav_utils.h"

#include "core/math/math_defs.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/rw_lock.h"

#include "servers/navigation_server_3d.h"

#include <KdTree2d.h>
#include <KdTree3d.h>
#include <RVOSimulator2d.h>
#include <RVOSimulator3d.h>

class NavAvoidanceSpace3D : public NavRid {
	RWLock space_rwlock;

	bool obstacles_dirty = false;
	bool agents_dirty = false;

	uint32_t iteration_id = 0;

	RVO2D::RVOSimulator2D rvo_simulation_2d;
	RVO3D::RVOSimulator3D rvo_simulation_3d;

	LocalVector<NavAgent *> active_2d_avoidance_agents;
	LocalVector<NavAgent *> active_3d_avoidance_agents;

	LocalVector<NavAgent *> agents;
	LocalVector<NavObstacle *> obstacles;
	real_t deltatime = 0.0;

	static bool use_threads;
	static bool avoidance_use_multiple_threads;
	static bool avoidance_use_high_priority_threads;

	int pm_agent_count = 0;
	int pm_obstacle_count = 0;

	void compute_single_step(uint32_t index, NavAgent **agent);
	void compute_single_avoidance_step_2d(uint32_t index, NavAgent **agent);
	void compute_single_avoidance_step_3d(uint32_t index, NavAgent **agent);
	void _update_rvo_simulation();
	void _update_rvo_obstacles_tree_2d();
	void _update_rvo_agents_tree_2d();
	void _update_rvo_agents_tree_3d();

public:
	uint32_t get_iteration_id() { return iteration_id; };

	bool has_agent(NavAgent *agent) const;
	void add_agent(NavAgent *agent);
	void remove_agent(NavAgent *agent);
	const LocalVector<NavAgent *> &get_agents() const { return agents; }

	bool has_obstacle(NavObstacle *obstacle) const;
	void add_obstacle(NavObstacle *obstacle);
	void remove_obstacle(NavObstacle *obstacle);
	const LocalVector<NavObstacle *> &get_obstacles() const { return obstacles; }

	void sync();
	void step(real_t p_deltatime);
	void dispatch_callbacks();

	int get_pm_agent_count() const { return pm_agent_count; }
	int get_pm_obstacle_count() const { return pm_obstacle_count; }

	NavAvoidanceSpace3D();
	~NavAvoidanceSpace3D();
};

#endif // NAV_AVOIDANCE_SPACE_3D_H
