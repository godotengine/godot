/**************************************************************************/
/*  nav_avoidance_space_3d.cpp                                            */
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

#include "nav_avoidance_space_3d.h"

#include "core/config/project_settings.h"
#include "core/object/worker_thread_pool.h"

#include <Obstacle2d.h>

bool NavAvoidanceSpace3D::use_threads = true;
bool NavAvoidanceSpace3D::avoidance_use_multiple_threads = true;
bool NavAvoidanceSpace3D::avoidance_use_high_priority_threads = true;

bool NavAvoidanceSpace3D::has_agent(NavAgent *agent) const {
	return agents.find(agent) >= 0;
}

void NavAvoidanceSpace3D::add_agent(NavAgent *agent) {
	if (agent->get_paused()) {
		// No point in adding a paused agent, it will add itself when unpaused again.
		return;
	}

	if (!has_agent(agent)) {
		agents.push_back(agent);
		agents_dirty = true;
	}

	if (agent->get_use_3d_avoidance()) {
		int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
		if (agent_3d_index < 0) {
			active_3d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	} else {
		int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
		if (agent_2d_index < 0) {
			active_2d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	}
}

void NavAvoidanceSpace3D::remove_agent(NavAgent *agent) {
	int64_t agent_index = agents.find(agent);
	if (agent_index >= 0) {
		agents.remove_at_unordered(agent_index);
		agents_dirty = true;
	}

	int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
	if (agent_3d_index >= 0) {
		active_3d_avoidance_agents.remove_at_unordered(agent_3d_index);
		agents_dirty = true;
	}
	int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
	if (agent_2d_index >= 0) {
		active_2d_avoidance_agents.remove_at_unordered(agent_2d_index);
		agents_dirty = true;
	}
}

bool NavAvoidanceSpace3D::has_obstacle(NavObstacle *obstacle) const {
	return obstacles.find(obstacle) >= 0;
}

void NavAvoidanceSpace3D::add_obstacle(NavObstacle *obstacle) {
	if (obstacle->get_paused()) {
		// No point in adding a paused obstacle, it will add itself when unpaused again.
		return;
	}

	if (!has_obstacle(obstacle)) {
		obstacles.push_back(obstacle);
		obstacles_dirty = true;
	}
}

void NavAvoidanceSpace3D::remove_obstacle(NavObstacle *obstacle) {
	int64_t obstacle_index = obstacles.find(obstacle);
	if (obstacle_index >= 0) {
		obstacles.remove_at_unordered(obstacle_index);
		obstacles_dirty = true;
	}
}

void NavAvoidanceSpace3D::sync() {
	RWLockWrite write_lock(space_rwlock);

	int _new_pm_agent_count = agents.size();
	int _new_pm_obstacle_count = obstacles.size();

	for (NavObstacle *obstacle : obstacles) {
		if (obstacle->check_dirty()) {
			obstacles_dirty = true;
		}
	}

	for (NavAgent *agent : agents) {
		if (agent->check_dirty()) {
			agents_dirty = true;
		}
	}

	if (obstacles_dirty || agents_dirty) {
		_update_rvo_simulation();
	}

	agents_dirty = false;
	obstacles_dirty = false;

	pm_agent_count = _new_pm_agent_count;
	pm_obstacle_count = _new_pm_obstacle_count;
}

void NavAvoidanceSpace3D::_update_rvo_obstacles_tree_2d() {
	int obstacle_vertex_count = 0;
	for (NavObstacle *obstacle : obstacles) {
		obstacle_vertex_count += obstacle->get_vertices().size();
	}

	// Cleaning old obstacles.
	for (size_t i = 0; i < rvo_simulation_2d.obstacles_.size(); ++i) {
		delete rvo_simulation_2d.obstacles_[i];
	}
	rvo_simulation_2d.obstacles_.clear();

	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree
	std::vector<RVO2D::Obstacle2D *> &raw_obstacles = rvo_simulation_2d.obstacles_;
	raw_obstacles.reserve(obstacle_vertex_count);

	// The following block is modified copy from RVO2D::AddObstacle()
	// Obstacles are linked and depend on all other obstacles.
	for (NavObstacle *obstacle : obstacles) {
		const Vector3 &_obstacle_position = obstacle->get_position();
		const Vector<Vector3> &_obstacle_vertices = obstacle->get_vertices();

		if (_obstacle_vertices.size() < 2) {
			continue;
		}

		std::vector<RVO2D::Vector2> rvo_2d_vertices;
		rvo_2d_vertices.reserve(_obstacle_vertices.size());

		uint32_t _obstacle_avoidance_layers = obstacle->get_avoidance_layers();
		real_t _obstacle_height = obstacle->get_height();

		for (const Vector3 &_obstacle_vertex : _obstacle_vertices) {
#ifdef TOOLS_ENABLED
			if (_obstacle_vertex.y != 0) {
				WARN_PRINT_ONCE("Y coordinates of static obstacle vertices are ignored. Please use obstacle position Y to change elevation of obstacle.");
			}
#endif
			rvo_2d_vertices.push_back(RVO2D::Vector2(_obstacle_vertex.x + _obstacle_position.x, _obstacle_vertex.z + _obstacle_position.z));
		}

		const size_t obstacleNo = raw_obstacles.size();

		for (size_t i = 0; i < rvo_2d_vertices.size(); i++) {
			RVO2D::Obstacle2D *rvo_2d_obstacle = new RVO2D::Obstacle2D();
			rvo_2d_obstacle->point_ = rvo_2d_vertices[i];
			rvo_2d_obstacle->height_ = _obstacle_height;
			rvo_2d_obstacle->elevation_ = _obstacle_position.y;

			rvo_2d_obstacle->avoidance_layers_ = _obstacle_avoidance_layers;

			if (i != 0) {
				rvo_2d_obstacle->prevObstacle_ = raw_obstacles.back();
				rvo_2d_obstacle->prevObstacle_->nextObstacle_ = rvo_2d_obstacle;
			}

			if (i == rvo_2d_vertices.size() - 1) {
				rvo_2d_obstacle->nextObstacle_ = raw_obstacles[obstacleNo];
				rvo_2d_obstacle->nextObstacle_->prevObstacle_ = rvo_2d_obstacle;
			}

			rvo_2d_obstacle->unitDir_ = normalize(rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)] - rvo_2d_vertices[i]);

			if (rvo_2d_vertices.size() == 2) {
				rvo_2d_obstacle->isConvex_ = true;
			} else {
				rvo_2d_obstacle->isConvex_ = (leftOf(rvo_2d_vertices[(i == 0 ? rvo_2d_vertices.size() - 1 : i - 1)], rvo_2d_vertices[i], rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)]) >= 0.0f);
			}

			rvo_2d_obstacle->id_ = raw_obstacles.size();

			raw_obstacles.push_back(rvo_2d_obstacle);
		}
	}

	rvo_simulation_2d.kdTree_->buildObstacleTree(raw_obstacles);
}

void NavAvoidanceSpace3D::_update_rvo_agents_tree_2d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO2D::Agent2D *> raw_agents;
	raw_agents.reserve(active_2d_avoidance_agents.size());
	for (NavAgent *agent : active_2d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_2d());
	}
	rvo_simulation_2d.kdTree_->buildAgentTree(raw_agents);
}

void NavAvoidanceSpace3D::_update_rvo_agents_tree_3d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO3D::Agent3D *> raw_agents;
	raw_agents.reserve(active_3d_avoidance_agents.size());
	for (NavAgent *agent : active_3d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_3d());
	}
	rvo_simulation_3d.kdTree_->buildAgentTree(raw_agents);
}

void NavAvoidanceSpace3D::_update_rvo_simulation() {
	if (obstacles_dirty) {
		_update_rvo_obstacles_tree_2d();
	}
	if (agents_dirty) {
		_update_rvo_agents_tree_2d();
		_update_rvo_agents_tree_3d();
	}
}

void NavAvoidanceSpace3D::compute_single_avoidance_step_2d(uint32_t index, NavAgent **agent) {
	NavAgent *nav_agent = (*(agent + index));
	nav_agent->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
	nav_agent->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
	nav_agent->get_rvo_agent_2d()->update(&rvo_simulation_2d);
	nav_agent->update();
}

void NavAvoidanceSpace3D::compute_single_avoidance_step_3d(uint32_t index, NavAgent **agent) {
	NavAgent *nav_agent = (*(agent + index));
	nav_agent->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
	nav_agent->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
	nav_agent->get_rvo_agent_3d()->update(&rvo_simulation_3d);
	nav_agent->update();
}

void NavAvoidanceSpace3D::step(real_t p_deltatime) {
	RWLockWrite write_lock(space_rwlock);

	deltatime = p_deltatime;

	rvo_simulation_2d.setTimeStep(float(deltatime));
	rvo_simulation_3d.setTimeStep(float(deltatime));

	if (active_2d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavAvoidanceSpace3D::compute_single_avoidance_step_2d, active_2d_avoidance_agents.ptr(), active_2d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents2D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_2d_avoidance_agents) {
				agent->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->update(&rvo_simulation_2d);
				agent->update();
			}
		}
	}

	if (active_3d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavAvoidanceSpace3D::compute_single_avoidance_step_3d, active_3d_avoidance_agents.ptr(), active_3d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents3D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_3d_avoidance_agents) {
				agent->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->update(&rvo_simulation_3d);
				agent->update();
			}
		}
	}

	iteration_id = iteration_id % UINT32_MAX + 1;
}

void NavAvoidanceSpace3D::dispatch_callbacks() {
	for (NavAgent *agent : active_2d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}

	for (NavAgent *agent : active_3d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}
}

NavAvoidanceSpace3D::NavAvoidanceSpace3D() {
	avoidance_use_multiple_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_multiple_threads");
	avoidance_use_high_priority_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_high_priority_threads");
}

NavAvoidanceSpace3D::~NavAvoidanceSpace3D() {
}
