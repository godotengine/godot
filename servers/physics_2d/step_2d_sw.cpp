/*************************************************************************/
/*  step_2d_sw.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "step_2d_sw.h"

#include "core/os/os.h"

#define BODY_ISLAND_COUNT_RESERVE 128
#define BODY_ISLAND_SIZE_RESERVE 512
#define ISLAND_COUNT_RESERVE 128
#define ISLAND_SIZE_RESERVE 512
#define CONSTRAINT_COUNT_RESERVE 1024

void Step2DSW::_populate_island(Body2DSW *p_body, LocalVector<Body2DSW *> &p_body_island, LocalVector<Constraint2DSW *> &p_constraint_island) {
	p_body->set_island_step(_step);

	if (p_body->get_mode() > PhysicsServer2D::BODY_MODE_KINEMATIC) {
		// Only dynamic bodies are tested for activation.
		p_body_island.push_back(p_body);
	}

	for (const List<Pair<Constraint2DSW *, int>>::Element *E = p_body->get_constraint_list().front(); E; E = E->next()) {
		Constraint2DSW *constraint = (Constraint2DSW *)E->get().first;
		if (constraint->get_island_step() == _step) {
			continue; // Already processed.
		}
		constraint->set_island_step(_step);
		p_constraint_island.push_back(constraint);
		all_constraints.push_back(constraint);

		for (int i = 0; i < constraint->get_body_count(); i++) {
			if (i == E->get().second) {
				continue;
			}
			Body2DSW *other_body = constraint->get_body_ptr()[i];
			if (other_body->get_island_step() == _step) {
				continue; // Already processed.
			}
			if (other_body->get_mode() == PhysicsServer2D::BODY_MODE_STATIC) {
				continue; // Static bodies don't connect islands.
			}
			_populate_island(other_body, p_body_island, p_constraint_island);
		}
	}
}

void Step2DSW::_setup_contraint(uint32_t p_constraint_index, void *p_userdata) {
	Constraint2DSW *constraint = all_constraints[p_constraint_index];
	constraint->setup(delta);
}

void Step2DSW::_pre_solve_island(LocalVector<Constraint2DSW *> &p_constraint_island) const {
	uint32_t constraint_count = p_constraint_island.size();
	uint32_t valid_constraint_count = 0;
	for (uint32_t constraint_index = 0; constraint_index < constraint_count; ++constraint_index) {
		Constraint2DSW *constraint = p_constraint_island[constraint_index];
		if (p_constraint_island[constraint_index]->pre_solve(delta)) {
			// Keep this constraint for solving.
			p_constraint_island[valid_constraint_count++] = constraint;
		}
	}
	p_constraint_island.resize(valid_constraint_count);
}

void Step2DSW::_solve_island(uint32_t p_island_index, void *p_userdata) const {
	const LocalVector<Constraint2DSW *> &constraint_island = constraint_islands[p_island_index];

	for (int i = 0; i < iterations; i++) {
		uint32_t constraint_count = constraint_island.size();
		for (uint32_t constraint_index = 0; constraint_index < constraint_count; ++constraint_index) {
			constraint_island[constraint_index]->solve(delta);
		}
	}
}

void Step2DSW::_check_suspend(LocalVector<Body2DSW *> &p_body_island) const {
	bool can_sleep = true;

	uint32_t body_count = p_body_island.size();
	for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
		Body2DSW *body = p_body_island[body_index];

		if (!body->sleep_test(delta)) {
			can_sleep = false;
		}
	}

	// Put all to sleep or wake up everyone.
	for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
		Body2DSW *body = p_body_island[body_index];

		bool active = body->is_active();

		if (active == can_sleep) {
			body->set_active(!can_sleep);
		}
	}
}

void Step2DSW::step(Space2DSW *p_space, real_t p_delta, int p_iterations) {
	p_space->lock(); // can't access space during this

	p_space->setup(); //update inertias, etc

	iterations = p_iterations;
	delta = p_delta;

	const SelfList<Body2DSW>::List *body_list = &p_space->get_active_body_list();

	/* INTEGRATE FORCES */

	uint64_t profile_begtime = OS::get_singleton()->get_ticks_usec();
	uint64_t profile_endtime = 0;

	int active_count = 0;

	const SelfList<Body2DSW> *b = body_list->first();
	while (b) {
		b->self()->integrate_forces(p_delta);
		b = b->next();
		active_count++;
	}

	p_space->set_active_objects(active_count);

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space2DSW::ELAPSED_TIME_INTEGRATE_FORCES, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* GENERATE CONSTRAINT ISLANDS FOR MOVING AREAS */

	uint32_t island_count = 0;

	const SelfList<Area2DSW>::List &aml = p_space->get_moved_area_list();

	while (aml.first()) {
		for (const Set<Constraint2DSW *>::Element *E = aml.first()->self()->get_constraints().front(); E; E = E->next()) {
			Constraint2DSW *constraint = E->get();
			if (constraint->get_island_step() == _step) {
				continue;
			}
			constraint->set_island_step(_step);

			// Each constraint can be on a separate island for areas as there's no solving phase.
			++island_count;
			if (constraint_islands.size() < island_count) {
				constraint_islands.resize(island_count);
			}
			LocalVector<Constraint2DSW *> &constraint_island = constraint_islands[island_count - 1];
			constraint_island.clear();

			all_constraints.push_back(constraint);
			constraint_island.push_back(constraint);
		}
		p_space->area_remove_from_moved_list((SelfList<Area2DSW> *)aml.first()); //faster to remove here
	}

	/* GENERATE CONSTRAINT ISLANDS FOR ACTIVE RIGID BODIES */

	b = body_list->first();

	uint32_t body_island_count = 0;

	while (b) {
		Body2DSW *body = b->self();

		if (body->get_island_step() != _step) {
			++body_island_count;
			if (body_islands.size() < body_island_count) {
				body_islands.resize(body_island_count);
			}
			LocalVector<Body2DSW *> &body_island = body_islands[body_island_count - 1];
			body_island.clear();
			body_island.reserve(BODY_ISLAND_SIZE_RESERVE);

			++island_count;
			if (constraint_islands.size() < island_count) {
				constraint_islands.resize(island_count);
			}
			LocalVector<Constraint2DSW *> &constraint_island = constraint_islands[island_count - 1];
			constraint_island.clear();
			constraint_island.reserve(ISLAND_SIZE_RESERVE);

			_populate_island(body, body_island, constraint_island);

			if (body_island.is_empty()) {
				--body_island_count;
			}

			if (constraint_island.is_empty()) {
				--island_count;
			}
		}
		b = b->next();
	}

	p_space->set_island_count((int)island_count);

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space2DSW::ELAPSED_TIME_GENERATE_ISLANDS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* SETUP CONSTRAINTS / PROCESS COLLISIONS */

	uint32_t total_contraint_count = all_constraints.size();
	work_pool.do_work(total_contraint_count, this, &Step2DSW::_setup_contraint, nullptr);

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space2DSW::ELAPSED_TIME_SETUP_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* PRE-SOLVE CONSTRAINT ISLANDS */

	// Warning: This doesn't run on threads, because it involves thread-unsafe processing.
	for (uint32_t island_index = 0; island_index < island_count; ++island_index) {
		_pre_solve_island(constraint_islands[island_index]);
	}

	/* SOLVE CONSTRAINT ISLANDS */

	// Warning: _solve_island modifies the constraint islands for optimization purpose,
	// their content is not reliable after these calls and shouldn't be used anymore.
	if (island_count > 1) {
		work_pool.do_work(island_count, this, &Step2DSW::_solve_island, nullptr);
	} else if (island_count > 0) {
		_solve_island(0);
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space2DSW::ELAPSED_TIME_SOLVE_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* INTEGRATE VELOCITIES */

	b = body_list->first();
	while (b) {
		const SelfList<Body2DSW> *n = b->next();
		b->self()->integrate_velocities(p_delta);
		b = n; // in case it shuts itself down
	}

	/* SLEEP / WAKE UP ISLANDS */

	for (uint32_t island_index = 0; island_index < body_island_count; ++island_index) {
		_check_suspend(body_islands[island_index]);
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space2DSW::ELAPSED_TIME_INTEGRATE_VELOCITIES, profile_endtime - profile_begtime);
		//profile_begtime=profile_endtime;
	}

	all_constraints.clear();

	p_space->update();
	p_space->unlock();
	_step++;
}

Step2DSW::Step2DSW() {
	_step = 1;

	body_islands.reserve(BODY_ISLAND_COUNT_RESERVE);
	constraint_islands.reserve(ISLAND_COUNT_RESERVE);
	all_constraints.reserve(CONSTRAINT_COUNT_RESERVE);

	work_pool.init();
}

Step2DSW::~Step2DSW() {
	work_pool.finish();
}
