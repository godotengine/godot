/*************************************************************************/
/*  step_3d_sw.cpp                                                       */
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

#include "step_3d_sw.h"
#include "joints_3d_sw.h"

#include "core/os/os.h"

#define BODY_ISLAND_COUNT_RESERVE 128
#define BODY_ISLAND_SIZE_RESERVE 512
#define ISLAND_COUNT_RESERVE 128
#define ISLAND_SIZE_RESERVE 512

void Step3DSW::_populate_island(Body3DSW *p_body, LocalVector<Body3DSW *> &p_body_island, LocalVector<Constraint3DSW *> &p_constraint_island) {
	p_body->set_island_step(_step);
	p_body_island.push_back(p_body);

	// Faster with reversed iterations.
	for (Map<Constraint3DSW *, int>::Element *E = p_body->get_constraint_map().back(); E; E = E->prev()) {
		Constraint3DSW *c = (Constraint3DSW *)E->key();
		if (c->get_island_step() == _step) {
			continue; //already processed
		}
		c->set_island_step(_step);
		p_constraint_island.push_back(c);

		for (int i = 0; i < c->get_body_count(); i++) {
			if (i == E->get()) {
				continue;
			}
			Body3DSW *b = c->get_body_ptr()[i];
			if (b->get_island_step() == _step || b->get_mode() == PhysicsServer3D::BODY_MODE_STATIC || b->get_mode() == PhysicsServer3D::BODY_MODE_KINEMATIC) {
				continue; //no go
			}
			_populate_island(c->get_body_ptr()[i], p_body_island, p_constraint_island);
		}
	}
}

void Step3DSW::_setup_island(LocalVector<Constraint3DSW *> &p_constraint_island, real_t p_delta) {
	uint32_t constraint_count = p_constraint_island.size();
	uint32_t valid_constraint_count = 0;
	for (uint32_t constraint_index = 0; constraint_index < constraint_count; ++constraint_index) {
		Constraint3DSW *constraint = p_constraint_island[constraint_index];
		if (p_constraint_island[constraint_index]->setup(p_delta)) {
			// Keep this constraint for solving.
			p_constraint_island[valid_constraint_count++] = constraint;
		}
	}
	p_constraint_island.resize(valid_constraint_count);
}

void Step3DSW::_solve_island(LocalVector<Constraint3DSW *> &p_constraint_island, int p_iterations, real_t p_delta) {
	int current_priority = 1;

	uint32_t constraint_count = p_constraint_island.size();
	while (constraint_count > 0) {
		for (int i = 0; i < p_iterations; i++) {
			// Go through all iterations.
			for (uint32_t constraint_index = 0; constraint_index < constraint_count; ++constraint_index) {
				p_constraint_island[constraint_index]->solve(p_delta);
			}
		}

		// Check priority to keep only higher priority constraints.
		uint32_t priority_constraint_count = 0;
		++current_priority;
		for (uint32_t constraint_index = 0; constraint_index < constraint_count; ++constraint_index) {
			Constraint3DSW *constraint = p_constraint_island[constraint_index];
			if (constraint->get_priority() >= current_priority) {
				// Keep this constraint for the next iteration.
				p_constraint_island[priority_constraint_count++] = constraint;
			}
		}
		constraint_count = priority_constraint_count;
	}
}

void Step3DSW::_check_suspend(const LocalVector<Body3DSW *> &p_body_island, real_t p_delta) {
	bool can_sleep = true;

	uint32_t body_count = p_body_island.size();
	for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
		Body3DSW *body = p_body_island[body_index];

		if (body->get_mode() == PhysicsServer3D::BODY_MODE_STATIC || body->get_mode() == PhysicsServer3D::BODY_MODE_KINEMATIC) {
			continue; // Ignore for static.
		}

		if (!body->sleep_test(p_delta)) {
			can_sleep = false;
		}
	}

	// Put all to sleep or wake up everyone.
	for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
		Body3DSW *body = p_body_island[body_index];

		if (body->get_mode() == PhysicsServer3D::BODY_MODE_STATIC || body->get_mode() == PhysicsServer3D::BODY_MODE_KINEMATIC) {
			continue; // Ignore for static.
		}

		bool active = body->is_active();

		if (active == can_sleep) {
			body->set_active(!can_sleep);
		}
	}
}

void Step3DSW::step(Space3DSW *p_space, real_t p_delta, int p_iterations) {
	p_space->lock(); // can't access space during this

	p_space->setup(); //update inertias, etc

	const SelfList<Body3DSW>::List *body_list = &p_space->get_active_body_list();

	const SelfList<SoftBody3DSW>::List *soft_body_list = &p_space->get_active_soft_body_list();

	/* INTEGRATE FORCES */

	uint64_t profile_begtime = OS::get_singleton()->get_ticks_usec();
	uint64_t profile_endtime = 0;

	int active_count = 0;

	const SelfList<Body3DSW> *b = body_list->first();
	while (b) {
		b->self()->integrate_forces(p_delta);
		b = b->next();
		active_count++;
	}

	/* UPDATE SOFT BODY MOTION */

	const SelfList<SoftBody3DSW> *sb = soft_body_list->first();
	while (sb) {
		sb->self()->predict_motion(p_delta);
		sb = sb->next();
		active_count++;
	}

	p_space->set_active_objects(active_count);

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space3DSW::ELAPSED_TIME_INTEGRATE_FORCES, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* GENERATE CONSTRAINT ISLANDS */

	b = body_list->first();

	uint32_t body_island_count = 0;
	uint32_t island_count = 0;

	while (b) {
		Body3DSW *body = b->self();

		if (body->get_island_step() != _step) {
			++body_island_count;
			if (body_islands.size() < body_island_count) {
				body_islands.resize(body_island_count);
			}
			LocalVector<Body3DSW *> &body_island = body_islands[body_island_count - 1];
			body_island.clear();
			body_island.reserve(BODY_ISLAND_SIZE_RESERVE);

			++island_count;
			if (constraint_islands.size() < island_count) {
				constraint_islands.resize(island_count);
			}
			LocalVector<Constraint3DSW *> &constraint_island = constraint_islands[island_count - 1];
			constraint_island.clear();
			constraint_island.reserve(ISLAND_SIZE_RESERVE);

			_populate_island(body, body_island, constraint_island);

			body_islands.push_back(body_island);

			if (constraint_island.is_empty()) {
				--island_count;
			}
		}
		b = b->next();
	}

	p_space->set_island_count((int)island_count);

	const SelfList<Area3DSW>::List &aml = p_space->get_moved_area_list();

	while (aml.first()) {
		for (const Set<Constraint3DSW *>::Element *E = aml.first()->self()->get_constraints().front(); E; E = E->next()) {
			Constraint3DSW *c = E->get();
			if (c->get_island_step() == _step) {
				continue;
			}
			c->set_island_step(_step);
			++island_count;
			if (constraint_islands.size() < island_count) {
				constraint_islands.resize(island_count);
			}
			LocalVector<Constraint3DSW *> &constraint_island = constraint_islands[island_count - 1];
			constraint_island.clear();
			constraint_island.push_back(c);
		}
		p_space->area_remove_from_moved_list((SelfList<Area3DSW> *)aml.first()); //faster to remove here
	}

	sb = soft_body_list->first();
	while (sb) {
		for (const Set<Constraint3DSW *>::Element *E = sb->self()->get_constraints().front(); E; E = E->next()) {
			Constraint3DSW *c = E->get();
			if (c->get_island_step() == _step) {
				continue;
			}
			c->set_island_step(_step);
			++island_count;
			if (constraint_islands.size() < island_count) {
				constraint_islands.resize(island_count);
			}
			LocalVector<Constraint3DSW *> &constraint_island = constraint_islands[island_count - 1];
			constraint_island.clear();
			constraint_island.push_back(c);
		}
		sb = sb->next();
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space3DSW::ELAPSED_TIME_GENERATE_ISLANDS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* SETUP CONSTRAINT ISLANDS */

	for (uint32_t island_index = 0; island_index < island_count; ++island_index) {
		_setup_island(constraint_islands[island_index], p_delta);
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space3DSW::ELAPSED_TIME_SETUP_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* SOLVE CONSTRAINT ISLANDS */

	for (uint32_t island_index = 0; island_index < island_count; ++island_index) {
		// Warning: _solve_island modifies the constraint islands for optimization purpose,
		// their content is not reliable after these calls and shouldn't be used anymore.
		_solve_island(constraint_islands[island_index], p_iterations, p_delta);
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space3DSW::ELAPSED_TIME_SOLVE_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* INTEGRATE VELOCITIES */

	b = body_list->first();
	while (b) {
		const SelfList<Body3DSW> *n = b->next();
		b->self()->integrate_velocities(p_delta);
		b = n;
	}

	/* SLEEP / WAKE UP ISLANDS */

	for (uint32_t island_index = 0; island_index < body_island_count; ++island_index) {
		_check_suspend(body_islands[island_index], p_delta);
	}

	/* UPDATE SOFT BODY CONSTRAINTS */

	sb = soft_body_list->first();
	while (sb) {
		sb->self()->solve_constraints(p_delta);
		sb = sb->next();
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(Space3DSW::ELAPSED_TIME_INTEGRATE_VELOCITIES, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	p_space->update();
	p_space->unlock();
	_step++;
}

Step3DSW::Step3DSW() {
	_step = 1;

	body_islands.reserve(BODY_ISLAND_COUNT_RESERVE);
	constraint_islands.reserve(ISLAND_COUNT_RESERVE);
}
