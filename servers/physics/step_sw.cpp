/*************************************************************************/
/*  step_sw.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "step_sw.h"
#include "joints_sw.h"

#include "os/os.h"

void StepSW::_populate_island(BodySW *p_body, BodySW **p_island, ConstraintSW **p_constraint_island) {

	p_body->set_island_step(_step);
	p_body->set_island_next(*p_island);
	*p_island = p_body;

	for (Map<ConstraintSW *, int>::Element *E = p_body->get_constraint_map().front(); E; E = E->next()) {

		ConstraintSW *c = (ConstraintSW *)E->key();
		if (c->get_island_step() == _step)
			continue; //already processed
		c->set_island_step(_step);
		c->set_island_next(*p_constraint_island);
		*p_constraint_island = c;

		for (int i = 0; i < c->get_body_count(); i++) {
			if (i == E->get())
				continue;
			BodySW *b = c->get_body_ptr()[i];
			if (b->get_island_step() == _step || b->get_mode() == PhysicsServer::BODY_MODE_STATIC || b->get_mode() == PhysicsServer::BODY_MODE_KINEMATIC)
				continue; //no go
			_populate_island(c->get_body_ptr()[i], p_island, p_constraint_island);
		}
	}
}

void StepSW::_setup_island(ConstraintSW *p_island, real_t p_delta) {

	ConstraintSW *ci = p_island;
	while (ci) {
		ci->setup(p_delta);
		//todo remove from island if process fails
		ci = ci->get_island_next();
	}
}

void StepSW::_solve_island(ConstraintSW *p_island, int p_iterations, real_t p_delta) {

	int at_priority = 1;

	while (p_island) {

		for (int i = 0; i < p_iterations; i++) {

			ConstraintSW *ci = p_island;
			while (ci) {
				ci->solve(p_delta);
				ci = ci->get_island_next();
			}
		}

		at_priority++;

		{
			ConstraintSW *ci = p_island;
			ConstraintSW *prev = NULL;
			while (ci) {
				if (ci->get_priority() < at_priority) {
					if (prev) {
						prev->set_island_next(ci->get_island_next()); //remove
					} else {
						p_island = ci->get_island_next();
					}
				} else {

					prev = ci;
				}

				ci = ci->get_island_next();
			}
		}
	}
}

void StepSW::_check_suspend(BodySW *p_island, real_t p_delta) {

	bool can_sleep = true;

	BodySW *b = p_island;
	while (b) {

		if (b->get_mode() == PhysicsServer::BODY_MODE_STATIC || b->get_mode() == PhysicsServer::BODY_MODE_KINEMATIC) {
			b = b->get_island_next();
			continue; //ignore for static
		}

		if (!b->sleep_test(p_delta))
			can_sleep = false;

		b = b->get_island_next();
	}

	//put all to sleep or wake up everyoen

	b = p_island;
	while (b) {

		if (b->get_mode() == PhysicsServer::BODY_MODE_STATIC || b->get_mode() == PhysicsServer::BODY_MODE_KINEMATIC) {
			b = b->get_island_next();
			continue; //ignore for static
		}

		bool active = b->is_active();

		if (active == can_sleep)
			b->set_active(!can_sleep);

		b = b->get_island_next();
	}
}

void StepSW::step(SpaceSW *p_space, real_t p_delta, int p_iterations) {

	p_space->lock(); // can't access space during this

	p_space->setup(); //update inertias, etc

	const SelfList<BodySW>::List *body_list = &p_space->get_active_body_list();

	/* INTEGRATE FORCES */

	uint64_t profile_begtime = OS::get_singleton()->get_ticks_usec();
	uint64_t profile_endtime = 0;

	int active_count = 0;

	const SelfList<BodySW> *b = body_list->first();
	while (b) {

		b->self()->integrate_forces(p_delta);
		b = b->next();
		active_count++;
	}

	p_space->set_active_objects(active_count);

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(SpaceSW::ELAPSED_TIME_INTEGRATE_FORCES, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* GENERATE CONSTRAINT ISLANDS */

	BodySW *island_list = NULL;
	ConstraintSW *constraint_island_list = NULL;
	b = body_list->first();

	int island_count = 0;

	while (b) {
		BodySW *body = b->self();

		if (body->get_island_step() != _step) {

			BodySW *island = NULL;
			ConstraintSW *constraint_island = NULL;
			_populate_island(body, &island, &constraint_island);

			island->set_island_list_next(island_list);
			island_list = island;

			if (constraint_island) {
				constraint_island->set_island_list_next(constraint_island_list);
				constraint_island_list = constraint_island;
				island_count++;
			}
		}
		b = b->next();
	}

	p_space->set_island_count(island_count);

	const SelfList<AreaSW>::List &aml = p_space->get_moved_area_list();

	while (aml.first()) {
		for (const Set<ConstraintSW *>::Element *E = aml.first()->self()->get_constraints().front(); E; E = E->next()) {

			ConstraintSW *c = E->get();
			if (c->get_island_step() == _step)
				continue;
			c->set_island_step(_step);
			c->set_island_next(NULL);
			c->set_island_list_next(constraint_island_list);
			constraint_island_list = c;
		}
		p_space->area_remove_from_moved_list((SelfList<AreaSW> *)aml.first()); //faster to remove here
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(SpaceSW::ELAPSED_TIME_GENERATE_ISLANDS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	//print_line("island count: "+itos(island_count)+" active count: "+itos(active_count));
	/* SETUP CONSTRAINT ISLANDS */

	{
		ConstraintSW *ci = constraint_island_list;
		while (ci) {

			_setup_island(ci, p_delta);
			ci = ci->get_island_list_next();
		}
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(SpaceSW::ELAPSED_TIME_SETUP_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* SOLVE CONSTRAINT ISLANDS */

	{
		ConstraintSW *ci = constraint_island_list;
		while (ci) {
			//iterating each island separatedly improves cache efficiency
			_solve_island(ci, p_iterations, p_delta);
			ci = ci->get_island_list_next();
		}
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(SpaceSW::ELAPSED_TIME_SOLVE_CONSTRAINTS, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	/* INTEGRATE VELOCITIES */

	b = body_list->first();
	while (b) {
		const SelfList<BodySW> *n = b->next();
		b->self()->integrate_velocities(p_delta);
		b = n;
	}

	/* SLEEP / WAKE UP ISLANDS */

	{
		BodySW *bi = island_list;
		while (bi) {

			_check_suspend(bi, p_delta);
			bi = bi->get_island_list_next();
		}
	}

	{ //profile
		profile_endtime = OS::get_singleton()->get_ticks_usec();
		p_space->set_elapsed_time(SpaceSW::ELAPSED_TIME_INTEGRATE_VELOCITIES, profile_endtime - profile_begtime);
		profile_begtime = profile_endtime;
	}

	p_space->update();
	p_space->unlock();
	_step++;
}

StepSW::StepSW() {

	_step = 1;
}
