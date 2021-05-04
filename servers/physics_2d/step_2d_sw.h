/*************************************************************************/
/*  step_2d_sw.h                                                         */
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

#ifndef STEP_2D_SW_H
#define STEP_2D_SW_H

#include "space_2d_sw.h"

#include "core/templates/local_vector.h"
#include "core/templates/thread_work_pool.h"

class Step2DSW {
	uint64_t _step;

	int iterations = 0;
	real_t delta = 0.0;

	ThreadWorkPool work_pool;

	LocalVector<LocalVector<Body2DSW *>> body_islands;
	LocalVector<LocalVector<Constraint2DSW *>> constraint_islands;
	LocalVector<Constraint2DSW *> all_constraints;

	void _populate_island(Body2DSW *p_body, LocalVector<Body2DSW *> &p_body_island, LocalVector<Constraint2DSW *> &p_constraint_island);
	void _setup_contraint(uint32_t p_constraint_index, void *p_userdata = nullptr);
	void _pre_solve_island(LocalVector<Constraint2DSW *> &p_constraint_island) const;
	void _solve_island(uint32_t p_island_index, void *p_userdata = nullptr) const;
	void _check_suspend(LocalVector<Body2DSW *> &p_body_island) const;

public:
	void step(Space2DSW *p_space, real_t p_delta, int p_iterations);
	Step2DSW();
	~Step2DSW();
};

#endif // STEP_2D_SW_H
