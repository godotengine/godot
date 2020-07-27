/*************************************************************************/
/*  net_utilities.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

/**
	@author AndreaCatania
*/

#include "interpolator.h"

#include "core/ustring.h"

void Interpolator::bind_methods() {
	ClassDB::bind_method(D_METHOD("register_variable", "default", "fallback"), &Interpolator::register_variable);
	ClassDB::bind_method(D_METHOD("set_variable_custom_interpolator", "var_id", "object", "function_name"), &Interpolator::set_variable_custom_interpolator);

	ClassDB::bind_method(D_METHOD("epoch_insert", "var_id", "value"), &Interpolator::epoch_insert);

	// TODO remove these below, was just a test.
	ClassDB::bind_method(D_METHOD("terminate_init"), &Interpolator::terminate_init);
	ClassDB::bind_method(D_METHOD("begin_write", "epoch"), &Interpolator::begin_write);
	ClassDB::bind_method(D_METHOD("end_write"), &Interpolator::end_write);

	BIND_ENUM_CONSTANT(FALLBACK_INTERPOLATE);
	BIND_ENUM_CONSTANT(FALLBACK_DEFAULT);
	BIND_ENUM_CONSTANT(FALLBACK_NEW);
	BIND_ENUM_CONSTANT(FALLBACK_OLD);
}

Interpolator::Interpolator() {
}

int Interpolator::register_variable(const Variant &p_default, Fallback p_fallback) {
	ERR_FAIL_COND_V_MSG(init_phase == false, -1, "You cannot add another variable at this point.");
	const uint32_t id = variables.size();
	VariableInfo vi;
	vi.id = id;
	vi.default_value = p_default;
	variables.push_back(vi);
	return id;
}

void Interpolator::set_variable_custom_interpolator(int p_var_id, Object *p_object, StringName p_function_name) {
#warning "Implement this"
	CRASH_NOW_MSG("TODO Implement this please.");
}

void Interpolator::terminate_init() {
	init_phase = false;
	buffer.resize(variables.size());
}

void Interpolator::begin_write(uint64_t p_epoch) {
}

void Interpolator::epoch_insert(int p_var_id, Variant p_value) {
}

void Interpolator::end_write() {
}

void Interpolator::pop_epoch(uint64_t p_epoch, LocalVector<Variant> &r_vector) {
	r_vector.clear();
}
