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

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "core/local_vector.h"
#include "core/object.h"

class Interpolator : public Object {
	GDCLASS(Interpolator, Object);

public:
	enum Fallback {
		FALLBACK_INTERPOLATE,
		FALLBACK_DEFAULT,
		FALLBACK_OLD_OR_NEAREST,
		FALLBACK_NEW_OR_NEAREST,
		FALLBACK_CUSTOM_INTERPOLATOR
	};

private:
	struct VariableInfo {
		// TODO Do we need a name?
		Variant default_value;
		Fallback fallback;
		ObjectID custom_interpolator_object;
		StringName custom_interpolator_function;
	};

	LocalVector<VariableInfo> variables;

	/// Epoch ids, sorted from youngest to oldest.
	LocalVector<uint32_t> epochs;
	/// Epoch data.
	LocalVector<Vector<Variant>> buffer;

	bool init_phase = true;
	uint32_t write_position = UINT32_MAX;
	uint32_t last_pop_epoch = 0;

	static void _bind_methods();

public:
	Interpolator();

	void clear();
	void reset();

	int register_variable(const Variant &p_default, Fallback p_fallback);
	void set_variable_default(int p_var_id, const Variant &p_default);
	void set_variable_custom_interpolator(int p_var_id, Object *p_object, StringName p_function_name);
	void terminate_init();

	/// Returns the epochs stored.
	uint32_t known_epochs_count() const;
	void begin_write(uint32_t p_epoch);
	void epoch_insert(int p_var_id, Variant p_value);
	void end_write();

	Vector<Variant> pop_epoch(uint32_t p_epoch, real_t p_fraction);
	uint32_t get_last_pop_epoch() const; // TODO do I need this? Remove if not.
	uint32_t get_youngest_epoch() const;
	uint32_t get_oldest_epoch() const;

	/// Returns the epochs count between the two last received time window.
	uint32_t epochs_between_last_time_window() const;
};

VARIANT_ENUM_CAST(Interpolator::Fallback);
#endif
