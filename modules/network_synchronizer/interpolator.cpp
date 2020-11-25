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

// TODO write unit tests to make sure all cases are covered.

#include "interpolator.h"

#include "core/ustring.h"

void Interpolator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_variable", "default", "fallback"), &Interpolator::register_variable);
	ClassDB::bind_method(D_METHOD("set_variable_default", "var_id", "default"), &Interpolator::set_variable_default);
	ClassDB::bind_method(D_METHOD("set_variable_custom_interpolator", "var_id", "object", "function_name"), &Interpolator::set_variable_custom_interpolator);

	ClassDB::bind_method(D_METHOD("epoch_insert", "var_id", "value"), &Interpolator::epoch_insert);
	ClassDB::bind_method(D_METHOD("pop_epoch", "epoch"), &Interpolator::pop_epoch);
	ClassDB::bind_method(D_METHOD("get_last_pop_epoch"), &Interpolator::get_last_pop_epoch);

	// TODO used to do the tests.
	//ClassDB::bind_method(D_METHOD("terminate_init"), &Interpolator::terminate_init);
	//ClassDB::bind_method(D_METHOD("begin_write", "epoch"), &Interpolator::begin_write);
	//ClassDB::bind_method(D_METHOD("end_write"), &Interpolator::end_write);

	BIND_ENUM_CONSTANT(FALLBACK_INTERPOLATE);
	BIND_ENUM_CONSTANT(FALLBACK_DEFAULT);
	BIND_ENUM_CONSTANT(FALLBACK_NEW_OR_NEAREST);
	BIND_ENUM_CONSTANT(FALLBACK_OLD_OR_NEAREST);
}

Interpolator::Interpolator() {}

void Interpolator::clear() {
	epochs.clear();
	buffer.clear();

	write_position = UINT32_MAX;
}

void Interpolator::reset() {
	variables.clear();
	epochs.clear();
	buffer.clear();

	init_phase = true;
	write_position = UINT32_MAX;
	last_pop_epoch = 0;
}

int Interpolator::register_variable(const Variant &p_default, Fallback p_fallback) {
	ERR_FAIL_COND_V_MSG(init_phase == false, -1, "You cannot add another variable at this point.");
	const uint32_t id = variables.size();
	variables.push_back(VariableInfo{ p_default, p_fallback, ObjectID(), StringName() });
	return id;
}

void Interpolator::set_variable_default(int p_var_id, const Variant &p_default) {
	ERR_FAIL_INDEX(p_var_id, variables.size());
	ERR_FAIL_COND(variables[p_var_id].default_value.get_type() != p_default.get_type());
	variables[p_var_id].default_value = p_default;
}

void Interpolator::set_variable_custom_interpolator(int p_var_id, Object *p_object, StringName p_function_name) {
	ERR_FAIL_COND_MSG(init_phase == false, "You cannot add another variable at this point.");
	ERR_FAIL_INDEX_MSG(p_var_id, int(variables.size()), "The variable_id passed is unknown.");
	variables[p_var_id].fallback = FALLBACK_CUSTOM_INTERPOLATOR;
	variables[p_var_id].custom_interpolator_object = p_object->get_instance_id();
	variables[p_var_id].custom_interpolator_function = p_function_name;
}

void Interpolator::terminate_init() {
	init_phase = false;
}

uint32_t Interpolator::known_epochs_count() const {
	return epochs.size();
}

void Interpolator::begin_write(uint32_t p_epoch) {
	ERR_FAIL_COND_MSG(write_position != UINT32_MAX, "You can't call this function twice.");
	ERR_FAIL_COND_MSG(init_phase, "You cannot write data while the buffer is not fully initialized, call `terminate_init`.");

	// Make room for this epoch.
	// Insert the epoch sorted in the buffer.
	write_position = UINT32_MAX;
	for (uint32_t i = 0; i < epochs.size(); i += 1) {
		if (epochs[i] >= p_epoch) {
			write_position = i;
			break;
		}
	}

	if (write_position < UINT32_MAX) {
		if (epochs[write_position] == p_epoch) {
			// This epoch already exists, nothing to do.
			return;
		} else {
			// Make room.
			epochs.push_back(UINT32_MAX);
			buffer.push_back(Vector<Variant>());
			// Sort the epochs.
			for (int i = epochs.size() - 2; i >= int(write_position); i -= 1) {
				epochs[uint32_t(i) + 1] = epochs[uint32_t(i)];
				buffer[uint32_t(i) + 1] = buffer[uint32_t(i)];
			}
			// Init the new epoch.
			epochs[write_position] = p_epoch;
			buffer[write_position].clear();
			buffer[write_position].resize(variables.size());
		}
	} else {
		// No sort needed.
		write_position = epochs.size();
		epochs.push_back(p_epoch);
		buffer.push_back(Vector<Variant>());
		buffer[write_position].resize(variables.size());
	}

	// Set defaults.
	Variant *ptr = buffer[write_position].ptrw();
	for (uint32_t i = 0; i < variables.size(); i += 1) {
		ptr[i] = variables[i].default_value;
	}
}

void Interpolator::epoch_insert(int p_var_id, Variant p_value) {
	ERR_FAIL_COND_MSG(write_position == UINT32_MAX, "Please call `begin_write` before.");
	ERR_FAIL_INDEX_MSG(p_var_id, int(variables.size()), "The variable_id passed is unknown.");
	const uint32_t var_id(p_var_id);
	ERR_FAIL_COND_MSG(variables[var_id].default_value.get_type() != p_value.get_type(), "The variable: " + itos(p_var_id) + " expects the variable type: " + Variant::get_type_name(variables[var_id].default_value.get_type()) + ", and not: " + Variant::get_type_name(p_value.get_type()));
	buffer[write_position].write[var_id] = p_value;
}

void Interpolator::end_write() {
	ERR_FAIL_COND_MSG(write_position == UINT32_MAX, "You can't call this function before starting the epoch with `begin_write`.");
	write_position = UINT32_MAX;
}

Variant interpolate(const Variant &p_v1, const Variant &p_v2, real_t p_delta) {
	ERR_FAIL_COND_V(p_v1.get_type() != p_v2.get_type(), p_v1);

	switch (p_v1.get_type()) {
		case Variant::Type::INT:
			return int(Math::round(Math::lerp(p_v1.operator real_t(), p_v2.operator real_t(), p_delta)));
		case Variant::Type::FLOAT:
			return Math::lerp(p_v1, p_v2, p_delta);
		case Variant::Type::VECTOR2:
			return p_v1.operator Vector2().lerp(p_v2.operator Vector2(), p_delta);
		case Variant::Type::VECTOR2I:
			return Vector2i(
					int(Math::round(Math::lerp(p_v1.operator Vector2i()[0], p_v2.operator Vector2i()[0], p_delta))),
					int(Math::round(Math::lerp(p_v1.operator Vector2i()[1], p_v2.operator Vector2i()[1], p_delta))));
		case Variant::Type::TRANSFORM2D:
			return p_v1.operator Transform2D().interpolate_with(p_v2.operator Transform2D(), p_delta);
		case Variant::Type::VECTOR3:
			return p_v1.operator Vector3().lerp(p_v2.operator Vector3(), p_delta);
		case Variant::Type::VECTOR3I:
			return Vector3i(
					int(Math::round(Math::lerp(p_v1.operator Vector3i()[0], p_v2.operator Vector3i()[0], p_delta))),
					int(Math::round(Math::lerp(p_v1.operator Vector3i()[1], p_v2.operator Vector3i()[1], p_delta))),
					int(Math::round(Math::lerp(p_v1.operator Vector3i()[2], p_v2.operator Vector3i()[2], p_delta))));
		case Variant::Type::QUAT:
			return p_v1.operator Quat().slerp(p_v2.operator Quat(), p_delta);
		case Variant::Type::BASIS:
			return p_v1.operator Basis().slerp(p_v2.operator Basis(), p_delta);
		case Variant::Type::TRANSFORM:
			return p_v1.operator Transform().interpolate_with(p_v2.operator Transform(), p_delta);
		default:
			return p_delta > 0.5 ? p_v2 : p_v1;
	}
}

Vector<Variant> Interpolator::pop_epoch(uint32_t p_epoch, real_t p_fraction) {
	ERR_FAIL_COND_V_MSG(init_phase, Vector<Variant>(), "You can't pop data if the interpolator is not fully initialized.");
	ERR_FAIL_COND_V_MSG(write_position != UINT32_MAX, Vector<Variant>(), "You can't pop data while writing the epoch");

	double epoch = double(p_epoch) + double(p_fraction);

	// Search the epoch.
	uint32_t position = UINT32_MAX;
	for (uint32_t i = 0; i < epochs.size(); i += 1) {
		if (static_cast<double>(epochs[i]) >= epoch) {
			position = i;
			break;
		}
	}

	ObjectID cache_object_id;
	Object *cache_object = nullptr;

	Vector<Variant> data;
	if (unlikely(position == UINT32_MAX)) {
		data.resize(variables.size());
		Variant *ptr = data.ptrw();
		if (buffer.size() == 0) {
			// No data found, set all to default.
			for (uint32_t i = 0; i < variables.size(); i += 1) {
				ptr[i] = variables[i].default_value;
			}
		} else {
			// No new data.
			for (uint32_t i = 0; i < variables.size(); i += 1) {
				switch (variables[i].fallback) {
					case FALLBACK_DEFAULT:
						ptr[i] = variables[i].default_value;
						break;
					case FALLBACK_INTERPOLATE: // No way to interpolate, so just send the nearest.
					case FALLBACK_NEW_OR_NEAREST: // No new data, so send the nearest.
					case FALLBACK_OLD_OR_NEAREST: // Just send the oldest, as desired.
						ptr[i] = buffer[buffer.size() - 1][i];
						break;
					case FALLBACK_CUSTOM_INTERPOLATOR:
						ptr[i] = variables[i].default_value;

						if (cache_object_id != variables[i].custom_interpolator_object) {
							ERR_CONTINUE_MSG(variables[i].custom_interpolator_object.is_null(), "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

							Object *o = ObjectDB::get_instance(variables[i].custom_interpolator_object);
							ERR_CONTINUE_MSG(o == nullptr, "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

							cache_object_id = variables[i].custom_interpolator_object;
							cache_object = o;
						}

						ptr[i] = cache_object->call(
								variables[i].custom_interpolator_function,
								epochs[buffer.size() - 1],
								buffer[buffer.size() - 1][i],
								-1,
								variables[i].default_value,
								0.0);

						if (ptr[i].get_type() != variables[i].default_value.get_type()) {
							ERR_PRINT("The variable: " + itos(i) + " custom interpolator [" + variables[i].custom_interpolator_function + "], returned a different variant type. Expected: " + Variant::get_type_name(variables[i].default_value.get_type()) + ", returned: " + Variant::get_type_name(ptr[i].get_type()));
							ptr[i] = variables[i].default_value;
						}
						break;
				}
			}
		}
	} else if (unlikely(ABS(epochs[position] - epoch) <= CMP_EPSILON)) {
		// Precise data.
		data = buffer[position];
	} else if (unlikely(position == 0)) {
		// No old data.
		data.resize(variables.size());
		Variant *ptr = data.ptrw();
		for (uint32_t i = 0; i < variables.size(); i += 1) {
			switch (variables[i].fallback) {
				case FALLBACK_DEFAULT:
					ptr[i] = variables[i].default_value;
					break;
				case FALLBACK_INTERPOLATE: // No way to interpolate, so just send the nearest.
				case FALLBACK_NEW_OR_NEAREST: // Just send the newer data as desired.
				case FALLBACK_OLD_OR_NEAREST: // No old data, so send nearest.
					ptr[i] = buffer[0][i];
					break;
				case FALLBACK_CUSTOM_INTERPOLATOR:
					ptr[i] = variables[i].default_value;
					if (cache_object_id != variables[i].custom_interpolator_object) {
						ERR_CONTINUE_MSG(variables[i].custom_interpolator_object.is_null(), "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

						Object *o = ObjectDB::get_instance(variables[i].custom_interpolator_object);
						ERR_CONTINUE_MSG(o == nullptr, "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

						cache_object_id = variables[i].custom_interpolator_object;
						cache_object = o;
					}

					ptr[i] = cache_object->call(
							variables[i].custom_interpolator_function,
							-1,
							variables[i].default_value,
							epochs[0],
							buffer[0][i],
							1.0);

					if (ptr[i].get_type() != variables[i].default_value.get_type()) {
						ERR_PRINT("The variable: " + itos(i) + " custom interpolator [" + variables[i].custom_interpolator_function + "], returned a different variant type. Expected: " + Variant::get_type_name(variables[i].default_value.get_type()) + ", returned: " + Variant::get_type_name(ptr[i].get_type()));
						ptr[i] = variables[i].default_value;
					}
					break;
			}
		}
	} else {
		// Enought data to do anything needed.
		data.resize(variables.size());
		Variant *ptr = data.ptrw();
		for (uint32_t i = 0; i < variables.size(); i += 1) {
			switch (variables[i].fallback) {
				case FALLBACK_DEFAULT:
					ptr[i] = variables[i].default_value;
					break;
				case FALLBACK_INTERPOLATE: {
					const real_t delta = (epoch - double(epochs[position - 1])) / double(epochs[position] - epochs[position - 1]);
					ptr[i] = interpolate(
							buffer[position - 1][i],
							buffer[position][i],
							delta);
				} break;
				case FALLBACK_NEW_OR_NEAREST:
					ptr[i] = buffer[position][i];
					break;
				case FALLBACK_OLD_OR_NEAREST:
					ptr[i] = buffer[position - 1][i];
					break;
				case FALLBACK_CUSTOM_INTERPOLATOR: {
					ptr[i] = variables[i].default_value;

					if (cache_object_id != variables[i].custom_interpolator_object) {
						ERR_CONTINUE_MSG(variables[i].custom_interpolator_object.is_null(), "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

						Object *o = ObjectDB::get_instance(variables[i].custom_interpolator_object);
						ERR_CONTINUE_MSG(o == nullptr, "The variable: " + itos(i) + " has a custom interpolator, but the function is invalid.");

						cache_object_id = variables[i].custom_interpolator_object;
						cache_object = o;
					}

					const real_t delta = (epoch - double(epochs[position - 1])) / double(epochs[position] - epochs[position - 1]);

					ptr[i] = cache_object->call(
							variables[i].custom_interpolator_function,
							epochs[position - 1],
							buffer[position - 1][i],
							epochs[position],
							buffer[position][i],
							delta);

					if (ptr[i].get_type() != variables[i].default_value.get_type()) {
						ERR_PRINT("The variable: " + itos(i) + " custom interpolator [" + variables[i].custom_interpolator_function + "], returned a different variant type. Expected: " + Variant::get_type_name(variables[i].default_value.get_type()) + ", returned: " + Variant::get_type_name(ptr[i].get_type()));
						ptr[i] = variables[i].default_value;
					}
				} break;
			}
		}
	}

	if (unlikely(position == UINT32_MAX)) {
		if (buffer.size() > 1) {
			// Remove all the elements but last. This happens when the p_epoch is
			// bigger than the one already stored into the queue.
			epochs[0] = epochs[buffer.size() - 1];
			buffer[0] = buffer[buffer.size() - 1];
			epochs.resize(1);
			buffer.resize(1);
		}
	} else if (position >= 2) {
		// TODO improve this by performing first the shifting then the resizing.
		// Remove the old elements, but leave the one used to interpolate.
		for (uint32_t i = 0; i < position - 1; i += 1) {
			epochs.remove(0);
			buffer.remove(0);
		}
	}

	// TODO this is no more valid since I'm using the fractional part.
	last_pop_epoch = MAX(p_epoch, last_pop_epoch);

	return data;
}

uint32_t Interpolator::get_last_pop_epoch() const {
	return last_pop_epoch;
}

uint32_t Interpolator::get_youngest_epoch() const {
	if (epochs.size() <= 0) {
		return UINT32_MAX;
	}
	return epochs[0];
}

uint32_t Interpolator::get_oldest_epoch() const {
	if (epochs.size() <= 0) {
		return UINT32_MAX;
	}
	return epochs[epochs.size() - 1];
}

uint32_t Interpolator::epochs_between_last_time_window() const {
	if (epochs.size() <= 1) {
		return 0;
	}

	return epochs[epochs.size() - 1] - epochs[epochs.size() - 2];
}
