/**************************************************************************/
/*  random_number_generator.cpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/random_number_generator.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/packed_float32_array.hpp>

namespace godot {

void RandomNumberGenerator::set_seed(uint64_t p_seed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("set_seed")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_seed_encoded;
	PtrToArg<int64_t>::encode(p_seed, &p_seed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seed_encoded);
}

uint64_t RandomNumberGenerator::get_seed() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("get_seed")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

void RandomNumberGenerator::set_state(uint64_t p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("set_state")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_state_encoded);
}

uint64_t RandomNumberGenerator::get_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("get_state")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint32_t RandomNumberGenerator::randi() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randi")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

float RandomNumberGenerator::randf() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randf")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float RandomNumberGenerator::randfn(float p_mean, float p_deviation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randfn")._native_ptr(), 837325100);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_mean_encoded;
	PtrToArg<double>::encode(p_mean, &p_mean_encoded);
	double p_deviation_encoded;
	PtrToArg<double>::encode(p_deviation, &p_deviation_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_mean_encoded, &p_deviation_encoded);
}

float RandomNumberGenerator::randf_range(float p_from, float p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randf_range")._native_ptr(), 4269894367);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_from_encoded, &p_to_encoded);
}

int32_t RandomNumberGenerator::randi_range(int32_t p_from, int32_t p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randi_range")._native_ptr(), 50157827);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	int64_t p_to_encoded;
	PtrToArg<int64_t>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_encoded, &p_to_encoded);
}

int64_t RandomNumberGenerator::rand_weighted(const PackedFloat32Array &p_weights) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("rand_weighted")._native_ptr(), 4189642986);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_weights);
}

void RandomNumberGenerator::randomize() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RandomNumberGenerator::get_class_static()._native_ptr(), StringName("randomize")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
