/**************************************************************************/
/*  audio_stream_generator.cpp                                            */
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

#include <godot_cpp/classes/audio_stream_generator.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioStreamGenerator::set_mix_rate(float p_hz) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("set_mix_rate")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_hz_encoded;
	PtrToArg<double>::encode(p_hz, &p_hz_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hz_encoded);
}

float AudioStreamGenerator::get_mix_rate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("get_mix_rate")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamGenerator::set_mix_rate_mode(AudioStreamGenerator::AudioStreamGeneratorMixRate p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("set_mix_rate_mode")._native_ptr(), 3354885803);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AudioStreamGenerator::AudioStreamGeneratorMixRate AudioStreamGenerator::get_mix_rate_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("get_mix_rate_mode")._native_ptr(), 3537132591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioStreamGenerator::AudioStreamGeneratorMixRate(0)));
	return (AudioStreamGenerator::AudioStreamGeneratorMixRate)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamGenerator::set_buffer_length(float p_seconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("set_buffer_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seconds_encoded;
	PtrToArg<double>::encode(p_seconds, &p_seconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seconds_encoded);
}

float AudioStreamGenerator::get_buffer_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamGenerator::get_class_static()._native_ptr(), StringName("get_buffer_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
