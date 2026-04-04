/**************************************************************************/
/*  audio_effect_compressor.cpp                                           */
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

#include <godot_cpp/classes/audio_effect_compressor.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioEffectCompressor::set_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float AudioEffectCompressor::get_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float AudioEffectCompressor::get_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_gain(float p_gain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_gain")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_gain_encoded;
	PtrToArg<double>::encode(p_gain, &p_gain_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gain_encoded);
}

float AudioEffectCompressor::get_gain() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_gain")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_attack_us(float p_attack_us) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_attack_us")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_attack_us_encoded;
	PtrToArg<double>::encode(p_attack_us, &p_attack_us_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_attack_us_encoded);
}

float AudioEffectCompressor::get_attack_us() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_attack_us")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_release_ms(float p_release_ms) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_release_ms")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_release_ms_encoded;
	PtrToArg<double>::encode(p_release_ms, &p_release_ms_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_release_ms_encoded);
}

float AudioEffectCompressor::get_release_ms() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_release_ms")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_mix(float p_mix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_mix")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mix_encoded;
	PtrToArg<double>::encode(p_mix, &p_mix_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mix_encoded);
}

float AudioEffectCompressor::get_mix() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_mix")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectCompressor::set_sidechain(const StringName &p_sidechain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("set_sidechain")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sidechain);
}

StringName AudioEffectCompressor::get_sidechain() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectCompressor::get_class_static()._native_ptr(), StringName("get_sidechain")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

} // namespace godot
