/**************************************************************************/
/*  audio_effect_limiter.cpp                                              */
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

#include <godot_cpp/classes/audio_effect_limiter.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioEffectLimiter::set_ceiling_db(float p_ceiling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("set_ceiling_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ceiling_encoded;
	PtrToArg<double>::encode(p_ceiling, &p_ceiling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ceiling_encoded);
}

float AudioEffectLimiter::get_ceiling_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("get_ceiling_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectLimiter::set_threshold_db(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("set_threshold_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float AudioEffectLimiter::get_threshold_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("get_threshold_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectLimiter::set_soft_clip_db(float p_soft_clip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("set_soft_clip_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_soft_clip_encoded;
	PtrToArg<double>::encode(p_soft_clip, &p_soft_clip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_soft_clip_encoded);
}

float AudioEffectLimiter::get_soft_clip_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("get_soft_clip_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectLimiter::set_soft_clip_ratio(float p_soft_clip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("set_soft_clip_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_soft_clip_encoded;
	PtrToArg<double>::encode(p_soft_clip, &p_soft_clip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_soft_clip_encoded);
}

float AudioEffectLimiter::get_soft_clip_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectLimiter::get_class_static()._native_ptr(), StringName("get_soft_clip_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
