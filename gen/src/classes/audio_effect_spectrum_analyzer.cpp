/**************************************************************************/
/*  audio_effect_spectrum_analyzer.cpp                                    */
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

#include <godot_cpp/classes/audio_effect_spectrum_analyzer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioEffectSpectrumAnalyzer::set_buffer_length(float p_seconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("set_buffer_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seconds_encoded;
	PtrToArg<double>::encode(p_seconds, &p_seconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seconds_encoded);
}

float AudioEffectSpectrumAnalyzer::get_buffer_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("get_buffer_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectSpectrumAnalyzer::set_tap_back_pos(float p_seconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("set_tap_back_pos")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seconds_encoded;
	PtrToArg<double>::encode(p_seconds, &p_seconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seconds_encoded);
}

float AudioEffectSpectrumAnalyzer::get_tap_back_pos() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("get_tap_back_pos")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioEffectSpectrumAnalyzer::set_fft_size(AudioEffectSpectrumAnalyzer::FFTSize p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("set_fft_size")._native_ptr(), 1202879215);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

AudioEffectSpectrumAnalyzer::FFTSize AudioEffectSpectrumAnalyzer::get_fft_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectSpectrumAnalyzer::get_class_static()._native_ptr(), StringName("get_fft_size")._native_ptr(), 3925405343);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioEffectSpectrumAnalyzer::FFTSize(0)));
	return (AudioEffectSpectrumAnalyzer::FFTSize)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
