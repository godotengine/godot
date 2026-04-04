/**************************************************************************/
/*  audio_stream.cpp                                                      */
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

#include <godot_cpp/classes/audio_stream.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_sample.hpp>
#include <godot_cpp/classes/audio_stream_playback.hpp>

namespace godot {

double AudioStream::get_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("get_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool AudioStream::is_monophonic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("is_monophonic")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<AudioStreamPlayback> AudioStream::instantiate_playback() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("instantiate_playback")._native_ptr(), 210135309);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamPlayback>()));
	return Ref<AudioStreamPlayback>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamPlayback>(_gde_method_bind, _owner));
}

bool AudioStream::can_be_sampled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("can_be_sampled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<AudioSample> AudioStream::generate_sample() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("generate_sample")._native_ptr(), 2646048999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioSample>()));
	return Ref<AudioSample>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioSample>(_gde_method_bind, _owner));
}

bool AudioStream::is_meta_stream() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStream::get_class_static()._native_ptr(), StringName("is_meta_stream")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<AudioStreamPlayback> AudioStream::_instantiate_playback() const {
	return Ref<AudioStreamPlayback>();
}

String AudioStream::_get_stream_name() const {
	return String();
}

double AudioStream::_get_length() const {
	return 0.0;
}

bool AudioStream::_is_monophonic() const {
	return false;
}

double AudioStream::_get_bpm() const {
	return 0.0;
}

int32_t AudioStream::_get_beat_count() const {
	return 0;
}

Dictionary AudioStream::_get_tags() const {
	return Dictionary();
}

TypedArray<Dictionary> AudioStream::_get_parameter_list() const {
	return TypedArray<Dictionary>();
}

bool AudioStream::_has_loop() const {
	return false;
}

int32_t AudioStream::_get_bar_beats() const {
	return 0;
}

} // namespace godot
