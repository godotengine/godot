/**************************************************************************/
/*  audio_stream_randomizer.cpp                                           */
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

#include <godot_cpp/classes/audio_stream_randomizer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioStreamRandomizer::add_stream(int32_t p_index, const Ref<AudioStream> &p_stream, float p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("add_stream")._native_ptr(), 1892018854);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_stream != nullptr ? &p_stream->_owner : nullptr), &p_weight_encoded);
}

void AudioStreamRandomizer::move_stream(int32_t p_index_from, int32_t p_index_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("move_stream")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_from_encoded;
	PtrToArg<int64_t>::encode(p_index_from, &p_index_from_encoded);
	int64_t p_index_to_encoded;
	PtrToArg<int64_t>::encode(p_index_to, &p_index_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_from_encoded, &p_index_to_encoded);
}

void AudioStreamRandomizer::remove_stream(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("remove_stream")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void AudioStreamRandomizer::set_stream(int32_t p_index, const Ref<AudioStream> &p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_stream")._native_ptr(), 111075094);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_stream != nullptr ? &p_stream->_owner : nullptr));
}

Ref<AudioStream> AudioStreamRandomizer::get_stream(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_stream")._native_ptr(), 2739380747);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStream>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<AudioStream>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStream>(_gde_method_bind, _owner, &p_index_encoded));
}

void AudioStreamRandomizer::set_stream_probability_weight(int32_t p_index, float p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_stream_probability_weight")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_weight_encoded);
}

float AudioStreamRandomizer::get_stream_probability_weight(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_stream_probability_weight")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void AudioStreamRandomizer::set_streams_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_streams_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t AudioStreamRandomizer::get_streams_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_streams_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamRandomizer::set_random_pitch(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_random_pitch")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float AudioStreamRandomizer::get_random_pitch() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_random_pitch")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamRandomizer::set_random_pitch_semitones(float p_semitones) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_random_pitch_semitones")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_semitones_encoded;
	PtrToArg<double>::encode(p_semitones, &p_semitones_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_semitones_encoded);
}

float AudioStreamRandomizer::get_random_pitch_semitones() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_random_pitch_semitones")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamRandomizer::set_random_volume_offset_db(float p_db_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_random_volume_offset_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_db_offset_encoded;
	PtrToArg<double>::encode(p_db_offset, &p_db_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_db_offset_encoded);
}

float AudioStreamRandomizer::get_random_volume_offset_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_random_volume_offset_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamRandomizer::set_playback_mode(AudioStreamRandomizer::PlaybackMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("set_playback_mode")._native_ptr(), 3950967023);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AudioStreamRandomizer::PlaybackMode AudioStreamRandomizer::get_playback_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamRandomizer::get_class_static()._native_ptr(), StringName("get_playback_mode")._native_ptr(), 3943055077);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioStreamRandomizer::PlaybackMode(0)));
	return (AudioStreamRandomizer::PlaybackMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
