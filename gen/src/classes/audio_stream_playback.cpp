/**************************************************************************/
/*  audio_stream_playback.cpp                                             */
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

#include <godot_cpp/classes/audio_stream_playback.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_sample_playback.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void AudioStreamPlayback::set_sample_playback(const Ref<AudioSamplePlayback> &p_playback_sample) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("set_sample_playback")._native_ptr(), 3195455091);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_playback_sample != nullptr ? &p_playback_sample->_owner : nullptr));
}

Ref<AudioSamplePlayback> AudioStreamPlayback::get_sample_playback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("get_sample_playback")._native_ptr(), 3482738536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioSamplePlayback>()));
	return Ref<AudioSamplePlayback>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioSamplePlayback>(_gde_method_bind, _owner));
}

PackedVector2Array AudioStreamPlayback::mix_audio(float p_rate_scale, int32_t p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("mix_audio")._native_ptr(), 3341291446);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	double p_rate_scale_encoded;
	PtrToArg<double>::encode(p_rate_scale, &p_rate_scale_encoded);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_rate_scale_encoded, &p_frames_encoded);
}

void AudioStreamPlayback::start(double p_from_pos) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("start")._native_ptr(), 1958160172);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_from_pos_encoded;
	PtrToArg<double>::encode(p_from_pos, &p_from_pos_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_pos_encoded);
}

void AudioStreamPlayback::seek(double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("seek")._native_ptr(), 1958160172);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_encoded);
}

void AudioStreamPlayback::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t AudioStreamPlayback::get_loop_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("get_loop_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

double AudioStreamPlayback::get_playback_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("get_playback_position")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool AudioStreamPlayback::is_playing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayback::get_class_static()._native_ptr(), StringName("is_playing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioStreamPlayback::_start(double p_from_pos) {}

void AudioStreamPlayback::_stop() {}

bool AudioStreamPlayback::_is_playing() const {
	return false;
}

int32_t AudioStreamPlayback::_get_loop_count() const {
	return 0;
}

double AudioStreamPlayback::_get_playback_position() const {
	return 0.0;
}

void AudioStreamPlayback::_seek(double p_position) {}

int32_t AudioStreamPlayback::_mix(AudioFrame *p_buffer, float p_rate_scale, int32_t p_frames) {
	return 0;
}

void AudioStreamPlayback::_tag_used_streams() {}

void AudioStreamPlayback::_set_parameter(const StringName &p_name, const Variant &p_value) {}

Variant AudioStreamPlayback::_get_parameter(const StringName &p_name) const {
	return Variant();
}

} // namespace godot
