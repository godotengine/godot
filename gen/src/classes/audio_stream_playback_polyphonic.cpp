/**************************************************************************/
/*  audio_stream_playback_polyphonic.cpp                                  */
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

#include <godot_cpp/classes/audio_stream_playback_polyphonic.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_stream.hpp>

namespace godot {

int64_t AudioStreamPlaybackPolyphonic::play_stream(const Ref<AudioStream> &p_stream, float p_from_offset, float p_volume_db, float p_pitch_scale, AudioServer::PlaybackType p_playback_type, const StringName &p_bus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlaybackPolyphonic::get_class_static()._native_ptr(), StringName("play_stream")._native_ptr(), 1846744803);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	double p_from_offset_encoded;
	PtrToArg<double>::encode(p_from_offset, &p_from_offset_encoded);
	double p_volume_db_encoded;
	PtrToArg<double>::encode(p_volume_db, &p_volume_db_encoded);
	double p_pitch_scale_encoded;
	PtrToArg<double>::encode(p_pitch_scale, &p_pitch_scale_encoded);
	int64_t p_playback_type_encoded;
	PtrToArg<int64_t>::encode(p_playback_type, &p_playback_type_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr), &p_from_offset_encoded, &p_volume_db_encoded, &p_pitch_scale_encoded, &p_playback_type_encoded, &p_bus);
}

void AudioStreamPlaybackPolyphonic::set_stream_volume(int64_t p_stream, float p_volume_db) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlaybackPolyphonic::get_class_static()._native_ptr(), StringName("set_stream_volume")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stream_encoded;
	PtrToArg<int64_t>::encode(p_stream, &p_stream_encoded);
	double p_volume_db_encoded;
	PtrToArg<double>::encode(p_volume_db, &p_volume_db_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stream_encoded, &p_volume_db_encoded);
}

void AudioStreamPlaybackPolyphonic::set_stream_pitch_scale(int64_t p_stream, float p_pitch_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlaybackPolyphonic::get_class_static()._native_ptr(), StringName("set_stream_pitch_scale")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stream_encoded;
	PtrToArg<int64_t>::encode(p_stream, &p_stream_encoded);
	double p_pitch_scale_encoded;
	PtrToArg<double>::encode(p_pitch_scale, &p_pitch_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stream_encoded, &p_pitch_scale_encoded);
}

bool AudioStreamPlaybackPolyphonic::is_stream_playing(int64_t p_stream) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlaybackPolyphonic::get_class_static()._native_ptr(), StringName("is_stream_playing")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_stream_encoded;
	PtrToArg<int64_t>::encode(p_stream, &p_stream_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_stream_encoded);
}

void AudioStreamPlaybackPolyphonic::stop_stream(int64_t p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlaybackPolyphonic::get_class_static()._native_ptr(), StringName("stop_stream")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stream_encoded;
	PtrToArg<int64_t>::encode(p_stream, &p_stream_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stream_encoded);
}

} // namespace godot
