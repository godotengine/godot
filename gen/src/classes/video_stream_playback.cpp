/**************************************************************************/
/*  video_stream_playback.cpp                                             */
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

#include <godot_cpp/classes/video_stream_playback.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

int32_t VideoStreamPlayback::mix_audio(int32_t p_num_frames, const PackedFloat32Array &p_buffer, int32_t p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayback::get_class_static()._native_ptr(), StringName("mix_audio")._native_ptr(), 93876830);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_num_frames_encoded;
	PtrToArg<int64_t>::encode(p_num_frames, &p_num_frames_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_num_frames_encoded, &p_buffer, &p_offset_encoded);
}

void VideoStreamPlayback::_stop() {}

void VideoStreamPlayback::_play() {}

bool VideoStreamPlayback::_is_playing() const {
	return false;
}

void VideoStreamPlayback::_set_paused(bool p_paused) {}

bool VideoStreamPlayback::_is_paused() const {
	return false;
}

double VideoStreamPlayback::_get_length() const {
	return 0.0;
}

double VideoStreamPlayback::_get_playback_position() const {
	return 0.0;
}

void VideoStreamPlayback::_seek(double p_time) {}

void VideoStreamPlayback::_set_audio_track(int32_t p_idx) {}

Ref<Texture2D> VideoStreamPlayback::_get_texture() const {
	return Ref<Texture2D>();
}

void VideoStreamPlayback::_update(double p_delta) {}

int32_t VideoStreamPlayback::_get_channels() const {
	return 0;
}

int32_t VideoStreamPlayback::_get_mix_rate() const {
	return 0;
}

} // namespace godot
