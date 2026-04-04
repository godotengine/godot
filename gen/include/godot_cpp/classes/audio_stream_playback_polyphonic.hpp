/**************************************************************************/
/*  audio_stream_playback_polyphonic.hpp                                  */
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

#pragma once

#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/classes/audio_stream_playback.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioStream;

class AudioStreamPlaybackPolyphonic : public AudioStreamPlayback {
	GDEXTENSION_CLASS(AudioStreamPlaybackPolyphonic, AudioStreamPlayback)

public:
	static const int INVALID_ID = -1;

	int64_t play_stream(const Ref<AudioStream> &p_stream, float p_from_offset = 0, float p_volume_db = 0, float p_pitch_scale = 1.0, AudioServer::PlaybackType p_playback_type = (AudioServer::PlaybackType)0, const StringName &p_bus = "Master");
	void set_stream_volume(int64_t p_stream, float p_volume_db);
	void set_stream_pitch_scale(int64_t p_stream, float p_pitch_scale);
	bool is_stream_playing(int64_t p_stream) const;
	void stop_stream(int64_t p_stream);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioStreamPlayback::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

