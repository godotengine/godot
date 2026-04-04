/**************************************************************************/
/*  audio_stream_playback_resampled.hpp                                   */
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

#include <godot_cpp/classes/audio_frame.hpp>
#include <godot_cpp/classes/audio_stream_playback.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioStreamPlaybackResampled : public AudioStreamPlayback {
	GDEXTENSION_CLASS(AudioStreamPlaybackResampled, AudioStreamPlayback)

public:
	void begin_resample();
	virtual int32_t _mix_resampled(AudioFrame *p_dst_buffer, int32_t p_frame_count);
	virtual float _get_stream_sampling_rate() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioStreamPlayback::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_mix_resampled), decltype(&T::_mix_resampled)>) {
			BIND_VIRTUAL_METHOD(T, _mix_resampled, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_stream_sampling_rate), decltype(&T::_get_stream_sampling_rate)>) {
			BIND_VIRTUAL_METHOD(T, _get_stream_sampling_rate, 1740695150);
		}
	}

public:
};

} // namespace godot

