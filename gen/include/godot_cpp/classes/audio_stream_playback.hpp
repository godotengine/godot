/**************************************************************************/
/*  audio_stream_playback.hpp                                             */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioSamplePlayback;
class StringName;

class AudioStreamPlayback : public RefCounted {
	GDEXTENSION_CLASS(AudioStreamPlayback, RefCounted)

public:
	void set_sample_playback(const Ref<AudioSamplePlayback> &p_playback_sample);
	Ref<AudioSamplePlayback> get_sample_playback() const;
	PackedVector2Array mix_audio(float p_rate_scale, int32_t p_frames);
	void start(double p_from_pos = 0.0);
	void seek(double p_time = 0.0);
	void stop();
	int32_t get_loop_count() const;
	double get_playback_position() const;
	bool is_playing() const;
	virtual void _start(double p_from_pos);
	virtual void _stop();
	virtual bool _is_playing() const;
	virtual int32_t _get_loop_count() const;
	virtual double _get_playback_position() const;
	virtual void _seek(double p_position);
	virtual int32_t _mix(AudioFrame *p_buffer, float p_rate_scale, int32_t p_frames);
	virtual void _tag_used_streams();
	virtual void _set_parameter(const StringName &p_name, const Variant &p_value);
	virtual Variant _get_parameter(const StringName &p_name) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_start), decltype(&T::_start)>) {
			BIND_VIRTUAL_METHOD(T, _start, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_stop), decltype(&T::_stop)>) {
			BIND_VIRTUAL_METHOD(T, _stop, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_playing), decltype(&T::_is_playing)>) {
			BIND_VIRTUAL_METHOD(T, _is_playing, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_loop_count), decltype(&T::_get_loop_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_loop_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_playback_position), decltype(&T::_get_playback_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_playback_position, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_seek), decltype(&T::_seek)>) {
			BIND_VIRTUAL_METHOD(T, _seek, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_mix), decltype(&T::_mix)>) {
			BIND_VIRTUAL_METHOD(T, _mix, 925936155);
		}
		if constexpr (!std::is_same_v<decltype(&B::_tag_used_streams), decltype(&T::_tag_used_streams)>) {
			BIND_VIRTUAL_METHOD(T, _tag_used_streams, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_parameter), decltype(&T::_set_parameter)>) {
			BIND_VIRTUAL_METHOD(T, _set_parameter, 3776071444);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_parameter), decltype(&T::_get_parameter)>) {
			BIND_VIRTUAL_METHOD(T, _get_parameter, 2760726917);
		}
	}

public:
};

} // namespace godot

