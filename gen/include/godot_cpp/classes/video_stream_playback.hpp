/**************************************************************************/
/*  video_stream_playback.hpp                                             */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class VideoStreamPlayback : public Resource {
	GDEXTENSION_CLASS(VideoStreamPlayback, Resource)

public:
	int32_t mix_audio(int32_t p_num_frames, const PackedFloat32Array &p_buffer = PackedFloat32Array(), int32_t p_offset = 0);
	virtual void _stop();
	virtual void _play();
	virtual bool _is_playing() const;
	virtual void _set_paused(bool p_paused);
	virtual bool _is_paused() const;
	virtual double _get_length() const;
	virtual double _get_playback_position() const;
	virtual void _seek(double p_time);
	virtual void _set_audio_track(int32_t p_idx);
	virtual Ref<Texture2D> _get_texture() const;
	virtual void _update(double p_delta);
	virtual int32_t _get_channels() const;
	virtual int32_t _get_mix_rate() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_stop), decltype(&T::_stop)>) {
			BIND_VIRTUAL_METHOD(T, _stop, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_play), decltype(&T::_play)>) {
			BIND_VIRTUAL_METHOD(T, _play, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_playing), decltype(&T::_is_playing)>) {
			BIND_VIRTUAL_METHOD(T, _is_playing, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_paused), decltype(&T::_set_paused)>) {
			BIND_VIRTUAL_METHOD(T, _set_paused, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_paused), decltype(&T::_is_paused)>) {
			BIND_VIRTUAL_METHOD(T, _is_paused, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_length), decltype(&T::_get_length)>) {
			BIND_VIRTUAL_METHOD(T, _get_length, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_playback_position), decltype(&T::_get_playback_position)>) {
			BIND_VIRTUAL_METHOD(T, _get_playback_position, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_seek), decltype(&T::_seek)>) {
			BIND_VIRTUAL_METHOD(T, _seek, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_audio_track), decltype(&T::_set_audio_track)>) {
			BIND_VIRTUAL_METHOD(T, _set_audio_track, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_texture), decltype(&T::_get_texture)>) {
			BIND_VIRTUAL_METHOD(T, _get_texture, 3635182373);
		}
		if constexpr (!std::is_same_v<decltype(&B::_update), decltype(&T::_update)>) {
			BIND_VIRTUAL_METHOD(T, _update, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_channels), decltype(&T::_get_channels)>) {
			BIND_VIRTUAL_METHOD(T, _get_channels, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_mix_rate), decltype(&T::_get_mix_rate)>) {
			BIND_VIRTUAL_METHOD(T, _get_mix_rate, 3905245786);
		}
	}

public:
};

} // namespace godot

