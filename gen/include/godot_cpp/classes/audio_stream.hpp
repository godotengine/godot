/**************************************************************************/
/*  audio_stream.hpp                                                      */
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
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioSample;
class AudioStreamPlayback;

class AudioStream : public Resource {
	GDEXTENSION_CLASS(AudioStream, Resource)

public:
	double get_length() const;
	bool is_monophonic() const;
	Ref<AudioStreamPlayback> instantiate_playback();
	bool can_be_sampled() const;
	Ref<AudioSample> generate_sample() const;
	bool is_meta_stream() const;
	virtual Ref<AudioStreamPlayback> _instantiate_playback() const;
	virtual String _get_stream_name() const;
	virtual double _get_length() const;
	virtual bool _is_monophonic() const;
	virtual double _get_bpm() const;
	virtual int32_t _get_beat_count() const;
	virtual Dictionary _get_tags() const;
	virtual TypedArray<Dictionary> _get_parameter_list() const;
	virtual bool _has_loop() const;
	virtual int32_t _get_bar_beats() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_instantiate_playback), decltype(&T::_instantiate_playback)>) {
			BIND_VIRTUAL_METHOD(T, _instantiate_playback, 3093715447);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_stream_name), decltype(&T::_get_stream_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_stream_name, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_length), decltype(&T::_get_length)>) {
			BIND_VIRTUAL_METHOD(T, _get_length, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_monophonic), decltype(&T::_is_monophonic)>) {
			BIND_VIRTUAL_METHOD(T, _is_monophonic, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_bpm), decltype(&T::_get_bpm)>) {
			BIND_VIRTUAL_METHOD(T, _get_bpm, 1740695150);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_beat_count), decltype(&T::_get_beat_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_beat_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_tags), decltype(&T::_get_tags)>) {
			BIND_VIRTUAL_METHOD(T, _get_tags, 3102165223);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_parameter_list), decltype(&T::_get_parameter_list)>) {
			BIND_VIRTUAL_METHOD(T, _get_parameter_list, 3995934104);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_loop), decltype(&T::_has_loop)>) {
			BIND_VIRTUAL_METHOD(T, _has_loop, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_bar_beats), decltype(&T::_get_bar_beats)>) {
			BIND_VIRTUAL_METHOD(T, _get_bar_beats, 3905245786);
		}
	}

public:
};

} // namespace godot

