/**************************************************************************/
/*  movie_writer.hpp                                                      */
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
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Image;
class String;
struct Vector2i;

class MovieWriter : public Object {
	GDEXTENSION_CLASS(MovieWriter, Object)

public:
	static void add_writer(MovieWriter *p_writer);
	virtual uint32_t _get_audio_mix_rate() const;
	virtual AudioServer::SpeakerMode _get_audio_speaker_mode() const;
	virtual bool _handles_file(const String &p_path) const;
	virtual Error _write_begin(const Vector2i &p_movie_size, uint32_t p_fps, const String &p_base_path);
	virtual Error _write_frame(const Ref<Image> &p_frame_image, const void *p_audio_frame_block);
	virtual void _write_end();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_audio_mix_rate), decltype(&T::_get_audio_mix_rate)>) {
			BIND_VIRTUAL_METHOD(T, _get_audio_mix_rate, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_audio_speaker_mode), decltype(&T::_get_audio_speaker_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_audio_speaker_mode, 2549190337);
		}
		if constexpr (!std::is_same_v<decltype(&B::_handles_file), decltype(&T::_handles_file)>) {
			BIND_VIRTUAL_METHOD(T, _handles_file, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_write_begin), decltype(&T::_write_begin)>) {
			BIND_VIRTUAL_METHOD(T, _write_begin, 1866453460);
		}
		if constexpr (!std::is_same_v<decltype(&B::_write_frame), decltype(&T::_write_frame)>) {
			BIND_VIRTUAL_METHOD(T, _write_frame, 2784607037);
		}
		if constexpr (!std::is_same_v<decltype(&B::_write_end), decltype(&T::_write_end)>) {
			BIND_VIRTUAL_METHOD(T, _write_end, 3218959716);
		}
	}

public:
};

} // namespace godot

