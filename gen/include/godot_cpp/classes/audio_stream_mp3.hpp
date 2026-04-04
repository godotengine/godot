/**************************************************************************/
/*  audio_stream_mp3.hpp                                                  */
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

#include <godot_cpp/classes/audio_stream.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;

class AudioStreamMP3 : public AudioStream {
	GDEXTENSION_CLASS(AudioStreamMP3, AudioStream)

public:
	static Ref<AudioStreamMP3> load_from_buffer(const PackedByteArray &p_stream_data);
	static Ref<AudioStreamMP3> load_from_file(const String &p_path);
	void set_data(const PackedByteArray &p_data);
	PackedByteArray get_data() const;
	void set_loop(bool p_enable);
	bool has_loop() const;
	void set_loop_offset(double p_seconds);
	double get_loop_offset() const;
	void set_bpm(double p_bpm);
	double get_bpm() const;
	void set_beat_count(int32_t p_count);
	int32_t get_beat_count() const;
	void set_bar_beats(int32_t p_count);
	int32_t get_bar_beats() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioStream::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

