/**************************************************************************/
/*  audio_stream_wav.hpp                                                  */
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
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;

class AudioStreamWAV : public AudioStream {
	GDEXTENSION_CLASS(AudioStreamWAV, AudioStream)

public:
	enum Format {
		FORMAT_8_BITS = 0,
		FORMAT_16_BITS = 1,
		FORMAT_IMA_ADPCM = 2,
		FORMAT_QOA = 3,
	};

	enum LoopMode {
		LOOP_DISABLED = 0,
		LOOP_FORWARD = 1,
		LOOP_PINGPONG = 2,
		LOOP_BACKWARD = 3,
	};

	static Ref<AudioStreamWAV> load_from_buffer(const PackedByteArray &p_stream_data, const Dictionary &p_options = Dictionary());
	static Ref<AudioStreamWAV> load_from_file(const String &p_path, const Dictionary &p_options = Dictionary());
	void set_data(const PackedByteArray &p_data);
	PackedByteArray get_data() const;
	void set_format(AudioStreamWAV::Format p_format);
	AudioStreamWAV::Format get_format() const;
	void set_loop_mode(AudioStreamWAV::LoopMode p_loop_mode);
	AudioStreamWAV::LoopMode get_loop_mode() const;
	void set_loop_begin(int32_t p_loop_begin);
	int32_t get_loop_begin() const;
	void set_loop_end(int32_t p_loop_end);
	int32_t get_loop_end() const;
	void set_mix_rate(int32_t p_mix_rate);
	int32_t get_mix_rate() const;
	void set_stereo(bool p_stereo);
	bool is_stereo() const;
	void set_tags(const Dictionary &p_tags);
	Dictionary get_tags() const;
	Error save_to_wav(const String &p_path);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioStream::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AudioStreamWAV::Format);
VARIANT_ENUM_CAST(AudioStreamWAV::LoopMode);

