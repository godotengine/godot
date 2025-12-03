/**************************************************************************/
/*  test_utils.cpp                                                        */
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

#include "core/io/dir_access.h"
#include "core/io/marshalls.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"

#include "scene/resources/audio_stream_wav.h"
#include "tests/test_utils.h"

String TestUtils::get_data_path(const String &p_file) {
	String data_path = "../tests/data";
	return get_executable_dir().path_join(data_path.path_join(p_file));
}

String TestUtils::get_executable_dir() {
	return OS::get_singleton()->get_executable_path().get_base_dir();
}

String TestUtils::get_temp_path(const String &p_suffix) {
	const String temp_base = OS::get_singleton()->get_cache_path().path_join("godot_test");
	DirAccess::make_dir_absolute(temp_base); // Ensure the directory exists.
	return temp_base.path_join(p_suffix);
}

float TestUtils::gen_wav(float frequency, float wav_rate, int wav_number) {
	// formula for generating a sin wave with given frequency.
	return Math::sin((Math::TAU * frequency / wav_rate) * wav_number);
}

/* Generates a 440Hz sin wave in channel 0 (mono channel or left stereo channel)
 * and a 261.63Hz wave in channel 1 (right stereo channel).
 * These waves correspond to the music notes A4 and C4 respectively.
 */
Vector<uint8_t> TestUtils::gen_pcm8(float wav_rate, int wav_count, bool stereo) {
	Vector<uint8_t> buffer;
	buffer.resize(stereo ? wav_count * 2 : wav_count);

	uint8_t *write_ptr = buffer.ptrw();
	for (int i = 0; i < buffer.size(); i++) {
		float wav;
		if (stereo) {
			if (i % 2 == 0) {
				wav = gen_wav(440, wav_rate, i / 2);
			} else {
				wav = gen_wav(261.63, wav_rate, i / 2);
			}
		} else {
			wav = gen_wav(440, wav_rate, i);
		}

		// Map sin wave to full range of 8-bit values.
		uint8_t wav_8bit = Math::fast_ftoi(((wav + 1) / 2) * UINT8_MAX);
		// Unlike the .wav format, AudioStreamWAV expects signed 8-bit wavs.
		uint8_t wav_8bit_signed = wav_8bit - (INT8_MAX + 1);
		write_ptr[i] = wav_8bit_signed;
	}

	return buffer;
}

// Same as gen_pcm8 but with 16-bit wavs.
Vector<uint8_t> TestUtils::gen_pcm16(float wav_rate, int wav_count, bool stereo) {
	Vector<uint8_t> buffer;
	buffer.resize(stereo ? wav_count * 4 : wav_count * 2);

	uint8_t *write_ptr = buffer.ptrw();
	for (int i = 0; i < buffer.size() / 2; i++) {
		float wav;
		if (stereo) {
			if (i % 2 == 0) {
				wav = gen_wav(440, wav_rate, i / 2);
			} else {
				wav = gen_wav(261.63, wav_rate, i / 2);
			}
		} else {
			wav = gen_wav(440, wav_rate, i);
		}

		// Map sin wave to full range of 16-bit values.
		uint16_t wav_16bit = Math::fast_ftoi(((wav + 1) / 2) * UINT16_MAX);
		// The .wav format expects wavs larger than 8 bits to be signed.
		uint16_t wav_16bit_signed = wav_16bit - (INT16_MAX + 1);
		encode_uint16(wav_16bit_signed, write_ptr + (i * 2));
	}

	return buffer;
}

Ref<AudioStreamWAV> TestUtils::gen_audio_stream_wav(AudioStreamWAV::Format data_format, bool stereo, double length, float wav_rate) {
	Vector<uint8_t> data;
	float wav_count = wav_rate * length;

	if (data_format == AudioStreamWAV::FORMAT_8_BITS) {
		data = gen_pcm8(wav_rate, wav_count, stereo);
	} else {
		data = gen_pcm16(wav_rate, wav_count, stereo);
	}

	Ref<AudioStreamWAV> stream_wav = memnew(AudioStreamWAV);
	stream_wav->set_mix_rate(wav_rate);
	stream_wav->set_format(data_format);
	stream_wav->set_stereo(stereo);
	stream_wav->set_data(data);

	return stream_wav;
}
