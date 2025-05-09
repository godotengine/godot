/**************************************************************************/
/*  test_audio_stream_wav.h                                               */
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

#pragma once

#include "scene/resources/audio_stream_wav.h"
#include "tests/test_macros.h"

namespace TestAudioStreamWAV {

// Default wav rate for test cases.
constexpr float WAV_RATE = 44100;
/* Default wav count for test cases. 1 second of audio is used so that the file can be listened
to manually if needed. */
constexpr int WAV_COUNT = WAV_RATE;

void run_test(String file_name, AudioStreamWAV::Format data_format, bool stereo, float wav_rate, float wav_count) {
	String save_path = TestUtils::get_temp_path(file_name);
	double length = double(wav_count / wav_rate);
	Ref<AudioStreamWAV> stream = TestUtils::gen_audio_stream_wav(data_format, stereo, length, wav_rate);
	Vector<uint8_t> expected_data;

	if (data_format == AudioStreamWAV::FORMAT_8_BITS) {
		expected_data = TestUtils::gen_pcm8(wav_rate, wav_count, stereo);
	} else {
		expected_data = TestUtils::gen_pcm16(wav_rate, wav_count, stereo);
	}

	CHECK(stream->get_mix_rate() == wav_rate);
	CHECK(stream->get_format() == data_format);
	CHECK(stream->is_stereo() == stereo);
	CHECK(stream->get_data() == expected_data);

	SUBCASE("Stream length is computed properly") {
		CHECK(stream->get_length() == doctest::Approx(length));
	}

	SUBCASE("Stream can be saved as .wav") {
		REQUIRE(stream->save_to_wav(save_path) == OK);

		Error error;
		Ref<FileAccess> wav_file = FileAccess::open(save_path, FileAccess::READ, &error);
		REQUIRE(error == OK);

		Dictionary options;
		Ref<AudioStreamWAV> loaded_stream = AudioStreamWAV::load_from_file(save_path, options);

		CHECK(loaded_stream->get_format() == stream->get_format());
		CHECK(loaded_stream->get_loop_mode() == stream->get_loop_mode());
		CHECK(loaded_stream->get_loop_begin() == stream->get_loop_begin());
		CHECK(loaded_stream->get_loop_end() == stream->get_loop_end());
		CHECK(loaded_stream->get_mix_rate() == stream->get_mix_rate());
		CHECK(loaded_stream->is_stereo() == stream->is_stereo());
		CHECK(loaded_stream->get_length() == stream->get_length());
		CHECK(loaded_stream->is_monophonic() == stream->is_monophonic());
		CHECK(loaded_stream->get_data() == stream->get_data());
	}
}

TEST_CASE("[Audio][AudioStreamWAV] Mono PCM8 format") {
	run_test("test_pcm8_mono.wav", AudioStreamWAV::FORMAT_8_BITS, false, WAV_RATE, WAV_COUNT);
}

TEST_CASE("[Audio][AudioStreamWAV] Mono PCM16 format") {
	run_test("test_pcm16_mono.wav", AudioStreamWAV::FORMAT_16_BITS, false, WAV_RATE, WAV_COUNT);
}

TEST_CASE("[Audio][AudioStreamWAV] Stereo PCM8 format") {
	run_test("test_pcm8_stereo.wav", AudioStreamWAV::FORMAT_8_BITS, true, WAV_RATE, WAV_COUNT);
}

TEST_CASE("[Audio][AudioStreamWAV] Stereo PCM16 format") {
	run_test("test_pcm16_stereo.wav", AudioStreamWAV::FORMAT_16_BITS, true, WAV_RATE, WAV_COUNT);
}

TEST_CASE("[Audio][AudioStreamWAV] Alternate mix rate") {
	run_test("test_pcm16_stereo_38000Hz.wav", AudioStreamWAV::FORMAT_16_BITS, true, 38000, 38000);
}

TEST_CASE("[Audio][AudioStreamWAV] save_to_wav() adds '.wav' file extension automatically") {
	String save_path = TestUtils::get_temp_path("test_wav_extension");
	Vector<uint8_t> test_data = TestUtils::gen_pcm8(WAV_RATE, WAV_COUNT, false);
	Ref<AudioStreamWAV> stream = memnew(AudioStreamWAV);
	stream->set_data(test_data);

	REQUIRE(stream->save_to_wav(save_path) == OK);
	Error error;
	Ref<FileAccess> wav_file = FileAccess::open(save_path + ".wav", FileAccess::READ, &error);
	CHECK(error == OK);
}

TEST_CASE("[Audio][AudioStreamWAV] Default values") {
	Ref<AudioStreamWAV> stream = memnew(AudioStreamWAV);
	CHECK(stream->get_format() == AudioStreamWAV::FORMAT_8_BITS);
	CHECK(stream->get_loop_mode() == AudioStreamWAV::LOOP_DISABLED);
	CHECK(stream->get_loop_begin() == 0);
	CHECK(stream->get_loop_end() == 0);
	CHECK(stream->get_mix_rate() == 44100);
	CHECK(stream->is_stereo() == false);
	CHECK(stream->get_length() == 0);
	CHECK(stream->is_monophonic() == false);
	CHECK(stream->get_data() == Vector<uint8_t>{});
	CHECK(stream->get_stream_name() == "");
}

TEST_CASE("[Audio][AudioStreamWAV] Save empty file") {
	run_test("test_empty.wav", AudioStreamWAV::FORMAT_8_BITS, false, WAV_RATE, 0);
}

TEST_CASE("[Audio][AudioStreamWAV] Saving IMA ADPCM is not supported") {
	String save_path = TestUtils::get_temp_path("test_adpcm.wav");
	Ref<AudioStreamWAV> stream = memnew(AudioStreamWAV);
	stream->set_format(AudioStreamWAV::FORMAT_IMA_ADPCM);
	ERR_PRINT_OFF;
	CHECK(stream->save_to_wav(save_path) == ERR_UNAVAILABLE);
	ERR_PRINT_ON;
}

} // namespace TestAudioStreamWAV
