/**************************************************************************/
/*  test_audio_stream_wav.cpp                                             */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_audio_stream_wav)

#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "scene/resources/audio/audio_stream_wav.h"
#include "servers/audio/audio_server.h"
#include "tests/test_utils.h"

namespace TestAudioStreamWAV {

// Default wav rate for test cases.
constexpr float WAV_RATE = 44100;
/* Default wav count for test cases. 1 second of audio is used so that the file can be listened
to manually if needed. */
constexpr int WAV_COUNT = WAV_RATE;

float gen_wav(float frequency, float wav_rate, int wav_number) {
	// formula for generating a sin wave with given frequency.
	return Math::sin((Math::TAU * frequency / wav_rate) * wav_number);
}

/* Generates a 440Hz sin wave in channel 0 (mono channel or left stereo channel)
 * and a 261.63Hz wave in channel 1 (right stereo channel).
 * These waves correspond to the music notes A4 and C4 respectively.
 */
Vector<uint8_t> gen_pcm8_test(float wav_rate, int wav_count, bool stereo) {
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

// Same as gen_pcm8_test but with 16-bit wavs.
Vector<uint8_t> gen_pcm16_test(float wav_rate, int wav_count, bool stereo) {
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

void run_test(String file_name, AudioStreamWAV::Format data_format, bool stereo, float wav_rate, float wav_count) {
	String save_path = TestUtils::get_temp_path(file_name);

	Vector<uint8_t> test_data;
	if (data_format == AudioStreamWAV::FORMAT_8_BITS) {
		test_data = gen_pcm8_test(wav_rate, wav_count, stereo);
	} else {
		test_data = gen_pcm16_test(wav_rate, wav_count, stereo);
	}

	Ref<AudioStreamWAV> stream = memnew(AudioStreamWAV);
	stream->set_mix_rate(wav_rate);
	CHECK(stream->get_mix_rate() == wav_rate);

	stream->set_format(data_format);
	CHECK(stream->get_format() == data_format);

	stream->set_stereo(stereo);
	CHECK(stream->is_stereo() == stereo);

	stream->set_data(test_data);
	CHECK(stream->get_data() == test_data);

	SUBCASE("Stream length is computed properly") {
		CHECK(stream->get_length() == doctest::Approx(double(wav_count / wav_rate)));
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
	Vector<uint8_t> test_data = gen_pcm8_test(WAV_RATE, WAV_COUNT, false);
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

/* Builds a mono 16-bit PCM stream whose frames all hold distinct, easily recognizable values,
 * so that the exact frame a mixer produced can be identified from its output. */
Ref<AudioStreamWAV> gen_loop_test_stream(const Vector<int16_t> &p_samples) {
	Vector<uint8_t> data;
	data.resize(p_samples.size() * 2);
	uint8_t *write_ptr = data.ptrw();
	for (int i = 0; i < p_samples.size(); i++) {
		encode_uint16((uint16_t)p_samples[i], write_ptr + i * 2);
	}

	Ref<AudioStreamWAV> stream = memnew(AudioStreamWAV);
	stream->set_format(AudioStreamWAV::FORMAT_16_BITS);
	stream->set_data(data);
	/* Matching the server's rate keeps the resampler at a 1:1 ratio, where cubic interpolation
	   reduces to passing each source frame through unchanged. */
	stream->set_mix_rate(AudioServer::get_singleton()->get_mix_rate());
	stream->set_loop_mode(AudioStreamWAV::LOOP_FORWARD);
	return stream;
}

/* `loop_end` is inclusive: it names the last frame that is played before wrapping back to
 * `loop_begin`. A forward loop must therefore replay the whole range on every pass, and must
 * never read outside the sample data. Regression test for GH-119778. */
TEST_CASE("[Audio][AudioStreamWAV] Forward loop plays every frame of the loop range") {
	const Vector<int16_t> samples = { 24000, 18000, 12000, 6000 };
	const int frame_count = samples.size();

	// Number of leading frames to discard, covering the resampler's cubic interpolation history.
	constexpr int PRIMING = 8;
	constexpr int MIX_FRAMES = 128;

	SUBCASE("Whole-file loop replays loop_begin on every pass") {
		Ref<AudioStreamWAV> stream = gen_loop_test_stream(samples);
		stream->set_loop_begin(0);
		stream->set_loop_end(frame_count - 1);

		Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
		REQUIRE(playback.is_valid());
		playback->start();

		AudioFrame buffer[MIX_FRAMES];
		playback->mix(buffer, 1.0, MIX_FRAMES);

		// Every mixed frame must be one of the source frames, never data from outside the stream.
		for (int i = PRIMING; i < MIX_FRAMES; i++) {
			bool found = false;
			for (int j = 0; j < frame_count; j++) {
				if (Math::is_equal_approx(buffer[i].left, samples[j] / 32767.0f)) {
					found = true;
					break;
				}
			}
			CHECK_MESSAGE(found, "Mixed frame ", i, " is not one of the source frames.");
		}

		/* Every source frame must still be reachable after wrapping. Treating `loop_end` as
		   exclusive when wrapping drops `loop_begin` from every pass after the first. */
		for (int j = 0; j < frame_count; j++) {
			bool found = false;
			for (int i = PRIMING; i < MIX_FRAMES; i++) {
				if (Math::is_equal_approx(buffer[i].left, samples[j] / 32767.0f)) {
					found = true;
					break;
				}
			}
			CHECK_MESSAGE(found, "Frame ", j, " is never played after the loop wraps.");
		}

		// The loop is periodic over its whole length, not a shorter range.
		for (int i = PRIMING; i < MIX_FRAMES - frame_count; i++) {
			CHECK(Math::is_equal_approx(buffer[i].left, buffer[i + frame_count].left));
		}
	}

	SUBCASE("Sub-range loop replays loop_begin on every pass") {
		const int loop_begin = 1;
		const int loop_end = 2;
		Ref<AudioStreamWAV> stream = gen_loop_test_stream(samples);
		stream->set_loop_begin(loop_begin);
		stream->set_loop_end(loop_end);

		Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
		REQUIRE(playback.is_valid());
		playback->start();

		AudioFrame buffer[MIX_FRAMES];
		playback->mix(buffer, 1.0, MIX_FRAMES);

		const int loop_length = loop_end - loop_begin + 1;
		for (int j = loop_begin; j <= loop_end; j++) {
			bool found = false;
			for (int i = PRIMING; i < MIX_FRAMES; i++) {
				if (Math::is_equal_approx(buffer[i].left, samples[j] / 32767.0f)) {
					found = true;
					break;
				}
			}
			CHECK_MESSAGE(found, "Frame ", j, " is never played after the loop wraps.");
		}
		for (int i = PRIMING; i < MIX_FRAMES - loop_length; i++) {
			CHECK(Math::is_equal_approx(buffer[i].left, buffer[i + loop_length].left));
		}
	}

	SUBCASE("Out-of-range loop_end does not read past the sample data") {
		Ref<AudioStreamWAV> stream = gen_loop_test_stream(samples);
		stream->set_loop_begin(0);
		// One past the last valid frame, which an inclusive `loop_end` can never address.
		stream->set_loop_end(frame_count);

		Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
		REQUIRE(playback.is_valid());
		playback->start();

		AudioFrame buffer[MIX_FRAMES];
		playback->mix(buffer, 1.0, MIX_FRAMES);

		for (int i = PRIMING; i < MIX_FRAMES; i++) {
			bool found = false;
			for (int j = 0; j < frame_count; j++) {
				if (Math::is_equal_approx(buffer[i].left, samples[j] / 32767.0f)) {
					found = true;
					break;
				}
			}
			CHECK_MESSAGE(found, "Mixed frame ", i, " was read from outside the sample data.");
		}
	}
}

} // namespace TestAudioStreamWAV
