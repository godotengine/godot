/**************************************************************************/
/*  test_audio_stream_synchronized.h                                      */
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

#include "modules/interactive_music/audio_stream_synchronized.h"
#include "scene/resources/audio_stream_wav.h"
#include "tests/test_macros.h"

namespace TestAudioStreamSynchronized {

void detach_stream_sync(Ref<AudioStreamSynchronized> audio_stream_synchronized, Ref<AudioStreamPlayback> playback_sync = nullptr) {
	for (int i = 0; i < audio_stream_synchronized->get_stream_count(); i++) {
		if (audio_stream_synchronized->get_sync_stream(i).is_valid()) {
			audio_stream_synchronized->get_sync_stream(i)->detach_from_objectdb();
		}

		if (playback_sync.is_valid()) {
			audio_stream_synchronized->get_sync_stream_playback(i)->detach_from_objectdb();
		}
	}

	if (playback_sync.is_valid()) {
		playback_sync->detach_from_objectdb();
	}

	audio_stream_synchronized->detach_from_objectdb();
}

Ref<AudioStreamSynchronized> gen_default_audio_stream_synchronized(double first_wav_length = 1, double second_wav_length = 1, bool is_looped = true) {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = memnew(AudioStreamSynchronized);
	Ref<AudioStreamWAV> stream_wav1 = TestUtils::gen_audio_stream_wav(AudioStreamWAV::FORMAT_8_BITS, true, first_wav_length);
	Ref<AudioStreamWAV> stream_wav2 = TestUtils::gen_audio_stream_wav(AudioStreamWAV::FORMAT_8_BITS, true, second_wav_length);

	if (is_looped) {
		float default_wav_rate = 44100;

		stream_wav1->set_loop_mode(AudioStreamWAV::LoopMode::LOOP_FORWARD);
		stream_wav1->set_loop_end(default_wav_rate * first_wav_length);

		stream_wav2->set_loop_mode(AudioStreamWAV::LoopMode::LOOP_FORWARD);
		stream_wav2->set_loop_end(default_wav_rate * second_wav_length);
	}

	audio_stream_synchronized->set_stream_count(2);
	audio_stream_synchronized->set_sync_stream(0, stream_wav1);
	audio_stream_synchronized->set_sync_stream(1, stream_wav2);

	return audio_stream_synchronized;
}

TEST_CASE("[Audio][AudioStreamSynchronized] Constructor") {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = memnew(AudioStreamSynchronized);
	CHECK(audio_stream_synchronized->get_bpm() == 0);
	CHECK(audio_stream_synchronized->get_beat_count() == 0);
	CHECK(audio_stream_synchronized->get_bar_beats() == 0);
	CHECK(audio_stream_synchronized->get_stream_count() == 0);
	CHECK(audio_stream_synchronized->has_loop() == false);
	CHECK(audio_stream_synchronized->get_length() == 0);
	CHECK(audio_stream_synchronized->is_meta_stream() == true);
	CHECK(audio_stream_synchronized->get_stream_name() == "Synchronized");
}

TEST_CASE("[Audio][AudioStreamSynchronized] set_sync_stream, get_sync_stream") {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = memnew(AudioStreamSynchronized);
	Ref<AudioStreamPlayback> playback_sync = audio_stream_synchronized->instantiate_playback();
	Ref<AudioStreamWAV> stream_wav1 = TestUtils::gen_audio_stream_wav(AudioStreamWAV::FORMAT_8_BITS, true);

	audio_stream_synchronized->set_stream_count(2);
	audio_stream_synchronized->set_sync_stream(0, stream_wav1);

	CHECK(audio_stream_synchronized->get_sync_stream(0)->get_instance_id() == stream_wav1->get_instance_id());

	SUBCASE("Stream should start at current position if set while playing") {
		playback_sync->start();
		playback_sync->seek(0.8);

		Ref<AudioStreamWAV> stream_wav2 = TestUtils::gen_audio_stream_wav(AudioStreamWAV::FORMAT_8_BITS, true);
		audio_stream_synchronized->set_sync_stream(1, stream_wav2);

		double wav2_expected_position = 0.8;

		Ref<AudioStreamPlayback> wav2_playback = audio_stream_synchronized->get_sync_stream_playback(1);
		CHECK(wav2_playback->get_playback_position() == doctest::Approx(wav2_expected_position));
		CHECK(audio_stream_synchronized->get_sync_stream(1)->get_instance_id() == stream_wav2->get_instance_id());

		playback_sync->stop();
	}

	detach_stream_sync(audio_stream_synchronized, playback_sync);
}

TEST_CASE("[Audio][AudioStreamSynchronized] get_length") {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = gen_default_audio_stream_synchronized(1, 1.5);

	double expected_length = 1.5;

	CHECK(audio_stream_synchronized->get_length() == expected_length);

	detach_stream_sync(audio_stream_synchronized);
}

TEST_CASE("[Audio][AudioStreamSynchronized] set_stream_count, get_stream_count") {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = gen_default_audio_stream_synchronized();

	audio_stream_synchronized->set_stream_count(5);

	CHECK(audio_stream_synchronized->get_stream_count() == 5);

	detach_stream_sync(audio_stream_synchronized);
}

TEST_CASE("[Audio][AudioStreamSynchronized][AudioStreamPlaybackSynchronized] seek, start, get_playback_position") {
	Ref<AudioStreamSynchronized> audio_stream_synchronized = gen_default_audio_stream_synchronized(1, 1.5);
	Ref<AudioStreamPlayback> playback_sync = audio_stream_synchronized->instantiate_playback();
	Ref<AudioStreamPlayback> wav1_playback = audio_stream_synchronized->get_sync_stream_playback(0);
	Ref<AudioStreamPlayback> wav2_playback = audio_stream_synchronized->get_sync_stream_playback(1);

	playback_sync->start();
	playback_sync->seek(1.3);

	double expected_position = 1.3;

	CHECK(playback_sync->get_playback_position() == doctest::Approx(expected_position));
	playback_sync->stop();

	SUBCASE("Whenever a stream has higher length than another, if the one with lower length has loop and seek value is higher than total length, it should adjust accordingly") {
		playback_sync->start();
		playback_sync->seek(1.3);

		double wav1_expected_position = 0.3;
		double wav2_expected_position = 1.3;

		CHECK(wav1_playback->get_playback_position() == doctest::Approx(wav1_expected_position));
		CHECK(wav2_playback->get_playback_position() == doctest::Approx(wav2_expected_position));

		playback_sync->stop();
	}

	SUBCASE("Same behavior as above must be applied with start") {
		playback_sync->start(1.3);

		double wav1_expected_position = 0.3;
		double wav2_expected_position = 1.3;

		CHECK(wav1_playback->get_playback_position() == doctest::Approx(wav1_expected_position));
		CHECK(wav2_playback->get_playback_position() == doctest::Approx(wav2_expected_position));

		playback_sync->stop();
	}

	detach_stream_sync(audio_stream_synchronized, playback_sync);
}

} // namespace TestAudioStreamSynchronized
