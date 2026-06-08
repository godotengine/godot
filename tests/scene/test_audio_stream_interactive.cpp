/**************************************************************************/
/*  test_audio_stream_interactive.cpp                                     */
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

TEST_FORCE_LINK(test_audio_stream_interactive)

#include "core/object/class_db.h"
#include "modules/interactive_music/audio_stream_interactive.h"
#include "modules/interactive_music/audio_stream_playlist.h"
#include "modules/interactive_music/audio_stream_synchronized.h"

namespace TestAudioStreamInteractive {

class DummyAudioPlayback : public AudioStreamPlayback {
	GDCLASS(DummyAudioPlayback, AudioStreamPlayback);

public:
	virtual void start(double p_from_pos = 0.0) override {}
	virtual void stop() override {}
	virtual bool is_playing() const override { return true; }
	virtual int get_loop_count() const override { return 0; }
	virtual double get_playback_position() const override { return 0.0; }
	virtual void seek(double p_time) override {}
	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return p_frames;
	}
};

class DummyAudioStream : public AudioStream {
	GDCLASS(DummyAudioStream, AudioStream);

public:
	virtual Ref<AudioStreamPlayback> instantiate_playback() override {
		Ref<DummyAudioPlayback> pb;
		pb.instantiate();
		return pb;
	}
	virtual String get_stream_name() const override { return "Dummy"; }
	virtual double get_length() const override { return 1.0; }
};

TEST_CASE("[Audio][AudioStreamInteractive] Default values") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();

	CHECK(stream->get_clip_count() == 0);
	CHECK(stream->get_initial_clip() == 0);
	CHECK(stream->get_stream_name() == "Transitioner");
	CHECK(stream->get_length() == 0.0);
	CHECK(stream->is_meta_stream() == true);
}

TEST_CASE("[Audio][AudioStreamInteractive] Clip count") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();

	stream->set_clip_count(5);
	CHECK(stream->get_clip_count() == 5);

	stream->set_clip_count(0);
	CHECK(stream->get_clip_count() == 0);

	ERR_PRINT_OFF;
	stream->set_clip_count(-1);
	CHECK(stream->get_clip_count() == 0);

	stream->set_clip_count(64);
	CHECK(stream->get_clip_count() == 0);
	ERR_PRINT_ON;

	stream->set_clip_count(63);
	CHECK(stream->get_clip_count() == 63);
}

TEST_CASE("[Audio][AudioStreamInteractive] Clip name") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();

	stream->set_clip_count(3);
	stream->set_clip_name(0, "Intro");
	CHECK(stream->get_clip_name(0) == "Intro");

	CHECK(stream->get_clip_name(AudioStreamInteractive::CLIP_ANY) == "All Clips");
}

TEST_CASE("[Audio][AudioStreamInteractive] Clip stream") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	stream->set_clip_count(3);

	CHECK(stream->get_clip_stream(0).is_null());

	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<DummyAudioStream> dummy;
	dummy.instantiate();

	stream->set_clip_stream(0, dummy);
	CHECK(stream->get_clip_stream(0) == dummy);
}

TEST_CASE("[Audio][AudioStreamInteractive] Clip auto advance") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	stream->set_clip_count(3);

	CHECK(stream->get_clip_auto_advance(0) == AudioStreamInteractive::AUTO_ADVANCE_DISABLED);

	stream->set_clip_auto_advance(0, AudioStreamInteractive::AUTO_ADVANCE_ENABLED);
	CHECK(stream->get_clip_auto_advance(0) == AudioStreamInteractive::AUTO_ADVANCE_ENABLED);

	stream->set_clip_auto_advance_next_clip(0, 2);
	CHECK(stream->get_clip_auto_advance_next_clip(0) == 2);
}

TEST_CASE("[Audio][AudioStreamInteractive] Initial clip") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	stream->set_clip_count(3);

	stream->set_initial_clip(2);
	CHECK(stream->get_initial_clip() == 2);

	ERR_PRINT_OFF;
	stream->set_initial_clip(-1);
	CHECK(stream->get_initial_clip() == 2);

	stream->set_initial_clip(3);
	CHECK(stream->get_initial_clip() == 2);
	ERR_PRINT_ON;
}

TEST_CASE("[Audio][AudioStreamInteractive] Transitions") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	stream->set_clip_count(3);

	CHECK_FALSE(stream->has_transition(0, 1));

	stream->add_transition(0, 1,
			AudioStreamInteractive::TRANSITION_FROM_TIME_NEXT_BEAT,
			AudioStreamInteractive::TRANSITION_TO_TIME_START,
			AudioStreamInteractive::FADE_CROSS,
			2.5,
			true,
			2,
			true);

	CHECK(stream->has_transition(0, 1));
	CHECK(stream->get_transition_from_time(0, 1) == AudioStreamInteractive::TRANSITION_FROM_TIME_NEXT_BEAT);
	CHECK(stream->get_transition_to_time(0, 1) == AudioStreamInteractive::TRANSITION_TO_TIME_START);
	CHECK(stream->get_transition_fade_mode(0, 1) == AudioStreamInteractive::FADE_CROSS);
	CHECK(stream->get_transition_fade_beats(0, 1) == doctest::Approx(2.5));
	CHECK(stream->is_transition_using_filler_clip(0, 1) == true);
	CHECK(stream->get_transition_filler_clip(0, 1) == 2);
	CHECK(stream->is_transition_holding_previous(0, 1) == true);

	PackedInt32Array list = stream->get_transition_list();
	REQUIRE(list.size() == 2);
	CHECK(list[0] == 0);
	CHECK(list[1] == 1);

	stream->erase_transition(0, 1);
	CHECK_FALSE(stream->has_transition(0, 1));
}

TEST_CASE("[Audio][AudioStreamInteractive] Playback basic") {
	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	stream->set_clip_count(2);

	Ref<DummyAudioStream> dummy1;
	dummy1.instantiate();
	Ref<DummyAudioStream> dummy2;
	dummy2.instantiate();

	stream->set_clip_stream(0, dummy1);
	stream->set_clip_stream(1, dummy2);
	stream->set_initial_clip(0);

	Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
	REQUIRE(playback.is_valid());

	CHECK_FALSE(playback->is_playing());
	playback->start();
	CHECK(playback->is_playing());

	Ref<AudioStreamPlaybackInteractive> pb_interactive = playback;
	REQUIRE(pb_interactive.is_valid());
	CHECK(pb_interactive->get_current_clip_index() == 0);

	pb_interactive->switch_to_clip(1);
	// We mix a few frames to let the playback process the switch request in mix thread
	AudioFrame buffer[1024];
	playback->mix(buffer, 1.0, 1024);
	// The mix should trigger the queue transition
	CHECK(pb_interactive->get_current_clip_index() == 1);

	playback->stop();
	CHECK_FALSE(playback->is_playing());
}

TEST_CASE("[Audio][AudioStreamPlaylist] Default values") {
	Ref<AudioStreamPlaylist> stream;
	stream.instantiate();

	CHECK(stream->get_stream_count() == 0);
	CHECK(stream->get_shuffle() == false);
	CHECK(stream->has_loop() == true);
	CHECK(stream->get_fade_time() == doctest::Approx(0.3));
	CHECK(stream->get_stream_name() == "Playlist");
	CHECK(stream->is_meta_stream() == true);
}

TEST_CASE("[Audio][AudioStreamPlaylist] Getters and setters") {
	Ref<AudioStreamPlaylist> stream;
	stream.instantiate();

	stream->set_stream_count(5);
	CHECK(stream->get_stream_count() == 5);

	stream->set_shuffle(true);
	CHECK(stream->get_shuffle() == true);

	stream->set_loop(false);
	CHECK(stream->has_loop() == false);

	stream->set_fade_time(0.5);
	CHECK(stream->get_fade_time() == doctest::Approx(0.5));

	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<DummyAudioStream> dummy;
	dummy.instantiate();
	stream->set_list_stream(0, dummy);
	CHECK(stream->get_list_stream(0) == dummy);
}

TEST_CASE("[Audio][AudioStreamPlaylist] Stream count boundary") {
	Ref<AudioStreamPlaylist> stream;
	stream.instantiate();

	ERR_PRINT_OFF;
	stream->set_stream_count(-1);
	CHECK(stream->get_stream_count() == 0);

	stream->set_stream_count(65);
	CHECK(stream->get_stream_count() == 0);
	ERR_PRINT_ON;

	stream->set_stream_count(64);
	CHECK(stream->get_stream_count() == 64);
}

TEST_CASE("[Audio][AudioStreamPlaylist] Playback basic") {
	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<AudioStreamPlaylist> stream;
	stream.instantiate();
	stream->set_stream_count(2);

	Ref<DummyAudioStream> dummy1;
	dummy1.instantiate();
	Ref<DummyAudioStream> dummy2;
	dummy2.instantiate();

	stream->set_list_stream(0, dummy1);
	stream->set_list_stream(1, dummy2);

	Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
	REQUIRE(playback.is_valid());

	CHECK_FALSE(playback->is_playing());
	playback->start();
	CHECK(playback->is_playing());

	playback->stop();
	CHECK_FALSE(playback->is_playing());
}

TEST_CASE("[Audio][AudioStreamSynchronized] Default values") {
	Ref<AudioStreamSynchronized> stream;
	stream.instantiate();

	CHECK(stream->get_stream_count() == 0);
	CHECK(stream->get_stream_name() == "Synchronized");
	CHECK(stream->is_meta_stream() == true);
}

TEST_CASE("[Audio][AudioStreamSynchronized] Getters and setters") {
	Ref<AudioStreamSynchronized> stream;
	stream.instantiate();

	stream->set_stream_count(5);
	CHECK(stream->get_stream_count() == 5);

	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<DummyAudioStream> dummy;
	dummy.instantiate();
	stream->set_sync_stream(0, dummy);
	CHECK(stream->get_sync_stream(0) == dummy);

	stream->set_sync_stream_volume(0, -6.0);
	CHECK(stream->get_sync_stream_volume(0) == doctest::Approx(-6.0));
}

TEST_CASE("[Audio][AudioStreamSynchronized] Stream count boundary") {
	Ref<AudioStreamSynchronized> stream;
	stream.instantiate();

	ERR_PRINT_OFF;
	stream->set_stream_count(-1);
	CHECK(stream->get_stream_count() == 0);

	stream->set_stream_count(33);
	CHECK(stream->get_stream_count() == 0);
	ERR_PRINT_ON;

	stream->set_stream_count(32);
	CHECK(stream->get_stream_count() == 32);
}

TEST_CASE("[Audio][AudioStreamSynchronized] Playback basic") {
	GDREGISTER_CLASS(DummyAudioPlayback);
	GDREGISTER_CLASS(DummyAudioStream);

	Ref<AudioStreamSynchronized> stream;
	stream.instantiate();
	stream->set_stream_count(2);

	Ref<DummyAudioStream> dummy1;
	dummy1.instantiate();
	Ref<DummyAudioStream> dummy2;
	dummy2.instantiate();

	stream->set_sync_stream(0, dummy1);
	stream->set_sync_stream(1, dummy2);

	Ref<AudioStreamPlayback> playback = stream->instantiate_playback();
	REQUIRE(playback.is_valid());

	CHECK_FALSE(playback->is_playing());
	playback->start();
	CHECK(playback->is_playing());

	playback->stop();
	CHECK_FALSE(playback->is_playing());
}

} // namespace TestAudioStreamInteractive
