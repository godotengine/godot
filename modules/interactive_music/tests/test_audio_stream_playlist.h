/**************************************************************************/
/*  test_audio_stream_playlist.cpp                                        */
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

#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "modules/interactive_music/audio_stream_playlist.h"

namespace TestAudioStreamPlaylist {

TEST_CASE("[Audio][AudioStreamPlaylist] Defaults") {
	Ref<AudioStreamPlaylist> playlist;
	playlist.instantiate();
	CHECK(playlist.is_valid());

	// Check default values.
	CHECK(playlist->get_fade_time() == float(0.3));
	CHECK(playlist->get_shuffle() == false);
	CHECK(playlist->has_loop() == true);
	CHECK(playlist->get_stream_count() == 0);
	CHECK(playlist->get_length() == 0.0);
	CHECK(playlist->is_meta_stream() == true);
	CHECK(playlist->get_bpm() == 0.0);

	SUBCASE("Shuffle") {
		playlist->set_shuffle(true);
		CHECK(playlist->get_shuffle());
	}

	SUBCASE("Fade time") {
		playlist->set_fade_time(0.2);
		CHECK(playlist->get_fade_time() == float(0.2));
	}

	SUBCASE("Loop") {
		playlist->set_loop(false);
		CHECK(playlist->has_loop() == false);
	}
}

TEST_CASE("[Audio][AudioStreamPlaybackPlaylist] Playback") {
	Ref<AudioStreamPlaylist> playlist;
	playlist.instantiate();
	CHECK(playlist.is_valid());

	Ref<AudioStreamPlaybackPlaylist> playback = playlist->instantiate_playback();
	CHECK(playback.is_valid());

	// Check playback defaults.
	CHECK_FALSE(playback->is_playing());
	CHECK(playback->get_loop_count() == 0);
	CHECK(playback->get_playback_position() == 0.0);
}

} // namespace TestAudioStreamPlaylist
