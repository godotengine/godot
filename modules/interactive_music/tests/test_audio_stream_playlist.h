/**************************************************************************/
/*  test_audio_stream_playlist.h                                          */
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

#include "servers/audio/audio_stream.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "../audio_stream_playlist.h"

namespace TestAudio_stream_playlist {

TEST_CASE("[AudioStreamPlaylist] test_initialized_and_default") {
	Ref<AudioStreamPlaylist> playlist;
	playlist.instantiate();
	CHECK(playlist.is_valid());

	// checking default settings
	CHECK(playlist->get_stream_name() == "Playlist");
	CHECK(playlist->get_shuffle() == false);
	CHECK(playlist->get_fade_time() == float(0.3));
	CHECK(playlist->has_loop() == true);
	CHECK(playlist->get_stream_count() == 0);
	CHECK(playlist->get_length() == 0.0);
	CHECK(playlist->is_meta_stream() == true);
	CHECK(playlist->get_bpm() == 0.0);

	//using some set methods
	playlist->set_loop(false); //we want these to fail for stress testing
	CHECK(playlist->has_loop() == true);
	playlist->set_shuffle(true);
	CHECK(playlist->get_shuffle() == false);
}

TEST_CASE("[AudioStreamPlaylist] test_instantiate_playback") {
	Ref<AudioStreamPlaylist> playlist;
	playlist.instantiate();
	CHECK(playlist.is_valid());

	Ref<AudioStreamPlayback> playback = playlist->instantiate_playback();
	CHECK(playback.is_valid());
}

} // namespace TestAudio_stream_playlist
