/**************************************************************************/
/*  test_audio_stream_interactive.h                                       */
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

#include "../audio_stream_interactive.h"

namespace TestAudio_stream_interactive {

TEST_CASE("[AudioStreamInteractive] test_is_initialized_and_default") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());

	//checking default values
	CHECK(stream->get_stream_name() == "Transitioner");
}

TEST_CASE("[AudioStreamInteractive] test_clip_count") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());
	CHECK(stream->get_initial_clip() == 0);

	CHECK(stream->get_clip_count() == 0);
}

TEST_CASE("[AudioStreamInteractive] test_clip_name") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());

	//checking setting and getting clip name
	StringName name = "file1";
	stream->set_clip_name(0, name);
	CHECK(stream->get_clip_name(0) == name);
}

TEST_CASE("[AudioStreamInteractive] test_instantiate_playback") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());

	Ref<AudioStreamPlaybackInteractive> playback = stream->instantiate_playback();
	CHECK(playback.is_valid());
}

TEST_CASE("[AudioStreamPlaybackInteractive] test_playback") {
	Ref<AudioStreamPlaybackInteractive> playback;
	playback.instantiate();
	CHECK(playback.is_valid());

	CHECK(playback->is_playing() == false);
	CHECK(playback->get_loop_count() == 0);
	CHECK(playback->get_playback_position() == 0.0);
}

} // namespace TestAudio_stream_interactive
