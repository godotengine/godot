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
#include "tests/test_utils.h"

#include "modules/interactive_music/audio_stream_interactive.h"

namespace TestAudioStreamInteractive {

TEST_CASE("[Audio][AudioStreamInteractive] Defaults") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());

	// Check default values.
	CHECK(stream->get_clip_count() == 0);
	CHECK(stream->get_initial_clip() == 0);
	CHECK(stream->get_transition_list().is_empty());
	CHECK(stream->get_length() == 0);
	CHECK(stream->is_meta_stream() == true);

	SUBCASE("Clip count setter") {
		stream->set_clip_count(1);
		CHECK(stream->get_clip_count() == 1);
	}

	SUBCASE("Clip name") {
		const StringName clip_name = "Test Name";
		stream->set_clip_name(0, clip_name);

		CHECK_MESSAGE(stream->get_clip_name(0) == clip_name, "Clip name should have been changed.");
	}
}

TEST_CASE("[Audio][AudioStreamPlaybackInteractive] Playback") {
	Ref<AudioStreamInteractive> stream;
	stream.instantiate();
	CHECK(stream.is_valid());

	Ref<AudioStreamPlaybackInteractive> playback = stream->instantiate_playback();
	CHECK(playback.is_valid());

	// Check playback defaults.
	CHECK_FALSE(playback->is_playing());
	CHECK(playback->get_loop_count() == 0);
	CHECK(playback->get_playback_position() == 0.0);
	CHECK_MESSAGE(playback->get_current_clip_index() == -1, "Expected value is -1 as there is no current clip.");
}

} // namespace TestAudioStreamInteractive