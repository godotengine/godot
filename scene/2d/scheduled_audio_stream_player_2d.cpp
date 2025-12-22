/**************************************************************************/
/*  scheduled_audio_stream_player_2d.cpp                                  */
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

#include "scheduled_audio_stream_player_2d.h"

#include "scene/audio/audio_stream_player_internal.h"
#include "scene/resources/audio_stream_playback_scheduled.h"

Ref<AudioStreamPlaybackScheduled> ScheduledAudioStreamPlayer2D::play_scheduled(double p_abs_time, double p_from_pos) {
	Ref<AudioStreamPlaybackScheduled> stream_playback_scheduled = internal->play_scheduled_basic();
	if (stream_playback_scheduled.is_null()) {
		return stream_playback_scheduled;
	}
	stream_playback_scheduled->set_scheduled_start_time(p_abs_time);
	_play_internal(stream_playback_scheduled, p_from_pos);

	return stream_playback_scheduled;
}

void ScheduledAudioStreamPlayer2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("play_scheduled", "absolute_time", "from_position"), &ScheduledAudioStreamPlayer2D::play_scheduled, DEFVAL(0.0));
}
