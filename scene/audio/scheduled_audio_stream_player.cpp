/**************************************************************************/
/*  scheduled_audio_stream_player.cpp                                     */
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

#include "scheduled_audio_stream_player.h"

#include "core/object/class_db.h"
#include "scene/audio/audio_stream_player_internal.h"
#include "servers/audio/audio_stream.h"

/*
TODOs:
- implement 2d/3d
*/

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_scheduled(double p_abs_time, double p_from_pos) {
	Ref<AudioStreamPlayback> stream_playback = internal->play_basic();
	if (stream_playback.is_null()) {
		return stream_playback;
	}
	AudioServer::get_singleton()->start_playback_stream(stream_playback, internal->bus, _get_volume_vector(), p_from_pos, internal->pitch_scale, p_abs_time);
	internal->ensure_playback_limit();

	// Sample handling.
	if (stream_playback->get_is_sample() && stream_playback->get_sample_playback().is_valid()) {
		WARN_PRINT_ED("Scheduled play does not support samples. Playing immediately.");

		Ref<AudioSamplePlayback> sample_playback = stream_playback->get_sample_playback();
		sample_playback->offset = p_from_pos;
		sample_playback->volume_vector = _get_volume_vector();
		sample_playback->bus = get_bus();

		AudioServer::get_singleton()->start_sample_playback(sample_playback);
	}

	return stream_playback;
}

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_scheduled_relative(Ref<AudioStreamPlayback> p_playback, double p_rel_time, double p_from_pos) {
	if (p_playback.is_null()) {
		WARN_PRINT_ED("Could not schedule relative playback as base playback was invalid. Playing relative to current time.");
		play_scheduled(AudioServer::get_singleton()->get_absolute_time() + p_rel_time, p_from_pos);
		return get_stream_playback();
	}

	// TODO: This doesn't actually work if the base playback either (1) was paused at any time, or (2) started from the middle of the stream
	double other_start_time = AudioServer::get_singleton()->get_playback_scheduled_start_time(p_playback);
	if (other_start_time == 0) {
		WARN_PRINT_ED("Could not schedule relative playback as base playback was invalid. Playing relative to current time.");
		play_scheduled(AudioServer::get_singleton()->get_absolute_time() + p_rel_time, p_from_pos);
		return get_stream_playback();
	}

	return play_scheduled(other_start_time + p_rel_time, p_from_pos);
}

void ScheduledAudioStreamPlayer::set_scheduled_start_time(Ref<AudioStreamPlayback> p_playback, double p_abs_time) {
	if (p_playback.is_null()) {
		return;
	}
	AudioServer::get_singleton()->set_playback_scheduled_start_time(p_playback, p_abs_time);
}

void ScheduledAudioStreamPlayer::set_scheduled_end_time(Ref<AudioStreamPlayback> p_playback, double p_abs_time) {
	if (p_playback.is_null()) {
		return;
	}
	AudioServer::get_singleton()->set_playback_scheduled_end_time(p_playback, p_abs_time);
}

bool ScheduledAudioStreamPlayer::is_playback_scheduled(Ref<AudioStreamPlayback> p_playback) {
	if (p_playback.is_null()) {
		return false;
	}
	return AudioServer::get_singleton()->is_playback_scheduled(p_playback);
}

void ScheduledAudioStreamPlayer::cancel_scheduled_playback(Ref<AudioStreamPlayback> p_playback) {
	if (p_playback.is_null()) {
		return;
	}
	if (is_playback_scheduled(p_playback)) {
		p_playback->stop();
		AudioServer::get_singleton()->stop_playback_stream(p_playback);
	}
}

void ScheduledAudioStreamPlayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("play_scheduled", "absolute_time", "from_position"), &ScheduledAudioStreamPlayer::play_scheduled, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("play_scheduled_relative", "playback", "relative_time", "from_position"), &ScheduledAudioStreamPlayer::play_scheduled_relative, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("set_scheduled_start_time", "playback", "absolute_time"), &ScheduledAudioStreamPlayer::set_scheduled_start_time);
	ClassDB::bind_method(D_METHOD("set_scheduled_end_time", "playback", "absolute_time"), &ScheduledAudioStreamPlayer::set_scheduled_end_time);
	ClassDB::bind_method(D_METHOD("is_playback_scheduled", "playback"), &ScheduledAudioStreamPlayer::is_playback_scheduled);
	ClassDB::bind_method(D_METHOD("cancel_scheduled_playback", "playback"), &ScheduledAudioStreamPlayer::cancel_scheduled_playback);
}
