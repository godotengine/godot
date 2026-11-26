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
#include "scene/resources/audio/audio_stream.h"
#include "servers/audio/audio_server.h"

/*
TODOs:
- implement 2d/3d
*/

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_delayed(double p_delay, double p_from_pos) {
	Ref<AudioStreamPlayback> stream_playback = internal->play_basic();
	if (stream_playback.is_null()) {
		return stream_playback;
	}

	if (!stream_playback->set_delay(p_delay)) {
		WARN_PRINT_ED("AudioStream type does not support scheduled playback. Playing immediately.");
	}

	AudioServer::get_singleton()->start_playback_stream(stream_playback, internal->bus, _get_volume_vector(), p_from_pos, internal->pitch_scale);
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

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_relative(Ref<AudioStreamPlayback> p_playback, double p_rel_time, double p_from_pos) {
	if (p_playback.is_null() || (!AudioServer::get_singleton()->is_playback_active(p_playback) && !AudioServer::get_singleton()->is_playback_paused(p_playback))) {
		WARN_PRINT_ED("Could not schedule relative playback as base playback was invalid. Playing relative to current time.");
		play_delayed(p_rel_time);
		return get_stream_playback();
	}

	// Start playback with big delay
	Ref<AudioStreamPlayback> new_playback = play_delayed(100, p_from_pos);

	// Calculate real delay with lock to ensure mixing is not happening
	AudioServer::get_singleton()->lock();
	double delay;
	if (p_rel_time < p_playback->get_playback_position()) {
		WARN_PRINT_ED(vformat("Sound was scheduled for relative time %.4f, which has already passed. Playing immediately.", p_rel_time));
		delay = 0;
	} else {
		delay = p_rel_time - p_playback->get_playback_position() + p_playback->get_delay();
	}
	new_playback->set_delay(delay);
	AudioServer::get_singleton()->unlock();

	return new_playback;
}

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_absolute(double p_abs_time, double p_from_pos) {
	// Start playback with big delay
	Ref<AudioStreamPlayback> new_playback = play_delayed(100, p_from_pos);

	// Calculate real delay with lock to ensure mixing is not happening
	AudioServer::get_singleton()->lock();
	double delay;
	if (p_abs_time < AudioServer::get_singleton()->get_absolute_time()) {
		WARN_PRINT_ED(vformat("Sound was scheduled for absolute time %.4f, which has already passed. Playing immediately.", p_abs_time));
		delay = 0;
	} else {
		delay = p_abs_time - AudioServer::get_singleton()->get_absolute_time();
	}
	new_playback->set_delay(delay);
	AudioServer::get_singleton()->unlock();

	return new_playback;
}

void ScheduledAudioStreamPlayer::set_delay(Ref<AudioStreamPlayback> p_playback, double p_delay) {
	if (p_playback.is_null()) {
		return;
	}
	if (!p_playback->set_delay(p_delay)) {
		WARN_PRINT_ED("AudioStreamPlayback type does not support scheduled playback.");
	}
}

void ScheduledAudioStreamPlayer::set_stop_position(Ref<AudioStreamPlayback> p_playback, double p_stop_position) {
	if (p_playback.is_null()) {
		return;
	}
	if (p_stop_position < p_playback->get_playback_position()) {
		WARN_PRINT_ED(vformat("Sound was scheduled to stop at relative time %.4f, which has already passed. Stopping immediately.", p_stop_position));
	}
	if (!p_playback->set_end_position(p_stop_position)) {
		WARN_PRINT_ED("AudioStreamPlayback type does not support scheduled playback.");
	}
}

void ScheduledAudioStreamPlayer::set_start_time_absolute(Ref<AudioStreamPlayback> p_playback, double p_abs_time) {
	if (p_playback.is_null()) {
		return;
	}

	// Calculate delay with lock to ensure mixing is not happening
	AudioServer::get_singleton()->lock();
	double delay;
	if (p_abs_time < AudioServer::get_singleton()->get_absolute_time()) {
		WARN_PRINT_ED(vformat("Sound was scheduled for absolute time %.4f, which has already passed. Playing immediately.", p_abs_time));
		delay = 0;
	} else {
		delay = p_abs_time - AudioServer::get_singleton()->get_absolute_time();
	}
	if (!p_playback->set_delay(delay)) {
		WARN_PRINT_ED("AudioStreamPlayback type does not support scheduled playback.");
	}
	AudioServer::get_singleton()->unlock();
}

void ScheduledAudioStreamPlayer::set_end_time_absolute(Ref<AudioStreamPlayback> p_playback, double p_abs_time) {
	if (p_playback.is_null()) {
		return;
	}

	// Calculate delay with lock to ensure mixing is not happening
	AudioServer::get_singleton()->lock();
	double stop_position;
	if (p_abs_time < AudioServer::get_singleton()->get_absolute_time()) {
		WARN_PRINT_ED(vformat("Sound was scheduled to stop at absolute time %.4f, which has already passed. Stopping immediately.", p_abs_time));
		stop_position = 0;
	} else {
		stop_position = p_abs_time - AudioServer::get_singleton()->get_absolute_time() - p_playback->get_delay();
	}
	if (!p_playback->set_end_position(stop_position)) {
		WARN_PRINT_ED("AudioStreamPlayback type does not support scheduled playback.");
	}
	AudioServer::get_singleton()->unlock();
}

bool ScheduledAudioStreamPlayer::is_playback_scheduled(Ref<AudioStreamPlayback> p_playback) {
	if (p_playback.is_null()) {
		return false;
	}
	return p_playback->get_delay() > 0;
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
	ClassDB::bind_method(D_METHOD("play_delayed", "delay", "from_position"), &ScheduledAudioStreamPlayer::play_delayed, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("play_relative", "playback", "relative_time", "from_position"), &ScheduledAudioStreamPlayer::play_relative, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("play_absolute", "absolute_time", "from_position"), &ScheduledAudioStreamPlayer::play_absolute, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("set_delay", "playback", "delay"), &ScheduledAudioStreamPlayer::set_delay);
	ClassDB::bind_method(D_METHOD("set_stop_position", "playback", "stop_position"), &ScheduledAudioStreamPlayer::set_stop_position);
	ClassDB::bind_method(D_METHOD("set_start_time_absolute", "playback", "absolute_time"), &ScheduledAudioStreamPlayer::set_start_time_absolute);
	ClassDB::bind_method(D_METHOD("set_end_time_absolute", "playback", "absolute_time"), &ScheduledAudioStreamPlayer::set_end_time_absolute);
	ClassDB::bind_method(D_METHOD("is_playback_scheduled", "playback"), &ScheduledAudioStreamPlayer::is_playback_scheduled);
	ClassDB::bind_method(D_METHOD("cancel_scheduled_playback", "playback"), &ScheduledAudioStreamPlayer::cancel_scheduled_playback);
}
