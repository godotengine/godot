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

void ScheduledAudioStreamPlayer::_mix_callback(void *p_schedule_params) {
	ERR_FAIL_NULL(p_schedule_params);
	ScheduleOnMixParams *params = static_cast<ScheduleOnMixParams *>(p_schedule_params);
	ERR_FAIL_NULL(params->self);
	params->self->_schedule_relative_playback(params);
	AudioServer::get_singleton()->remove_mix_callback(_mix_callback, p_schedule_params);
	delete params;
}

// Called on the audio thread
void ScheduledAudioStreamPlayer::_schedule_relative_playback(ScheduleOnMixParams *p_params) {
	Ref<AudioStreamPlayback> base_playback = p_params->base_playback;
	Ref<AudioStreamPlayback> new_playback = p_params->new_playback;
	ERR_FAIL_COND(base_playback.is_null() || new_playback.is_null());

	// Now that we are on the audio thread, we can safely calculate the delay
	// relative to the other playback.
	double delay = p_params->rel_time - base_playback->get_playback_position() + AudioServer::get_singleton()->get_playback_delay(base_playback);
	AudioServer::get_singleton()->set_playback_delay(new_playback, delay);
}

Ref<AudioStreamPlayback> ScheduledAudioStreamPlayer::play_delayed(double p_delay, double p_from_pos) {
	Ref<AudioStreamPlayback> stream_playback = internal->play_basic();
	if (stream_playback.is_null()) {
		return stream_playback;
	}
	AudioServer::get_singleton()->start_playback_stream(stream_playback, internal->bus, _get_volume_vector(), p_from_pos, internal->pitch_scale, p_delay);
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
	if (p_playback.is_null() || !AudioServer::get_singleton()->is_playback_active(p_playback) && !AudioServer::get_singleton()->is_playback_paused(p_playback)) {
		WARN_PRINT_ED("Could not schedule relative playback as base playback was invalid. Playing relative to current time.");
		play_delayed(p_rel_time, p_from_pos);
		return get_stream_playback();
	}

	// Start it with a large delay. This will be fixed in the mix callback.
	Ref<AudioStreamPlayback> new_playback = play_delayed(100, p_from_pos);
	if (new_playback.is_null() || new_playback->get_is_sample() && new_playback->get_sample_playback().is_valid()) {
		return new_playback;
	}

	ScheduleOnMixParams *params = new ScheduleOnMixParams();
	params->self = this;
	params->base_playback = p_playback;
	params->new_playback = new_playback;
	params->rel_time = p_rel_time;

	// Schedule the new playback to have its delay calculated in the audio
	// thread.
	//
	// FIXME: Mix callbacks are run in reverse-order of how they are added due
	// to the implementation of SafeList. This causes issues if trying to
	// "chain" the playbacks together. For example:
	// 1. Schedule playback A with delay 1
	// 2. Schedule playback B relative to A
	// 3. Schedule playback C relative to B
	// The actual order of delay calculations is A (on main thread), C (2nd
	// added callback), then B (1st added callback), which will cause the B
	// delay to be calculated incorrectly. The AudioServer needs to ensure the
	// callbacks are called in forward-order.
	AudioServer::get_singleton()->add_mix_callback(_mix_callback, params);

	return new_playback;
}

void ScheduledAudioStreamPlayer::set_delay(Ref<AudioStreamPlayback> p_playback, double p_delay) {
	if (p_playback.is_null()) {
		return;
	}
	AudioServer::get_singleton()->set_playback_delay(p_playback, p_delay);
}

void ScheduledAudioStreamPlayer::set_stop_position(Ref<AudioStreamPlayback> p_playback, double p_stop_position) {
	if (p_playback.is_null()) {
		return;
	}
	AudioServer::get_singleton()->set_playback_stop_position(p_playback, p_stop_position);
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
	ClassDB::bind_method(D_METHOD("play_delayed", "delay", "from_position"), &ScheduledAudioStreamPlayer::play_delayed, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("play_relative", "playback", "relative_time", "from_position"), &ScheduledAudioStreamPlayer::play_relative, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("set_delay", "playback", "delay"), &ScheduledAudioStreamPlayer::set_delay);
	ClassDB::bind_method(D_METHOD("set_stop_position", "playback", "stop_position"), &ScheduledAudioStreamPlayer::set_stop_position);
	ClassDB::bind_method(D_METHOD("is_playback_scheduled", "playback"), &ScheduledAudioStreamPlayer::is_playback_scheduled);
	ClassDB::bind_method(D_METHOD("cancel_scheduled_playback", "playback"), &ScheduledAudioStreamPlayer::cancel_scheduled_playback);
}
