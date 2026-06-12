/**************************************************************************/
/*  scheduled_audio_stream_player.h                                       */
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

#include "scene/audio/audio_stream_player.h"

class ScheduledAudioStreamPlayer : public AudioStreamPlayer {
	GDCLASS(ScheduledAudioStreamPlayer, AudioStreamPlayer);

protected:
	static void _bind_methods();

public:
	Ref<AudioStreamPlayback> play_scheduled(double p_abs_time, double p_from_pos = 0.0);
	Ref<AudioStreamPlayback> play_scheduled_relative(Ref<AudioStreamPlayback> p_playback, double p_rel_time, double p_from_pos = 0.0);

	void set_scheduled_start_time(Ref<AudioStreamPlayback> p_playback, double p_abs_time);
	void set_scheduled_end_time(Ref<AudioStreamPlayback> p_playback, double p_abs_time);

	bool is_playback_scheduled(Ref<AudioStreamPlayback> p_playback);

	void cancel_scheduled_playback(Ref<AudioStreamPlayback> p_playback);
};
