/*************************************************************************/
/*  event_player.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef EVENT_PLAYER_H
#define EVENT_PLAYER_H

#include "scene/main/node.h"
#include "scene/resources/event_stream.h"
class EventPlayer : public Node {

	OBJ_TYPE(EventPlayer, Node);

	enum {
		MAX_CHANNELS = 256
	};

	Ref<EventStreamPlayback> playback;
	Ref<EventStream> stream;
	bool paused;
	bool autoplay;
	bool loops;
	float volume;

	float tempo_scale;
	float pitch_scale;

	float channel_volume[MAX_CHANNELS];
	bool _play;
	void _set_play(bool p_play);
	bool _get_play() const;

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void set_stream(const Ref<EventStream> &p_stream);
	Ref<EventStream> get_stream() const;

	void play();
	void stop();
	bool is_playing() const;

	void set_paused(bool p_paused);
	bool is_paused() const;

	void set_loop(bool p_enable);
	bool has_loop() const;

	void set_volume(float p_vol);
	float get_volume() const;

	void set_volume_db(float p_db);
	float get_volume_db() const;

	void set_pitch_scale(float p_scale);
	float get_pitch_scale() const;

	void set_tempo_scale(float p_scale);
	float get_tempo_scale() const;

	String get_stream_name() const;

	int get_loop_count() const;

	float get_pos() const;
	void seek_pos(float p_time);
	float get_length() const;
	void set_autoplay(bool p_vol);
	bool has_autoplay() const;

	void set_channel_volume(int p_channel, float p_volume);
	float get_channel_volume(int p_channel) const;

	float get_channel_last_note_time(int p_channel) const;

	EventPlayer();
	~EventPlayer();
};

#endif // EVENT_PLAYER_H
