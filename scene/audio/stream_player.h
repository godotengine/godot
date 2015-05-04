/*************************************************************************/
/*  stream_player.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef STREAM_PLAYER_H
#define STREAM_PLAYER_H

#include "scene/resources/audio_stream.h"
#include "scene/main/node.h"

class StreamPlayer : public Node {

	OBJ_TYPE(StreamPlayer,Node);

	Ref<AudioStream> stream;
	RID stream_rid;
	bool paused;
	bool autoplay;
	bool loops;
	float volume;

	bool _play;
	void _set_play(bool p_play);
	bool _get_play() const;
protected:
	void _notification(int p_what);

	static void _bind_methods();
public:

	void set_stream(const Ref<AudioStream> &p_stream);
	Ref<AudioStream> get_stream() const;

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

	String get_stream_name() const;

	int get_loop_count() const;

	float get_pos() const;
	void seek_pos(float p_time);
	float get_length() const;
	void set_autoplay(bool p_vol);
	bool has_autoplay() const;


	StreamPlayer();
	~StreamPlayer();
};

#endif // AUDIO_STREAM_PLAYER_H
