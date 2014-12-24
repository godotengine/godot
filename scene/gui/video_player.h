/*************************************************************************/
/*  video_player.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef VIDEO_PLAYER_H
#define VIDEO_PLAYER_H

#include "scene/resources/video_stream.h"
#include "scene/gui/control.h"

class VideoPlayer : public Control {

	OBJ_TYPE(VideoPlayer,Control);

	Ref<VideoStream> stream;
	RID stream_rid;

	Ref<ImageTexture> texture;
	Image last_frame;

	bool paused;
	bool autoplay;
	float volume;
	bool expand;
	bool loops;

protected:

	static void _bind_methods();
	void _notification(int p_notification);

public:

	Size2 get_minimum_size() const;
	void set_expand(bool p_expand);
	bool has_expand() const;


	void set_stream(const Ref<VideoStream> &p_stream);
	Ref<VideoStream> get_stream() const;

	void play();
	void stop();
	bool is_playing() const;

	void set_paused(bool p_paused);
	bool is_paused() const;

	void set_volume(float p_vol);
	float get_volume() const;

	void set_volume_db(float p_db);
	float get_volume_db() const;

	String get_stream_name() const;
	float get_stream_pos() const;

	void set_autoplay(bool p_vol);
	bool has_autoplay() const;

	VideoPlayer();
	~VideoPlayer();
};

#endif
