/*************************************************************************/
/*  video_player.cpp                                                     */
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
#include "video_player.h"

void VideoPlayer::_notification(int p_notification) {

	switch (p_notification) {

		case NOTIFICATION_ENTER_TREE: {

			//set_idle_process(false); //don't annoy
			if (stream.is_valid() && autoplay && !get_tree()->is_editor_hint())
				play();
		} break;

		case NOTIFICATION_PROCESS: {

			if (stream.is_null())
				return;
			if (paused)
				return;
			if (!stream->is_playing())
				return;

			stream->update(get_tree()->get_idle_process_time());
			int prev_width = texture->get_width();
			stream->pop_frame(texture);
			if (prev_width == 0) {
				update();
				minimum_size_changed();
			};

		} break;

		case NOTIFICATION_DRAW: {

			if (texture.is_null())
				return;
			if (texture->get_width() == 0)
				return;

			Size2 s=expand?get_size():texture->get_size();
			RID ci = get_canvas_item();
			printf("drawing with size %f, %f\n", s.x, s.y);
			draw_texture_rect(texture,Rect2(Point2(),s),false);

		} break;
	};

};

Size2 VideoPlayer::get_minimum_size() const {

	if (!expand && !texture.is_null())
		return texture->get_size();
	else
		return Size2();
}

void VideoPlayer::set_expand(bool p_expand) {

	expand=p_expand;
	update();
	minimum_size_changed();
}

bool VideoPlayer::has_expand() const {

	return expand;
}


void VideoPlayer::set_stream(const Ref<VideoStream> &p_stream) {

	stop();

	texture = Ref<ImageTexture>(memnew(ImageTexture));

	stream=p_stream;
	if (!stream.is_null()) {

		stream->set_loop(loops);
		stream->set_paused(paused);
	}

};

Ref<VideoStream> VideoPlayer::get_stream() const {

	return stream;
};

void VideoPlayer::play() {

	ERR_FAIL_COND(!is_inside_tree());
	if (stream.is_null())
		return;
	stream->play();
	set_process(true);
};

void VideoPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (stream.is_null())
		return;

	stream->stop();
	set_process(false);
};

bool VideoPlayer::is_playing() const {

	if (stream.is_null())
		return false;

	return stream->is_playing();
};

void VideoPlayer::set_paused(bool p_paused) {

	paused=p_paused;
	if (stream.is_valid()) {
		stream->set_paused(p_paused);
		set_process(!p_paused);
	};
};

bool VideoPlayer::is_paused() const {

	return paused;
};

void VideoPlayer::set_volume(float p_vol) {

	volume=p_vol;
};

float VideoPlayer::get_volume() const {

	return volume;
};

void VideoPlayer::set_volume_db(float p_db) {

	if (p_db<-79)
		set_volume(0);
	else
		set_volume(Math::db2linear(p_db));
};

float VideoPlayer::get_volume_db() const {

	if (volume==0)
		return -80;
	else
		return Math::linear2db(volume);
};


String VideoPlayer::get_stream_name() const {

	if (stream.is_null())
		return "<No Stream>";
	return stream->get_name();
};

float VideoPlayer::get_stream_pos() const {

	if (stream.is_null())
		return 0;
	return stream->get_pos();
};


void VideoPlayer::set_autoplay(bool p_enable) {

	autoplay=p_enable;
};

bool VideoPlayer::has_autoplay() const {

	return autoplay;
};

void VideoPlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_stream","stream:Stream"),&VideoPlayer::set_stream);
	ObjectTypeDB::bind_method(_MD("get_stream:Stream"),&VideoPlayer::get_stream);

	ObjectTypeDB::bind_method(_MD("play"),&VideoPlayer::play);
	ObjectTypeDB::bind_method(_MD("stop"),&VideoPlayer::stop);

	ObjectTypeDB::bind_method(_MD("is_playing"),&VideoPlayer::is_playing);

	ObjectTypeDB::bind_method(_MD("set_paused","paused"),&VideoPlayer::set_paused);
	ObjectTypeDB::bind_method(_MD("is_paused"),&VideoPlayer::is_paused);

	ObjectTypeDB::bind_method(_MD("set_volume","volume"),&VideoPlayer::set_volume);
	ObjectTypeDB::bind_method(_MD("get_volume"),&VideoPlayer::get_volume);

	ObjectTypeDB::bind_method(_MD("set_volume_db","db"),&VideoPlayer::set_volume_db);
	ObjectTypeDB::bind_method(_MD("get_volume_db"),&VideoPlayer::get_volume_db);

	ObjectTypeDB::bind_method(_MD("get_stream_name"),&VideoPlayer::get_stream_name);

	ObjectTypeDB::bind_method(_MD("get_stream_pos"),&VideoPlayer::get_stream_pos);

	ObjectTypeDB::bind_method(_MD("set_autoplay","enabled"),&VideoPlayer::set_autoplay);
	ObjectTypeDB::bind_method(_MD("has_autoplay"),&VideoPlayer::has_autoplay);

	ObjectTypeDB::bind_method(_MD("set_expand","enable"), &VideoPlayer::set_expand );
	ObjectTypeDB::bind_method(_MD("has_expand"), &VideoPlayer::has_expand );


	ADD_PROPERTY( PropertyInfo(Variant::OBJECT, "stream/stream", PROPERTY_HINT_RESOURCE_TYPE,"VideoStream"), _SCS("set_stream"), _SCS("get_stream") );
//	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/loop"), _SCS("set_loop"), _SCS("has_loop") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL, "stream/volume_db", PROPERTY_HINT_RANGE,"-80,24,0.01"), _SCS("set_volume_db"), _SCS("get_volume_db") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/autoplay"), _SCS("set_autoplay"), _SCS("has_autoplay") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/paused"), _SCS("set_paused"), _SCS("is_paused") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "expand" ), _SCS("set_expand"),_SCS("has_expand") );
}


VideoPlayer::VideoPlayer() {

	volume=1;
	loops = false;
	paused = false;
	autoplay = false;
	expand = true;
	loops = false;
};

VideoPlayer::~VideoPlayer() {

	if (stream_rid.is_valid())
		AudioServer::get_singleton()->free(stream_rid);
};

