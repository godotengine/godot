/*************************************************************************/
/*  stream_player.cpp                                                    */
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
#include "stream_player.h"


void StreamPlayer::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			//set_idle_process(false); //don't annoy
			if (stream.is_valid() && autoplay && !get_tree()->is_editor_hint())
				play();
		} break;
		case NOTIFICATION_EXIT_TREE: {

			stop(); //wathever it may be doing, stop
		} break;
	}
}



void StreamPlayer::set_stream(const Ref<AudioStream> &p_stream) {

	stop();

	if (stream_rid.is_valid())
		AudioServer::get_singleton()->free(stream_rid);
	stream_rid=RID();

	stream=p_stream;
	if (!stream.is_null()) {

		stream->set_loop(loops);
		stream->set_paused(paused);
		stream_rid=AudioServer::get_singleton()->audio_stream_create(stream->get_audio_stream());
	}


}

Ref<AudioStream> StreamPlayer::get_stream() const {

	return stream;
}


void StreamPlayer::play() {

	ERR_FAIL_COND(!is_inside_tree());
	if (stream.is_null())
		return;
	if (stream->is_playing())
		stop();

	stream->play();
	AudioServer::get_singleton()->stream_set_active(stream_rid,true);
	AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,volume);
//	if (stream->get_update_mode()!=AudioStream::UPDATE_NONE)
//		set_idle_process(true);

}

void StreamPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (stream.is_null())
		return;

	AudioServer::get_singleton()->stream_set_active(stream_rid,false);
	stream->stop();
	//set_idle_process(false);
}

bool StreamPlayer::is_playing() const {

	if (stream.is_null())
		return false;

	return stream->is_playing();
}

void StreamPlayer::set_loop(bool p_enable) {

	loops=p_enable;
	if (stream.is_null())
		return;
	stream->set_loop(loops);

}
bool StreamPlayer::has_loop() const {

	return loops;
}

void StreamPlayer::set_volume(float p_vol) {

	volume=p_vol;
	if (stream_rid.is_valid())
		AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,volume);
}

float StreamPlayer::get_volume() const {

	return volume;
}

void StreamPlayer::set_volume_db(float p_db) {

	if (p_db<-79)
		set_volume(0);
	else
		set_volume(Math::db2linear(p_db));
}

float StreamPlayer::get_volume_db() const {

	if (volume==0)
		return -80;
	else
		return Math::linear2db(volume);
}


String StreamPlayer::get_stream_name() const  {

	if (stream.is_null())
		return "<No Stream>";
	return stream->get_name();

}

int StreamPlayer::get_loop_count() const  {

	if (stream.is_null())
		return 0;
	return stream->get_loop_count();

}

float StreamPlayer::get_pos() const  {

	if (stream.is_null())
		return 0;
	return stream->get_pos();

}

float StreamPlayer::get_length() const {

	if (stream.is_null())
		return 0;
	return stream->get_length();
}
void StreamPlayer::seek_pos(float p_time) {

	if (stream.is_null())
		return;
	return stream->seek_pos(p_time);

}

void StreamPlayer::set_autoplay(bool p_enable) {

	autoplay=p_enable;
}

bool StreamPlayer::has_autoplay() const {

	return autoplay;
}

void StreamPlayer::set_paused(bool p_paused) {

	paused=p_paused;
	if (stream.is_valid())
		stream->set_paused(p_paused);
}

bool StreamPlayer::is_paused() const {

	return paused;
}

void StreamPlayer::_set_play(bool p_play) {

	_play=p_play;
	if (is_inside_tree()) {
		if(_play)
			play();
		else
			stop();
	}

}

bool StreamPlayer::_get_play() const{

	return _play;
}


void StreamPlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_stream","stream:Stream"),&StreamPlayer::set_stream);
	ObjectTypeDB::bind_method(_MD("get_stream:Stream"),&StreamPlayer::get_stream);

	ObjectTypeDB::bind_method(_MD("play"),&StreamPlayer::play);
	ObjectTypeDB::bind_method(_MD("stop"),&StreamPlayer::stop);

	ObjectTypeDB::bind_method(_MD("is_playing"),&StreamPlayer::is_playing);

	ObjectTypeDB::bind_method(_MD("set_paused","paused"),&StreamPlayer::set_paused);
	ObjectTypeDB::bind_method(_MD("is_paused"),&StreamPlayer::is_paused);

	ObjectTypeDB::bind_method(_MD("set_loop","enabled"),&StreamPlayer::set_loop);
	ObjectTypeDB::bind_method(_MD("has_loop"),&StreamPlayer::has_loop);

	ObjectTypeDB::bind_method(_MD("set_volume","volume"),&StreamPlayer::set_volume);
	ObjectTypeDB::bind_method(_MD("get_volume"),&StreamPlayer::get_volume);

	ObjectTypeDB::bind_method(_MD("set_volume_db","db"),&StreamPlayer::set_volume_db);
	ObjectTypeDB::bind_method(_MD("get_volume_db"),&StreamPlayer::get_volume_db);

	ObjectTypeDB::bind_method(_MD("get_stream_name"),&StreamPlayer::get_stream_name);
	ObjectTypeDB::bind_method(_MD("get_loop_count"),&StreamPlayer::get_loop_count);

	ObjectTypeDB::bind_method(_MD("get_pos"),&StreamPlayer::get_pos);
	ObjectTypeDB::bind_method(_MD("seek_pos","time"),&StreamPlayer::seek_pos);

	ObjectTypeDB::bind_method(_MD("set_autoplay","enabled"),&StreamPlayer::set_autoplay);
	ObjectTypeDB::bind_method(_MD("has_autoplay"),&StreamPlayer::has_autoplay);

	ObjectTypeDB::bind_method(_MD("get_length"),&StreamPlayer::get_length);

	ObjectTypeDB::bind_method(_MD("_set_play","play"),&StreamPlayer::_set_play);
	ObjectTypeDB::bind_method(_MD("_get_play"),&StreamPlayer::_get_play);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT, "stream/stream", PROPERTY_HINT_RESOURCE_TYPE,"AudioStream"), _SCS("set_stream"), _SCS("get_stream") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/play"), _SCS("_set_play"), _SCS("_get_play") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/loop"), _SCS("set_loop"), _SCS("has_loop") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL, "stream/volume_db", PROPERTY_HINT_RANGE,"-80,24,0.01"), _SCS("set_volume_db"), _SCS("get_volume_db") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/autoplay"), _SCS("set_autoplay"), _SCS("has_autoplay") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/paused"), _SCS("set_paused"), _SCS("is_paused") );
}


StreamPlayer::StreamPlayer() {

	volume=1;
	loops=false;
	paused=false;
	autoplay=false;
	_play=false;
}

StreamPlayer::~StreamPlayer() {
	if (stream_rid.is_valid())
		AudioServer::get_singleton()->free(stream_rid);

}
