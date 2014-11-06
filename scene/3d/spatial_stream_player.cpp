/*************************************************************************/
/*  spatial_stream_player.cpp                                            */
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
#include "spatial_stream_player.h"



void SpatialStreamPlayer::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_WORLD: {

//			set_idle_process(false); //don't annoy
		} break;
		case NOTIFICATION_PROCESS: {

//			if (!stream.is_null())
//				stream->update();

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			stop(); //wathever it may be doing, stop
		} break;
	}
}



void SpatialStreamPlayer::set_stream(const Ref<AudioStream> &p_stream) {

	stop();

	stream=p_stream;
	if (!stream.is_null()) {

		stream->set_loop(loops);
	}


}

Ref<AudioStream> SpatialStreamPlayer::get_stream() const {

	return stream;
}


void SpatialStreamPlayer::play() {

	if (!is_inside_tree())
		return;
	if (stream.is_null())
		return;
	if (stream->is_playing())
		stop();
	stream->play();
	SpatialSoundServer::get_singleton()->source_set_audio_stream(get_source_rid(),stream->get_audio_stream());
	//if (stream->get_update_mode()!=AudioStream::UPDATE_NONE)
	//	set_idle_process(true);

}

void SpatialStreamPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (stream.is_null())
		return;

	SpatialSoundServer::get_singleton()->source_set_audio_stream(get_source_rid(),NULL);
	stream->stop();
	//set_idle_process(false);
}

bool SpatialStreamPlayer::is_playing() const {

	if (stream.is_null())
		return false;

	return stream->is_playing();
}

void SpatialStreamPlayer::set_loop(bool p_enable) {

	loops=p_enable;
	if (stream.is_null())
		return;
	stream->set_loop(loops);

}
bool SpatialStreamPlayer::has_loop() const {

	return loops;
}



String SpatialStreamPlayer::get_stream_name() const  {

	if (stream.is_null())
		return "<No Stream>";
	return stream->get_name();

}

int SpatialStreamPlayer::get_loop_count() const  {

	if (stream.is_null())
		return 0;
	return stream->get_loop_count();

}

float SpatialStreamPlayer::get_pos() const  {

	if (stream.is_null())
		return 0;
	return stream->get_pos();

}
void SpatialStreamPlayer::seek_pos(float p_time) {

	if (stream.is_null())
		return;
	return stream->seek_pos(p_time);

}

void SpatialStreamPlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_stream","stream:Stream"),&SpatialStreamPlayer::set_stream);
	ObjectTypeDB::bind_method(_MD("get_stream:Stream"),&SpatialStreamPlayer::get_stream);

	ObjectTypeDB::bind_method(_MD("play"),&SpatialStreamPlayer::play);
	ObjectTypeDB::bind_method(_MD("stop"),&SpatialStreamPlayer::stop);

	ObjectTypeDB::bind_method(_MD("is_playing"),&SpatialStreamPlayer::is_playing);

	ObjectTypeDB::bind_method(_MD("set_loop","enabled"),&SpatialStreamPlayer::set_loop);
	ObjectTypeDB::bind_method(_MD("has_loop"),&SpatialStreamPlayer::has_loop);

	ObjectTypeDB::bind_method(_MD("get_stream_name"),&SpatialStreamPlayer::get_stream_name);
	ObjectTypeDB::bind_method(_MD("get_loop_count"),&SpatialStreamPlayer::get_loop_count);

	ObjectTypeDB::bind_method(_MD("get_pos"),&SpatialStreamPlayer::get_pos);
	ObjectTypeDB::bind_method(_MD("seek_pos","time"),&SpatialStreamPlayer::seek_pos);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT, "stream/stream", PROPERTY_HINT_RESOURCE_TYPE,"AudioStream"), _SCS("set_stream"),_SCS("get_stream") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/loop"), _SCS("set_loop"),_SCS("has_loop") );

}


SpatialStreamPlayer::SpatialStreamPlayer() {

	loops=false;
}

SpatialStreamPlayer::~SpatialStreamPlayer() {

}


