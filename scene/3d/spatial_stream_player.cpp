/*************************************************************************/
/*  spatial_stream_player.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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



int SpatialStreamPlayer::InternalStream::get_channel_count() const {

	return player->sp_get_channel_count();
}
void SpatialStreamPlayer::InternalStream::set_mix_rate(int p_rate){

	return player->sp_set_mix_rate(p_rate);
}
bool SpatialStreamPlayer::InternalStream::mix(int32_t *p_buffer,int p_frames){

	return player->sp_mix(p_buffer,p_frames);
}
void SpatialStreamPlayer::InternalStream::update(){

	player->sp_update();
}


int SpatialStreamPlayer::sp_get_channel_count() const {

	return playback->get_channels();
}

void SpatialStreamPlayer::sp_set_mix_rate(int p_rate){

	server_mix_rate=p_rate;
}

bool SpatialStreamPlayer::sp_mix(int32_t *p_buffer,int p_frames) {

	if (resampler.is_ready() && !paused) {
		return resampler.mix(p_buffer,p_frames);
	}

	return false;
}

void SpatialStreamPlayer::sp_update() {

	_THREAD_SAFE_METHOD_
	if (!paused && resampler.is_ready() && playback.is_valid()) {

		if (!playback->is_playing()) {
			//stream depleted data, but there's still audio in the ringbuffer
			//check that all this audio has been flushed before stopping the stream
			int to_mix = resampler.get_total() - resampler.get_todo();
			if (to_mix==0) {
				stop();
				return;
			}

			return;
		}

		int todo =resampler.get_todo();
		int wrote = playback->mix(resampler.get_write_buffer(),todo);
		resampler.write(wrote);
	}
}



void SpatialStreamPlayer::_notification(int p_what) {

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



void SpatialStreamPlayer::set_stream(const Ref<AudioStream> &p_stream) {

	stop();

	stream=p_stream;

	if (!stream.is_null()) {
		playback=stream->instance_playback();
		playback->set_loop(loops);
		playback->set_loop_restart_time(loop_point);
		AudioServer::get_singleton()->lock();
		resampler.setup(playback->get_channels(),playback->get_mix_rate(),server_mix_rate,buffering_ms,playback->get_minimum_buffer_size());
		AudioServer::get_singleton()->unlock();
	} else {
		AudioServer::get_singleton()->lock();
		resampler.clear();
		playback.unref();
		AudioServer::get_singleton()->unlock();
	}
}

Ref<AudioStream> SpatialStreamPlayer::get_stream() const {

	return stream;
}


void SpatialStreamPlayer::play(float p_from_offset) {

	ERR_FAIL_COND(!is_inside_tree());
	if (playback.is_null())
		return;
	if (playback->is_playing())
		stop();

	_THREAD_SAFE_METHOD_
	playback->play(p_from_offset);
	//feed the ringbuffer as long as no update callback is going on
	sp_update();

	SpatialSoundServer::get_singleton()->source_set_audio_stream(get_source_rid(),&internal_stream);

	/*
	AudioServer::get_singleton()->stream_set_active(stream_rid,true);
	AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,volume);
	if (stream->get_update_mode()!=AudioStream::UPDATE_NONE)
		set_idle_process(true);
	*/

}

void SpatialStreamPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (playback.is_null())
		return;

	_THREAD_SAFE_METHOD_
	//AudioServer::get_singleton()->stream_set_active(stream_rid,false);
	SpatialSoundServer::get_singleton()->source_set_audio_stream(get_source_rid(),NULL);
	playback->stop();
	resampler.flush();
	//set_idle_process(false);
}

bool SpatialStreamPlayer::is_playing() const {

	if (playback.is_null())
		return false;

	return playback->is_playing();
}

void SpatialStreamPlayer::set_loop(bool p_enable) {

	loops=p_enable;
	if (playback.is_null())
		return;
	playback->set_loop(loops);

}
bool SpatialStreamPlayer::has_loop() const {

	return loops;
}

void SpatialStreamPlayer::set_volume(float p_vol) {

	volume=p_vol;
	if (stream_rid.is_valid())
		AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,volume);
}

float SpatialStreamPlayer::get_volume() const {

	return volume;
}

void SpatialStreamPlayer::set_loop_restart_time(float p_secs) {

	loop_point=p_secs;
	if (playback.is_valid())
		playback->set_loop_restart_time(p_secs);
}

float SpatialStreamPlayer::get_loop_restart_time() const {

	return loop_point;
}


void SpatialStreamPlayer::set_volume_db(float p_db) {

	if (p_db<-79)
		set_volume(0);
	else
		set_volume(Math::db2linear(p_db));
}

float SpatialStreamPlayer::get_volume_db() const {

	if (volume==0)
		return -80;
	else
		return Math::linear2db(volume);
}


String SpatialStreamPlayer::get_stream_name() const  {

	if (stream.is_null())
		return "<No Stream>";
	return stream->get_name();

}

int SpatialStreamPlayer::get_loop_count() const  {

	if (playback.is_null())
		return 0;
	return playback->get_loop_count();

}

float SpatialStreamPlayer::get_pos() const  {

	if (playback.is_null())
		return 0;
	return playback->get_pos();

}

float SpatialStreamPlayer::get_length() const {

	if (playback.is_null())
		return 0;
	return playback->get_length();
}
void SpatialStreamPlayer::seek_pos(float p_time) {

	if (playback.is_null())
		return;
	return playback->seek_pos(p_time);

}

void SpatialStreamPlayer::set_autoplay(bool p_enable) {

	autoplay=p_enable;
}

bool SpatialStreamPlayer::has_autoplay() const {

	return autoplay;
}

void SpatialStreamPlayer::set_paused(bool p_paused) {

	paused=p_paused;
	/*
	if (stream.is_valid())
		stream->set_paused(p_paused);
	*/
}

bool SpatialStreamPlayer::is_paused() const {

	return paused;
}

void SpatialStreamPlayer::_set_play(bool p_play) {

	_play=p_play;
	if (is_inside_tree()) {
		if(_play)
			play();
		else
			stop();
	}

}

bool SpatialStreamPlayer::_get_play() const{

	return _play;
}

void SpatialStreamPlayer::set_buffering_msec(int p_msec) {

	buffering_ms=p_msec;
}

int SpatialStreamPlayer::get_buffering_msec() const{

	return buffering_ms;
}



void SpatialStreamPlayer::_bind_methods() {

	ClassDB::bind_method(_MD("set_stream","stream:AudioStream"),&SpatialStreamPlayer::set_stream);
	ClassDB::bind_method(_MD("get_stream:AudioStream"),&SpatialStreamPlayer::get_stream);

	ClassDB::bind_method(_MD("play","offset"),&SpatialStreamPlayer::play,DEFVAL(0));
	ClassDB::bind_method(_MD("stop"),&SpatialStreamPlayer::stop);

	ClassDB::bind_method(_MD("is_playing"),&SpatialStreamPlayer::is_playing);

	ClassDB::bind_method(_MD("set_paused","paused"),&SpatialStreamPlayer::set_paused);
	ClassDB::bind_method(_MD("is_paused"),&SpatialStreamPlayer::is_paused);

	ClassDB::bind_method(_MD("set_loop","enabled"),&SpatialStreamPlayer::set_loop);
	ClassDB::bind_method(_MD("has_loop"),&SpatialStreamPlayer::has_loop);

	ClassDB::bind_method(_MD("set_volume","volume"),&SpatialStreamPlayer::set_volume);
	ClassDB::bind_method(_MD("get_volume"),&SpatialStreamPlayer::get_volume);

	ClassDB::bind_method(_MD("set_volume_db","db"),&SpatialStreamPlayer::set_volume_db);
	ClassDB::bind_method(_MD("get_volume_db"),&SpatialStreamPlayer::get_volume_db);

	ClassDB::bind_method(_MD("set_buffering_msec","msec"),&SpatialStreamPlayer::set_buffering_msec);
	ClassDB::bind_method(_MD("get_buffering_msec"),&SpatialStreamPlayer::get_buffering_msec);

	ClassDB::bind_method(_MD("set_loop_restart_time","secs"),&SpatialStreamPlayer::set_loop_restart_time);
	ClassDB::bind_method(_MD("get_loop_restart_time"),&SpatialStreamPlayer::get_loop_restart_time);

	ClassDB::bind_method(_MD("get_stream_name"),&SpatialStreamPlayer::get_stream_name);
	ClassDB::bind_method(_MD("get_loop_count"),&SpatialStreamPlayer::get_loop_count);

	ClassDB::bind_method(_MD("get_pos"),&SpatialStreamPlayer::get_pos);
	ClassDB::bind_method(_MD("seek_pos","time"),&SpatialStreamPlayer::seek_pos);

	ClassDB::bind_method(_MD("set_autoplay","enabled"),&SpatialStreamPlayer::set_autoplay);
	ClassDB::bind_method(_MD("has_autoplay"),&SpatialStreamPlayer::has_autoplay);

	ClassDB::bind_method(_MD("get_length"),&SpatialStreamPlayer::get_length);

	ClassDB::bind_method(_MD("_set_play","play"),&SpatialStreamPlayer::_set_play);
	ClassDB::bind_method(_MD("_get_play"),&SpatialStreamPlayer::_get_play);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE,"AudioStream"), _SCS("set_stream"), _SCS("get_stream") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "play"), _SCS("_set_play"), _SCS("_get_play") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "loop"), _SCS("set_loop"), _SCS("has_loop") );
	ADD_PROPERTY( PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE,"-80,24,0.01"), _SCS("set_volume_db"), _SCS("get_volume_db") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "autoplay"), _SCS("set_autoplay"), _SCS("has_autoplay") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "paused"), _SCS("set_paused"), _SCS("is_paused") );
	ADD_PROPERTY( PropertyInfo(Variant::INT, "loop_restart_time"), _SCS("set_loop_restart_time"), _SCS("get_loop_restart_time") );
	ADD_PROPERTY( PropertyInfo(Variant::INT, "buffering_ms"), _SCS("set_buffering_msec"), _SCS("get_buffering_msec") );
}


SpatialStreamPlayer::SpatialStreamPlayer() {

	volume=1;
	loops=false;
	paused=false;
	autoplay=false;
	_play=false;
	server_mix_rate=1;
	internal_stream.player=this;
	stream_rid=AudioServer::get_singleton()->audio_stream_create(&internal_stream);
	buffering_ms=500;
	loop_point=0;

}

SpatialStreamPlayer::~SpatialStreamPlayer() {
	AudioServer::get_singleton()->free(stream_rid);
	resampler.clear();


}
