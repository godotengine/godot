/*************************************************************************/
/*  video_player.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "os/os.h"
#include "servers/audio_server.h"
/*

int VideoPlayer::InternalStream::get_channel_count() const {

	return player->sp_get_channel_count();
}
void VideoPlayer::InternalStream::set_mix_rate(int p_rate){

	return player->sp_set_mix_rate(p_rate);
}
bool VideoPlayer::InternalStream::mix(int32_t *p_buffer,int p_frames){

	return player->sp_mix(p_buffer,p_frames);
}
void VideoPlayer::InternalStream::update(){

	player->sp_update();
}
*/

int VideoPlayer::sp_get_channel_count() const {

	return playback->get_channels();
}

void VideoPlayer::sp_set_mix_rate(int p_rate) {

	server_mix_rate = p_rate;
}

bool VideoPlayer::sp_mix(int32_t *p_buffer, int p_frames) {

	if (resampler.is_ready()) {
		return resampler.mix(p_buffer, p_frames);
	}

	return false;
}

void VideoPlayer::sp_update() {
#if 0
	_THREAD_SAFE_METHOD_
	//update is unused
	if (!paused && playback.is_valid()) {

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
#endif
}

int VideoPlayer::_audio_mix_callback(void *p_udata, const int16_t *p_data, int p_frames) {

	VideoPlayer *vp = (VideoPlayer *)p_udata;

	int todo = MIN(vp->resampler.get_todo(), p_frames);

	int16_t *wb = vp->resampler.get_write_buffer();
	int c = vp->resampler.get_channel_count();

	for (int i = 0; i < todo * c; i++) {
		wb[i] = p_data[i];
	}
	vp->resampler.write(todo);
	return todo;
}

void VideoPlayer::_notification(int p_notification) {

	switch (p_notification) {

		case NOTIFICATION_ENTER_TREE: {

			if (stream.is_valid() && autoplay && !get_tree()->is_editor_hint()) {
				play();
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {

			if (stream.is_null())
				return;
			if (paused)
				return;
			if (!playback->is_playing())
				return;

			double audio_time = USEC_TO_SEC(OS::get_singleton()->get_ticks_usec()); //AudioServer::get_singleton()->get_mix_time();

			double delta = last_audio_time == 0 ? 0 : audio_time - last_audio_time;
			last_audio_time = audio_time;
			if (delta == 0)
				return;

			playback->update(delta);

		} break;

		case NOTIFICATION_DRAW: {

			if (texture.is_null())
				return;
			if (texture->get_width() == 0)
				return;

			Size2 s = expand ? get_size() : texture->get_size();
			draw_texture_rect(texture, Rect2(Point2(), s), false);

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

	expand = p_expand;
	update();
	minimum_size_changed();
}

bool VideoPlayer::has_expand() const {

	return expand;
}

void VideoPlayer::set_stream(const Ref<VideoStream> &p_stream) {

	stop();

	stream = p_stream;
	if (stream.is_valid()) {
		stream->set_audio_track(audio_track);
		playback = stream->instance_playback();
	} else {
		playback = Ref<VideoStreamPlayback>();
	}

	if (!playback.is_null()) {
		playback->set_loop(loops);
		playback->set_paused(paused);
		texture = playback->get_texture();

		const int channels = playback->get_channels();

		AudioServer::get_singleton()->lock();
		if (channels > 0)
			resampler.setup(channels, playback->get_mix_rate(), server_mix_rate, buffering_ms, 0);
		else
			resampler.clear();
		AudioServer::get_singleton()->unlock();

		if (channels > 0)
			playback->set_mix_callback(_audio_mix_callback, this);

	} else {
		texture.unref();
		AudioServer::get_singleton()->lock();
		resampler.clear();
		AudioServer::get_singleton()->unlock();
	}

	update();
};

Ref<VideoStream> VideoPlayer::get_stream() const {

	return stream;
};

void VideoPlayer::play() {

	ERR_FAIL_COND(!is_inside_tree());
	if (playback.is_null())
		return;
	playback->stop();
	playback->play();
	set_process_internal(true);
	//	AudioServer::get_singleton()->stream_set_active(stream_rid,true);
	//	AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,volume);
	last_audio_time = 0;
};

void VideoPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (playback.is_null())
		return;

	playback->stop();
	//	AudioServer::get_singleton()->stream_set_active(stream_rid,false);
	resampler.flush();
	set_process_internal(false);
	last_audio_time = 0;
};

bool VideoPlayer::is_playing() const {

	if (playback.is_null())
		return false;

	return playback->is_playing();
};

void VideoPlayer::set_paused(bool p_paused) {

	paused = p_paused;
	if (playback.is_valid()) {
		playback->set_paused(p_paused);
		set_process_internal(!p_paused);
	};
	last_audio_time = 0;
};

bool VideoPlayer::is_paused() const {

	return paused;
}

void VideoPlayer::set_buffering_msec(int p_msec) {

	buffering_ms = p_msec;
}

int VideoPlayer::get_buffering_msec() const {

	return buffering_ms;
}

void VideoPlayer::set_audio_track(int p_track) {
	audio_track = p_track;
}

int VideoPlayer::get_audio_track() const {

	return audio_track;
}

void VideoPlayer::set_volume(float p_vol) {

	volume = p_vol;
};

float VideoPlayer::get_volume() const {

	return volume;
};

void VideoPlayer::set_volume_db(float p_db) {

	if (p_db < -79)
		set_volume(0);
	else
		set_volume(Math::db2linear(p_db));
};

float VideoPlayer::get_volume_db() const {

	if (volume == 0)
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

	if (playback.is_null())
		return 0;
	return playback->get_pos();
};

Ref<Texture> VideoPlayer::get_video_texture() {

	if (playback.is_valid())
		return playback->get_texture();

	return Ref<Texture>();
}

void VideoPlayer::set_autoplay(bool p_enable) {

	autoplay = p_enable;
};

bool VideoPlayer::has_autoplay() const {

	return autoplay;
};

void VideoPlayer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stream", "stream:VideoStream"), &VideoPlayer::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream:VideoStream"), &VideoPlayer::get_stream);

	ClassDB::bind_method(D_METHOD("play"), &VideoPlayer::play);
	ClassDB::bind_method(D_METHOD("stop"), &VideoPlayer::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &VideoPlayer::is_playing);

	ClassDB::bind_method(D_METHOD("set_paused", "paused"), &VideoPlayer::set_paused);
	ClassDB::bind_method(D_METHOD("is_paused"), &VideoPlayer::is_paused);

	ClassDB::bind_method(D_METHOD("set_volume", "volume"), &VideoPlayer::set_volume);
	ClassDB::bind_method(D_METHOD("get_volume"), &VideoPlayer::get_volume);

	ClassDB::bind_method(D_METHOD("set_volume_db", "db"), &VideoPlayer::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &VideoPlayer::get_volume_db);

	ClassDB::bind_method(D_METHOD("set_audio_track", "track"), &VideoPlayer::set_audio_track);
	ClassDB::bind_method(D_METHOD("get_audio_track"), &VideoPlayer::get_audio_track);

	ClassDB::bind_method(D_METHOD("get_stream_name"), &VideoPlayer::get_stream_name);

	ClassDB::bind_method(D_METHOD("get_stream_pos"), &VideoPlayer::get_stream_pos);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enabled"), &VideoPlayer::set_autoplay);
	ClassDB::bind_method(D_METHOD("has_autoplay"), &VideoPlayer::has_autoplay);

	ClassDB::bind_method(D_METHOD("set_expand", "enable"), &VideoPlayer::set_expand);
	ClassDB::bind_method(D_METHOD("has_expand"), &VideoPlayer::has_expand);

	ClassDB::bind_method(D_METHOD("set_buffering_msec", "msec"), &VideoPlayer::set_buffering_msec);
	ClassDB::bind_method(D_METHOD("get_buffering_msec"), &VideoPlayer::get_buffering_msec);

	ClassDB::bind_method(D_METHOD("get_video_texture:Texture"), &VideoPlayer::get_video_texture);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "audio_track", PROPERTY_HINT_RANGE, "0,128,1"), "set_audio_track", "get_audio_track");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "VideoStream"), "set_stream", "get_stream");
	//ADD_PROPERTY( PropertyInfo(Variant::BOOL, "stream/loop"), "set_loop", "has_loop") ;
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE, "-80,24,0.01"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "has_autoplay");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "paused"), "set_paused", "is_paused");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand"), "set_expand", "has_expand");
}

VideoPlayer::VideoPlayer() {

	volume = 1;
	loops = false;
	paused = false;
	autoplay = false;
	expand = true;
	loops = false;

	audio_track = 0;

	buffering_ms = 500;
	server_mix_rate = 44100;

	//	internal_stream.player=this;
	//	stream_rid=AudioServer::get_singleton()->audio_stream_create(&internal_stream);
	last_audio_time = 0;
};

VideoPlayer::~VideoPlayer() {

	//	if (stream_rid.is_valid())
	//		AudioServer::get_singleton()->free(stream_rid);
	resampler.clear(); //Not necessary here, but make in consistent with other "stream_player" classes
};
