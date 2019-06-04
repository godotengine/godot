/*************************************************************************/
/*  event_player.cpp                                                     */
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
#include "event_player.h"

void EventPlayer::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			//set_idle_process(false); //don't annoy
			if (playback.is_valid() && autoplay && !get_tree()->is_editor_hint())
				play();
		} break;
		case NOTIFICATION_EXIT_TREE: {

			stop(); //wathever it may be doing, stop
		} break;
	}
}

void EventPlayer::set_stream(const Ref<EventStream> &p_stream) {

	stop();
	stream = p_stream;
	if (stream.is_valid())
		playback = stream->instance_playback();
	else
		playback.unref();

	if (playback.is_valid()) {

		playback->set_loop(loops);
		playback->set_paused(paused);
		playback->set_volume(volume);
		for (int i = 0; i < (MIN(MAX_CHANNELS, stream->get_channel_count())); i++)
			playback->set_channel_volume(i, channel_volume[i]);
	}
}

Ref<EventStream> EventPlayer::get_stream() const {

	return stream;
}

void EventPlayer::play() {

	ERR_FAIL_COND(!is_inside_tree());
	if (playback.is_null()) {
		return;
	}
	if (playback->is_playing()) {
		AudioServer::get_singleton()->lock();
		stop();
		AudioServer::get_singleton()->unlock();
	}

	AudioServer::get_singleton()->lock();
	playback->play();
	AudioServer::get_singleton()->unlock();
}

void EventPlayer::stop() {

	if (!is_inside_tree())
		return;
	if (playback.is_null())
		return;

	AudioServer::get_singleton()->lock();
	playback->stop();
	AudioServer::get_singleton()->unlock();
}

bool EventPlayer::is_playing() const {

	if (playback.is_null())
		return false;

	return playback->is_playing();
}

void EventPlayer::set_loop(bool p_enable) {

	loops = p_enable;
	if (playback.is_null())
		return;
	playback->set_loop(loops);
}
bool EventPlayer::has_loop() const {

	return loops;
}

void EventPlayer::set_volume(float p_volume) {

	volume = p_volume;
	if (playback.is_valid())
		playback->set_volume(volume);
}

float EventPlayer::get_volume() const {

	return volume;
}

void EventPlayer::set_volume_db(float p_db) {

	if (p_db < -79)
		set_volume(0);
	else
		set_volume(Math::db2linear(p_db));
}

float EventPlayer::get_volume_db() const {

	if (volume == 0)
		return -80;
	else
		return Math::linear2db(volume);
}

void EventPlayer::set_pitch_scale(float p_pitch_scale) {

	pitch_scale = p_pitch_scale;
	if (playback.is_valid())
		playback->set_pitch_scale(pitch_scale);
}

float EventPlayer::get_pitch_scale() const {

	return pitch_scale;
}

void EventPlayer::set_tempo_scale(float p_tempo_scale) {

	tempo_scale = p_tempo_scale;
	if (playback.is_valid())
		playback->set_tempo_scale(tempo_scale);
}

float EventPlayer::get_tempo_scale() const {

	return tempo_scale;
}

String EventPlayer::get_stream_name() const {

	if (stream.is_null())
		return "<No Stream>";
	return stream->get_name();
}

int EventPlayer::get_loop_count() const {

	if (playback.is_null())
		return 0;
	return playback->get_loop_count();
}

float EventPlayer::get_pos() const {

	if (playback.is_null())
		return 0;
	return playback->get_pos();
}

float EventPlayer::get_length() const {

	if (stream.is_null())
		return 0;
	return stream->get_length();
}
void EventPlayer::seek_pos(float p_time) {

	if (playback.is_null())
		return;
	return playback->seek_pos(p_time);
}

void EventPlayer::set_autoplay(bool p_enable) {

	autoplay = p_enable;
}

bool EventPlayer::has_autoplay() const {

	return autoplay;
}

void EventPlayer::set_paused(bool p_paused) {

	paused = p_paused;
	if (playback.is_valid())
		playback->set_paused(p_paused);
}

bool EventPlayer::is_paused() const {

	return paused;
}

void EventPlayer::_set_play(bool p_play) {

	_play = p_play;
	if (is_inside_tree()) {
		if (_play)
			play();
		else
			stop();
	}
}

bool EventPlayer::_get_play() const {

	return _play;
}

void EventPlayer::set_channel_volume(int p_channel, float p_volume) {

	ERR_FAIL_INDEX(p_channel, MAX_CHANNELS);
	channel_volume[p_channel] = p_volume;
	if (playback.is_valid())
		playback->set_channel_volume(p_channel, p_volume);
}

float EventPlayer::get_channel_volume(int p_channel) const {

	ERR_FAIL_INDEX_V(p_channel, MAX_CHANNELS, 0);
	return channel_volume[p_channel];
}

float EventPlayer::get_channel_last_note_time(int p_channel) const {

	if (playback.is_valid())
		return playback->get_last_note_time(p_channel);

	return 0;
}

void EventPlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_stream", "stream:EventStream"), &EventPlayer::set_stream);
	ObjectTypeDB::bind_method(_MD("get_stream:EventStream"), &EventPlayer::get_stream);

	ObjectTypeDB::bind_method(_MD("play"), &EventPlayer::play);
	ObjectTypeDB::bind_method(_MD("stop"), &EventPlayer::stop);

	ObjectTypeDB::bind_method(_MD("is_playing"), &EventPlayer::is_playing);

	ObjectTypeDB::bind_method(_MD("set_paused", "paused"), &EventPlayer::set_paused);
	ObjectTypeDB::bind_method(_MD("is_paused"), &EventPlayer::is_paused);

	ObjectTypeDB::bind_method(_MD("set_loop", "enabled"), &EventPlayer::set_loop);
	ObjectTypeDB::bind_method(_MD("has_loop"), &EventPlayer::has_loop);

	ObjectTypeDB::bind_method(_MD("set_volume", "volume"), &EventPlayer::set_volume);
	ObjectTypeDB::bind_method(_MD("get_volume"), &EventPlayer::get_volume);

	ObjectTypeDB::bind_method(_MD("set_pitch_scale", "pitch_scale"), &EventPlayer::set_pitch_scale);
	ObjectTypeDB::bind_method(_MD("get_pitch_scale"), &EventPlayer::get_pitch_scale);

	ObjectTypeDB::bind_method(_MD("set_tempo_scale", "tempo_scale"), &EventPlayer::set_tempo_scale);
	ObjectTypeDB::bind_method(_MD("get_tempo_scale"), &EventPlayer::get_tempo_scale);

	ObjectTypeDB::bind_method(_MD("set_volume_db", "db"), &EventPlayer::set_volume_db);
	ObjectTypeDB::bind_method(_MD("get_volume_db"), &EventPlayer::get_volume_db);

	ObjectTypeDB::bind_method(_MD("get_stream_name"), &EventPlayer::get_stream_name);
	ObjectTypeDB::bind_method(_MD("get_loop_count"), &EventPlayer::get_loop_count);

	ObjectTypeDB::bind_method(_MD("get_pos"), &EventPlayer::get_pos);
	ObjectTypeDB::bind_method(_MD("seek_pos", "time"), &EventPlayer::seek_pos);

	ObjectTypeDB::bind_method(_MD("get_length"), &EventPlayer::get_length);

	ObjectTypeDB::bind_method(_MD("set_autoplay", "enabled"), &EventPlayer::set_autoplay);
	ObjectTypeDB::bind_method(_MD("has_autoplay"), &EventPlayer::has_autoplay);

	ObjectTypeDB::bind_method(_MD("set_channel_volume", "channel", "channel_volume"), &EventPlayer::set_channel_volume);
	ObjectTypeDB::bind_method(_MD("get_channel_volume", "channel"), &EventPlayer::get_channel_volume);
	ObjectTypeDB::bind_method(_MD("get_channel_last_note_time", "channel"), &EventPlayer::get_channel_last_note_time);

	ObjectTypeDB::bind_method(_MD("_set_play", "play"), &EventPlayer::_set_play);
	ObjectTypeDB::bind_method(_MD("_get_play"), &EventPlayer::_get_play);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream/stream", PROPERTY_HINT_RESOURCE_TYPE, "EventStream"), _SCS("set_stream"), _SCS("get_stream"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream/play"), _SCS("_set_play"), _SCS("_get_play"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream/loop"), _SCS("set_loop"), _SCS("has_loop"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "stream/volume_db", PROPERTY_HINT_RANGE, "-80,24,0.01"), _SCS("set_volume_db"), _SCS("get_volume_db"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "stream/pitch_scale", PROPERTY_HINT_RANGE, "0.001,16,0.001"), _SCS("set_pitch_scale"), _SCS("get_pitch_scale"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "stream/tempo_scale", PROPERTY_HINT_RANGE, "0.001,16,0.001"), _SCS("set_tempo_scale"), _SCS("get_tempo_scale"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream/autoplay"), _SCS("set_autoplay"), _SCS("has_autoplay"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream/paused"), _SCS("set_paused"), _SCS("is_paused"));
}

EventPlayer::EventPlayer() {

	volume = 1;
	loops = false;
	paused = false;
	autoplay = false;
	_play = false;
	pitch_scale = 1.0;
	tempo_scale = 1.0;
	for (int i = 0; i < MAX_CHANNELS; i++)
		channel_volume[i] = 1.0;
}

EventPlayer::~EventPlayer() {
}
